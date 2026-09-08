/* ----------------------------------------------------------------------
   OpenEdge extension of SPARTA UpdateKokkos.
   Adds the OpenEdge Boris pusher (E/B from plasma compute or fix
   background mesh views) and the spatial-sheath device machinery.
   This file is the canonical Kokkos update implementation.
------------------------------------------------------------------------- */

#ifndef SPARTA_UPDATE_KOKKOS_H
#define SPARTA_UPDATE_KOKKOS_H

#include "update.h"
#include "kokkos_type.h"
#include "particle.h"
#include "grid_kokkos.h"
#include "domain_kokkos.h"
#include "kokkos_copy.h"
#include "surf_collide_diffuse_kokkos.h"
#include "surf_collide_specular_kokkos.h"
#include "surf_collide_vanish_kokkos.h"
#include "surf_collide_piston_kokkos.h"
#include "surf_collide_transparent_kokkos.h"
#include "surf_collide_toroidal_kokkos.h"
#include "gca_kokkos.h"
#include "compute_boundary_kokkos.h"
#include "compute_surf_kokkos.h"
#include "pusher_kokkos.h"
#include "sheath_models_kokkos.h"

namespace SPARTA_NS {

#define KOKKOS_MAX_SURF_COLL_PER_TYPE 2
#define KOKKOS_MAX_TOT_SURF_COLL 10
#define KOKKOS_MAX_BLIST 2
#define KOKKOS_MAX_SLIST 3

struct s_UPDATE_REDUCE {
  int ntouch_one,nexit_one,nboundary_one,
      entryexit,ncomm_one,
      nscheck_one,nscollide_one,nreact_one,nstuck,
      naxibad,error_flag,ncaplost;
  KOKKOS_INLINE_FUNCTION
  s_UPDATE_REDUCE() {
    ntouch_one = nexit_one = nboundary_one = ncomm_one = 0;
    nscheck_one = nscollide_one = nreact_one = nstuck = naxibad = 0;
    ncaplost = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_UPDATE_REDUCE &rhs) {
    ntouch_one += rhs.ntouch_one; nexit_one += rhs.nexit_one;
    nboundary_one += rhs.nboundary_one; ncomm_one += rhs.ncomm_one;
    nscheck_one += rhs.nscheck_one; nscollide_one += rhs.nscollide_one;
    nreact_one += rhs.nreact_one; nstuck += rhs.nstuck; naxibad += rhs.naxibad;
    ncaplost += rhs.ncaplost;
  }
};
typedef struct s_UPDATE_REDUCE UPDATE_REDUCE;

template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
struct TagUpdateMove{};

// OpenEdge gate 9b: device plasma-cache fill (retires the per-step
// host pcache loop and its full particle+custom D2H/H2D round-trip)
struct TagUpdatePcacheFill{};

class UpdateKokkos : public Update {
 public:
  typedef UPDATE_REDUCE value_type;

  DAT::tdual_int_1d k_mlist;
  DAT::tdual_int_1d k_mlist_small;

  UpdateKokkos(class SPARTA *);
  ~UpdateKokkos();
  void init();
  void setup();
  void run(int);

  template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int&) const;

  template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int&, UPDATE_REDUCE&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagUpdatePcacheFill, const int&) const;

 private:

  double dt;
  int field_active[3];

  double dx,dy,dz,Lx,Ly,Lz;
  double xlo,ylo,zlo,xhi,yhi,zhi;
  int ncx,ncy,ncz;
  GridKokkos::hash_type hash_kk;

  t_cell_1d d_cells;
  t_sinfo_1d d_sinfo;
  t_pcell_1d d_pcells;

  Kokkos::Crs<int, DeviceType, void, int> d_csurfs;
  Kokkos::Crs<int, DeviceType, void, int> d_csplits;
  Kokkos::Crs<int, DeviceType, void, int> d_csubs;

  t_line_1d d_lines;
  t_tri_1d d_tris;
  t_particle_1d d_particles;

  // cross-field diffusion displacements. Host fix path: mirrors
  // Update::dx_cd, uploaded each step in run(). Device fix path
  // (cross_field_diffusion/kk): filled in place by the fix's kernel
  // (oe_cd_dev = 1) and the upload is skipped.
  int oe_cd_dev;
  DAT::t_float_2d_lr d_dx_cd;
  DAT::t_float_2d_lr::HostMirror h_dx_cd;
  t_species_1d d_species;  // OpenEdge: species data for charge/mass lookup

  // Base SPARTA field perturbation (fstyle)
  DAT::t_float_2d_lr d_fieldfix_array_particle;
  DAT::t_float_2d_lr d_fieldfix_array_grid;
  class KokkosBase* KKBaseFieldFix;

  // OpenEdge: plasma compute device view (bypass field fixes)
  // Boris kernel reads B directly from compute columns
  DAT::t_float_2d_lr d_oe_plasma_compute;
  class KokkosBase *oe_plasma_kkbase;  // live source of d_oe_plasma_compute
  int oe_bx_col, oe_by_col, oe_bz_col;  // column indices for B in compute

  // OpenEdge: device-resident equilibrium psi map (Phase A of pusher port).
  // Filled by binding to ComputePlasmaFieldsKokkos at init time. When
  // oe_has_equilibrium = 1, oe_boris3d uses bilinear point-query off the
  // psi grid; otherwise it falls back to cell-center reads from
  // d_oe_plasma_compute.
  DAT::t_float_1d d_oe_equ_r;
  DAT::t_float_1d d_oe_equ_z;
  DAT::t_float_2d_lr d_oe_equ_psi;
  double oe_equ_btf, oe_equ_rtf;
  int oe_equ_jm, oe_equ_km;
  int oe_has_equilibrium;
  // native equilibrium B maps (slag b05b4687): preferred over
  // psi-derived B when present, matching the CPU equ_bfield_at chain
  int oe_has_equ_bmaps;
  // constant-B fix background (no mesh, no psi map): 1 = cylindrical
  // (const_br, const_bz, const_bt), 2 = Cartesian const_bcart
  int oe_has_const_b;
  double oe_const_br, oe_const_bz, oe_const_bt, oe_const_bcart[3];
  void bind_oe_equ_from_fix(class FixBackground *pd);
  void bind_oe_psi();

  // fix reflect/psi (psi-contour core boundary) on the device mover:
  // bilinear normalized-psi map copied from the fix, CPU-identical
  // bisection crossing + specular reflection, per-species absorb tallies
  int oe_psi_on, oe_psi_action, oe_psi_imix, oe_psi_nw, oe_psi_nh, oe_pw_slot_on;
  double oe_psi_thr, oe_psi_axis, oe_psi_b;
  DAT::t_float_1d d_oe_psi_r, d_oe_psi_z, d_oe_psi_map, d_oe_pw;
  DAT::t_int_2d d_oe_s2g;
  Kokkos::View<double*, DeviceType> d_oe_psi_ev, d_oe_psi_ph;
  Kokkos::View<double*, DeviceType>::HostMirror h_oe_psi_ev, h_oe_psi_ph;
  Kokkos::View<int, DeviceType> d_oe_psi_bad;
  Kokkos::View<int, DeviceType>::HostMirror h_oe_psi_bad;

  // pusher switch_log_file on the device hybrid: bounded per-pass event
  // buffer (id, oldmode, newmode, reason code, d_start, d_end, d_sw,
  // e_pre, e_post, replay) drained on the host through Pusher::log_switch
  int oe_swlog_on, oe_swlog_cap;
  Kokkos::View<double*[10], DeviceType> d_oe_swlog;
  Kokkos::View<double*[10], DeviceType>::HostMirror h_oe_swlog;
  Kokkos::View<int, DeviceType> d_oe_swlog_n;
  Kokkos::View<int, DeviceType>::HostMirror h_oe_swlog_n;
  KOKKOS_INLINE_FUNCTION
  void oe_swlog_push(int id, int oldmode, int newmode, int reason,
                     double d_start, double d_end, double d_sw,
                     double e_pre, double e_post, int replay) const {
    const int k = Kokkos::atomic_fetch_add(&d_oe_swlog_n(), 1);
    if (k >= oe_swlog_cap) return;      // overflow counted by the host
    d_oe_swlog(k,0) = (double) id;  d_oe_swlog(k,1) = oldmode; d_oe_swlog(k,2) = newmode;
    d_oe_swlog(k,3) = reason;       d_oe_swlog(k,4) = d_start; d_oe_swlog(k,5) = d_end;
    d_oe_swlog(k,6) = d_sw;         d_oe_swlog(k,7) = e_pre;   d_oe_swlog(k,8) = e_post;
    d_oe_swlog(k,9) = replay;
  }

  KOKKOS_INLINE_FUNCTION
  int oe_psi_bracket(const DAT::t_float_1d &g, int n, double x) const {
    if (x <= g(0)) return 0;
    if (x >= g(n-1)) return n-2;
    int lo = 0, hi = n-1;               // largest i with g(i) <= x
    while (hi - lo > 1) { const int mid = (lo+hi)/2; if (g(mid) <= x) lo = mid; else hi = mid; }
    if (lo > n-2) lo = n-2;
    return lo;
  }
  KOKKOS_INLINE_FUNCTION
  void oe_psi_rz(const double *xyz, double &R, double &Z) const {
    if (oe_dim == 3) { R = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]); Z = xyz[2]; }
    else if (oe_axisymmetric) { Z = xyz[0]; R = xyz[1]; }
    else { R = xyz[0]; Z = xyz[1]; }
  }
  // CPU FixReflectPsi::psi_norm_gradient: returns normalized psi, fills gradients
  KOKKOS_INLINE_FUNCTION
  double oe_psi_norm_grad(double R, double Z, double &gR, double &gZ) const {
    gR = gZ = 0.0;
    const int nw = oe_psi_nw, nh = oe_psi_nh;
    const double Rc = Kokkos::fmin(Kokkos::fmax(R, d_oe_psi_r(0)), d_oe_psi_r(nw-1));
    const double Zc = Kokkos::fmin(Kokkos::fmax(Z, d_oe_psi_z(0)), d_oe_psi_z(nh-1));
    const int i = oe_psi_bracket(d_oe_psi_r, nw, Rc);
    const int j = oe_psi_bracket(d_oe_psi_z, nh, Zc);
    const double dr = d_oe_psi_r(i+1) - d_oe_psi_r(i);
    const double dz = d_oe_psi_z(j+1) - d_oe_psi_z(j);
    const double dpsi = oe_psi_b - oe_psi_axis;
    if (Kokkos::fabs(dr) < 1.0e-30 || Kokkos::fabs(dz) < 1.0e-30 ||
        Kokkos::fabs(dpsi) < 1.0e-30) return 1.0;
    const double t = Kokkos::fmin(Kokkos::fmax((Rc - d_oe_psi_r(i))/dr, 0.0), 1.0);
    const double u = Kokkos::fmin(Kokkos::fmax((Zc - d_oe_psi_z(j))/dz, 0.0), 1.0);
    const double p00 = d_oe_psi_map(j*nw+i),     p10 = d_oe_psi_map(j*nw+i+1);
    const double p01 = d_oe_psi_map((j+1)*nw+i), p11 = d_oe_psi_map((j+1)*nw+i+1);
    const double psi = (1.0-t)*(1.0-u)*p00 + t*(1.0-u)*p10 + (1.0-t)*u*p01 + t*u*p11;
    gR = ((1.0-u)*(p10-p00) + u*(p11-p01)) / (dr*dpsi);
    gZ = ((1.0-t)*(p01-p00) + t*(p11-p10)) / (dz*dpsi);
    return (psi - oe_psi_axis) / dpsi;
  }
  KOKKOS_INLINE_FUNCTION
  double oe_psi_norm_at(const double *xyz) const {
    double R, Z, gR, gZ;
    oe_psi_rz(xyz, R, Z);
    return oe_psi_norm_grad(R, Z, gR, gZ);
  }
  // CPU FixReflectPsi::segment_crossing
  KOKKOS_INLINE_FUNCTION
  bool oe_psi_crossing(const double *x0, const double *x1, double &fraction,
                       double *normal) const {
    const double p0 = oe_psi_norm_at(x0);
    const double p1 = oe_psi_norm_at(x1);
    if (p0 < oe_psi_thr || p1 >= oe_psi_thr) return false;
    double lo = 0.0, hi = 1.0, xc[3];
    for (int iter = 0; iter < 60; iter++) {
      const double mid = 0.5*(lo+hi);
      for (int k = 0; k < 3; k++) xc[k] = x0[k] + mid*(x1[k]-x0[k]);
      if (oe_psi_norm_at(xc) >= oe_psi_thr) lo = mid; else hi = mid;
    }
    fraction = 0.5*(lo+hi);
    for (int k = 0; k < 3; k++) xc[k] = x0[k] + fraction*(x1[k]-x0[k]);
    double R, Z, gR, gZ;
    oe_psi_rz(xc, R, Z);
    oe_psi_norm_grad(R, Z, gR, gZ);
    if (oe_dim == 3) {
      if (R <= 1.0e-30) return false;
      normal[0] = gR*xc[0]/R; normal[1] = gR*xc[1]/R; normal[2] = gZ;
    } else if (oe_axisymmetric) { normal[0] = gZ; normal[1] = gR; normal[2] = 0.0; }
    else { normal[0] = gR; normal[1] = gZ; normal[2] = 0.0; }
    const double nmag = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (!(nmag > 1.0e-20) || !Kokkos::isfinite(nmag)) return false;
    normal[0] /= nmag; normal[1] /= nmag; normal[2] /= nmag;
    return true;
  }
  KOKKOS_INLINE_FUNCTION
  bool oe_const_bfield_slot(const double *xq, double *Bout) const;
  DAT::t_float_2d_lr d_oe_equ_br, d_oe_equ_bt, d_oe_equ_bz;
  int oe_dim, oe_axisymmetric;        // cached domain layout for point-query

  // OpenEdge: device-resident triangulation B (Phase B). Mesh path takes
  // precedence over equilibrium when both are loaded — matches CPU.
  DAT::t_float_1d d_oe_mesh_vtx_r;
  DAT::t_float_1d d_oe_mesh_vtx_z;
  DAT::t_int_1d d_oe_mesh_tri;
  DAT::t_float_1d d_oe_mesh_tri_br;
  DAT::t_float_1d d_oe_mesh_tri_bz;
  DAT::t_float_1d d_oe_mesh_tri_bt;
  DAT::t_float_1d d_oe_mesh_tri_rmin;
  DAT::t_float_1d d_oe_mesh_tri_rmax;
  DAT::t_float_1d d_oe_mesh_tri_zmin;
  DAT::t_float_1d d_oe_mesh_tri_zmax;
  DAT::t_int_1d d_oe_hash_offset;
  DAT::t_int_1d d_oe_hash_entries;
  double oe_mesh_hash_rmin, oe_mesh_hash_zmin;
  double oe_mesh_hash_dr,   oe_mesh_hash_dz;
  int    oe_mesh_hash_nr,   oe_mesh_hash_nz;
  int    oe_mesh_ntri;
  int    oe_has_mesh_b;

  // OpenEdge: background mesh E-field (E = -grad phi from the plasma
  // file), flattened per-tri like the B views; consumed by oe_boris3d
  int    oe_has_mesh_e;
  DAT::t_float_1d d_oe_mesh_tri_er, d_oe_mesh_tri_ez, d_oe_mesh_tri_et;

  // OpenEdge gate 9: per-tri ion density + parallel flow (coulomb drag)
  // and grad-T fields (thermal force), flattened like te/ti/ne. The
  // device fix kernels bind these via friendship.
  int    oe_has_mesh_drag;    // ni + upar present
  int    oe_has_mesh_gradte;  // grad_te_r/z present
  int    oe_has_mesh_gradti;  // grad_ti_r/z present
  DAT::t_float_1d d_oe_mesh_tri_ni, d_oe_mesh_tri_upar;
  // gradients are PER MESH CELL (host pd_grad samples the SPARTA-cell
  // centroid's mesh cell via cell_mesh_cell, NOT the particle's tri)
  DAT::t_float_1d d_oe_meshcell_gter, d_oe_meshcell_gtez;
  DAT::t_float_1d d_oe_meshcell_gtir, d_oe_meshcell_gtiz;
  friend class FixCoulombBackgroundKokkos;
  friend class FixForceThermalKokkos;
  friend class FixCrossFieldDiffusionKokkos;

  // build the device mesh B/E views directly from FixBackground for
  // decks whose plasma provider is the fix (static SOLPS/SOLEDGE3X file)
  void build_oe_mesh_from_fix();

  // OpenEdge gate 9b: device plasma-cache fill. Capability decided once
  // per run() (oe_pcache_dev); the kernel samples the masked slots from
  // the device mesh views (tri-constant scalars, mesh/equ B) and applies
  // the sheath Boltzmann ne correction with the mover's element
  // refinement — exact CPU cache_plasma_particles() semantics for the
  // supported mask. Unsupported configs keep the host fill.
  void cache_plasma_particles_device();
  int oe_pcache_dev;    // 1 = device fill active this run
  int oe_pc_mask;       // pcache_need_mask captured for the kernel
  int oe_pc_csg;        // sheath Boltzmann ne correction active
  DAT::t_float_1d d_pc_te, d_pc_ti, d_pc_ne, d_pc_ni, d_pc_vpar;
  DAT::t_float_1d d_pc_bx, d_pc_by, d_pc_bz;
  int oe_pc_ncells;               // nlocal+nghost at fill time
  int oe_pc_diag_warned;
  Kokkos::View<int[6], DeviceType> d_pc_diag;

  // OpenEdge: Boris config. Hybrid/GCA pusher modes are NOT
  // supported on the device — the old oe_hybrid3d port encoded physics
  // since removed from the CPU pusher (pre-selector sheath force, old
  // switching without the Boris shell / trial-replay) and was deleted
  // 2026-08-26; UpdateKokkos::init() errors out instead.
  int oe_pusher_subcycles;
  double oe_echarge;
  // per-species pusher bypass (global pusher ... skip <mixture>): dust
  // grains advect ballistically even when charged, as on the CPU
  DAT::t_int_1d d_oe_pusher_skip;

  // OpenEdge Phase D (rev 2, CPU-parity): spatial-mode sheath data.
  //
  // Fix-background provider (production monoblock deck): per-ELEMENT
  // coefficient cache, the device mirror of the CPU per-element cache
  // (Pusher::build_sheath_cache_entry_3d builds each row on the host, so
  // plasma/B queries and coefficient prep are CPU-identical). One row per
  // surf element:
  //   [0]  state (1 = active; 0 = inactive: no te/ne/B at the centroid)
  //   [1]  d_max            — engagement cut-off (m), from sheath_auto_dmax
  //   [2]  phi_total_eV  [3] lambdaD_m  [4] lmps_m  [5] inv_lD
  //   [6]  inv_lmps  [7] K1_scaled  [8] K2  [9] phi_slow_eV
  //   [10] phi_fast_eV  [11] e_anchor_vpm
  // Compute provider: per-cell raw plasma (te, ti, ne, br, bt, bz); the
  // derived quantities are computed per particle on the device exactly as
  // the CPU per-particle path does.
  DAT::tdual_float_2d_lr k_oe_sheath_elem;
  DAT::t_float_2d_lr     d_oe_sheath_elem;
  DAT::tdual_float_2d_lr k_oe_sheath_cellplasma;
  DAT::t_float_2d_lr     d_oe_sheath_cellplasma;
  // per-cell nearest-surf element from the sheath geom compute
  // (ComputeNearestSurfGrid::midx_grid), refined per particle on device
  // against the cell's csurfs — mirrors CPU pusher.cpp refinement.
  DAT::tdual_int_1d      k_oe_midx_gcell;
  DAT::t_int_1d          d_oe_midx_gcell;
  int    oe_sheath_provider;   // 0 = none, 1 = fix (per-element), 2 = compute (per-cell)
  int    oe_sheath_sgroupbit;  // surf group mask of the sheath geom compute
  // decomposition stamps: fix balance re-decomposes the grid mid-run
  // (every 200 steps in the monoblock deck), which invalidates every
  // local-cell-indexed map; run() rebuilds the cache when these change
  int     oe_sheath_stamp_n;
  cellint oe_sheath_stamp_id;
  double oe_sheath_mD_amu;
  double oe_sheath_dmax_user;  // global pusher sheath dmax (0 = auto)
  double oe_col_x0, oe_col_y0; // column axis for cyl->Cartesian rotations
  // fix-provider per-particle fallback plasma (te/ti/ne flattened per
  // mesh tri, same layout as the E views) for wall elements whose
  // centroid sits outside the plasma-mesh footprint (CPU falls back to a
  // per-particle query there).
  DAT::t_float_1d d_oe_mesh_tri_te, d_oe_mesh_tri_ti, d_oe_mesh_tri_ne;
  int    oe_has_mesh_plasma;
  // per-particle customs of the spatial-mode potential impulse
  // (sheath_bank / sheath_phiprev), rebound each move() attempt; the
  // _backup twins snapshot them across a react/retry replay
  DAT::t_float_1d d_oe_sheath_bank;
  DAT::t_float_1d d_oe_sheath_phiprev;
  DAT::t_float_1d d_oe_sheath_bank_backup;
  DAT::t_float_1d d_oe_sheath_phiprev_backup;
  int    oe_has_sheath_customs;

  // OpenEdge Phase B: hybrid/GCA pusher configuration + GC-state customs
  int    oe_pusher_mode;          // Pusher::PusherMode (0 boris, 1 hybrid, 2 gca)
  int    oe_gca_integrator;       // Pusher::GCAIntegrator (0 rk4, 1 simple, 2 rk2)
  int    oe_boris_near_rhol, oe_gc_wall_flux;
  double oe_gca_switch, oe_boris_near;
  int    oe_has_gca_customs;
  int    oe_gc_hooks;             // OE_GC_HOOKS bitmask (diagnostic A/B): 1 collision
                                  // invalidate, 2 kick displace; default all on
  DAT::t_float_1d d_oe_gca_x, d_oe_gca_y, d_oe_gca_z, d_oe_gca_vpar,
                  d_oe_gca_mu, d_oe_gca_mode, d_oe_gca_valid, d_oe_gca_chi;
  DAT::t_float_1d d_oe_gca_backup[8];
  void build_oe_sheath_cache();
  // Spatial-sheath engagement diagnostics (device twins of the CPU
  // sheath_diag_* counters; gated on `global pusher ... dump yes`).
  // [0]=nactive (moves with a live sheath) [1]=nengage (subcycles with a
  // nonzero impulse) [2]=nreflect (turning-point reflections)
  int    oe_sheath_diag;
  DAT::t_int_1d d_oe_shd_counts;
  Kokkos::View<double*, DeviceType> d_oe_shd_esum;   // [0]=sum|E| [1]=max|E|
  long   oe_trace_id;   // OE_SHEATH_TRACE_ID per-particle trace (-1 = off)

  KKCopy<GridKokkos> grid_kk_copy;
  KKCopy<DomainKokkos> domain_kk_copy;

  int sc_type_list[KOKKOS_MAX_TOT_SURF_COLL];
  int sc_map[KOKKOS_MAX_TOT_SURF_COLL];
  KKCopy<SurfCollideSpecularKokkos> sc_kk_specular_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideDiffuseKokkos> sc_kk_diffuse_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideVanishKokkos> sc_kk_vanish_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollidePistonKokkos> sc_kk_piston_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideTransparentKokkos> sc_kk_transparent_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideToroidalKokkos> sc_kk_toroidal_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];

  //KKCopy<ComputeSurfKokkos> blist_active_copy[KOKKOS_MAX_GLIST];
  KKCopy<ComputeSurfKokkos> slist_active_copy[KOKKOS_MAX_SLIST];
  KKCopy<ComputeBoundaryKokkos> blist_active_copy[KOKKOS_MAX_BLIST];

  ComputeBoundaryKokkos tmp_compute_boundary_kk;
  ComputeSurfKokkos tmp_compute_surf_kk;

  typedef Kokkos::DualView<int[15], DeviceType::array_layout, DeviceType> tdual_int_14;
  typedef tdual_int_14::t_dev t_int_14;
  typedef tdual_int_14::t_host t_host_int_14;
  t_int_14 d_scalars;
  t_host_int_14 h_scalars;

  DAT::t_int_scalar d_ntouch_one;     HAT::t_int_scalar h_ntouch_one;
  DAT::t_int_scalar d_nexit_one;      HAT::t_int_scalar h_nexit_one;
  DAT::t_int_scalar d_nboundary_one;  HAT::t_int_scalar h_nboundary_one;
  DAT::t_int_scalar d_nmigrate;       HAT::t_int_scalar h_nmigrate;
  DAT::t_int_scalar d_entryexit;      HAT::t_int_scalar h_entryexit;
  DAT::t_int_scalar d_ncomm_one;      HAT::t_int_scalar h_ncomm_one;
  DAT::t_int_scalar d_nscheck_one;    HAT::t_int_scalar h_nscheck_one;
  DAT::t_int_scalar d_nscollide_one;  HAT::t_int_scalar h_nscollide_one;
  DAT::t_int_scalar d_nreact_one;     HAT::t_int_scalar h_nreact_one;
  DAT::t_int_scalar d_nstuck;         HAT::t_int_scalar h_nstuck;
  DAT::t_int_scalar d_ncaplost;       HAT::t_int_scalar h_ncaplost;
  DAT::t_int_scalar d_naxibad;        HAT::t_int_scalar h_naxibad;
  DAT::t_int_scalar d_error_flag;     HAT::t_int_scalar h_error_flag;
  DAT::t_int_scalar d_retry;          HAT::t_int_scalar h_retry;
  DAT::t_int_scalar d_nlocal;         HAT::t_int_scalar h_nlocal;

  void backup();
  void restore();
  t_particle_1d d_particles_backup;

  void tally_set(bigint);

  KOKKOS_INLINE_FUNCTION
  void axi_remap(double *x, double *v) const {
    double ynew = x[1], znew = x[2];
    x[1] = sqrt(ynew*ynew + znew*znew); x[2] = 0.0;
    double rn = ynew / x[1], wn = znew / x[1];
    double vy = v[1], vz = v[2];
    v[1] = vy*rn + vz*wn; v[2] = -vy*wn + vz*rn;
  };

  typedef void (UpdateKokkos::*FnPtr)();
  FnPtr moveptr;
  template <int, int, int, int> void move();

  KOKKOS_INLINE_FUNCTION
  void field2d(double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0]; x[1] += dtsq*field[1];
    v[0] += dt*field[0];   v[1] += dt*field[1];
  };
  KOKKOS_INLINE_FUNCTION
  void field3d(double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0]; x[1] += dtsq*field[1]; x[2] += dtsq*field[2];
    v[0] += dt*field[0];   v[1] += dt*field[1];   v[2] += dt*field[2];
  };
  KOKKOS_INLINE_FUNCTION
  void field_per_particle(int i, int icell, double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    auto &d_array = d_fieldfix_array_particle;
    int icol = 0;
    if (field_active[0]) { x[0] += dtsq*d_array(i,icol); v[0] += dt*d_array(i,icol); icol++; }
    if (field_active[1]) { x[1] += dtsq*d_array(i,icol); v[1] += dt*d_array(i,icol); icol++; }
    if (field_active[2]) { x[2] += dtsq*d_array(i,icol); v[2] += dt*d_array(i,icol); icol++; }
  };
  KOKKOS_INLINE_FUNCTION
  void field_per_grid(int i, int icell, double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    auto &d_array = d_fieldfix_array_grid;
    int icol = 0;
    if (field_active[0]) { x[0] += dtsq*d_array(icell,icol); v[0] += dt*d_array(icell,icol); icol++; }
    if (field_active[1]) { x[1] += dtsq*d_array(icell,icol); v[1] += dt*d_array(icell,icol); icol++; }
    if (field_active[2]) { x[2] += dtsq*d_array(icell,icol); v[2] += dt*d_array(icell,icol); icol++; }
  };

  // OpenEdge: device-callable Boris 3D pusher (reads E/B from grid fix views)
  // OpenEdge Phase B: device hybrid/GCA pusher (3D). Mirrors
  // Pusher::push_hybrid_3d / sample_gca_fields; Boris delegation =
  // oe_boris3d. GC state lives in the gca_* particle customs.
  KOKKOS_INLINE_FUNCTION
  bool oe_sample_gca_fields(const double *xpos, int icell,
                            GCAKokkos::Fields &F) const;
  KOKKOS_INLINE_FUNCTION
  double oe_near_signed(int midx, const double *p) const;
  KOKKOS_INLINE_FUNCTION
  void oe_hybrid3d(int i, int icell, double dt,
                   double *x, double *v, double *xnew,
                   double charge, double mass) const;
  // 2D / axisymmetric kick-drift Boris (device twin of Pusher::push_boris_2d,
  // without the spatial sheath -- 2D sheath errors out at init)
  KOKKOS_INLINE_FUNCTION
  void oe_boris2d(int i, int icell, double dt,
                  double *x, double *v, double *xnew,
                  double charge, double mass) const;

  KOKKOS_INLINE_FUNCTION
  void oe_boris3d(int i, int icell, double dt_full,
                  double *x, double *v, double *xnew,
                  double charge, double mass) const;

  KOKKOS_INLINE_FUNCTION
  int split3d(int, double*) const;
  KOKKOS_INLINE_FUNCTION
  int split2d(int, double*) const;
};

}

#endif
