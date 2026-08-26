/* ----------------------------------------------------------------------
   OpenEdge extension of SPARTA UpdateKokkos.
   Adds Boris/GCA pusher support with separate E-field and B-field fixes.
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
      naxibad,error_flag;
  KOKKOS_INLINE_FUNCTION
  s_UPDATE_REDUCE() {
    ntouch_one = nexit_one = nboundary_one = ncomm_one = 0;
    nscheck_one = nscollide_one = nreact_one = nstuck = naxibad = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_UPDATE_REDUCE &rhs) {
    ntouch_one += rhs.ntouch_one; nexit_one += rhs.nexit_one;
    nboundary_one += rhs.nboundary_one; ncomm_one += rhs.ncomm_one;
    nscheck_one += rhs.nscheck_one; nscollide_one += rhs.nscollide_one;
    nreact_one += rhs.nreact_one; nstuck += rhs.nstuck; naxibad += rhs.naxibad;
  }
};
typedef struct s_UPDATE_REDUCE UPDATE_REDUCE;

template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
struct TagUpdateMove{};

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

  // cross-field diffusion displacements (host fix -> device mover);
  // mirrors Update::dx_cd, uploaded each step in run()
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
  int oe_bx_col, oe_by_col, oe_bz_col;  // column indices for B in compute
  int oe_ex_col, oe_ey_col, oe_ez_col;  // column indices for E (sheath)

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

  // build the device mesh B/E views directly from FixBackground for
  // decks whose plasma provider is the fix (static SOLPS/SOLEDGE3X file)
  void build_oe_mesh_from_fix();

  // OpenEdge: Boris config. Hybrid/GCA (oe_pusher_mode != 0) is NOT
  // supported on the device — the old oe_hybrid3d port encoded physics
  // since removed from the CPU pusher (pre-selector sheath force, old
  // switching without the Boris shell / trial-replay) and was deleted
  // 2026-08-26; UpdateKokkos::init() errors out instead.
  int oe_pusher_subcycles;
  int oe_pusher_mode;     // Pusher::PUSHER_* (must be 0 = Boris under Kokkos)
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
  void build_oe_sheath_cache();

  KKCopy<GridKokkos> grid_kk_copy;
  KKCopy<DomainKokkos> domain_kk_copy;

  int sc_type_list[KOKKOS_MAX_TOT_SURF_COLL];
  int sc_map[KOKKOS_MAX_TOT_SURF_COLL];
  KKCopy<SurfCollideSpecularKokkos> sc_kk_specular_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideDiffuseKokkos> sc_kk_diffuse_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideVanishKokkos> sc_kk_vanish_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollidePistonKokkos> sc_kk_piston_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideTransparentKokkos> sc_kk_transparent_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];

  //KKCopy<ComputeSurfKokkos> blist_active_copy[KOKKOS_MAX_GLIST];
  KKCopy<ComputeSurfKokkos> slist_active_copy[KOKKOS_MAX_SLIST];
  KKCopy<ComputeBoundaryKokkos> blist_active_copy[KOKKOS_MAX_BLIST];

  ComputeBoundaryKokkos tmp_compute_boundary_kk;
  ComputeSurfKokkos tmp_compute_surf_kk;

  typedef Kokkos::DualView<int[14], DeviceType::array_layout, DeviceType> tdual_int_14;
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
