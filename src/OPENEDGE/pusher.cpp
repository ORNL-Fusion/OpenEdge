/* ----------------------------------------------------------------------
   OpenEdge Pusher implementation.
   Phase-3 refactor: class Pusher owns all charged-particle pusher state
   (mode, subcycles, plasma provider, GCA custom-attr indices, bad_dt
   limits) and the three large push kernels:
       Pusher::push_boris_2d   (Boris on 2D positions, 3D velocity)
       Pusher::push_boris_3d   (Boris on full 3D)
       Pusher::push_hybrid_3d  (hybrid Boris/GCA dispatcher)
   Update has a single Pusher* member and delegates from move()/init()/
   global(). Pusher methods reach Update-resident state (efield/bfield
   fix indices, sheath_*) through the Pointers-base `update->X`.
   Sister header pusher.h carries the inline math (Boris kick + GCA RHS/RK4).

   Abdourahmane Diaw,  diawa@ornl.gov
   Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include <cmath>
#include "spatype.h"
#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"
#include "math_const.h"
#include "particle.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "domain.h"
#include "comm.h"
#include "grid.h"
#include "surf.h"
#include "math_extra.h"
#include "pusher.h"
#include "openedge_geom.h"
#include "random_mars.h"
#include "sheath_models.h"
#include "compute_nearest_surf_grid.h"
#include "compute_plasma_fields.h"
#include "fix_background.h"
#include "geometry.h"
#include "input.h"
#include "memory.h"
#include "error.h"
#include <algorithm>

using namespace SPARTA_NS;
using namespace MathConst;

// File-scope helpers — kept private to this translation unit. Mirrors the
// anonymous-namespace symbols used by these pusher kernels in update.cpp;
// defining them again at file scope here is safe because each .cpp
// compiles independently.
namespace {

enum {NOFIELD, CFIELD, PFIELD, GFIELD};   // matches update.cpp

inline void xyz_to_rz(const double xyz[3], int dim, int axi, double &R, double &Z)
{
  if (axi) {              // 2D true axi: SPARTA x = Z-axis, y = R-radial
    Z = xyz[0];
    R = xyz[1];
  } else if (dim == 2) {  // 2D Cartesian (legacy)
    R = xyz[0];
    Z = xyz[1];
  } else {                // 3D Cartesian
    R = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    Z = xyz[2];
  }
}

inline void grad_from_fix(const FixBackground *pd, const std::vector<double> &field,
                          double R, double Z, double &d_dr, double &d_dz)
{
  d_dr = 0.0;
  d_dz = 0.0;
  if (!pd || field.empty() || pd->nr < 2 || pd->nz < 2) return;

  const double dR = std::max(1.0e-9, 0.5 * std::fabs(pd->rvals[1] - pd->rvals[0]));
  const double dZ = std::max(1.0e-9, 0.5 * std::fabs(pd->zvals[1] - pd->zvals[0]));
  d_dr = (pd->interp2D(field, R + dR, Z) - pd->interp2D(field, R - dR, Z)) / (2.0 * dR);
  d_dz = (pd->interp2D(field, R, Z + dZ) - pd->interp2D(field, R, Z - dZ)) / (2.0 * dZ);
}

// icell (when >= 0) routes every mesh-field lookup through FixBackground's
// O(1) cell-indexed cache instead of a per-field hash-grid triangle search
// (~16 searches per call otherwise — dominant sheath-prefetch cost).
// grad_from_fix stays position-based: it finite-differences the field, so
// cell-constant lookups would return a zero gradient.
inline PlasmaFileParams query_plasma_from_fix(const FixBackground *pd,
                                              const double xyz[3], int dim, int axi,
                                              int icell = -1)
{
  PlasmaFileParams P{};
  if (!pd) return P;

  double R, Z;
  xyz_to_rz(xyz, dim, axi, R, Z);

  P.temp_e = pd->interp2D(pd->temp_e, R, Z, icell);
  P.dens_e = pd->interp2D(pd->dens_e, R, Z, icell);
  P.temp_i = pd->interp2D(pd->temp_i, R, Z, icell);
  P.dens_i = pd->interp2D(pd->dens_i, R, Z, icell);
  P.parr_flow = pd->interp2D(pd->parr_flow, R, Z, icell);
  P.parr_flow_r = pd->interp2D(pd->parr_flow_r, R, Z, icell);
  P.parr_flow_t = pd->interp2D(pd->parr_flow_t, R, Z, icell);
  P.parr_flow_z = pd->interp2D(pd->parr_flow_z, R, Z, icell);
  P.grad_temp_e_r = pd->interp2D(pd->grad_te_r, R, Z, icell);
  P.grad_temp_e_t = pd->interp2D(pd->grad_te_t, R, Z, icell);
  P.grad_temp_e_z = pd->interp2D(pd->grad_te_z, R, Z, icell);
  P.grad_temp_i_r = pd->interp2D(pd->grad_ti_r, R, Z, icell);
  P.grad_temp_i_t = pd->interp2D(pd->grad_ti_t, R, Z, icell);
  P.grad_temp_i_z = pd->interp2D(pd->grad_ti_z, R, Z, icell);
  P.epar = pd->interp2D(pd->epar, R, Z, icell);
  grad_from_fix(pd, pd->dens_e, R, Z, P.grad_dens_e_r, P.grad_dens_e_z);
  P.grad_dens_e_t = 0.0;
  return P;
}

inline MagneticFieldFileDataParams query_bfield_from_fix(const FixBackground *pd,
                                                         const double xyz[3], int /*dim*/, int /*axi*/,
                                                         int icell = -1, int iparticle = -1)
{
  if (!pd) return MagneticFieldFileDataParams{};
  return pd->query_bfield_at_point(xyz, icell, iparticle);
}

inline double sheath_auto_dmax(double te_eV, double ti_eV, double ne_m3,
                                double bmag_T, double alpha_deg,
                                double mD_amu, double user_ceiling)
{
  constexpr double QE_LOC   = 1.602176634e-19;
  constexpr double AMU_LOC  = 1.66053906660e-27;
  constexpr double EPS0_LOC = 8.8541878128e-12;
  const double mD_kg = std::max(mD_amu * AMU_LOC, 1.0e-99);
  const double lambdaD = std::sqrt(EPS0_LOC * std::max(te_eV, 1.0e-12)
                                   / (std::max(ne_m3, 1.0e-60) * QE_LOC));
  // vth_d: 1D effective thermal speed for rho_i (not Bohm cs).
  const double vth_d = std::sqrt(std::max(te_eV + ti_eV, 0.0) * QE_LOC
                                 / (2.0 * mD_kg));
  const double omega_ci = QE_LOC * std::max(std::fabs(bmag_T), 1.0e-20) / mD_kg;
  const double rho_i = vth_d / std::max(omega_ci, 1.0e-99);
  // MPS normal-direction thickness is a few rho_i, roughly angle-independent
  // (Chodura ~sqrt(6) rho_i; grazing-incidence PIC shows a few rho_i). The
  // former rho_i*tan(alpha_from_normal) factor diverged at grazing incidence
  // (tan 88deg ~ 28) and engulfed the whole domain in "sheath".
  (void)alpha_deg;
  // user_ceiling (global pusher sheath dmax) > 0 sets the extent explicitly
  // (e.g. to cover the long MPS tail at grazing incidence); 0 = auto.
  if (user_ceiling > 0.0) return user_ceiling;
  return std::max(5.0 * rho_i, 10.0 * lambdaD);
}

// Additional attracting potential drop carried by a target surface tile.
// The custom array is [Vdc, Vrf_peak, phase_rad], with wall voltage measured
// relative to the local quasineutral plasma.  Only negative wall voltage adds
// to the present floating-sheath boundary model.
inline double sheath_waveform_drop(Update *update, Surf *surf, int midx)
{
  const int idx = update->sheath_waveform_custom;
  if (idx < 0 || midx < 0 || midx >= surf->nlocal + surf->nghost)
    return 0.0;
  double **wave = surf->edarray_local[surf->ewhich[idx]];
  if (!wave) return 0.0;
  const double now = update->time +
      (update->ntimestep - update->time_last_update) * update->dt;
  const double vwall = wave[midx][0] + wave[midx][1] *
      std::sin(2.0*M_PI*update->sheath_frequency_hz*now + wave[midx][2]);
  return std::max(0.0, -vwall);
}

}  // namespace

/* ----------------------------------------------------------------------
   Pusher class — ctor/dtor.
   Constructor sets all state members to safe defaults (matches the
   previous in-Update initialisation). Destructor frees the heap-owned
   plasma_cid string.
------------------------------------------------------------------------- */

Pusher::Pusher(SPARTA *sparta) : Pointers(sparta)
{
  pusher_mode         = PUSHER_BORIS;
  pusher_plasma_cid   = NULL;
  pusher_plasma_cidx  = -1;
  pusher_plasma_fidx  = -1;
  pusher_subcycles    = 1;
  pusher_gca_switch   = 2.5;
  pusher_boris_near   = 0.0;
  pusher_gca_integrator = GCA_RK4;
  sheath_cache_enabled = 0;
  sheath_diag_nactive = 0;
  sheath_diag_nengage = 0;
  sheath_diag_emax = 0.0;
  sheath_diag_esum = 0.0;
  sheath_diag_nreflect = 0;
  sheath_diag_nescape = 0;
  pusher_dump_flag    = 0;
  pusher_dump_every   = 1;
  pusher_bad_dt_check = 1;
  pusher_bad_dt_warned = 0;
  pusher_bad_dt_limit = 0.1;

  gca_x_custom    = -1;
  gca_y_custom    = -1;
  gca_z_custom    = -1;
  gca_vpar_custom = -1;
  gca_mu_custom   = -1;
  gca_on_custom   = -1;
}

Pusher::~Pusher()
{
  delete [] pusher_plasma_cid;
}

/* ----------------------------------------------------------------------
   Build one spatial-sheath cache entry for wall element `midx`.

   Evaluates the geometry, plasma (Te, Ti, ne), B, Chodura angle, cut-off
   distance and Coulette-Manfredi coefficients at the wall element MIDPOINT
   — the physical sheath edge — rather than at a particle position. For a
   static fix-background plasma every input is invariant, so this runs once
   per element and is reused by every near-wall particle for the whole run.
   Only called when sheath_cache_enabled (static fix-background plasma), so
   the plasma provider is always pusher_plasma_fidx here.
------------------------------------------------------------------------- */

void Pusher::build_sheath_cache_entry(int midx, SheathElemCache &C)
{
  const int  dim = domain->dimension;
  const bool axi = domain->axisymmetric;

  C.state = -1;   // assume inactive until proven otherwise

  Surf::Line *ln = &surf->lines[midx];

  // Unit normal (SPARTA slots -> cylindrical (nR, nZ, 0)).
  double nx_raw = ln->norm[0], ny_raw = ln->norm[1];
  const double nmag = std::sqrt(nx_raw*nx_raw + ny_raw*ny_raw);
  if (nmag > 0.0) { nx_raw /= nmag; ny_raw /= nmag; }
  const double n_slot[3] = {nx_raw, ny_raw, 0.0};
  double nR_tmp = 0.0, nZ_tmp = 0.0, nphi_tmp = 0.0;
  OpenEdge::sparta_v_to_RZphi(n_slot, dim, axi, 0.0, nR_tmp, nZ_tmp, nphi_tmp);
  C.nR = nR_tmp;
  C.nZ = nZ_tmp;

  // Wall element midpoint, in SPARTA slots and in cylindrical (R, Z).
  const double xmid_slot[3] = {0.5*(ln->p1[0]+ln->p2[0]),
                               0.5*(ln->p1[1]+ln->p2[1]), 0.0};
  OpenEdge::sparta_to_RZ(xmid_slot, dim, axi, C.sR, C.sZ, 0.0, 0.0);

  auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
  if (!pd) return;

  // Plasma at the wall midpoint (sheath-edge conditions).
  PlasmaFileParams pf = query_plasma_from_fix(pd, xmid_slot, dim, axi);
  const double te = pf.temp_e, ti = pf.temp_i, ne = pf.dens_e;

  // B at the wall midpoint (cylindrical), for |B| and the Chodura angle.
  double Br = 0.0, Bz = 0.0, Bt = 0.0;
  if (pd->has_bfield) pd->bfield_at(C.sR, C.sZ, Br, Bz, Bt);
  const double bmag = std::sqrt(Br*Br + Bz*Bz + Bt*Bt);

  if (!(te > 0.0 && ne > 0.0 && bmag > 0.0)) return;   // stays inactive

  const double bvec[3] = {Br, Bz, Bt};
  const double nvec[3] = {C.nR, C.nZ, 0.0};
  SheathModels::ChoduraMetrics cm =
    SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
  const double alpha_deg = cm.alpha_deg;

  C.d_max  = sheath_auto_dmax(te, ti, ne, bmag, alpha_deg,
                              update->sheath_mD_amu, update->sheath_dmax);
  C.coeffs = SheathModels::sheath_prepare_coulette_manfredi(
                 te, ti, ne, bmag, alpha_deg, update->sheath_mD_amu, 0.0);
  // Total sheath potential drop (V) = potential at the wall (d=0), used as
  // the escape barrier in sheath boundary mode.
  C.phi_total = SheathModels::sheath_phi_at_distance(C.coeffs, 0.0);
  C.state = 1;   // active

  // One-time-per-element field profile dump (first few elements) so a run
  // can see whether the model produces the expected strong near-wall field.
  static int ndbg = 0;
  if (pusher_dump_flag && ndbg < 6) {
    ndbg++;
    const double e0  = SheathModels::sheath_emag_at_distance(C.coeffs, 0.0);
    const double elD = SheathModels::sheath_emag_at_distance(C.coeffs, C.coeffs.lambdaD_m);
    const double edm = SheathModels::sheath_emag_at_distance(C.coeffs, C.d_max);
    printf("  [rank %d] sheath elem midx=%d Te=%.1f Ti=%.1f ne=%.2e B=%.2f "
           "alpha=%.1f lambdaD=%.2e d_max=%.2e | E(0)=%.3e E(lD)=%.3e "
           "E(dmax)=%.3e V/m\n",
           comm->me, midx, te, ti, ne, bmag, alpha_deg, C.coeffs.lambdaD_m,
           C.d_max, e0, elD, edm);
  }
}

/* ----------------------------------------------------------------------
   3D twin of build_sheath_cache_entry: one entry per wall TRIANGLE.

   Caches only the plasma-derived quantities (d_max, Coulette-Manfredi
   coefficients, phi_total) evaluated at the triangle CENTROID — the
   physical sheath edge. Geometry (normal, centroid reference point) is
   cheap and stays per-particle in push_boris_3d, so the 2D-specific
   nR/nZ/sR/sZ fields of SheathElemCache are left unused here. Static
   fix-background only (sheath_cache_enabled), so this runs once per
   element instead of a ~16-field plasma query per particle-move.
------------------------------------------------------------------------- */

void Pusher::build_sheath_cache_entry_3d(int midx, SheathElemCache &C)
{
  C.state = -1;   // assume inactive until proven otherwise

  Surf::Tri *tr = &surf->tris[midx];
  double nx = tr->norm[0], ny = tr->norm[1], nz = tr->norm[2];
  const double nmag = std::sqrt(nx*nx + ny*ny + nz*nz);
  if (nmag > 0.0) { nx /= nmag; ny /= nmag; nz /= nmag; }

  const double xmid[3] = {(tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0,
                          (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0,
                          (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0};

  auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
  if (!pd) return;

  // Plasma at the triangle centroid (sheath-edge conditions).
  PlasmaFileParams pf = query_plasma_from_fix(pd, xmid, 3,
                                              domain->axisymmetric);
  const double te = pf.temp_e, ti = pf.temp_i, ne = pf.dens_e;

  // B at the centroid: cylindrical query, rotated to Cartesian at the
  // centroid azimuth (about the column axis) for the Chodura angle.
  double Br = 0.0, Bz = 0.0, Bt = 0.0;
  const double rx = xmid[0] - pd->column_x0;
  const double ry = xmid[1] - pd->column_y0;
  const double rxy = std::sqrt(rx*rx + ry*ry);
  if (pd->has_bfield) pd->bfield_at(rxy, xmid[2], Br, Bz, Bt);
  double bvec[3];
  if (rxy > 1.0e-20) {
    const double cphi = rx / rxy, sphi = ry / rxy;
    bvec[0] = Br * cphi - Bt * sphi;
    bvec[1] = Br * sphi + Bt * cphi;
    bvec[2] = Bz;
  } else {
    bvec[0] = Br; bvec[1] = 0.0; bvec[2] = Bz;
  }
  const double bmag = std::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1] +
                                bvec[2]*bvec[2]);

  if (!(te > 0.0 && ne > 0.0 && bmag > 0.0)) return;   // stays inactive

  const double nvec[3] = {nx, ny, nz};
  SheathModels::ChoduraMetrics cm =
    SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);

  C.d_max  = sheath_auto_dmax(te, ti, ne, bmag, cm.alpha_deg,
                              update->sheath_mD_amu, update->sheath_dmax);
  C.coeffs = SheathModels::sheath_prepare_coulette_manfredi(
                 te, ti, ne, bmag, cm.alpha_deg, update->sheath_mD_amu, 0.0);
  C.phi_total = SheathModels::sheath_phi_at_distance(C.coeffs, 0.0);
  C.state = 1;
}

/* ----------------------------------------------------------------------
   Boris pusher for 2D (x,y) positions with full 3-component velocity
------------------------------------------------------------------------- */

void Pusher::push_boris_2d(int i, int icell, double dt,
                           double *x, double *v, double *xnew,
                           double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Boris pusher requires positive particle mass");
  // Fast path for neutrals: pure advection, no E/B field reads or Boris algebra.
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * update->echarge) / mass;
  const int nsub = (pusher_subcycles > 0) ? pusher_subcycles : 1;
  const double dt_sub = dt / static_cast<double>(nsub);

  const int dim = domain->dimension;
  const bool axi = domain->axisymmetric;

  double xcur[2] = {x[0], x[1]};
  double zcur = x[2];
  // Positions stay in SPARTA slot order (layout-agnostic for the 2D
  // position advance). Velocity, E and B are lifted into physical
  // cylindrical (R, Z, phi) via openedge_geom helpers, then rotated
  // into a right-handed (R, phi, Z) basis for the Boris cross product.
  // Supported layouts:
  //   - 2D Cartesian (legacy): SPARTA x = R, y = Z, z = phi
  //   - 2D axisymmetric:       SPARTA x = Z, y = R, z = phi
  // Both map to the same physics via OpenEdge::sparta_to_RZ /
  // sparta_v_to_RZphi / RZphi_force_to_sparta.
  //
  // Axisymmetric mode is kick-drift: the Boris kicks below update the
  // velocity at FIXED position, and the returned move is the single
  // straight segment xnew = x + dt*v. Update::move()'s axi machinery
  // (axi_horizontal_line, axi_line_intersect, axi_remap) reconstructs
  // the trajectory as x + t*v with constant v; handing it a curved,
  // subcycled endpoint makes the face/surface crossing tests disagree
  // with xnew, and particles that end up outside their cell after
  // axi_remap are discarded (naxibad). Gyration across the step is
  // captured geometrically by axi_remap, so dt must resolve the
  // gyroperiod (bad_dt_check guards this).
  double vcur[3] = {v[0], v[1], v[2]};
  double E_slot[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};   // cylindrical (BR, BZ, Bphi)

  // Cache E-field once per Boris call. Priority mirrors B-field below:
  // fix background's mesh/e_{r,z,t} (loaded from the plasma code's native
  // potential) is preferred; fall back to fix efield/grid when pd has no
  // E at this point or no fix background is registered.
  if (pusher_plasma_fidx >= 0) {
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
    if (pd) {
      const double xyz[3] = {xcur[0], xcur[1], (dim == 3) ? xcur[2] : 0.0};
      double ERp = 0.0, EZp = 0.0, Etp = 0.0;
      if (pd->query_efield_at_point(xyz, ERp, EZp, Etp, icell, i)) {
        const double phi_p = (dim == 3)
          ? std::atan2(xyz[1] - pd->column_y0, xyz[0] - pd->column_x0)
          : 0.0;
        OpenEdge::RZphi_force_to_sparta(ERp, EZp, Etp, dim, axi, phi_p,
                                        E_slot[0], E_slot[1], E_slot[2]);
      }
    }
  }
  if (E_slot[0] == 0.0 && E_slot[1] == 0.0 && E_slot[2] == 0.0
      && update->eperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[update->efieldfix], (update->efstyle == GFIELD),
                                   update->efield_active, i, icell, E_slot);

  double ER = 0.0, EZ = 0.0, Ephi = 0.0;
  OpenEdge::sparta_v_to_RZphi(E_slot, dim, axi, 0.0, ER, EZ, Ephi);

  // Cache B-field once via point query at initial position.
  // Particle displacement per full step (~v*dt ~ 10μm) is negligible
  // compared to the B-field scale length, so re-querying per subcycle is
  // unnecessary.
  if (pusher_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[pusher_plasma_cidx];
    ComputePlasmaFields *cp_bf = dynamic_cast<ComputePlasmaFields *>(cp_base);
    if (cp_bf) {
      const double xyz[3] = {xcur[0], xcur[1], 0.0};
      MagneticFieldFileDataParams Bcyl = cp_bf->query_bfield_at_point(xyz);
      if (Bcyl.Bmag > 0.0) {
        B[0] = Bcyl.br;
        B[1] = Bcyl.bz;
        B[2] = Bcyl.bt;
      }
    }
  } else if (pusher_plasma_fidx >= 0) {
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
    if (pd && pd->has_bfield) {
      const double xyz[3] = {xcur[0], xcur[1], 0.0};
      double R = 0.0, Z = 0.0;
      OpenEdge::sparta_to_RZ(xyz, dim, axi, R, Z,
                             pd->column_x0, pd->column_y0);
      double Br = 0.0, Bz = 0.0, Bt = 0.0;
      pd->bfield_at(R, Z, Br, Bz, Bt, icell, i);
      B[0] = Br;
      B[1] = Bz;
      B[2] = Bt;
    }
  }
  if (B[0] == 0.0 && B[1] == 0.0 && B[2] == 0.0 && update->bperturbflag) {
    // fix bfield/grid returns SPARTA slot order; lift to cylindrical.
    double B_slot[3] = {0.0, 0.0, 0.0};
    BorisGrid::read_field_from_fix(modify->fix[update->bfieldfix], (update->bfstyle == GFIELD),
                                   update->bfield_active, i, icell, B_slot);
    double BR = 0.0, BZ = 0.0, Bphi = 0.0;
    OpenEdge::sparta_v_to_RZphi(B_slot, dim, axi, 0.0, BR, BZ, Bphi);
    B[0] = BR;
    B[1] = BZ;
    B[2] = Bphi;
  }

  // --- Pre-fetch sheath data for the spatial-sheath E-field. Geometry,
  //     plasma and the derived Coulette-Manfredi coefficients are constant
  //     across subcycles; for a static fix-background plasma they are also
  //     constant per wall element across the whole run, so they come from a
  //     per-element cache (build once, reuse for every near-wall particle).
  //     Other plasma sources use the per-particle fallback below. Only the
  //     side test (sh_d0_sign) is genuinely per-particle.
  double sh_nR = 0.0, sh_nZ = 0.0;           // unit normal in cylindrical
  double sh_sR = 0.0, sh_sZ = 0.0;           // wall reference point (R, Z)
  double sh_d_max = 0.0;
  double sh_d0_sign = 0.0;
  double sh_phi_total = 0.0;                 // total sheath potential [V]
  int    sh_active = 0;
  int    sh_midx = -1;
  SheathModels::SheathEmagCoeffs sh_coeffs;

  if (update->sheath_flag && !update->sheath_kick && update->sheath_geom_cidx >= 0 &&
      (pusher_plasma_cidx >= 0 || pusher_plasma_fidx >= 0)) {
    Compute *cg = modify->compute[update->sheath_geom_cidx];
    int gcell = icell;
    Grid::ChildCell *cells_tmp = grid->cells;
    if (cells_tmp[icell].nsplit <= 0 && cells_tmp[icell].isplit >= 0)
      gcell = grid->sinfo[cells_tmp[icell].isplit].icell;

    auto *csg = dynamic_cast<ComputeNearestSurfGrid *>(cg);
    if (csg) {
      int midx = csg->midx_grid[gcell];
      // Refine midx when the parent cell holds multiple surface segments
      // (e.g. near corners); pick the one closest to the PARTICLE.
      Grid::ChildCell *pc = &grid->cells[gcell];
      if (pc->nsurf > 0) {
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          if (!(surf->lines[m].mask & sbit)) continue;
          Surf::Line *ln = &surf->lines[m];
          const double d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                                     (x[1]-ln->p1[1])*ln->norm[1]);
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0 && sheath_cache_enabled) {
        sh_midx = midx;
        // ---- Cached path (static fix-background plasma) ----
        const int nsurf_all = surf->nlocal + surf->nghost;
        if ((int) sheath_cache.size() != nsurf_all)
          sheath_cache.assign(nsurf_all, SheathElemCache{});
        if (midx < nsurf_all) {
          SheathElemCache &C = sheath_cache[midx];
          if (C.state == 0) build_sheath_cache_entry(midx, C);
          if (C.state == 1) {
            sh_nR = C.nR; sh_nZ = C.nZ;
            sh_sR = C.sR; sh_sZ = C.sZ;
            sh_d_max = C.d_max;
            sh_phi_total = C.phi_total;
            sh_coeffs = C.coeffs;
            sh_active = 1;
          }
        }
      } else if (midx >= 0) {
        sh_midx = midx;
        // ---- Per-particle fallback (compute plasma or non-static fix):
        //      evaluate geometry, plasma, B and the coefficients at the
        //      particle position, as before. ----
        Surf::Line *ln = &surf->lines[midx];
        double nx_raw = ln->norm[0];
        double ny_raw = ln->norm[1];
        const double nmag = std::sqrt(nx_raw*nx_raw + ny_raw*ny_raw);
        if (nmag > 0.0) { nx_raw /= nmag; ny_raw /= nmag; }
        const double n_slot[3] = {nx_raw, ny_raw, 0.0};
        double nR_tmp = 0.0, nZ_tmp = 0.0, nphi_tmp = 0.0;
        OpenEdge::sparta_v_to_RZphi(n_slot, dim, axi, 0.0,
                                     nR_tmp, nZ_tmp, nphi_tmp);
        sh_nR = nR_tmp;
        sh_nZ = nZ_tmp;

        const double xmid_slot[3] = {0.5*(ln->p1[0]+ln->p2[0]),
                                     0.5*(ln->p1[1]+ln->p2[1]),
                                     0.0};
        OpenEdge::sparta_to_RZ(xmid_slot, dim, axi, sh_sR, sh_sZ, 0.0, 0.0);

        double sh_te = 0.0, sh_ti = 0.0, sh_ne = 0.0;
        if (pusher_plasma_cidx >= 0) {
          Compute *cp_base = modify->compute[pusher_plasma_cidx];
          auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
          if (cp) {
            sh_te = cp->plasma_arr[gcell].temp_e;
            sh_ti = cp->plasma_arr[gcell].temp_i;
            sh_ne = cp->plasma_arr[gcell].dens_e;
          }
        } else {
          auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
          if (pd) {
            PlasmaFileParams sh_pf = query_plasma_from_fix(pd, x, dim, axi, icell);
            sh_te = sh_pf.temp_e;
            sh_ti = sh_pf.temp_i;
            sh_ne = sh_pf.dens_e;
          }
        }

        const double sh_bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
        if (sh_te > 0.0 && sh_ne > 0.0 && sh_bmag > 0.0) {
          const double bvec[3] = {B[0], B[1], B[2]};
          const double nvec[3] = {sh_nR, sh_nZ, 0.0};
          SheathModels::ChoduraMetrics cm =
            SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
          const double sh_alpha_deg = cm.alpha_deg;
          sh_d_max = sheath_auto_dmax(sh_te, sh_ti, sh_ne, sh_bmag,
                                      sh_alpha_deg, update->sheath_mD_amu,
                                      update->sheath_dmax);
          sh_coeffs = SheathModels::sheath_prepare_coulette_manfredi(
                          sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha_deg,
                          update->sheath_mD_amu, 0.0);
          sh_phi_total = SheathModels::sheath_phi_at_distance(sh_coeffs, 0.0);
          sh_active = 1;
        }
      }
    }
  }

  if (update->sheath_boundary && sh_active)
    sh_phi_total += sheath_waveform_drop(update, surf, sh_midx);

  // Per-particle: which side of the wall is the particle on? One dot
  // product, always recomputed. The signed distance seeds the running
  // d used by the spatial-mode potential impulse in the subcycle loop.
  double sh_d_cur = 0.0;
  if (sh_active) {
    double R0 = 0.0, Z0 = 0.0;
    const double xyz0[3] = {xcur[0], xcur[1], 0.0};
    OpenEdge::sparta_to_RZ(xyz0, dim, axi, R0, Z0, 0.0, 0.0);
    const double d0 = (R0 - sh_sR)*sh_nR + (Z0 - sh_sZ)*sh_nZ;
    sh_d0_sign = (d0 >= 0.0) ? 1.0 : -1.0;
    sh_d_cur = d0;
    sheath_diag_nactive++;
  }

  // --- Sheath BOUNDARY mode: sub-grid potential barrier (prompt redep) ---
  // The Debye sheath (~lambda_D) is below the grid, so instead of resolving
  // its E-field we treat it as a thin potential sheet at the wall and apply
  // an energy-conserving impulse to the ion's wall-normal velocity:
  //   outbound ion, normal KE < Z e phi_total  -> reflect (can't escape;
  //                                                this IS prompt redeposition)
  //   outbound ion, normal KE >= Z e phi_total  -> decelerate (escapes)
  // A per-particle "paid" flag makes the escape charge fire once per transit
  // (the plasma-side band d_max is cm-scale, so an escaping ion would
  // otherwise be re-charged for the ~100 steps it takes to cross it). The
  // reflect branch leaves the ion inbound, so it re-arms naturally next step.
  // barrier engages only inside the sheath band (see boris3D twin)
  if (update->sheath_boundary && sh_active && sh_phi_total > 0.0 &&
      sh_d0_sign > 0.0 && sh_d_cur <= sh_d_max) {
    int *paid_vec = (update->sheath_paid_custom >= 0)
      ? particle->eivec[particle->ewhich[update->sheath_paid_custom]] : nullptr;
    double vR = 0.0, vZ = 0.0, vphi = 0.0;
    OpenEdge::sparta_v_to_RZphi(vcur, dim, axi, 0.0, vR, vZ, vphi);
    const double vn = vR*sh_nR + vZ*sh_nZ;   // >0 = outbound (into fluid)

    if (vn <= 0.0) {
      if (paid_vec) paid_vec[i] = 0;          // inbound: re-arm the barrier
    } else if (!paid_vec || paid_vec[i] == 0) {
      const double Zc = std::fabs(charge);
      const double barrier_J = Zc * update->echarge * sh_phi_total;
      const double KEn = 0.5 * mass * vn * vn;
      double dvn;                             // amount to remove from vn (>=0)
      if (KEn < barrier_J) {
        dvn = 2.0 * vn;                       // reflect: vn -> -vn (elastic)
        sheath_diag_nreflect++;
      } else {
        const double vn_new = std::sqrt(vn*vn - 2.0*barrier_J/mass);
        dvn = vn - vn_new;                    // decelerate: escapes
        if (paid_vec) paid_vec[i] = 1;
        sheath_diag_nescape++;
      }
      const double vR2 = vR - dvn*sh_nR;
      const double vZ2 = vZ - dvn*sh_nZ;
      const double v_cyl[3] = {vR2, vZ2, vphi};
      OpenEdge::RZphi_force_to_sparta(v_cyl[0], v_cyl[1], v_cyl[2], dim, axi,
                                       0.0, vcur[0], vcur[1], vcur[2]);
    }
  }

  const double Brhs[3] = {B[0], B[2], B[1]};

  // spatial-mode lifetime energy ledger (see registration in Update::init)
  double *sh_bank_vec = (update->sheath_bank_custom >= 0)
    ? particle->edvec[particle->ewhich[update->sheath_bank_custom]] : nullptr;
  // phi reference from the previous move (stored phi+1; 0 = unset, e.g.
  // newly ionized): pays element/profile switches between moves instead
  // of re-seeding the potential for free
  double *sh_phiprev_vec = (update->sheath_phiprev_custom >= 0)
    ? particle->edvec[particle->ewhich[update->sheath_phiprev_custom]] : nullptr;
  int sh_phi_pending = 0;
  double sh_phi_ref = 0.0;
  if (sh_phiprev_vec) {
    if (sh_active && !update->sheath_boundary) {
      if (sh_phiprev_vec[i] > 0.0) {
        sh_phi_ref = sh_phiprev_vec[i] - 1.0;
        sh_phi_pending = 1;
      }
    } else sh_phiprev_vec[i] = 1.0;   // out of band: known phi = 0
  }

  for (int isub = 0; isub < nsub; isub++) {

    if (pusher_bad_dt_check && !pusher_bad_dt_warned) {
      const double bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
      const double bad = std::fabs(qm) * bmag * dt_sub;
      if (bad > pusher_bad_dt_limit) {
        if (comm->me == 0)
          error->warning(FLERR, "OpenEdge Boris warning: |q/m|*|B|*dt_sub is large");
        pusher_bad_dt_warned = 1;
      }
    }

    double xold[2] = {xcur[0], xcur[1]};

    // Spatial-mode sheath is applied AFTER the velocity/position update
    // below as an energy-consistent potential impulse; the sheath no
    // longer enters the Boris E-field force. The old per-subcycle force
    // (with plasma-side and d_max gates on per-step frozen wall geometry)
    // was non-conservative and pumped gyrating near-wall ions.
    const double Erhs[3] = {ER, Ephi, EZ};

    double vR = 0.0, vZ = 0.0, vphi = 0.0;
    OpenEdge::sparta_v_to_RZphi(vcur, dim, axi, 0.0, vR, vZ, vphi);
    double vrhs[3] = {vR, vphi, vZ};

    BorisGrid::push_velocity(qm, dt_sub, Erhs, Brhs, vrhs);

    OpenEdge::RZphi_force_to_sparta(vrhs[0], vrhs[2], vrhs[1], dim, axi, 0.0,
                                     vcur[0], vcur[1], vcur[2]);

    if (!axi) {
      xcur[0] += vcur[0] * dt_sub;
      xcur[1] += vcur[1] * dt_sub;
      zcur += vcur[2] * dt_sub;
    }

    // Spatial-mode sheath: exact work of the sheath potential over this
    // subcycle's normal displacement,
    //   dKE = Z e [phi(d_new) - phi(d_old)],
    // phi clamped to phi(0) for d <= 0 (no force behind the wall plane;
    // in/out crossings symmetric, so no energy pocket). Outbound ions that
    // cannot climb the remaining potential reflect elastically at the
    // turning point. d advances with the post-push normal velocity, which
    // matches the position update above exactly in planar 2D; in axi
    // (position applied by the outer move) it is the same straight-line
    // prediction the move will take.
    if (sh_active && !update->sheath_boundary) {
      const double vn = vrhs[0]*sh_nR + vrhs[2]*sh_nZ;
      const double d_old = sh_d_cur;
      const double d_new = d_old + vn * dt_sub;
      sh_d_cur = d_new;
      if (std::min(d_old, d_new) < sh_d_max) {
        const double phi_old_geo = SheathModels::sheath_phi_at_distance(
            sh_coeffs, std::max(d_old, 0.0));
        // first engagement this move: phi_old = last move's stored phi, so
        // a reference-element switch is charged as work, not a free teleport
        const double phi_old = sh_phi_pending ? sh_phi_ref : phi_old_geo;
        sh_phi_pending = 0;
        const double phi_new = SheathModels::sheath_phi_at_distance(
            sh_coeffs, std::max(d_new, 0.0));
        double dKE_J =
            std::fabs(charge) * update->echarge * (phi_new - phi_old);
        // lifetime ledger cap: net energy given may never exceed Z e phi_tot
        if (sh_bank_vec && dKE_J > 0.0) {
          const double room =
              std::fabs(charge) * update->echarge * sh_phi_total
              - sh_bank_vec[i];
          if (dKE_J > room) dKE_J = (room > 0.0) ? room : 0.0;
        }
        if (dKE_J != 0.0) {
          const double s2 = vn*vn + 2.0*dKE_J/mass;
          double vn_new;
          if (s2 >= 0.0) {
            vn_new = (vn >= 0.0) ? std::sqrt(s2) : -std::sqrt(s2);
            if (sh_bank_vec) sh_bank_vec[i] += dKE_J;
          } else {
            // turning point: elastic reflection. Bounce the ledger AND the
            // position back to d_old — the climb to d_new was never paid
            // for, so keeping it would let the ion descend from unpaid
            // height and pump energy on every bounce.
            vn_new = -vn;
            sh_d_cur = d_old;
            if (!axi) {
              xcur[0] -= (d_new - d_old) * sh_nR;
              xcur[1] -= (d_new - d_old) * sh_nZ;
            }
          }
          const double dvn = vn_new - vn;
          vrhs[0] += dvn * sh_nR;
          vrhs[2] += dvn * sh_nZ;
          OpenEdge::RZphi_force_to_sparta(vrhs[0], vrhs[2], vrhs[1], dim, axi,
                                           0.0, vcur[0], vcur[1], vcur[2]);
          // diagnostics: report the equivalent field seen this subcycle
          sheath_diag_nengage++;
          const double emag_diag = SheathModels::sheath_emag_at_distance(
              sh_coeffs, std::max(std::min(d_old, d_new), 0.0));
          sheath_diag_esum += emag_diag;
          if (emag_diag > sheath_diag_emax) sheath_diag_emax = emag_diag;
        }
        // remember phi at the endpoint for next move's reference payment
        if (sh_phiprev_vec)
          sh_phiprev_vec[i] = 1.0 + SheathModels::sheath_phi_at_distance(
              sh_coeffs, std::max(sh_d_cur, 0.0));
      } else if (sh_phiprev_vec) sh_phiprev_vec[i] = 1.0;
    }

    if (pusher_dump_flag && (update->ntimestep % pusher_dump_every == 0) && i == 0) {
      // Print on the first local particle of WHATEVER rank owns it.
      // With source-biased decomp (fix balance rcb part) rank 0 often
      // holds zero particles, so a `me == 0` gate silently suppresses
      // the whole diagnostic. Tag the rank so output stays legible.
      printf("boris2D rank=%d step=%lld icell=%d sub=%d/%d qm=%g E_rpz=(%g,%g,%g) B_rpz=(%g,%g,%g) sh=%d d_max=%g\n",
             comm->me, (long long) update->ntimestep, icell, isub+1, nsub, qm,
             Erhs[0], Erhs[1], Erhs[2], Brhs[0], Brhs[1], Brhs[2],
             sh_active, sh_d_max);
    }

    // --- Per-subcycle wall / cell-exit guard (2D) ---
    // Two distinct leak modes when sheath E accelerates a particle near
    // the wall:
    //   (a) xold->xcur segment crosses a wall line IN THE CURRENT CELL.
    //       Clip xnew to the intersection point so SPARTA's move loop
    //       sees a trajectory that exactly touches the wall instead of
    //       one that punched through (important for grazing geometry
    //       where a straight-line x->xnew past the wall can miss
    //       entirely).
    //   (b) xcur leaves this cell without hitting a surface here —
    //       return immediately so SPARTA's outer move loop picks up
    //       cell migration and runs the standard surface-crossing
    //       check in each traversed cell.
    // Without (b) the subcycle keeps pushing in the wrong cell and the
    // final x->xnew straight line can miss divertor walls with grazing
    // angles.
    // Skipped in axi: position is fixed during the kicks, and the exact
    // curved-face crossing tests in Update::move() handle the drift.

    if (nsub > 1 && !axi) {
      int gcell = icell;
      Grid::ChildCell *cells_local = grid->cells;
      if (cells_local[icell].nsplit <= 0 && cells_local[icell].isplit >= 0)
        gcell = grid->sinfo[cells_local[icell].isplit].icell;

      // (a) in-cell wall crossing — clip to intersection point.
      int nsurf_cell = cells_local[gcell].nsurf;
      if (nsurf_cell > 0) {
        surfint *csurfs_local = cells_local[gcell].csurfs;
        Surf::Line *lines_local = surf->lines;
        double xc[2];
        double param;
        int side;
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = static_cast<int>(csurfs_local[m]);
          Surf::Line *line = &lines_local[isurf];
          if (Geometry::line_line_intersect(xold, xcur,
                                             line->p1, line->p2,
                                             line->norm, xc, param, side)) {
            v[0] = vcur[0];
            v[1] = vcur[1];
            v[2] = vcur[2];
            // Clip to intersection; keep the toroidal slot consistent
            // with the fraction of the subcycle that was traversed.
            xnew[0] = xc[0];
            xnew[1] = xc[1];
            xnew[2] = zcur - vcur[2] * dt_sub * (1.0 - param);
            return;
          }
        }
      }

      // (b) cell-exit — bail so SPARTA handles the remainder.
      // Use half-open [lo, hi) to match grid->id_find_child so a
      // particle exactly at `hi` is recognized as having left.
      const double *clo = cells_local[gcell].lo;
      const double *chi = cells_local[gcell].hi;
      if (xcur[0] < clo[0] || xcur[0] >= chi[0] ||
          xcur[1] < clo[1] || xcur[1] >= chi[1]) {
        v[0] = vcur[0];
        v[1] = vcur[1];
        v[2] = vcur[2];
        xnew[0] = xcur[0];
        xnew[1] = xcur[1];
        xnew[2] = zcur;
        return;
      }
    }
  }

  v[0] = vcur[0];
  v[1] = vcur[1];
  v[2] = vcur[2];
  if (axi) {
    // Kick-drift: single linear segment with the post-kick velocity,
    // so that xnew = x + dt*v holds exactly for the axi tracing.
    xnew[0] = x[0] + vcur[0] * dt;
    xnew[1] = x[1] + vcur[1] * dt;
    xnew[2] = x[2] + vcur[2] * dt;
  } else {
    xnew[0] = xcur[0];
    xnew[1] = xcur[1];
    xnew[2] = zcur;
  }
}

/* ----------------------------------------------------------------------
   Boris pusher for 3D cartesian coordinates
------------------------------------------------------------------------- */

void Pusher::push_boris_3d(int i, int icell, double dt,
                            double *x, double *v, double *xnew,
                            double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Boris pusher requires positive particle mass");

  // Fast path for neutrals: pure advection, no E/B field reads or Boris algebra.
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * update->echarge) / mass;
  const int nsub = (pusher_subcycles > 0) ? pusher_subcycles : 1;
  const double dt_sub = dt / static_cast<double>(nsub);

  double xcur[3] = {x[0], x[1], x[2]};
  double vcur[3] = {v[0], v[1], v[2]};

  // --- Pre-fetch per-particle sheath data from grid-cached computes ---
  // Grid cell's cached nearest-surface geometry and plasma parameters.
  // These are invariant during subcycling (grid data doesn't change mid-step).

  double sh_nx = 0.0, sh_ny = 0.0, sh_nz = 0.0;  // raw surface normal (unit)
  double sh_sref[3] = {0.0, 0.0, 0.0};  // reference point on nearest surface element
  double sh_te = 0.0, sh_ti = 0.0, sh_ne = 0.0;
  double sh_bmag = 0.0, sh_alpha_deg = 90.0;
  int sh_active = 0;
  int sh_midx = -1;
  int sh_from_cache = 0;   // d_max/coeffs taken from the per-element cache
  double sh_d_max = 0.0;
  SheathModels::SheathEmagCoeffs sh_coeffs;

  if (update->sheath_flag && update->sheath_geom_cidx >= 0 &&
      (pusher_plasma_cidx >= 0 || pusher_plasma_fidx >= 0)) {
    Compute *cg = modify->compute[update->sheath_geom_cidx];

    // If particle is in a sub-cell (split cell), resolve to parent cell
    // for geometry/plasma lookup — the compute skips sub-cells.
    int gcell = icell;
    Grid::ChildCell *cells_tmp = grid->cells;
    if (cells_tmp[icell].nsplit <= 0 && cells_tmp[icell].isplit >= 0)
      gcell = grid->sinfo[cells_tmp[icell].isplit].icell;

    // Get nearest surface element index from geometry compute
    auto *csg = dynamic_cast<ComputeNearestSurfGrid *>(cg);
    if (csg) {
      int midx = csg->midx_grid[gcell];

      // When the parent cell contains surface elements, refine midx by
      // finding the surface nearest to the PARTICLE position (not the
      // cell center used by the compute).  This fixes wrong-face
      // selection when a thin slab (top+bottom+side faces) intersects a
      // single cell and the cell center sits between the faces.
      Grid::ChildCell *pc = &grid->cells[gcell];
      if (pc->nsurf > 0) {
        const int dim = domain->dimension;
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          double d;
          if (dim == 2) {
            if (!(surf->lines[m].mask & sbit)) continue;
            Surf::Line *ln = &surf->lines[m];
            d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                          (x[1]-ln->p1[1])*ln->norm[1]);
          } else {
            if (!(surf->tris[m].mask & sbit)) continue;
            Surf::Tri *tr = &surf->tris[m];
            d = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                          (x[1]-tr->p1[1])*tr->norm[1] +
                          (x[2]-tr->p1[2])*tr->norm[2]);
          }
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0) {
        sh_midx = midx;
        // Use RAW triangle/line normal directly from the surface element,
        // not the per-cell flipped version.  This avoids normal sign flips
        // in split cells where the cell center is on the opposite side of
        // the surface from the particle.
        if (domain->dimension == 2) {
          Surf::Line *ln = &surf->lines[midx];
          sh_nx = ln->norm[0];
          sh_ny = ln->norm[1];
          sh_nz = 0.0;
          sh_sref[0] = 0.5*(ln->p1[0]+ln->p2[0]);
          sh_sref[1] = 0.5*(ln->p1[1]+ln->p2[1]);
          sh_sref[2] = 0.0;
        } else {
          Surf::Tri *tr = &surf->tris[midx];
          sh_nx = tr->norm[0];
          sh_ny = tr->norm[1];
          sh_nz = tr->norm[2];
          sh_sref[0] = (tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0;
          sh_sref[1] = (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0;
          sh_sref[2] = (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0;
        }
        const double nmag = std::sqrt(sh_nx*sh_nx + sh_ny*sh_ny + sh_nz*sh_nz);
        if (nmag > 0.0) {
          sh_nx /= nmag;  sh_ny /= nmag;  sh_nz /= nmag;
        }

        // ---- Cached path (static fix-background plasma): d_max and the
        // Coulette-Manfredi coefficients come from the per-element cache,
        // built once at the triangle centroid. Skips the ~16-field plasma
        // query + B query + Chodura prep on EVERY particle-move — the
        // dominant sheath cost in 3D. Mirrors the push_boris_2d cache.
        if (sheath_cache_enabled && domain->dimension == 3 &&
            pusher_plasma_fidx >= 0) {
          const int nsurf_all = surf->nlocal + surf->nghost;
          if ((int) sheath_cache.size() != nsurf_all)
            sheath_cache.assign(nsurf_all, SheathElemCache{});
          if (midx < nsurf_all) {
            SheathElemCache &C = sheath_cache[midx];
            if (C.state == 0) build_sheath_cache_entry_3d(midx, C);
            if (C.state == 1) {
              sh_d_max = C.d_max;
              sh_coeffs = C.coeffs;
              sh_active = 1;
              sh_from_cache = 1;
            }
            // state -1: element evaluated but inactive (no plasma/B there)
          }
        }

        double br = 0.0, bt = 0.0, bz = 0.0;
        if (sh_from_cache) {
          // per-particle plasma/B/Chodura evaluation not needed
        } else if (pusher_plasma_cidx >= 0) {
          Compute *cp_base = modify->compute[pusher_plasma_cidx];
          auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
          if (cp) {
            sh_te = cp->plasma_arr[gcell].temp_e;
            sh_ti = cp->plasma_arr[gcell].temp_i;
            sh_ne = cp->plasma_arr[gcell].dens_e;
            br = cp->mag_arr[gcell].br;
            bt = cp->mag_arr[gcell].bt;
            bz = cp->mag_arr[gcell].bz;
          }
        } else if (pusher_plasma_fidx >= 0) {
          auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
          if (pd) {
            PlasmaFileParams sh_pf = query_plasma_from_fix(pd, x, 3, domain->axisymmetric, icell);
            MagneticFieldFileDataParams sh_bf = query_bfield_from_fix(pd, x, 3, domain->axisymmetric, icell, i);
            sh_te = sh_pf.temp_e;
            sh_ti = sh_pf.temp_i;
            sh_ne = sh_pf.dens_e;
            br = sh_bf.br;
            bt = sh_bf.bt;
            bz = sh_bf.bz;
          }
        }

        if (sh_te > 0.0 && sh_ne > 0.0) {
          // Convert cylindrical B to Cartesian at particle position
          // (azimuth about the column axis, not the domain origin)
          double sh_x0 = 0.0, sh_y0 = 0.0;
          if (pusher_plasma_cidx >= 0) {
            auto *cpo = dynamic_cast<ComputePlasmaFields *>(
                modify->compute[pusher_plasma_cidx]);
            if (cpo) { sh_x0 = cpo->plasma_data.column_x0;
                       sh_y0 = cpo->plasma_data.column_y0; }
          } else if (pusher_plasma_fidx >= 0) {
            auto *pdo = dynamic_cast<FixBackground *>(
                modify->fix[pusher_plasma_fidx]);
            if (pdo) { sh_x0 = pdo->column_x0; sh_y0 = pdo->column_y0; }
          }
          const double rx = x[0] - sh_x0, ry = x[1] - sh_y0;
          const double rmag = std::sqrt(rx*rx + ry*ry);
          double bvec[3];
          if (rmag > 1.0e-20) {
            const double cphi = rx / rmag, sphi = ry / rmag;
            bvec[0] = br * cphi - bt * sphi;
            bvec[1] = br * sphi + bt * cphi;
            bvec[2] = bz;
          } else {
            bvec[0] = br;  bvec[1] = 0.0;  bvec[2] = bz;
          }
          sh_bmag = std::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1] + bvec[2]*bvec[2]);

          // Chodura angle: angle between B and surface normal
          // (chodura_metrics uses abs(B·n) so result is independent of normal sign)
          if (sh_bmag > 0.0) {
            double nvec[3] = {sh_nx, sh_ny, sh_nz};
            SheathModels::ChoduraMetrics cm =
              SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
            sh_alpha_deg = cm.alpha_deg;
          }

          sh_active = (sh_te > 0.0 && sh_ne > 0.0);
        }
      }
    }
  }

  // Record which side of the wall the particle starts on (sign of d_raw).
  // During subcycling, only apply sheath E-field while particle remains on
  // this side.  If it overshoots past the wall, skip E-field to prevent
  // reverse-field deceleration that causes energy loss.
  double sh_d0_sign = 0.0;
  double sh_d0 = 0.0;
  if (sh_active) {
    sh_d0 =
      (xcur[0] - sh_sref[0]) * sh_nx
    + (xcur[1] - sh_sref[1]) * sh_ny
    + (xcur[2] - sh_sref[2]) * sh_nz;
    sh_d0_sign = (sh_d0 >= 0.0) ? 1.0 : -1.0;
  }

  // Physics-derived sheath cut-off distance and Coulette-Manfredi
  // coefficients, hoisted out of the subcycle loop (Te, ne, B, alpha are
  // constant across subcycles). Skipped when the per-element cache already
  // supplied both (sh_from_cache).
  if (sh_active && !sh_from_cache) {
    sh_d_max = sheath_auto_dmax(sh_te, sh_ti, sh_ne, sh_bmag,
                                sh_alpha_deg, update->sheath_mD_amu,
                                update->sheath_dmax);
    sh_coeffs = SheathModels::sheath_prepare_coulette_manfredi(
                    sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha_deg,
                    update->sheath_mD_amu, 0.0);
  }

  // spatial-mode lifetime energy ledger + total potential for its cap
  double *sh_bank_vec = (update->sheath_bank_custom >= 0)
    ? particle->edvec[particle->ewhich[update->sheath_bank_custom]] : nullptr;
  double sh_phi_tot_sp = 0.0;
  if (sh_active && sh_bank_vec)
    sh_phi_tot_sp = SheathModels::sheath_phi_at_distance(sh_coeffs, 0.0);
  // phi reference from the previous move (stored phi+1; 0 = unset, e.g.
  // newly ionized): pays element/profile switches between moves instead
  // of re-seeding the potential for free
  double *sh_phiprev_vec = (update->sheath_phiprev_custom >= 0)
    ? particle->edvec[particle->ewhich[update->sheath_phiprev_custom]] : nullptr;
  int sh_phi_pending = 0;
  double sh_phi_ref = 0.0;
  if (sh_phiprev_vec) {
    if (sh_active && !update->sheath_kick && !update->sheath_boundary) {
      if (sh_phiprev_vec[i] > 0.0) {
        sh_phi_ref = sh_phiprev_vec[i] - 1.0;
        sh_phi_pending = 1;
      }
    } else sh_phiprev_vec[i] = 1.0;   // out of band: known phi = 0
  }

  // --- Sheath BOUNDARY mode: sub-grid potential barrier (prompt redep) ---
  // 3D port of the push_boris_2d barrier impulse: an outbound ion with
  // wall-normal KE below Z e phi_total reflects (cannot escape the sheath
  // potential — this IS prompt redeposition); one with KE above it
  // decelerates once per transit (per-particle "paid" flag, re-armed on
  // inbound motion). Velocity and normal are both Cartesian here, so no
  // cylindrical round-trip is needed.
  // barrier engages only inside the sheath band: the potential sheet
  // lives at the wall, not anywhere in the volume
  if (update->sheath_boundary && sh_active && sh_d0_sign > 0.0 &&
      sh_d0 <= sh_d_max) {
    const double sh_phi_total =
        SheathModels::sheath_phi_at_distance(sh_coeffs, 0.0) +
        sheath_waveform_drop(update, surf, sh_midx);
    if (sh_phi_total > 0.0) {
      int *paid_vec = (update->sheath_paid_custom >= 0)
        ? particle->eivec[particle->ewhich[update->sheath_paid_custom]] : nullptr;
      const double vn = vcur[0]*sh_nx + vcur[1]*sh_ny + vcur[2]*sh_nz;
      // vn > 0 = outbound (into fluid, along the outward surface normal)

      if (vn <= 0.0) {
        if (paid_vec) paid_vec[i] = 0;          // inbound: re-arm the barrier
      } else if (!paid_vec || paid_vec[i] == 0) {
        const double Zc = std::fabs(charge);
        const double barrier_J = Zc * update->echarge * sh_phi_total;
        const double KEn = 0.5 * mass * vn * vn;
        double dvn;                             // amount to remove from vn (>=0)
        if (KEn < barrier_J) {
          dvn = 2.0 * vn;                       // reflect: vn -> -vn (elastic)
          sheath_diag_nreflect++;
        } else {
          const double vn_new = std::sqrt(vn*vn - 2.0*barrier_J/mass);
          dvn = vn - vn_new;                    // decelerate: escapes
          if (paid_vec) paid_vec[i] = 1;
          sheath_diag_nescape++;
        }
        vcur[0] -= dvn * sh_nx;
        vcur[1] -= dvn * sh_ny;
        vcur[2] -= dvn * sh_nz;
      }
    }
  }

  // Cache B-field once via point query at initial position.
  double B_cached[3] = {0.0, 0.0, 0.0};
  if (pusher_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[pusher_plasma_cidx];
    ComputePlasmaFields *cp_bf = dynamic_cast<ComputePlasmaFields *>(cp_base);
    if (cp_bf) {
      MagneticFieldFileDataParams Bcyl = cp_bf->query_bfield_at_point(xcur);
      if (Bcyl.Bmag > 0.0) {
        // azimuth about the column axis, not the domain origin
        const double rx = xcur[0] - cp_bf->plasma_data.column_x0;
        const double ry = xcur[1] - cp_bf->plasma_data.column_y0;
        const double rxy = std::sqrt(rx*rx + ry*ry);
        double cphi = 1.0, sphi = 0.0;
        if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
        B_cached[0] = Bcyl.br * cphi - Bcyl.bt * sphi;
        B_cached[1] = Bcyl.br * sphi + Bcyl.bt * cphi;
        B_cached[2] = Bcyl.bz;
      }
    }
  } else if (pusher_plasma_fidx >= 0) {
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
    if (pd && pd->has_bfield) {
      double Br = 0.0, Bz = 0.0, Bt = 0.0;
      const double rx = xcur[0] - pd->column_x0;
      const double ry = xcur[1] - pd->column_y0;
      const double rxy = std::sqrt(rx * rx + ry * ry);
      double cphi = 1.0, sphi = 0.0;
      if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
      pd->bfield_at(rxy, xcur[2], Br, Bz, Bt, icell, i);
      B_cached[0] = Br * cphi - Bt * sphi;
      B_cached[1] = Br * sphi + Bt * cphi;
      B_cached[2] = Bz;
    }
  }
  if (B_cached[0] == 0.0 && B_cached[1] == 0.0 && B_cached[2] == 0.0 && update->bperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[update->bfieldfix], (update->bfstyle == GFIELD),
                                   update->bfield_active, i, icell, B_cached);

  for (int isub = 0; isub < nsub; isub++) {
    double E[3] = {0.0, 0.0, 0.0};
    double B[3] = {B_cached[0], B_cached[1], B_cached[2]};

    if (pusher_plasma_fidx >= 0) {
      auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
      if (pd) {
        const double xyz[3] = {xcur[0], xcur[1], xcur[2]};
        double ERp = 0.0, EZp = 0.0, Etp = 0.0;
        if (pd->query_efield_at_point(xyz, ERp, EZp, Etp, icell, i)) {
          const double phi_p = std::atan2(xyz[1] - pd->column_y0,
                                          xyz[0] - pd->column_x0);
          OpenEdge::RZphi_force_to_sparta(ERp, EZp, Etp, 3, false, phi_p,
                                          E[0], E[1], E[2]);
        }
      }
    }
    if (E[0] == 0.0 && E[1] == 0.0 && E[2] == 0.0 && update->eperturbflag)
      BorisGrid::read_field_from_fix(modify->fix[update->efieldfix], (update->efstyle == GFIELD),
                                     update->efield_active, i, icell, E);

    // Fall back to grid-stored B-field if no cached value
    if (B[0] == 0.0 && B[1] == 0.0 && B[2] == 0.0 && update->bperturbflag)
      BorisGrid::read_field_from_fix(modify->fix[update->bfieldfix], (update->bfstyle == GFIELD),
                                     update->bfield_active, i, icell, B);

    // Spatial-mode sheath is applied AFTER the position update below as an
    // energy-consistent potential impulse, not as an E-field force here.
    // The old per-subcycle force (with plasma-side and d_max gates on
    // per-step frozen wall geometry) was non-conservative: a gyrating ion
    // dipping behind the nearest tile's plane without colliding pocketed
    // the inbound sheath work every orbit and ran away.

    if (pusher_bad_dt_check && !pusher_bad_dt_warned) {
      const double bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
      const double bad = std::fabs(qm) * bmag * dt_sub;
      if (bad > pusher_bad_dt_limit) {
        if (comm->me == 0)
          error->warning(FLERR, "OpenEdge Boris warning: |q/m|*|B|*dt_sub is large");
        pusher_bad_dt_warned = 1;
      }
    }

    double xold[3] = {xcur[0], xcur[1], xcur[2]};

    BorisGrid::push_velocity(qm, dt_sub, E, B, vcur);
    xcur[0] += vcur[0] * dt_sub;
    xcur[1] += vcur[1] * dt_sub;
    xcur[2] += vcur[2] * dt_sub;

    // Spatial-mode sheath: exact work of the sheath potential over this
    // subcycle's normal displacement,
    //   dKE = Z e [phi(d_new) - phi(d_old)],
    // phi clamped to phi(0) for d <= 0 so there is no force behind the
    // wall plane and in/out crossings are symmetric (no energy pocket).
    // Outbound ions that cannot climb the remaining potential reflect
    // elastically at the turning point. Skipped when both endpoints are
    // beyond d_max (phi variation there is negligible and symmetric).
    if (sh_active && !update->sheath_kick && !update->sheath_boundary) {
      const double d_old =
        (xold[0] - sh_sref[0]) * sh_nx
      + (xold[1] - sh_sref[1]) * sh_ny
      + (xold[2] - sh_sref[2]) * sh_nz;
      const double d_new =
        (xcur[0] - sh_sref[0]) * sh_nx
      + (xcur[1] - sh_sref[1]) * sh_ny
      + (xcur[2] - sh_sref[2]) * sh_nz;
      if (std::min(d_old, d_new) < sh_d_max) {
        const double phi_old_geo = SheathModels::sheath_phi_at_distance(
            sh_coeffs, std::max(d_old, 0.0));
        // first engagement this move: phi_old = last move's stored phi, so
        // a reference-element switch is charged as work, not a free teleport
        const double phi_old = sh_phi_pending ? sh_phi_ref : phi_old_geo;
        sh_phi_pending = 0;
        const double phi_new = SheathModels::sheath_phi_at_distance(
            sh_coeffs, std::max(d_new, 0.0));
        double dKE_J =
            std::fabs(charge) * update->echarge * (phi_new - phi_old);
        // lifetime ledger cap: net energy given may never exceed Z e phi_tot
        if (sh_bank_vec && dKE_J > 0.0) {
          const double room =
              std::fabs(charge) * update->echarge * sh_phi_tot_sp
              - sh_bank_vec[i];
          if (dKE_J > room) dKE_J = (room > 0.0) ? room : 0.0;
        }
        double sh_d_fin = d_new;
        if (dKE_J != 0.0) {
          const double vn = vcur[0]*sh_nx + vcur[1]*sh_ny + vcur[2]*sh_nz;
          const double s2 = vn*vn + 2.0*dKE_J/mass;
          double vn_new;
          if (s2 >= 0.0) {
            vn_new = (vn >= 0.0) ? std::sqrt(s2) : -std::sqrt(s2);
            if (sh_bank_vec) sh_bank_vec[i] += dKE_J;
          } else {
            // turning point: elastic reflection. Bounce the position back
            // to d_old — the climb to d_new was never paid for, so keeping
            // it would let the ion descend from unpaid height and pump
            // energy on every bounce.
            vn_new = -vn;
            xcur[0] -= (d_new - d_old) * sh_nx;
            xcur[1] -= (d_new - d_old) * sh_ny;
            xcur[2] -= (d_new - d_old) * sh_nz;
            sh_d_fin = d_old;
          }
          const double dvn = vn_new - vn;
          vcur[0] += dvn * sh_nx;
          vcur[1] += dvn * sh_ny;
          vcur[2] += dvn * sh_nz;
          // diagnostics: report the equivalent field seen this subcycle
          sheath_diag_nengage++;
          const double emag_diag = SheathModels::sheath_emag_at_distance(
              sh_coeffs, std::max(std::min(d_old, d_new), 0.0));
          sheath_diag_esum += emag_diag;
          if (emag_diag > sheath_diag_emax) sheath_diag_emax = emag_diag;
        }
        // remember phi at the endpoint for next move's reference payment
        if (sh_phiprev_vec)
          sh_phiprev_vec[i] = 1.0 + SheathModels::sheath_phi_at_distance(
              sh_coeffs, std::max(sh_d_fin, 0.0));
      } else if (sh_phiprev_vec) sh_phiprev_vec[i] = 1.0;
    }

    if (pusher_dump_flag && (update->ntimestep % pusher_dump_every == 0) && i == 0) {
      // Print on first local particle of any rank; rcb-part decomp
      // can leave rank 0 empty. Tag the rank so output stays legible.
      printf("boris3D rank=%d step=%lld icell=%d sub=%d/%d qm=%g E=(%g,%g,%g) B=(%g,%g,%g)\n",
             comm->me, (long long) update->ntimestep, icell, isub+1, nsub, qm,
             E[0], E[1], E[2], B[0], B[1], B[2]);
    }

    // Per-subcycle surface crossing guard.
    // When subcycling, the move loop only sees the straight line from x
    // (start of timestep) to xnew (end of all subcycles).  If the curved
    // gyro-orbit crosses a surface during an intermediate subcycle but the
    // endpoints are on the same side, the crossing is invisible to the
    // move loop and the particle leaks through.
    //
    // Fix: after each subcycle, test the straight-line segment xold→xcur
    // against every triangle in the particle's grid cell.  If a crossing
    // is found, stop subcycling immediately and return xcur to the move
    // loop.  The move loop's straight-line check from x to xnew will then
    // see the particle on the far side of the surface and handle the
    // collision normally.
    //
    // This check is skipped when nsub==1 (no subcycling) since the move
    // loop already handles that single segment.

    if (nsub > 1) {
      int gcell = icell;
      Grid::ChildCell *cells_local = grid->cells;
      if (cells_local[icell].nsplit <= 0 && cells_local[icell].isplit >= 0)
        gcell = grid->sinfo[cells_local[icell].isplit].icell;

      // (a) In-cell wall hit — clip xnew to intersection point so the
      //     outer move loop sees a trajectory that touches the wall
      //     exactly, not one that punched through. Critical for
      //     grazing divertor geometry.
      int nsurf_cell = cells_local[gcell].nsurf;
      if (nsurf_cell > 0) {
        surfint *csurfs_local = cells_local[gcell].csurfs;
        Surf::Tri *tris_local = surf->tris;
        double xc[3];
        double param;
        int side;
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = static_cast<int>(csurfs_local[m]);
          Surf::Tri *tri = &tris_local[isurf];
          if (Geometry::line_tri_intersect(xold, xcur,
                                           tri->p1, tri->p2, tri->p3,
                                           tri->norm, xc, param, side)) {
            v[0] = vcur[0];
            v[1] = vcur[1];
            v[2] = vcur[2];
            xnew[0] = xc[0];
            xnew[1] = xc[1];
            xnew[2] = xc[2];
            return;
          }
        }
      }

      // (b) Cell-exit — bail so SPARTA's outer move loop handles cell
      //     migration and per-cell surface detection on the remainder
      //     of the straight-line path (fixes the "wrong-cell csurfs"
      //     blind spot for subcycles that cross a cell boundary).
      //     Use half-open [lo, hi) to match grid->id_find_child so a
      //     particle exactly at `hi` is recognized as having left.
      const double *clo = cells_local[gcell].lo;
      const double *chi = cells_local[gcell].hi;
      if (xcur[0] < clo[0] || xcur[0] >= chi[0] ||
          xcur[1] < clo[1] || xcur[1] >= chi[1] ||
          xcur[2] < clo[2] || xcur[2] >= chi[2]) {
        v[0] = vcur[0];
        v[1] = vcur[1];
        v[2] = vcur[2];
        xnew[0] = xcur[0];
        xnew[1] = xcur[1];
        xnew[2] = xcur[2];
        return;
      }
    }
  }

  v[0] = vcur[0];
  v[1] = vcur[1];
  v[2] = vcur[2];
  xnew[0] = xcur[0];
  xnew[1] = xcur[1];
  xnew[2] = xcur[2];
}

/* ----------------------------------------------------------------------
   Hybrid Boris/GCA 3D pusher
   Uses full Boris when Larmor radius is well-resolved by the B gradient
   scale, and switches to GCA when the gyration is fast (small rho_L).
   Criterion: use GCA when L_B < switch_factor * rho_L
   where L_B = B / |grad B| and rho_L = v_perp / (|q/m| * B)
------------------------------------------------------------------------- */

void Pusher::push_hybrid_3d(int i, int icell, double dt,
                              double *x, double *v, double *xnew,
                              double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Hybrid pusher requires positive particle mass");

  // Neutrals: pure advection
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * update->echarge) / mass;
  const double qm_abs = std::fabs(qm);

  double *gca_x_vec = NULL;
  double *gca_y_vec = NULL;
  double *gca_z_vec = NULL;
  double *gca_vpar_vec = NULL;
  double *gca_mu_vec = NULL;
  double *gca_on_vec = NULL;
  if (gca_x_custom >= 0 && gca_y_custom >= 0 && gca_z_custom >= 0 &&
      gca_vpar_custom >= 0 && gca_mu_custom >= 0 && gca_on_custom >= 0) {
    gca_x_vec = particle->edvec[particle->ewhich[gca_x_custom]];
    gca_y_vec = particle->edvec[particle->ewhich[gca_y_custom]];
    gca_z_vec = particle->edvec[particle->ewhich[gca_z_custom]];
    gca_vpar_vec = particle->edvec[particle->ewhich[gca_vpar_custom]];
    gca_mu_vec = particle->edvec[particle->ewhich[gca_mu_custom]];
    gca_on_vec = particle->edvec[particle->ewhich[gca_on_custom]];
  }
  const bool have_gca_state =
    (gca_x_vec && gca_y_vec && gca_z_vec && gca_vpar_vec && gca_mu_vec && gca_on_vec);

  // --- Read E and B fields ---
  double E[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};

  ComputePlasmaFields *cp_bfield = NULL;
  FixBackground *pd_bfield = NULL;
  if (pusher_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[pusher_plasma_cidx];
    cp_bfield = dynamic_cast<ComputePlasmaFields *>(cp_base);
  } else if (pusher_plasma_fidx >= 0) {
    pd_bfield = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
  }
  // azimuth for cylindrical->Cartesian rotations is about the column axis,
  // not the domain origin
  const double col_x0 = cp_bfield ? cp_bfield->plasma_data.column_x0
                      : (pd_bfield ? pd_bfield->column_x0 : 0.0);
  const double col_y0 = cp_bfield ? cp_bfield->plasma_data.column_y0
                      : (pd_bfield ? pd_bfield->column_y0 : 0.0);

  if (pd_bfield) {
    const double xyz[3] = {x[0], x[1], x[2]};
    double ERp = 0.0, EZp = 0.0, Etp = 0.0;
    if (pd_bfield->query_efield_at_point(xyz, ERp, EZp, Etp, icell, i)) {
      const double phi_p = std::atan2(xyz[1] - pd_bfield->column_y0,
                                      xyz[0] - pd_bfield->column_x0);
      OpenEdge::RZphi_force_to_sparta(ERp, EZp, Etp, 3, false, phi_p,
                                      E[0], E[1], E[2]);
    }
  }
  if (E[0] == 0.0 && E[1] == 0.0 && E[2] == 0.0 && update->eperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[update->efieldfix], (update->efstyle == GFIELD),
                                   update->efield_active, i, icell, E);

  MagneticFieldFileDataParams Bcyl{};
  bool have_point_b = false;
  if (cp_bfield || pd_bfield) {
    // GCA requires consistent spatial derivatives of B. Prefer the loaded
    // equilibrium for this query; providers retain mesh/grid fallbacks.
    Bcyl = cp_bfield ? cp_bfield->query_bfield_at_point(x, true)
                     : pd_bfield->query_bfield_at_point(x, icell, i, true);
    if (Bcyl.Bmag > 0.0) {
      have_point_b = true;
      if (domain->dimension == 2) {
        if (domain->axisymmetric) {
          // 2D axi slots: x=Z, y=R, z=toroidal
          B[0] = Bcyl.bz;
          B[1] = Bcyl.br;
          B[2] = Bcyl.bt;
        } else {
          // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
          B[0] = Bcyl.br;
          B[1] = Bcyl.bz;
          B[2] = Bcyl.bt;
        }
      } else {
        const double rx = x[0] - col_x0, ry = x[1] - col_y0;
        const double rxy = std::sqrt(rx*rx + ry*ry);
        double cphi = 1.0, sphi = 0.0;
        if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
        B[0] = Bcyl.br * cphi - Bcyl.bt * sphi;
        B[1] = Bcyl.br * sphi + Bcyl.bt * cphi;
        B[2] = Bcyl.bz;
      }
    }
  }

  if (!have_point_b && update->bperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[update->bfieldfix], (update->bfstyle == GFIELD),
                                   update->bfield_active, i, icell, B);

  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

  // --- Read grad(|B|) from point-query if available, else cell cache ---
  double gradBmag_cart[3] = {0.0, 0.0, 0.0};
  double kappa_cart[3] = {0.0, 0.0, 0.0};
  double curlb_cart[3] = {0.0, 0.0, 0.0};
  double gradBmag_magnitude = 0.0;

  if (Bmag > 0.0) {
    if (have_point_b) {
      if (domain->dimension == 2) {
        if (domain->axisymmetric) {
          // 2D axi slots: x=Z, y=R, z=toroidal
          gradBmag_cart[0] = Bcyl.dBmag_dz;
          gradBmag_cart[1] = Bcyl.dBmag_dr;
          gradBmag_cart[2] = 0.0;
        } else {
          // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
          gradBmag_cart[0] = Bcyl.dBmag_dr;
          gradBmag_cart[1] = Bcyl.dBmag_dz;
          gradBmag_cart[2] = 0.0;
        }
      } else {
        const double rx = x[0] - col_x0, ry = x[1] - col_y0;
        const double rxy = std::sqrt(rx*rx + ry*ry);
        if (rxy > 1.0e-20) {
          const double cphi = rx / rxy, sphi = ry / rxy;
          gradBmag_cart[0] = Bcyl.dBmag_dr * cphi;
          gradBmag_cart[1] = Bcyl.dBmag_dr * sphi;
        } else {
          gradBmag_cart[0] = Bcyl.dBmag_dr;
          gradBmag_cart[1] = 0.0;
        }
        gradBmag_cart[2] = Bcyl.dBmag_dz;
      }
    }
    // No cell-cached fallback: if point query failed, gradBmag stays zero.

    gradBmag_magnitude = std::sqrt(gradBmag_cart[0]*gradBmag_cart[0] +
                                   gradBmag_cart[1]*gradBmag_cart[1] +
                                   gradBmag_cart[2]*gradBmag_cart[2]);

    // Compute kappa and curl(b̂) at particle position from point-query
    // B-component derivatives, avoiding cell-center geom_arr cache.
    if (have_point_b && Bcyl.Bmag > 0.0) {
      const double invBm = 1.0 / Bcyl.Bmag;
      const double bR = Bcyl.br * invBm;
      const double bphi = Bcyl.bt * invBm;
      const double bZ = Bcyl.bz * invBm;

      // ∂b̂_i/∂x = (1/|B|)(∂B_i/∂x - b̂_i ∂|B|/∂x)
      const double dbR_dR = invBm * (Bcyl.dBr_dr - bR * Bcyl.dBmag_dr);
      const double dbR_dZ = invBm * (Bcyl.dBr_dz - bR * Bcyl.dBmag_dz);
      const double dbphi_dR = invBm * (Bcyl.dBt_dr - bphi * Bcyl.dBmag_dr);
      const double dbphi_dZ = invBm * (Bcyl.dBt_dz - bphi * Bcyl.dBmag_dz);
      const double dbZ_dR = invBm * (Bcyl.dBz_dr - bZ * Bcyl.dBmag_dr);
      const double dbZ_dZ = invBm * (Bcyl.dBz_dz - bZ * Bcyl.dBmag_dz);

      double R_pt, Z_pt_unused;
      OpenEdge::sparta_to_RZ(x, domain->dimension, domain->axisymmetric,
                              R_pt, Z_pt_unused, col_x0, col_y0);
      if (R_pt < 1.0e-10) R_pt = 1.0e-10;
      const double invR_pt = 1.0 / R_pt;

      // κ = (b̂·∇)b̂ in cylindrical (axisymmetric, ∂/∂φ = 0)
      const double kR = bR * dbR_dR + bZ * dbR_dZ - bphi * bphi * invR_pt;
      const double kphi = bR * dbphi_dR + bZ * dbphi_dZ + bR * bphi * invR_pt;
      const double kZ = bR * dbZ_dR + bZ * dbZ_dZ;

      // curl(b̂) in cylindrical (axisymmetric, ∂/∂φ = 0)
      const double cR = -dbphi_dZ;
      const double cphi_c = dbR_dZ - dbZ_dR;
      const double cZ = bphi * invR_pt + dbphi_dR;

      if (domain->dimension == 2) {
        if (domain->axisymmetric) {
          // 2D axi slots: x=Z, y=R, z=toroidal
          kappa_cart[0] = kZ;   kappa_cart[1] = kR;   kappa_cart[2] = kphi;
          curlb_cart[0] = cZ;   curlb_cart[1] = cR;   curlb_cart[2] = cphi_c;
        } else {
          // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
          kappa_cart[0] = kR;   kappa_cart[1] = kZ;   kappa_cart[2] = kphi;
          curlb_cart[0] = cR;   curlb_cart[1] = cZ;   curlb_cart[2] = cphi_c;
        }
      } else {
        const double rx = x[0] - col_x0, ry = x[1] - col_y0;
        const double rxy = std::sqrt(rx*rx + ry*ry);
        double cphi_a = 1.0, sphi_a = 0.0;
        if (rxy > 1.0e-20) { cphi_a = rx / rxy; sphi_a = ry / rxy; }

        kappa_cart[0] = kR * cphi_a - kphi * sphi_a;
        kappa_cart[1] = kR * sphi_a + kphi * cphi_a;
        kappa_cart[2] = kZ;

        curlb_cart[0] = cR * cphi_a - cphi_c * sphi_a;
        curlb_cart[1] = cR * sphi_a + cphi_c * cphi_a;
        curlb_cart[2] = cZ;
      }
    }
    // No cell-cached fallback: if point query failed, kappa/curl_b stay zero.
  }

  // --- Per-particle sheath E-field (same approach as boris3D) ---
  if (update->sheath_flag && !update->sheath_kick && update->sheath_geom_cidx >= 0 &&
      (pusher_plasma_cidx >= 0 || pusher_plasma_fidx >= 0)) {
    Compute *cg = modify->compute[update->sheath_geom_cidx];

    // Resolve sub-cell to parent for geometry/plasma lookup
    int gcell = icell;
    Grid::ChildCell *cells_h = grid->cells;
    if (cells_h[icell].nsplit <= 0 && cells_h[icell].isplit >= 0)
      gcell = grid->sinfo[cells_h[icell].isplit].icell;

    auto *csg = dynamic_cast<ComputeNearestSurfGrid *>(cg);
    if (csg) {
      int midx = csg->midx_grid[gcell];

      // Refine midx using particle position when cell contains surfaces
      Grid::ChildCell *pc = &grid->cells[gcell];
      if (pc->nsurf > 0) {
        const int dim = domain->dimension;
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          double d;
          if (dim == 2) {
            if (!(surf->lines[m].mask & sbit)) continue;
            Surf::Line *ln = &surf->lines[m];
            d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                          (x[1]-ln->p1[1])*ln->norm[1]);
          } else {
            if (!(surf->tris[m].mask & sbit)) continue;
            Surf::Tri *tr = &surf->tris[m];
            d = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                          (x[1]-tr->p1[1])*tr->norm[1] +
                          (x[2]-tr->p1[2])*tr->norm[2]);
          }
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0) {
        // Use RAW triangle/line normal from the surface element
        double sh_nx, sh_ny, sh_nz;
        double sref[3];
        if (domain->dimension == 2) {
          Surf::Line *ln = &surf->lines[midx];
          sh_nx = ln->norm[0]; sh_ny = ln->norm[1]; sh_nz = 0.0;
          sref[0] = 0.5*(ln->p1[0]+ln->p2[0]);
          sref[1] = 0.5*(ln->p1[1]+ln->p2[1]);
          sref[2] = 0.0;
        } else {
          Surf::Tri *tr = &surf->tris[midx];
          sh_nx = tr->norm[0]; sh_ny = tr->norm[1]; sh_nz = tr->norm[2];
          sref[0] = (tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0;
          sref[1] = (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0;
          sref[2] = (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0;
        }
        const double nmag = std::sqrt(sh_nx*sh_nx + sh_ny*sh_ny + sh_nz*sh_nz);
        if (nmag > 0.0) { sh_nx /= nmag; sh_ny /= nmag; sh_nz /= nmag; }

        // Signed distance: positive = particle on +nhat side
        const double d_raw =
          (x[0]-sref[0])*sh_nx + (x[1]-sref[1])*sh_ny + (x[2]-sref[2])*sh_nz;

        // Sheath coefficients + d_max + phi_total: per-element cache when
        // the plasma is a static fix background (built once at the element
        // midpoint/centroid), else per-particle evaluation. The cached
        // path removes the ~16-field plasma query and full CM prep that
        // used to run on every hybrid particle-move.
        SheathModels::SheathEmagCoeffs hc;
        double h_d_max = 0.0, h_phi_total = 0.0;
        int h_active = 0;
        if (sheath_cache_enabled && pusher_plasma_fidx >= 0) {
          const int nsurf_all = surf->nlocal + surf->nghost;
          if ((int) sheath_cache.size() != nsurf_all)
            sheath_cache.assign(nsurf_all, SheathElemCache{});
          if (midx < nsurf_all) {
            SheathElemCache &C = sheath_cache[midx];
            if (C.state == 0) {
              if (domain->dimension == 3) build_sheath_cache_entry_3d(midx, C);
              else build_sheath_cache_entry(midx, C);
            }
            if (C.state == 1) {
              hc = C.coeffs;
              h_d_max = C.d_max;
              h_phi_total = C.phi_total;
              h_active = 1;
            }
          }
        } else {
          ComputePlasmaFields *cp = nullptr;
          FixBackground *pd = nullptr;
          if (pusher_plasma_cidx >= 0) {
            Compute *cp_base = modify->compute[pusher_plasma_cidx];
            cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
          } else if (pusher_plasma_fidx >= 0) {
            pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
          }
          if (cp || pd) {
            // Point-query plasma data at particle position (dim-correct:
            // the old code passed dim=2 unconditionally, which mis-mapped
            // (R,Z) for 3D hybrid runs).
            PlasmaFileParams sh_pf = cp ? cp->query_plasma_at_point(x)
                                        : query_plasma_from_fix(pd, x, domain->dimension,
                                                                domain->axisymmetric, icell);
            const double sh_te = sh_pf.temp_e;
            const double sh_ne = sh_pf.dens_e;
            const double sh_ti = sh_pf.temp_i;
            if (sh_te > 0.0 && sh_ne > 0.0) {
              // Reuse B[] and Bmag already computed at particle position
              double bvec[3] = {B[0], B[1], B[2]};
              const double sh_bmag = Bmag;
              double sh_alpha_deg = 90.0;
              if (sh_bmag > 0.0) {
                double nvec[3] = {sh_nx, sh_ny, sh_nz};
                SheathModels::ChoduraMetrics cm =
                  SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
                sh_alpha_deg = cm.alpha_deg;
              }
              h_d_max = sheath_auto_dmax(sh_te, sh_ti, sh_ne, sh_bmag,
                                         sh_alpha_deg, update->sheath_mD_amu,
                                         update->sheath_dmax);
              hc = SheathModels::sheath_prepare_coulette_manfredi(
                       sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha_deg,
                       update->sheath_mD_amu, 0.0);
              h_phi_total = SheathModels::sheath_phi_at_distance(hc, 0.0);
              h_active = 1;
            }
          }
        }

        if (h_active && update->sheath_boundary) {
          // Boundary mode: no spatial field (it would double-count the
          // inbound wall kick applied in the move loop). Apply the
          // outbound escape barrier on the particle velocity instead,
          // with the same reflect/decelerate/paid semantics as the Boris
          // pushers. Limitation: a GCA particle carrying persistent
          // guiding-center state re-reads that state after this kick, so
          // the barrier is only fully effective on the Boris fallback and
          // fresh GC inits — use boris_near for trustworthy near-wall
          // boundary-mode physics in hybrid/GCA runs.
          if (h_phi_total > 0.0 && d_raw > 0.0 && d_raw <= h_d_max) {
            int *paid_vec = (update->sheath_paid_custom >= 0)
              ? particle->eivec[particle->ewhich[update->sheath_paid_custom]] : nullptr;
            const double vn = v[0]*sh_nx + v[1]*sh_ny + v[2]*sh_nz;
            if (vn <= 0.0) {
              if (paid_vec) paid_vec[i] = 0;
            } else if (!paid_vec || paid_vec[i] == 0) {
              const double Zc = std::fabs(charge);
              const double barrier_J = Zc * update->echarge * h_phi_total;
              const double KEn = 0.5 * mass * vn * vn;
              double dvn;
              if (KEn < barrier_J) {
                dvn = 2.0 * vn;                     // reflect: prompt redep
                sheath_diag_nreflect++;
              } else {
                const double vn_new = std::sqrt(vn*vn - 2.0*barrier_J/mass);
                dvn = vn - vn_new;                  // decelerate: escapes
                if (paid_vec) paid_vec[i] = 1;
                sheath_diag_nescape++;
              }
              v[0] -= dvn * sh_nx;
              v[1] -= dvn * sh_ny;
              v[2] -= dvn * sh_nz;
            }
          }
        } else if (h_active && d_raw > 0.0 && d_raw < h_d_max) {
          // Spatial mode: frozen-field sheath E for this step. The GCA
          // RHS integrates the frozen E consistently over the step (no
          // gyration, so the Boris gyro-pumping loop does not arise);
          // residual non-conservation is limited to step-boundary
          // geometry switches. Use boris_near for resolved near-wall
          // gyro physics.
          const double emag = SheathModels::sheath_emag_at_distance(hc, d_raw);
          // E into the wall along -n (inward-normal convention).
          E[0] -= emag * sh_nx;
          E[1] -= emag * sh_ny;
          E[2] -= emag * sh_nz;
          sheath_diag_nengage++;
          sheath_diag_esum += emag;
          if (emag > sheath_diag_emax) sheath_diag_emax = emag;
        }
      }
    }
  }

  // --- Switching criterion ---
  // mode=gca: always GCA (no Boris fallback) when B is available.
  // mode=hybrid: use rho_L/L_B criterion below.
  bool use_gca = false;
  if (pusher_mode == PUSHER_GCA && Bmag > 0.0 && qm_abs > 0.0) use_gca = true;

  // boris_near override: when > 0 and a sheath_geom (compute nearest_surf/grid)
  // is configured, force Boris (gyro-resolved) for particles within
  // pusher_boris_near metres of the cell's nearest surf in that group.
  // Use cgeom's per-cell midx_grid (already populated) -> compute distance to
  // that one surf only. Cheap and good enough for visualization buffering.
  if (use_gca && pusher_boris_near > 0.0 && update->sheath_geom_cidx >= 0) {
    Compute *cg_bn = modify->compute[update->sheath_geom_cidx];
    auto *csg_bn = dynamic_cast<ComputeNearestSurfGrid *>(cg_bn);
    if (csg_bn) {
      int gcell_bn = icell;
      Grid::ChildCell *cells_bn = grid->cells;
      if (cells_bn[icell].nsplit <= 0 && cells_bn[icell].isplit >= 0)
        gcell_bn = grid->sinfo[cells_bn[icell].isplit].icell;
      const int midx_bn = csg_bn->midx_grid[gcell_bn];
      if (midx_bn >= 0) {
        double dnear;
        if (domain->dimension == 2) {
          Surf::Line *ln = &surf->lines[midx_bn];
          dnear = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                            (x[1]-ln->p1[1])*ln->norm[1]);
        } else {
          Surf::Tri *tr = &surf->tris[midx_bn];
          dnear = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                            (x[1]-tr->p1[1])*tr->norm[1] +
                            (x[2]-tr->p1[2])*tr->norm[2]);
        }
        if (dnear < pusher_boris_near) use_gca = false;
      }
    }
  }

  if (!use_gca && Bmag > 0.0 && qm_abs > 0.0) {
    const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
    double v_perp = 0.0;
    if (have_gca_state && gca_on_vec[i] > 0.5) {
      // In persistent GCA mode, use stored mu to evaluate rho_L.
      const double mu_eff = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
      const double vperp2 = (2.0 * mu_eff * Bmag) / mass;
      v_perp = (vperp2 > 0.0) ? std::sqrt(vperp2) : 0.0;
    } else {
      const double v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
      const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
      double vperp2 = v2 - v_par * v_par;
      if (vperp2 < 0.0) vperp2 = 0.0;
      v_perp = std::sqrt(vperp2);
    }

    const double rho_L = GCAPusher::larmor_radius(v_perp, qm_abs, Bmag);
    const double L_B = GCAPusher::grad_b_length(Bmag, gradBmag_magnitude);

    // Switching criterion: use GCA when L_B < switch_factor * rho_L
    // i.e., the gradient scale is smaller than the Larmor orbit
    // Equivalently: rho_L is large relative to gradient scale → fast gyration
    // Actually: use GCA when rho_L is SMALL (fast gyration), i.e.,
    // the particle gyrates many times within one gradient scale length.
    // GCA when d_char < 2.5 * rho_L  →  rho_L > L_B / 2.5
    // Rearranged: use GCA when rho_L < L_B / switch_factor
    // This means: gradient is gentle relative to orbit → GCA is valid
    if (rho_L > 0.0 && L_B < 1.0e19) {
      use_gca = (rho_L < L_B / pusher_gca_switch);
    }
  }

  if (use_gca) {
    // --- GCA path ---
    GCAPusher::GCAState gca;
    if (have_gca_state && gca_on_vec[i] > 0.5) {
      gca.X[0] = gca_x_vec[i];
      gca.X[1] = gca_y_vec[i];
      gca.X[2] = gca_z_vec[i];
      gca.v_par = gca_vpar_vec[i];
      gca.mu = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
    } else {
      gca = GCAPusher::init_from_particle(x, v, mass, B);
    }

    // GCA integration. E, B, and their derivatives are sampled once above
    // and frozen over this timestep for all three integrators.
    //   rk4 (default): 4 full-Littlejohn RHS evaluations.
    //   rk2          : 2 full-Littlejohn RHS evaluations (midpoint method).
    //   simple       : 1 reduced RHS without the B* curvature/curl(b) terms.
    // Thus rk2 halves the full-RHS algebra relative to rk4, but it does not
    // eliminate additional mesh queries: neither method performs a stage
    // field query in the current frozen-field formulation.
    if (pusher_gca_integrator == GCA_SIMPLE) {
      GCAPusher::push_gca(qm, dt, mass, E, B, gradBmag_cart, gca);
    } else if (pusher_gca_integrator == GCA_RK2) {
      GCAPusher::push_gca_rk2(qm, dt, mass, E, B, Bmag, gradBmag_cart,
                              kappa_cart, curlb_cart, gca);
    } else {
      GCAPusher::push_gca_rk4(qm, dt, mass, E, B, Bmag, gradBmag_cart,
                              kappa_cart, curlb_cart, gca);
    }

    if (have_gca_state) {
      gca_x_vec[i] = gca.X[0];
      gca_y_vec[i] = gca.X[1];
      gca_z_vec[i] = gca.X[2];
      gca_vpar_vec[i] = gca.v_par;
      gca_mu_vec[i] = gca.mu;
      gca_on_vec[i] = 1.0;
    }

    // Keep persistent GC state, but reconstruct full v for diagnostics and
    // clean Boris fallback if regime switching happens.
    const double phi_golden = 0.6180339887498949;
    const double two_pi = 2.0 * M_PI;
    const double omega_c = std::fabs(qm) * Bmag;
    const double phase_turns =
      particle->particles[i].id * phi_golden +
      (omega_c * dt * static_cast<double>(update->ntimestep)) / two_pi;
    double rand_u = phase_turns - std::floor(phase_turns);
    GCAPusher::gca_to_particle(gca, B, mass, rand_u, xnew, v);
    // Pure-GCA mode: advect the particle AT the guiding center (no gyro
    // offset). The offset only matters for surface-impact geometry; with
    // it, particles whose GC sits within rho_L of a periodic cap jitter
    // across it every step (cap ping-pong). Velocity keeps the full
    // reconstruction for diagnostics and Boris handoff.
    if (pusher_mode == PUSHER_GCA) {
      xnew[0] = gca.X[0];
      xnew[1] = gca.X[1];
      xnew[2] = gca.X[2];
    }
  } else {
    // --- Boris path (with subcycling) ---
    const int nsub = (charge != 0.0 && pusher_subcycles > 0) ? pusher_subcycles : 1;
    const double dt_sub = dt / static_cast<double>(nsub);

    double xcur[3] = {x[0], x[1], x[2]};
    double vcur[3] = {v[0], v[1], v[2]};

    // Cache B-field once at initial position for subcycling
    if (cp_bfield || pd_bfield) {
      MagneticFieldFileDataParams Bc = cp_bfield
          ? cp_bfield->query_bfield_at_point(xcur)
          : pd_bfield->query_bfield_at_point(xcur, icell, i);
      if (Bc.Bmag > 0.0) {
        if (domain->dimension == 2) {
          if (domain->axisymmetric) {
            // 2D axi slots: x=Z, y=R, z=toroidal
            B[0] = Bc.bz;
            B[1] = Bc.br;
            B[2] = Bc.bt;
          } else {
            // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
            B[0] = Bc.br;
            B[1] = Bc.bz;
            B[2] = Bc.bt;
          }
        } else {
          const double rx = xcur[0] - col_x0, ry = xcur[1] - col_y0;
          const double rxy = std::sqrt(rx*rx + ry*ry);
          double cphi = 1.0, sphi = 0.0;
          if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
          B[0] = Bc.br * cphi - Bc.bt * sphi;
          B[1] = Bc.br * sphi + Bc.bt * cphi;
          B[2] = Bc.bz;
        }
      }
    }

    for (int isub = 0; isub < nsub; isub++) {
      BorisGrid::push_velocity(qm, dt_sub, E, B, vcur);
      xcur[0] += vcur[0] * dt_sub;
      xcur[1] += vcur[1] * dt_sub;
      xcur[2] += vcur[2] * dt_sub;
    }

    v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
    xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];

    if (have_gca_state) gca_on_vec[i] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   Pusher per-run init: resolve plasma compute/fix IDs (the GCA branch
   needs a per-grid compute plasma/fields for grad|B|, curvature, and
   curl(b̂)) and lazily register the persistent guiding-center custom
   particle attributes used by the hybrid pusher.

   Called from Update::init() after fixes are registered but before
   Modify::init() runs (so compute->has_equilibrium has not been
   populated yet — that's checked at runtime in gca_rhs and the GCA
   integrators).
------------------------------------------------------------------------- */

void Pusher::init()
{
  if (pusher_mode != PUSHER_HYBRID && pusher_mode != PUSHER_GCA) return;

  // hybrid/gca still integrate positions inside the pusher, which breaks
  // the linear-segment contract of the axisymmetric mover (see
  // push_boris_2d). Refuse rather than silently lose particles.
  if (domain->axisymmetric)
    error->all(FLERR,
               "global pusher mode hybrid/gca not supported for "
               "axisymmetric domains; use mode boris");

  if (!pusher_plasma_cid)
    error->all(FLERR,"global gca requires plasma provider ID");
  pusher_plasma_cidx = modify->find_compute(pusher_plasma_cid);
  pusher_plasma_fidx = -1;
  if (pusher_plasma_cidx >= 0) {
    if (!modify->compute[pusher_plasma_cidx]->per_grid_flag)
      error->all(FLERR,"global gca: plasma compute must be per-grid");
  } else {
    pusher_plasma_fidx = modify->find_fix(pusher_plasma_cid);
    if (pusher_plasma_fidx < 0)
      error->all(FLERR,"global gca: plasma provider ID not found");
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
    if (!pd)
      error->all(FLERR,
                 "global gca: plasma fix provider must be style background");
  }

  // GCA needs smooth B-field derivatives (grad|B|, curvature, curl(b̂))
  // from an equilibrium psi map. When the embedded /equilibrium/* group
  // in plasma.h5 (or an explicit `equilibrium <file>` keyword) is
  // present, ComputePlasmaFields / FixBackground query_bfield_at_point
  // picks it up; otherwise grad-|B| sample returns the unmodified B and
  // the GCA reverts to its simpler kernel.

  // Persistent guiding-center state per particle. Keep this state across
  // timesteps to avoid re-initializing from instantaneous gyromotion
  // every step.
  const int custom_double = 1;
  if (gca_x_custom < 0) {
    gca_x_custom = particle->find_custom((char *) "gca_x");
    if (gca_x_custom < 0)
      gca_x_custom = particle->add_custom((char *) "gca_x", custom_double, 0);
  }
  if (gca_y_custom < 0) {
    gca_y_custom = particle->find_custom((char *) "gca_y");
    if (gca_y_custom < 0)
      gca_y_custom = particle->add_custom((char *) "gca_y", custom_double, 0);
  }
  if (gca_z_custom < 0) {
    gca_z_custom = particle->find_custom((char *) "gca_z");
    if (gca_z_custom < 0)
      gca_z_custom = particle->add_custom((char *) "gca_z", custom_double, 0);
  }
  if (gca_vpar_custom < 0) {
    gca_vpar_custom = particle->find_custom((char *) "gca_vpar");
    if (gca_vpar_custom < 0)
      gca_vpar_custom = particle->add_custom((char *) "gca_vpar", custom_double, 0);
  }
  if (gca_mu_custom < 0) {
    gca_mu_custom = particle->find_custom((char *) "gca_mu");
    if (gca_mu_custom < 0)
      gca_mu_custom = particle->add_custom((char *) "gca_mu", custom_double, 0);
  }
  if (gca_on_custom < 0) {
    gca_on_custom = particle->find_custom((char *) "gca_on");
    if (gca_on_custom < 0)
      gca_on_custom = particle->add_custom((char *) "gca_on", custom_double, 0);
  }
}

/* ----------------------------------------------------------------------
   `global pusher ...` keyword parser. Called from Update::global() when
   the user passes `global pusher`. Handles the full sub-keyword tree
   (mode, subcycles, plasma, gca_switch, gca_integrator, dump, dump_every,
   bad_dt_check, bad_dt_limit, sheath off|kick|spatial [geom mD_amu]).

   Caller has already matched the literal "pusher" token; on entry
   `iarg` points at it. We advance past "pusher" and then consume sub-
   keywords until we either run out of args or hit a token that isn't
   a recognised pusher sub-keyword (in which case control returns to
   the outer global-keyword loop).
------------------------------------------------------------------------- */

void Pusher::global_keyword(int narg, char **arg, int &iarg)
{
  iarg++;  // skip "pusher"
  while (iarg < narg) {
    if (strcmp(arg[iarg], "mode") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher mode");
      if (strcmp(arg[iarg+1], "boris") == 0) pusher_mode = PUSHER_BORIS;
      else if (strcmp(arg[iarg+1], "hybrid") == 0) pusher_mode = PUSHER_HYBRID;
      else if (strcmp(arg[iarg+1], "gca") == 0) pusher_mode = PUSHER_GCA;
      else error->all(FLERR,
        "global pusher mode must be boris, hybrid, or gca");
      iarg += 2;
    } else if (strcmp(arg[iarg], "subcycles") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher subcycles");
      pusher_subcycles = input->inumeric(FLERR, arg[iarg+1]);
      if (pusher_subcycles <= 0)
        error->all(FLERR, "global pusher subcycles must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "plasma") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher plasma");
      delete [] pusher_plasma_cid;
      int n = strlen(arg[iarg+1]) + 1;
      pusher_plasma_cid = new char[n];
      strcpy(pusher_plasma_cid, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gca_switch") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher gca_switch");
      pusher_gca_switch = input->numeric(FLERR, arg[iarg+1]);
      if (pusher_gca_switch <= 0.0)
        error->all(FLERR, "global pusher gca_switch must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_near") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher boris_near");
      pusher_boris_near = input->numeric(FLERR, arg[iarg+1]);
      if (pusher_boris_near < 0.0)
        error->all(FLERR, "global pusher boris_near must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "gca_integrator") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher gca_integrator");
      if (strcmp(arg[iarg+1], "rk4") == 0)         pusher_gca_integrator = GCA_RK4;
      else if (strcmp(arg[iarg+1], "simple") == 0) pusher_gca_integrator = GCA_SIMPLE;
      else if (strcmp(arg[iarg+1], "rk2") == 0)    pusher_gca_integrator = GCA_RK2;
      else error->all(FLERR,
        "global pusher gca_integrator must be rk2, rk4, or simple");
      iarg += 2;
    } else if (strcmp(arg[iarg], "dump") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher dump");
      if (strcmp(arg[iarg+1], "yes") == 0) pusher_dump_flag = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) pusher_dump_flag = 0;
      else error->all(FLERR, "global pusher dump must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "dump_every") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher dump_every");
      pusher_dump_every = input->inumeric(FLERR, arg[iarg+1]);
      if (pusher_dump_every <= 0)
        error->all(FLERR, "global pusher dump_every must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "bad_dt_check") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher bad_dt_check");
      if (strcmp(arg[iarg+1], "yes") == 0) pusher_bad_dt_check = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) pusher_bad_dt_check = 0;
      else error->all(FLERR, "global pusher bad_dt_check must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "bad_dt_limit") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher bad_dt_limit");
      pusher_bad_dt_limit = input->numeric(FLERR, arg[iarg+1]);
      if (pusher_bad_dt_limit <= 0.0)
        error->all(FLERR, "global pusher bad_dt_limit must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "sheath") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher sheath");
      const char *mode = arg[iarg+1];
      if (strcmp(mode, "off") == 0)         { update->sheath_flag = 0; update->sheath_kick = 0; update->sheath_boundary = 0; }
      else if (strcmp(mode, "kick") == 0)    { update->sheath_flag = 1; update->sheath_kick = 1; update->sheath_boundary = 0; }
      else if (strcmp(mode, "spatial") == 0) { update->sheath_flag = 1; update->sheath_kick = 0; update->sheath_boundary = 0; }
      else if (strcmp(mode, "boundary") == 0){ update->sheath_flag = 1; update->sheath_kick = 0; update->sheath_boundary = 1; }
      else error->all(FLERR, "global pusher sheath must be off|kick|spatial|boundary");
      iarg += 2;
      while (iarg < narg) {
        if (strcmp(arg[iarg], "geom") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher sheath geom");
          delete [] update->sheath_geom_cid;
          int n = strlen(arg[iarg+1]) + 1;
          update->sheath_geom_cid = new char[n];
          strcpy(update->sheath_geom_cid, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "mD_amu") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher sheath mD_amu");
          update->sheath_mD_amu = input->numeric(FLERR, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "dmax") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher sheath dmax");
          update->sheath_dmax = input->numeric(FLERR, arg[iarg+1]);
          if (update->sheath_dmax < 0.0)
            error->all(FLERR, "global pusher sheath dmax must be >= 0");
          iarg += 2;
        } else if (strcmp(arg[iarg], "waveform") == 0) {
          if (iarg + 2 >= narg)
            error->all(FLERR,
              "global pusher sheath waveform needs <surface_attr> <frequency_hz>");
          delete [] update->sheath_waveform_attr;
          int n = strlen(arg[iarg+1]) + 1;
          update->sheath_waveform_attr = new char[n];
          strcpy(update->sheath_waveform_attr, arg[iarg+1]);
          update->sheath_frequency_hz = input->numeric(FLERR, arg[iarg+2]);
          if (update->sheath_frequency_hz < 0.0)
            error->all(FLERR,
              "global pusher sheath waveform frequency must be >= 0");
          iarg += 3;
        } else break;
      }
      if (update->sheath_flag && !update->sheath_geom_cid)
        error->all(FLERR, "global pusher sheath kick|spatial requires geom <ID>");
    } else break;  // next keyword belongs to a different global option
  }
}
