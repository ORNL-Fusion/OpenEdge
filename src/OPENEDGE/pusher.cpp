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
#include "mixture.h"
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

// golden-ratio conjugate: decorrelates the per-particle-id gyro-phase
// seed used when reconstructing full v from guiding-center state
const double GCA_PHASE_GOLDEN = 0.6180339887498949;

// A1 flux weight p(phi) ~ max(-vn(phi),0) with vn = a + cx cos + cy sin:
// deterministic golden-sequence rejection seeded by the hash phase.
// Shared by the 3D pre-step sampler and the axi impact materialization —
// arithmetic must stay identical (certified in the B2 oblique test).
static double flux_phase_sample(double a, double cx, double cy, double u0)
{
  const double cmag = std::sqrt(cx*cx + cy*cy);
  const double wmax = -a + cmag;   // max of -vn over phase
  if (wmax <= 0.0) return u0;
  const double two_pi = 2.0 * M_PI;
  double u = u0;
  for (int trial = 0; trial < 32; trial++) {
    const double ph = u * two_pi;
    const double w = -(a + cx*std::cos(ph) + cy*std::sin(ph));
    const double ua = std::fmod(u * 971.0 + 0.372549, 1.0);
    if (w > 0.0 && ua * wmax <= w) return u;
    u = std::fmod(u + GCA_PHASE_GOLDEN, 1.0);
  }
  return u0;
}

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
inline double sheath_waveform_drop(Update *update, Surf *surf, int midx,
                                   double t_offset = 0.0)
{
  const int idx = update->sheath_waveform_custom;
  if (idx < 0 || midx < 0 || midx >= surf->nlocal + surf->nghost)
    return 0.0;
  double **wave = surf->edarray_local[surf->ewhich[idx]];
  if (!wave) return 0.0;
  const double now = update->time +
      (update->ntimestep - update->time_last_update) * update->dt + t_offset;
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
  pusher_boris_near_rhol = 0;
  switch_log_file     = NULL;
  switch_log_fp       = NULL;
  pusher_gc_wall_flux = 0;
  pusher_skip_mix     = NULL;
  pusher_skip_flag    = NULL;
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
  gca_mode_custom  = -1;
  gca_valid_custom = -1;
  gca_chi_custom   = -1;
}

Pusher::~Pusher()
{
  delete [] pusher_plasma_cid;
  if (switch_log_fp) fclose(switch_log_fp);
  delete [] switch_log_file;
  delete [] pusher_skip_mix;
  delete [] pusher_skip_flag;
}

/* ----------------------------------------------------------------------
   resolve the skip mixture (dust grains etc.) to a per-species flag —
   called from Update::init once mixtures exist
------------------------------------------------------------------------- */

void Pusher::resolve_skip_species()
{
  delete [] pusher_skip_flag;
  pusher_skip_flag = NULL;
  if (!pusher_skip_mix) return;
  const int imix = particle->find_mixture(pusher_skip_mix);
  if (imix < 0)
    error->all(FLERR, "global pusher skip: mixture ID does not exist");
  Mixture *mix = particle->mixture[imix];
  pusher_skip_flag = new int[particle->nspecies]();
  for (int m = 0; m < mix->nspecies; m++)
    pusher_skip_flag[mix->species[m]] = 1;
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

  // B at the centroid via the FULL point query (mesh -> equilibrium ->
  // constant/bcart); the (R,Z)-only bfield_at overload cannot serve a
  // Cartesian bcart field, which silently deactivated the sheath here.
  MagneticFieldFileDataParams Bq = pd->query_bfield_at_point(xmid);
  const double rx = xmid[0] - pd->column_x0;
  const double ry = xmid[1] - pd->column_y0;
  const double rxy = std::sqrt(rx*rx + ry*ry);
  double bvec[3];
  if (rxy > 1.0e-20) {
    const double cphi = rx / rxy, sphi = ry / rxy;
    bvec[0] = Bq.br * cphi - Bq.bt * sphi;
    bvec[1] = Bq.br * sphi + Bq.bt * cphi;
    bvec[2] = Bq.bz;
  } else {
    bvec[0] = Bq.br; bvec[1] = 0.0; bvec[2] = Bq.bz;
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
  // skip mixture (dust grains): pure advection even when charged
  if (pusher_skip_flag &&
      pusher_skip_flag[particle->particles[i].ispecies]) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }
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
    // stale post-rebalance cell index: skip rather than read past midx_grid
    if (csg && gcell >= 0 && gcell < csg->nglocal) {
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
  // The "paid" flag is a TRANSIT state: the escape charge fires once per
  // band transit and re-arms only when the particle actually LEAVES the
  // sheath band — a vn-sign re-arm would charge oblique-field ions
  // repeatedly, since normal gyro-velocity flips sign inside the band.
  // barrier engages only inside the sheath band (see boris3D twin)
  if (update->sheath_boundary && sh_active && update->sheath_paid_custom >= 0 &&
      sh_d0_sign > 0.0 && sh_d_cur > sh_d_max) {
    int *st = particle->eivec[particle->ewhich[update->sheath_paid_custom]];
    if (st) st[i] = SH_OUTSIDE;   // verified band exit: transit complete
  }
  if (update->sheath_boundary && sh_active && sh_d0_sign > 0.0 &&
      sh_d_cur <= sh_d_max) {
    // unified evaluator (cache path); prefetch value as fallback
    double phi_here = sheath_phi_wall(sh_midx, 0.0);
    if (phi_here < 0.0) phi_here = sh_phi_total;
    if (phi_here > 0.0) {
    int *st = (update->sheath_paid_custom >= 0)
      ? particle->eivec[particle->ewhich[update->sheath_paid_custom]] : nullptr;
    if (st && st[i] == SH_OUTSIDE) st[i] = SH_ARMED;   // band entry
    double vR = 0.0, vZ = 0.0, vphi = 0.0;
    OpenEdge::sparta_v_to_RZphi(vcur, dim, axi, 0.0, vR, vZ, vphi);
    const double vn = vR*sh_nR + vZ*sh_nZ;   // >0 = outbound (into fluid)

    if (vn > 0.0) {
      const double Zc = std::fabs(charge);
      const double barrier_J = Zc * update->echarge * phi_here;
      const double KEn = 0.5 * mass * vn * vn;
      double dvn = 0.0;                       // amount to remove from vn
      if (KEn < barrier_J) {
        // sub-barrier: ALWAYS reflect (elastic, repeatable) — the
        // confinement mechanism; see SheathTransit doc in pusher.h
        dvn = 2.0 * vn;
        sheath_diag_nreflect++;
      } else if (!st || st[i] != SH_APPLIED) {
        const double vn_new = std::sqrt(vn*vn - 2.0*barrier_J/mass);
        dvn = vn - vn_new;                    // decelerate: escapes
        if (st) st[i] = SH_APPLIED;           // exactly once per transit
        sheath_diag_nescape++;
      }
      const double vR2 = vR - dvn*sh_nR;
      const double vZ2 = vZ - dvn*sh_nZ;
      const double v_cyl[3] = {vR2, vZ2, vphi};
      OpenEdge::RZphi_force_to_sparta(v_cyl[0], v_cyl[1], v_cyl[2], dim, axi,
                                       0.0, vcur[0], vcur[1], vcur[2]);
    }
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

  // skip mixture (dust grains): pure advection even when charged
  if (pusher_skip_flag &&
      pusher_skip_flag[particle->particles[i].ispecies]) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

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
    // stale post-rebalance cell index: skip rather than read past midx_grid
    if (csg && gcell >= 0 && gcell < csg->nglocal) {
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

          // require B > 0 like the 2D fallback and the cache builders:
          // with B = 0, sheath_auto_dmax's rho_i blows up and a spurious
          // alpha = 90 sheath would engulf the whole domain
          sh_active = (sh_bmag > 0.0);
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
    sheath_diag_nactive++;   // 3D near-wall count (2D counts in its own block)
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

  // Per-particle sheath trace (parity debugging): OE_SHEATH_TRACE_ID=<id>
  static const long sh_trace_id =
      getenv("OE_SHEATH_TRACE_ID") ? atol(getenv("OE_SHEATH_TRACE_ID")) : -1;
  const bool sh_trace =
      (sh_trace_id >= 0 && (long) particle->particles[i].id == sh_trace_id);
  if (sh_trace)
    printf("SHTRACE cpu step %lld id %ld pre: active=%d cache=%d midx=%d "
           "dmax=%.9e n=(%.9e,%.9e,%.9e) sref=(%.9e,%.9e,%.9e)\n",
           (long long) update->ntimestep, sh_trace_id, sh_active,
           sh_from_cache, sh_midx, sh_d_max, sh_nx, sh_ny, sh_nz,
           sh_sref[0], sh_sref[1], sh_sref[2]);

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
  // decelerates once per TRANSIT. The paid flag is a transit state:
  // re-armed only when the particle actually LEAVES the sheath band —
  // with oblique B the normal gyro-velocity changes sign inside the band
  // and a vn-sign re-arm would charge the barrier repeatedly.
  if (update->sheath_boundary && sh_active && update->sheath_paid_custom >= 0 &&
      sh_d0_sign > 0.0 && sh_d0 > sh_d_max) {
    int *st = particle->eivec[particle->ewhich[update->sheath_paid_custom]];
    if (st) st[i] = SH_OUTSIDE;   // verified band exit: transit complete
  }
  if (update->sheath_boundary && sh_active && sh_d0_sign > 0.0 &&
      sh_d0 <= sh_d_max) {
    // unified evaluator (cache path); per-particle coeffs fallback for
    // non-static plasma sources
    double sh_phi_total = sheath_phi_wall(sh_midx, 0.0);
    if (sh_phi_total < 0.0)
      sh_phi_total = SheathModels::sheath_phi_at_distance(sh_coeffs, 0.0) +
                     sheath_waveform_drop(update, surf, sh_midx);
    if (sh_phi_total > 0.0) {
      int *st = (update->sheath_paid_custom >= 0)
        ? particle->eivec[particle->ewhich[update->sheath_paid_custom]] : nullptr;
      if (st && st[i] == SH_OUTSIDE) st[i] = SH_ARMED;   // band entry
      const double vn = vcur[0]*sh_nx + vcur[1]*sh_ny + vcur[2]*sh_nz;
      // vn > 0 = outbound (into fluid, along the outward surface normal)

      if (vn > 0.0) {
        const double Zc = std::fabs(charge);
        const double barrier_J = Zc * update->echarge * sh_phi_total;
        const double KEn = 0.5 * mass * vn * vn;
        double dvn = 0.0;                       // amount to remove from vn
        if (KEn < barrier_J) {
          // sub-barrier: ALWAYS reflect (elastic, repeatable) — this is
          // the confinement mechanism; see SheathTransit doc
          dvn = 2.0 * vn;
          sheath_diag_nreflect++;
        } else if (!st || st[i] != SH_APPLIED) {
          const double vn_new = std::sqrt(vn*vn - 2.0*barrier_J/mass);
          dvn = vn - vn_new;                    // decelerate: escapes
          if (st) st[i] = SH_APPLIED;           // exactly once per transit
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
      // full point query (mesh -> equilibrium -> constant/bcart) — the
      // (R,Z)-only bfield_at overload cannot serve a Cartesian bcart field
      MagneticFieldFileDataParams Bcyl = pd->query_bfield_at_point(xcur, icell, i);
      if (Bcyl.Bmag > 0.0) {
        const double rx = xcur[0] - pd->column_x0;
        const double ry = xcur[1] - pd->column_y0;
        const double rxy = std::sqrt(rx * rx + ry * ry);
        double cphi = 1.0, sphi = 0.0;
        if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
        B_cached[0] = Bcyl.br * cphi - Bcyl.bt * sphi;
        B_cached[1] = Bcyl.br * sphi + Bcyl.bt * cphi;
        B_cached[2] = Bcyl.bz;
      }
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
        if (sh_trace)
          printf("SHTRACE cpu step %lld sub %d eng: d_old=%.9e d_new=%.9e "
                 "phi_old=%.9e phi_new=%.9e dKE=%.9e bank=%.9e\n",
                 (long long) update->ntimestep, isub, d_old, d_new,
                 phi_old, phi_new, dKE_J, sh_bank_vec ? sh_bank_vec[i] : -1.0);
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
            sheath_diag_nreflect++;   // spatial-mode turning point
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
   Sample E, B, |B|, grad|B|, curvature and curl(b̂) at xpos into F,
   slot-mapped for 2D axi / 2D Cartesian / 3D. Serves both the k1 fields
   and the per-stage resampling of the GCA integrators. Returns 1 when
   the point B query (with equilibrium derivatives) succeeded; on failure
   the derivative entries stay zero and B falls back to a field fix.
------------------------------------------------------------------------- */

bool Pusher::sample_gca_fields(const double *xpos, int icell, int i,
                               GCAPusher::GCAFields &F)
{
  F = GCAPusher::GCAFields();

  ComputePlasmaFields *cp = NULL;
  FixBackground *pd = NULL;
  if (pusher_plasma_cidx >= 0)
    cp = dynamic_cast<ComputePlasmaFields *>(modify->compute[pusher_plasma_cidx]);
  else if (pusher_plasma_fidx >= 0)
    pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);

  const int dim = domain->dimension;
  const bool axi = domain->axisymmetric;
  const double col_x0 = cp ? cp->plasma_data.column_x0 : (pd ? pd->column_x0 : 0.0);
  const double col_y0 = cp ? cp->plasma_data.column_y0 : (pd ? pd->column_y0 : 0.0);

  // E: fix background native potential first, fix efield/grid fallback.
  // e_valid records whether a real sample was obtained — an exactly-zero
  // field is a VALID value and must not read as a failed query.
  if (pd) {
    double ERp = 0.0, EZp = 0.0, Etp = 0.0;
    if (pd->query_efield_at_point(xpos, ERp, EZp, Etp, icell, i)) {
      const double phi_p = (dim == 3)
        ? std::atan2(xpos[1] - col_y0, xpos[0] - col_x0) : 0.0;
      OpenEdge::RZphi_force_to_sparta(ERp, EZp, Etp, dim, axi, phi_p,
                                      F.E[0], F.E[1], F.E[2]);
      F.e_valid = true;
    }
  }
  if (!F.e_valid && update->eperturbflag) {
    BorisGrid::read_field_from_fix(modify->fix[update->efieldfix],
                                   (update->efstyle == GFIELD),
                                   update->efield_active, i, icell, F.E);
    F.e_valid = true;
  }

  // B + derivatives: prefer the loaded equilibrium (GCA needs smooth grads)
  MagneticFieldFileDataParams Bcyl{};
  bool have_point_b = false;
  if (cp || pd) {
    Bcyl = cp ? cp->query_bfield_at_point(xpos, true)
              : pd->query_bfield_at_point(xpos, icell, i, true);
    if (Bcyl.Bmag > 0.0) have_point_b = true;
  }
  F.derivs_valid = have_point_b && Bcyl.derivatives_valid;

  double cphi = 1.0, sphi = 0.0;   // 3D azimuth about the column axis
  if (dim == 3) {
    const double rx = xpos[0] - col_x0, ry = xpos[1] - col_y0;
    const double rxy = std::sqrt(rx*rx + ry*ry);
    if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
  }

  if (have_point_b) {
    if (dim == 2) {
      if (axi) {
        // 2D axi slots: x=Z, y=R, z=toroidal
        F.B[0] = Bcyl.bz;  F.B[1] = Bcyl.br;  F.B[2] = Bcyl.bt;
        F.gradBmag[0] = Bcyl.dBmag_dz;
        F.gradBmag[1] = Bcyl.dBmag_dr;
      } else {
        // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
        F.B[0] = Bcyl.br;  F.B[1] = Bcyl.bz;  F.B[2] = Bcyl.bt;
        F.gradBmag[0] = Bcyl.dBmag_dr;
        F.gradBmag[1] = Bcyl.dBmag_dz;
      }
    } else {
      F.B[0] = Bcyl.br * cphi - Bcyl.bt * sphi;
      F.B[1] = Bcyl.br * sphi + Bcyl.bt * cphi;
      F.B[2] = Bcyl.bz;
      F.gradBmag[0] = Bcyl.dBmag_dr * cphi;
      F.gradBmag[1] = Bcyl.dBmag_dr * sphi;
      F.gradBmag[2] = Bcyl.dBmag_dz;
    }
  } else if (update->bperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[update->bfieldfix],
                                   (update->bfstyle == GFIELD),
                                   update->bfield_active, i, icell, F.B);

  F.Bmag = std::sqrt(F.B[0]*F.B[0] + F.B[1]*F.B[1] + F.B[2]*F.B[2]);

  // kappa and curl(b̂) from point-query B-component derivatives.
  // The cylindrical formulas below assume d/dphi = 0; for a
  // non-axisymmetric source (bcart) they manufacture fake curvature
  // ~1/R near the column axis — skip them (kappa = curl = 0 exactly
  // for a uniform Cartesian field).
  if (have_point_b && Bcyl.Bmag > 0.0 && Bcyl.axisymmetric_source) {
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
    OpenEdge::sparta_to_RZ(xpos, dim, axi, R_pt, Z_pt_unused, col_x0, col_y0);
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

    if (dim == 2) {
      if (axi) {
        // 2D axi slots: x=Z, y=R, z=toroidal
        F.kappa[0] = kZ;   F.kappa[1] = kR;   F.kappa[2] = kphi;
        F.curl_b[0] = cZ;  F.curl_b[1] = cR;  F.curl_b[2] = cphi_c;
      } else {
        // 2D Cartesian (legacy): x=R, y=Z, z=toroidal
        F.kappa[0] = kR;   F.kappa[1] = kZ;   F.kappa[2] = kphi;
        F.curl_b[0] = cR;  F.curl_b[1] = cZ;  F.curl_b[2] = cphi_c;
      }
    } else {
      F.kappa[0] = kR * cphi - kphi * sphi;
      F.kappa[1] = kR * sphi + kphi * cphi;
      F.kappa[2] = kZ;
      F.curl_b[0] = cR * cphi - cphi_c * sphi;
      F.curl_b[1] = cR * sphi + cphi_c * cphi;
      F.curl_b[2] = cZ;
    }
  }

  return have_point_b;
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

  // skip mixture (dust grains): pure advection even when charged — their
  // forces (gravity, drag, ablation) live in the grain fixes
  if (pusher_skip_flag &&
      pusher_skip_flag[particle->particles[i].ispecies]) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

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
  double *gca_mode_vec = NULL;
  double *gca_valid_vec = NULL;
  double *gca_chi_vec = NULL;
  if (gca_x_custom >= 0 && gca_y_custom >= 0 && gca_z_custom >= 0 &&
      gca_vpar_custom >= 0 && gca_mu_custom >= 0 && gca_mode_custom >= 0 &&
      gca_valid_custom >= 0 && gca_chi_custom >= 0) {
    gca_x_vec = particle->edvec[particle->ewhich[gca_x_custom]];
    gca_y_vec = particle->edvec[particle->ewhich[gca_y_custom]];
    gca_z_vec = particle->edvec[particle->ewhich[gca_z_custom]];
    gca_vpar_vec = particle->edvec[particle->ewhich[gca_vpar_custom]];
    gca_mu_vec = particle->edvec[particle->ewhich[gca_mu_custom]];
    gca_mode_vec = particle->edvec[particle->ewhich[gca_mode_custom]];
    gca_valid_vec = particle->edvec[particle->ewhich[gca_valid_custom]];
    gca_chi_vec = particle->edvec[particle->ewhich[gca_chi_custom]];
  }
  const bool have_gca_state =
    (gca_x_vec && gca_y_vec && gca_z_vec && gca_vpar_vec && gca_mu_vec &&
     gca_mode_vec && gca_valid_vec && gca_chi_vec);

  // --- Fields at particle position (k1 sample; RK stages resample) ---
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

  GCAPusher::GCAFields F;
  sample_gca_fields(x, icell, i, F);
  double *E = F.E;
  double *B = F.B;
  const double Bmag = F.Bmag;

  const double gradBmag_magnitude =
    std::sqrt(F.gradBmag[0]*F.gradBmag[0] + F.gradBmag[1]*F.gradBmag[1] +
              F.gradBmag[2]*F.gradBmag[2]);

  // --- Sheath: intentionally NOT applied here (review round 3) ---
  // The pre-selector sheath block let an outbound super-barrier particle
  // set `paid`, have its velocity overwritten by GC->Boris
  // materialization, and bypass the real Boris barrier. Sheath physics
  // for hybrid/GCA flows through exactly one path each: the mover's
  // inbound wall kick and the delegated Boris kernels' outbound barrier,
  // until the unified sheath event evaluator (PLAN.md) replaces both.
  // Esh stays zero: the GCA stages see no sheath field (GCA inside the
  // sheath band is the A1 operator's job, not a volume force).
  double Esh[3] = {0.0, 0.0, 0.0};
  (void) Esh;

  // switch-event observability: previous branch + residence before this
  // step's decision (chi is reset on GCA commit, so capture it now)
  const double prev_mode = have_gca_state ? gca_mode_vec[i] : -1.0;
  const double prev_chi  = have_gca_state ? gca_chi_vec[i]  : 0.0;
  const char *sw_reason = "criterion";
  double d_end_log = -1.0;

  // rho_L from stored mu when the GC state is valid, else from v. Used by
  // the adiabaticity criterion and the k*rho_L switch shell. Start-of-step
  // value; production switching requires the max over the proposed step
  // (identical for the uniform-B Stage A test).
  double rho_L = 0.0;
  if (Bmag > 0.0 && qm_abs > 0.0) {
    double vperp2;
    if (have_gca_state && gca_valid_vec[i] > 0.5) {
      const double mu_eff = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
      vperp2 = (2.0 * mu_eff * Bmag) / mass;
    } else {
      const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
      const double v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
      vperp2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] - v_par * v_par;
    }
    if (vperp2 < 0.0) vperp2 = 0.0;
    rho_L = GCAPusher::larmor_radius(std::sqrt(vperp2), qm_abs, Bmag);
  }

  // --- Switching criterion ---
  // mode=gca: always GCA when B is available.
  // mode=hybrid: rho_L/L_B adiabaticity — use GCA when the gradient scale
  // is gentle relative to the Larmor orbit (rho_L < L_B / switch_factor).
  // The shell checks below are evaluated LAST so nothing re-enables GCA
  // at the wall.
  bool use_gca = false;
  if (Bmag > 0.0 && qm_abs > 0.0) {
    if (pusher_mode == PUSHER_GCA) {
      use_gca = true;
      // fail-safe even in pure mode: a particle whose Larmor radius
      // rivals the gradient scale must never be GC-transformed — a
      // charged dust grain (q/m ~ 1e-3 C/kg, rho_L ~ 1e2 m) teleports
      // by (v x bhat)/Omega into the wall at its first charged step.
      // Kick-drift Boris is the exact regime for Omega*dt << 1.
      if (rho_L > 0.0 && F.derivs_valid) {
        const double L_B = GCAPusher::grad_b_length(Bmag, gradBmag_magnitude);
        use_gca = (rho_L < L_B / pusher_gca_switch);
      }
    }
    else {
      const double L_B = GCAPusher::grad_b_length(Bmag, gradBmag_magnitude);
      // derivs_valid: equilibrium/constant source — a uniform field
      // (L_B -> inf) is then a VALID GCA regime. Mesh/fallback B carries
      // no gradients, so its zeros must not read as uniform: stay Boris.
      if (rho_L > 0.0 && F.derivs_valid)
        use_gca = (rho_L < L_B / pusher_gca_switch);
    }
  }

  // --- Boris shell (C1/C2) ---
  // Switch distance d_sw: metres, or k*rho_L per particle (boris_near
  // <k> rhoL). Plane metric to the cell's nearest sheath_geom surf —
  // exact for the flat-wall Stage A geometry; production switching is
  // specified in examples/validation/pushers/hybrid/PLAN.md (swept distance to
  // bounded elements).
  int midx_bn = -1;
  if ((pusher_boris_near > 0.0 || pusher_gc_wall_flux) &&
      update->sheath_geom_cidx >= 0) {
    auto *csg_bn = dynamic_cast<ComputeNearestSurfGrid *>(
        modify->compute[update->sheath_geom_cidx]);
    if (csg_bn) {
      int gcell_bn = icell;
      Grid::ChildCell *cells_bn = grid->cells;
      if (cells_bn[icell].nsplit <= 0 && cells_bn[icell].isplit >= 0)
        gcell_bn = grid->sinfo[cells_bn[icell].isplit].icell;
      // stale post-rebalance cell index: leave midx_bn unset rather than
      // read past midx_grid
      if (gcell_bn >= 0 && gcell_bn < csg_bn->nglocal)
        midx_bn = csg_bn->midx_grid[gcell_bn];
    }
  }
  // SIGNED plane distance (positive = plasma side along the surf
  // normal). Signed matters: a segment whose endpoints BOTH sit outside
  // the shell can still cross the wall plane entirely (leap-through) —
  // absolute distances cannot see that.
  auto near_signed = [&](const double *p) -> double {
    if (midx_bn < 0) return 1.0e30;
    if (domain->dimension == 2) {
      Surf::Line *ln = &surf->lines[midx_bn];
      return (p[0]-ln->p1[0])*ln->norm[0] +
             (p[1]-ln->p1[1])*ln->norm[1];
    }
    Surf::Tri *tr = &surf->tris[midx_bn];
    return (p[0]-tr->p1[0])*tr->norm[0] +
           (p[1]-tr->p1[1])*tr->norm[1] +
           (p[2]-tr->p1[2])*tr->norm[2];
  };
  double d_sw = -1.0, d_start = 1.0e30;
  if (pusher_boris_near > 0.0 && midx_bn >= 0) {
    d_sw = pusher_boris_near_rhol ? pusher_boris_near * rho_L
                                  : pusher_boris_near;
    d_start = std::fabs(near_signed(x));
    if (use_gca && d_sw > 0.0) {
      if (d_start < d_sw) {
        use_gca = false;                     // already inside the shell
        sw_reason = "shell_start";
      } else if (have_gca_state && gca_chi_vec[i] > 0.0 &&
                 (gca_chi_vec[i] < 4.0*M_PI || d_start < 2.0*d_sw)) {
        // C2 hysteresis: after a switch, stay Boris until >= 2
        // gyroperiods of residence AND clear of twice the shell.
        use_gca = false;
        sw_reason = "hysteresis_hold";
      }
    }
  }

  bool ran_gca = false;
  if (use_gca) {
    // --- GCA path ---
    GCAPusher::GCAState gca;
    // 2D-cart slot triple (x=R, y=Z, z=phi) is LEFT-handed, so the RHS
    // cross products (ExB, BxgradB, B*) would flip every drift. Conjugate
    // the toroidal components into a right-handed frame for the GCA and
    // flip the reconstructed v back. Axi slots (Z, R, phi) and 3D
    // Cartesian are already right-handed.
    const bool lh2d = (domain->dimension == 2 && !domain->axisymmetric);
    auto flipF = [lh2d](GCAPusher::GCAFields &Ff) {
      if (!lh2d) return;
      Ff.E[2] = -Ff.E[2];
      Ff.B[2] = -Ff.B[2];
      Ff.kappa[2] = -Ff.kappa[2];
      Ff.curl_b[2] = -Ff.curl_b[2];
    };
    GCAPusher::GCAFields Fk = F;
    flipF(Fk);

    if (have_gca_state && gca_valid_vec[i] > 0.5) {
      gca.X[0] = gca_x_vec[i];
      gca.X[1] = gca_y_vec[i];
      gca.X[2] = gca_z_vec[i];
      gca.v_par = gca_vpar_vec[i];
      gca.mu = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
    } else {
      const double vinit[3] = {v[0], v[1], lh2d ? -v[2] : v[2]};
      gca = GCAPusher::init_from_particle(x, vinit, mass, qm, Fk.B);
    }

    // GCA integration.
    //   rk4 (default): 4 full-Littlejohn RHS evaluations.
    //   rk2          : 2 full-Littlejohn RHS evaluations (midpoint method).
    //   simple       : 1 reduced RHS without the B* curvature/curl(b) terms.
    // rk2/rk4 resample E, B and derivatives at each stage position via
    // sample_gca_fields; a failed stage query keeps the frozen k1 fields,
    // and a failed stage E query keeps the k1 E (fields are continuous,
    // so stale beats zero). Frozen sheath Esh rides on top.
    auto fields_at = [&](const double *Xs, GCAPusher::GCAFields &FS) {
      GCAPusher::GCAFields Ft;
      if (!sample_gca_fields(Xs, icell, i, Ft)) return;
      if (Ft.e_valid) {
        Ft.E[0] += Esh[0]; Ft.E[1] += Esh[1]; Ft.E[2] += Esh[2];
      }
      flipF(Ft);
      if (!Ft.e_valid) {
        // no E sample at the stage point: keep the k1 E (already
        // flipped + sheath-augmented); an exact zero sample is kept
        Ft.E[0] = FS.E[0]; Ft.E[1] = FS.E[1]; Ft.E[2] = FS.E[2];
      }
      FS = Ft;
    };
    if (pusher_gca_integrator == GCA_SIMPLE) {
      GCAPusher::push_gca(qm, dt, mass, Fk.E, Fk.B, Fk.gradBmag, gca);
    } else if (pusher_gca_integrator == GCA_RK2) {
      GCAPusher::push_gca_rk2(qm, dt, mass, Fk, gca, fields_at);
    } else {
      GCAPusher::push_gca_rk4(qm, dt, mass, Fk, gca, fields_at);
    }

    // C1 trial/replay: the advance above was a TRIAL (stored state is
    // untouched). If the swept GC chord entered the Boris shell, discard
    // it and redo the whole step with Boris from the pre-trial state —
    // an endpoint-only check would let a large GCA step hop the shell.
    if (d_sw > 0.0) {
      const double s1 = near_signed(gca.X);
      d_end_log = s1;
      // shell entered if either endpoint is inside OR the segment
      // crossed the wall plane outright (sign change = leap-through)
      const double s0 = near_signed(x);
      if (std::min(std::fabs(s0), std::fabs(s1)) < d_sw || s0 * s1 <= 0.0)
        sw_reason = "swept";
    }
    if (d_sw > 0.0 && strcmp(sw_reason, "swept") == 0) {
      // trial discarded; !ran_gca path below replays in Boris
    } else {
      if (have_gca_state) {
        if (switch_log_file && prev_mode < 0.5) {
          const double e_pre = 0.5*mass*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
          const double e_post = 0.5*mass*gca.v_par*gca.v_par + gca.mu*Bmag;
          log_switch(particle->particles[i].id, 0, 1,
                     (prev_chi > 0.0) ? "reentry" : "start",
                     d_start, -1.0, d_sw, e_pre, e_post, 0);
        }
        gca_x_vec[i] = gca.X[0];
        gca_y_vec[i] = gca.X[1];
        gca_z_vec[i] = gca.X[2];
        gca_vpar_vec[i] = gca.v_par;
        gca_mu_vec[i] = gca.mu;
        gca_mode_vec[i] = 1.0;
        gca_valid_vec[i] = 1.0;
        gca_chi_vec[i] = 0.0;
      }

      // Keep persistent GC state, but reconstruct full v for diagnostics
      // and clean Boris fallback if regime switching happens.
      // Reconstruction uses fields at the END-of-step GC position
      // (step-start B contaminates the dumped speed/mu diagnostics).
      GCAPusher::GCAFields Fe = Fk;
      {
        const double ye[4] = {gca.X[0], gca.X[1], gca.X[2], gca.v_par};
        fields_at(ye, Fe);
      }
      const double two_pi = 2.0 * M_PI;
      const double omega_c = std::fabs(qm) * Fe.Bmag;
      const double phase_turns =
        particle->particles[i].id * GCA_PHASE_GOLDEN +
        (omega_c * dt * static_cast<double>(update->ntimestep)) / two_pi;
      double rand_u = phase_turns - std::floor(phase_turns);
      // A1 (gc_wall flux, 3D): when the GC can reach the wall this step,
      // sample the reconstruction gyrophase from the first-passage flux
      // weight p(phi) ~ max(-vn(phi), 0) instead of the uniform hash —
      // A0's uniform phase provably underweights normal-incidence
      // impacts vs the resolved-orbit first passage.
      if (pusher_gc_wall_flux && midx_bn >= 0 && domain->dimension == 3 &&
          Fe.Bmag > 0.0) {
        const double vperp_e = std::sqrt(2.0 * gca.mu * Fe.Bmag / mass);
        const double rho_e = vperp_e / (qm_abs * Fe.Bmag);
        const double s1 = near_signed(gca.X);
        if (s1 < 2.0 * rho_e + std::fabs(gca.v_par) * dt) {
          Surf::Tri *trw = &surf->tris[midx_bn];
          const double nw[3] = {trw->norm[0], trw->norm[1], trw->norm[2]};
          const double bh[3] = {Fe.B[0]/Fe.Bmag, Fe.B[1]/Fe.Bmag,
                                Fe.B[2]/Fe.Bmag};
          double e1w[3], e2w[3];
          GCAPusher::gca_perp_basis(bh, e1w, e2w);
          const double a  = gca.v_par *
              (bh[0]*nw[0] + bh[1]*nw[1] + bh[2]*nw[2]);
          const double cx = vperp_e *
              (e1w[0]*nw[0] + e1w[1]*nw[1] + e1w[2]*nw[2]);
          const double cy = vperp_e *
              (e2w[0]*nw[0] + e2w[1]*nw[1] + e2w[2]*nw[2]);
          rand_u = flux_phase_sample(a, cx, cy, rand_u);
        }
      }
      // qm=0: no gyro offset on the ADVECTED position — GCA particles
      // ride at the guiding center every step (offset jitter from the
      // per-step phase hash loses particles at boundaries/caps). The
      // offset is materialized once, at the Boris handoff below.
      GCAPusher::gca_to_particle(gca, Fe.B, mass, 0.0, rand_u, xnew, v);
      if (lh2d) v[2] = -v[2];   // back to slot frame
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
      // 2D: the GC advance is cylindrical (R evolves directly in the y or
      // x slot), so hand the mover a pure in-plane chord: v = (xnew-x)/dt,
      // v[2] = 0, making axi_remap the identity and mid-move re-entries
      // (xnew = x + dtremain*v) retrace the same chord. Toroidal GC motion
      // is axisymmetric-irrelevant; physical v_par/mu live in the GC state.
      if (domain->dimension == 2) {
        xnew[2] = 0.0;
        v[0] = (xnew[0] - x[0]) / dt;
        v[1] = (xnew[1] - x[1]) / dt;
        v[2] = 0.0;
      }
      ran_gca = true;
    }
  }

  if (!ran_gca) {
    // --- Boris fallback: delegate to the guarded kernels ---
    double xstart[3] = {x[0], x[1], x[2]};
    // push_boris_2d carries the planar per-subcycle wall/cell guards and
    // the axi kick-drift contract; push_boris_3d carries the 3D
    // triangle-clip and cell-exit guards. On a GCA handoff rebuild
    // physical v from the stored GC state first (v may hold the 2D
    // chord or a stale reconstruction — never trust it across a switch).
    if (have_gca_state && gca_valid_vec[i] > 0.5 && Bmag > 0.0) {
      // Phase-space materialization about the STORED guiding center via
      // the orbit-state API (value semantics — particle storage is NOT
      // mutated; an offset crossing a cell boundary with stale icell let
      // the mover's recovery fake a wall collision, the non-terminal-
      // vanish episode). Boris integrates from the LOCAL materialized
      // start; the mover traces the GC -> endpoint chord with its
      // normal surface checks. Proper rehoming: PLAN.md contract.
      const double pt = particle->particles[i].id * GCA_PHASE_GOLDEN;
      MaterializedOrbit mo =
          materialize_orbit(i, B, qm, mass, pt - std::floor(pt));
      if (mo.valid) {
        v[0] = mo.v[0]; v[1] = mo.v[1]; v[2] = mo.v[2];
        xstart[0] = mo.x[0];
        xstart[1] = mo.x[1];
        if (domain->dimension == 3) xstart[2] = mo.x[2];
      }
    }
    if (switch_log_file && prev_mode > 0.5) {
      // GCA -> Boris: pre = stored GC energy, post = materialized KE
      // (17-digit event record; the conversion must agree to fp precision)
      double e_pre = -1.0;
      if (have_gca_state) {
        const double mu_l = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
        e_pre = 0.5*mass*gca_vpar_vec[i]*gca_vpar_vec[i] + mu_l*Bmag;
      }
      const double e_post = 0.5*mass*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      log_switch(particle->particles[i].id, 1, 0, sw_reason,
                 d_start, d_end_log, d_sw, e_pre, e_post,
                 strcmp(sw_reason, "swept") == 0 ? 1 : 0);
    }
    if (have_gca_state) {
      gca_mode_vec[i] = 0.0;
      gca_valid_vec[i] = 0.0;   // Boris advances x; stored X goes stale
      gca_chi_vec[i] += qm_abs * Bmag * dt;   // residence gyroangle
    }
    if (domain->dimension == 2)
      push_boris_2d(i, icell, dt, xstart, v, xnew, charge, mass);
    else
      push_boris_3d(i, icell, dt, xstart, v, xnew, charge, mass);
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

  // Axi: the GCA path advances the guiding center in cylindrical (R,Z)
  // and returns the linear chord v = (xnew-x)/dt the axi mover expects;
  // the Boris fallback delegates to push_boris_2d (kick-drift), so dt
  // must resolve the gyroperiod for fallback particles (bad_dt_check).

  if (!pusher_plasma_cid)
    error->all(FLERR,"global pusher plasma requires a provider ID");
  pusher_plasma_cidx = modify->find_compute(pusher_plasma_cid);
  pusher_plasma_fidx = -1;
  if (pusher_plasma_cidx >= 0) {
    if (!modify->compute[pusher_plasma_cidx]->per_grid_flag)
      error->all(FLERR,"global pusher plasma: compute must be per-grid");
  } else {
    pusher_plasma_fidx = modify->find_fix(pusher_plasma_cid);
    if (pusher_plasma_fidx < 0)
      error->all(FLERR,"global pusher plasma: provider ID not found");
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher_plasma_fidx]);
    if (!pd)
      error->all(FLERR,
                 "global pusher plasma: fix provider must be style background");
  }

  // GCA needs smooth B-field derivatives (grad|B|, curvature, curl(b̂))
  // from an equilibrium psi map. When the embedded /equilibrium/* group
  // in plasma.h5 (or an explicit `equilibrium <file>` keyword) is
  // present, ComputePlasmaFields / FixBackground query_bfield_at_point
  // picks it up; otherwise grad-|B| sample returns the unmodified B and
  // the GCA reverts to its simpler kernel.

  register_gca_custom();
}

/* ----------------------------------------------------------------------
   Persistent guiding-center state per particle (kept across timesteps to
   avoid re-initializing from instantaneous gyromotion every step). Also
   called at `global pusher` parse time so dumps defined later in the
   input script can reference p_gca_* attributes.
------------------------------------------------------------------------- */

void Pusher::register_gca_custom()
{
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
  if (gca_mode_custom < 0) {
    gca_mode_custom = particle->find_custom((char *) "gca_mode");
    if (gca_mode_custom < 0)
      gca_mode_custom = particle->add_custom((char *) "gca_mode", custom_double, 0);
  }
  if (gca_valid_custom < 0) {
    gca_valid_custom = particle->find_custom((char *) "gca_valid");
    if (gca_valid_custom < 0)
      gca_valid_custom = particle->add_custom((char *) "gca_valid", custom_double, 0);
  }
  if (gca_chi_custom < 0) {
    gca_chi_custom = particle->find_custom((char *) "gca_chi");
    if (gca_chi_custom < 0)
      gca_chi_custom = particle->add_custom((char *) "gca_chi", custom_double, 0);
  }
}


/* ----------------------------------------------------------------------
   Unified sheath wall potential: ONE evaluator for the inbound wall kick
   and the outbound barrier — Coulette-Manfredi base + RF waveform at the
   exact event time, from the per-element cache. Divergent per-site
   formulas passed the static test only because they reduce to the same
   floating potential there; RF breaks that silently.
------------------------------------------------------------------------- */

double Pusher::sheath_phi_wall(int midx, double t_offset)
{
  if (!sheath_cache_enabled || midx < 0) return -1.0;
  const int nsurf_all = surf->nlocal + surf->nghost;
  if (midx >= nsurf_all) return -1.0;
  if ((int) sheath_cache.size() != nsurf_all)
    sheath_cache.assign(nsurf_all, SheathElemCache{});
  SheathElemCache &C = sheath_cache[midx];
  if (C.state == 0) {
    if (domain->dimension == 3) build_sheath_cache_entry_3d(midx, C);
    else build_sheath_cache_entry(midx, C);
  }
  if (C.state != 1) return -1.0;
  return SheathModels::sheath_phi_at_distance(C.coeffs, 0.0) +
         sheath_waveform_drop(update, surf, midx, t_offset);
}

/* ----------------------------------------------------------------------
   Orbit-state API (Stage B prerequisite). Value semantics: nothing here
   mutates particle position/velocity storage; sync_/apply_/invalidate_
   touch ONLY the persistent GC custom attributes.
------------------------------------------------------------------------- */

MaterializedOrbit Pusher::materialize_orbit(int i, const double B[3],
                                            double qm, double mass,
                                            double phase01)
{
  MaterializedOrbit mo{};
  if (gca_x_custom < 0 || gca_valid_custom < 0) return mo;
  double *xv = particle->edvec[particle->ewhich[gca_x_custom]];
  double *yv = particle->edvec[particle->ewhich[gca_y_custom]];
  double *zv = particle->edvec[particle->ewhich[gca_z_custom]];
  double *pv = particle->edvec[particle->ewhich[gca_vpar_custom]];
  double *mv = particle->edvec[particle->ewhich[gca_mu_custom]];
  double *vv = particle->edvec[particle->ewhich[gca_valid_custom]];
  if (!xv || vv[i] < 0.5) return mo;

  GCAPusher::GCAState g;
  g.X[0] = xv[i]; g.X[1] = yv[i]; g.X[2] = zv[i];
  g.v_par = pv[i];
  g.mu = (mv[i] > 0.0) ? mv[i] : 0.0;

  // 2D-cart legacy slots are LEFT-handed: materialize in the conjugated
  // (toroidal-negated) frame and flip v[2] back, matching the GCA branch.
  const bool lh2d = (domain->dimension == 2 && !domain->axisymmetric);
  const double Bh[3] = {B[0], B[1], lh2d ? -B[2] : B[2]};
  GCAPusher::gca_to_particle(g, Bh, mass, qm, phase01, mo.x, mo.v);
  if (lh2d) mo.v[2] = -mo.v[2];
  if (domain->dimension == 2) mo.x[2] = 0.0;   // slot z stays 0
  mo.xtrace[0] = g.X[0]; mo.xtrace[1] = g.X[1]; mo.xtrace[2] = g.X[2];
  mo.B[0] = B[0]; mo.B[1] = B[1]; mo.B[2] = B[2];
  mo.icell = -1;   // NOT rehomed — the PLAN relocation contract is open
  mo.phase = phase01;
  mo.valid = true;
  return mo;
}

bool Pusher::materialize_impact_velocity(int i, int icell, int midx,
                                         double *v_io)
{
  // A1 for axi: a GCA particle rides its guiding center, so the axi
  // mover's chord velocity reaches the wall — swap in the physical gyro
  // velocity, flux-weighted for first passage, at the collision site.
  // 3D samples pre-step in push_hybrid_3d (the straight-chord trace
  // tolerates v != chord); the axi retrace contract xnew = x + dt*v
  // does not, hence this site. 2D-cart excluded (conjugation unported).
  if (!pusher_gc_wall_flux || midx < 0) return false;
  if (domain->dimension != 2 || !domain->axisymmetric) return false;
  if (gca_x_custom < 0 || gca_valid_custom < 0) return false;
  double *vv = particle->edvec[particle->ewhich[gca_valid_custom]];
  if (!vv || vv[i] < 0.5) return false;

  GCAPusher::GCAState g;
  g.X[0] = particle->edvec[particle->ewhich[gca_x_custom]][i];
  g.X[1] = particle->edvec[particle->ewhich[gca_y_custom]][i];
  g.X[2] = particle->edvec[particle->ewhich[gca_z_custom]][i];
  g.v_par = particle->edvec[particle->ewhich[gca_vpar_custom]][i];
  g.mu = particle->edvec[particle->ewhich[gca_mu_custom]][i];
  if (g.mu < 0.0) g.mu = 0.0;

  GCAPusher::GCAFields F;
  if (!sample_gca_fields(g.X, icell, i, F) || F.Bmag <= 0.0) return false;
  const int isp = particle->particles[i].ispecies;
  const double mass = particle->species[isp].mass;

  const double *nw = surf->lines[midx].norm;   // slot frame, norm[2] = 0
  const double bh[3] = {F.B[0]/F.Bmag, F.B[1]/F.Bmag, F.B[2]/F.Bmag};
  double e1w[3], e2w[3];
  GCAPusher::gca_perp_basis(bh, e1w, e2w);
  const double vperp = std::sqrt(2.0 * g.mu * F.Bmag / mass);
  const double a  = g.v_par * (bh[0]*nw[0] + bh[1]*nw[1] + bh[2]*nw[2]);
  const double cx = vperp * (e1w[0]*nw[0] + e1w[1]*nw[1] + e1w[2]*nw[2]);
  const double cy = vperp * (e2w[0]*nw[0] + e2w[1]*nw[1] + e2w[2]*nw[2]);

  const double omega_c = std::fabs(particle->species[isp].charge *
      update->echarge) / mass * F.Bmag;
  const double pt = particle->particles[i].id * GCA_PHASE_GOLDEN +
      (omega_c * update->dt * static_cast<double>(update->ntimestep)) /
      (2.0 * M_PI);
  const double u = flux_phase_sample(a, cx, cy, pt - std::floor(pt));

  double xd[3];
  GCAPusher::gca_to_particle(g, F.B, mass, 0.0, u, xd, v_io);
  // post-collision products (reflection, sputter continuation) carry a
  // new phase space — force re-init from the particle next step
  invalidate_gc(i, GC_INVAL_COLLISION);
  return true;
}

bool Pusher::sync_gc_velocity(int i, const double *v, const double B[3],
                              double mass)
{
  // velocity-only sync REQUIRES a currently-valid GC state: it must not
  // resurrect a stale stored position
  if (gca_vpar_custom < 0 || gca_valid_custom < 0) return false;
  if (particle->edvec[particle->ewhich[gca_valid_custom]][i] < 0.5)
    return false;
  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) { invalidate_gc(i, GC_INVAL_COLLISION); return false; }
  const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
  const double vpar = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
  double vperp2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] - vpar*vpar;
  if (vperp2 < 0.0) vperp2 = 0.0;
  particle->edvec[particle->ewhich[gca_vpar_custom]][i] = vpar;
  particle->edvec[particle->ewhich[gca_mu_custom]][i] =
      mass * vperp2 / (2.0 * Bmag);
  // GC position kept: a velocity-only operator (e.g. Coulomb) must not
  // apply a positional gyro-offset
  return true;
}


bool Pusher::apply_parallel_impulse(int i, double dvpar)
{
  if (gca_vpar_custom < 0 || gca_valid_custom < 0) return false;
  if (particle->edvec[particle->ewhich[gca_valid_custom]][i] < 0.5)
    return false;                        // refuse to kick invalid state
  particle->edvec[particle->ewhich[gca_vpar_custom]][i] += dvpar;
  return true;
}

bool Pusher::apply_gc_displacement(int i, const double *dx)
{
  if (gca_x_custom < 0 || gca_valid_custom < 0) return false;
  if (particle->edvec[particle->ewhich[gca_valid_custom]][i] < 0.5)
    return false;                        // refuse to move invalid state
  particle->edvec[particle->ewhich[gca_x_custom]][i] += dx[0];
  particle->edvec[particle->ewhich[gca_y_custom]][i] += dx[1];
  particle->edvec[particle->ewhich[gca_z_custom]][i] += dx[2];
  return true;
}

void Pusher::invalidate_gc(int i, int reason)
{
  (void) reason;   // reasons feed diagnostics once operators land
  if (gca_valid_custom < 0) return;
  particle->edvec[particle->ewhich[gca_valid_custom]][i] = 0.0;
}

/* ----------------------------------------------------------------------
   GCA<->Boris switch event log (per-rank CSV, 17-digit). Proves which
   transition happened, where, and with what energy bookkeeping — the
   dump-based view cannot see same-step switch+impact at large dt.
------------------------------------------------------------------------- */

void Pusher::log_switch(int id, int oldmode, int newmode, const char *reason,
                        double d_start, double d_end, double d_sw,
                        double e_pre, double e_post, int replay)
{
  if (!switch_log_file) return;
  if (!switch_log_fp) {
    char *fn = new char[strlen(switch_log_file) + 32];
    sprintf(fn, "%s.rank%d", switch_log_file, comm->me);
    switch_log_fp = fopen(fn, "w");
    delete [] fn;
    if (!switch_log_fp)
      error->one(FLERR, "Cannot open pusher switch_log file");
    fprintf(switch_log_fp,
      "timestep,id,oldmode,newmode,reason,d_start,d_end,d_sw,e_pre,e_post,replay\n");
  }
  fprintf(switch_log_fp, "%lld,%d,%d,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
          (long long) update->ntimestep, id, oldmode, newmode, reason,
          d_start, d_end, d_sw, e_pre, e_post, replay);
  fflush(switch_log_fp);   // events are rare; the dtor is not guaranteed
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
      // register p_gca_* now so dumps defined later in the script parse
      if (pusher_mode == PUSHER_HYBRID || pusher_mode == PUSHER_GCA)
        register_gca_custom();
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
    } else if (strcmp(arg[iarg], "skip") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher skip");
      delete [] pusher_skip_mix;
      int n = strlen(arg[iarg+1]) + 1;
      pusher_skip_mix = new char[n];
      strcpy(pusher_skip_mix, arg[iarg+1]);
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
      // optional units token: shell = k*rho_L per particle
      pusher_boris_near_rhol = 0;
      if (iarg + 2 < narg && strcmp(arg[iarg+2], "rhoL") == 0) {
        pusher_boris_near_rhol = 1;
        iarg++;
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "gc_wall") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher gc_wall");
      if (strcmp(arg[iarg+1], "a0") == 0)        pusher_gc_wall_flux = 0;
      else if (strcmp(arg[iarg+1], "flux") == 0) pusher_gc_wall_flux = 1;
      else error->all(FLERR, "global pusher gc_wall must be a0 or flux");
      iarg += 2;
    } else if (strcmp(arg[iarg], "switch_log") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global pusher switch_log");
      delete [] switch_log_file;
      int n = strlen(arg[iarg+1]) + 1;
      switch_log_file = new char[n];
      strcpy(switch_log_file, arg[iarg+1]);
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
        error->all(FLERR, "global pusher sheath kick|boundary|spatial requires geom <ID>");
    } else break;  // next keyword belongs to a different global option
  }
}
