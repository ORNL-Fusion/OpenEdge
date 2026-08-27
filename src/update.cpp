/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
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
#include "collide.h"
#include "grid.h"
#include "surf.h"
#include "surf_collide.h"
#include "surf_react.h"
#include "input.h"
#include "output.h"
#include "geometry.h"
#include "random_mars.h"
#include "timer.h"
#include "math_extra.h"
#include "pusher.h"
#include "openedge_geom.h"
#include "random_mars.h"
#include "sheath_models.h"
#include "compute_nearest_surf_grid.h"
#include "compute_plasma_fields.h"
#include "fix_background.h"
#include "fix_cross_field_diffusion.h"
#include "fix_force_thermal.h"
#include "fix_coulomb_base.h"
#include "fix_volume_chem_adas.h"
#include "memory.h"
#include "error.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace SPARTA_NS;

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // same as Domain
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain
enum{OUTSIDE,INSIDE,ONSURF2OUT,ONSURF2IN};      // several files
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Grid
enum{TALLYAUTO,TALLYREDUCE,TALLYRVOUS};         // same as Surf
enum{PERAUTO,PERCELL,PERSURF};                  // several files
enum{NOFIELD,CFIELD,PFIELD,GFIELD};             // several files

namespace {

inline void xyz_to_rz(const double xyz[3], int dim, int axi, double &R, double &Z)
{
  if (axi) {          // 2D true axi: SPARTA x = Z-axis, y = R-radial
    Z = xyz[0];
    R = xyz[1];
  } else if (dim == 2) {  // 2D Cartesian (legacy)
    R = xyz[0];
    Z = xyz[1];
  } else {            // 3D Cartesian
    R = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    Z = xyz[2];
  }
}

// Physics-derived sheath engagement cut-off distance, in meters.
//   max( 5 * L_MPS, 10 * lambdaD )
// where L_MPS = rho_i * tan(alpha_n), with alpha_n = angle(B, wall
// normal) as returned by chodura_metrics. If user_ceiling > 0 it is
// applied as an additional upper bound (legacy `global sheath dmax`).
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

// icell (when >= 0) routes mesh-field lookups through FixBackground's
// O(1) cell-indexed cache instead of per-field triangle searches.
inline PlasmaFileParams query_plasma_from_fix(const FixBackground *pd, const double xyz[3], int dim, int axi,
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
                                                         const double xyz[3], int dim, int axi,
                                                         int icell = -1, int iparticle = -1)
{
  MagneticFieldFileDataParams B{};
  if (!pd || !pd->has_bfield) return B;

  xyz_to_rz(xyz, dim, axi, B.r, B.z);
  pd->bfield_at(B.r, B.z, B.br, B.bz, B.bt, icell, iparticle);
  B.Bmag = std::sqrt(B.br * B.br + B.bt * B.bt + B.bz * B.bz);
  return B;
}

// ---- Fused bilinear stencil for FixBackground ----
// Build once per particle (R,Z); reuse across every field interp.
// Replaces ~21 redundant clamp/index/weight computations with 1.
struct PdStencil2D {
  int c00;        // base flat index = iz0*nr + ir0
  int row;        // row stride = nr
  double w00, w01, w10, w11;
  bool valid;
};

inline PdStencil2D make_pd_stencil(const FixBackground *pd, double R, double Z)
{
  PdStencil2D st{};
  if (!pd || pd->nr < 2 || pd->nz < 2) return st;
  const int nr = pd->nr;
  const int nz = pd->nz;

  const double Rc = std::min(std::max(R, pd->rvals.front()), pd->rvals.back());
  const double Zc = std::min(std::max(Z, pd->zvals.front()), pd->zvals.back());
  const double dr = pd->rvals[1] - pd->rvals[0];
  const double dz = pd->zvals[1] - pd->zvals[0];
  const double fi = (Rc - pd->rvals.front()) / dr;
  const double fj = (Zc - pd->zvals.front()) / dz;
  const int ir0 = std::max(0, std::min((int)fi, nr - 2));
  const int iz0 = std::max(0, std::min((int)fj, nz - 2));
  const double s = std::max(0.0, std::min(1.0, fi - ir0));
  const double t = std::max(0.0, std::min(1.0, fj - iz0));

  st.c00 = iz0 * nr + ir0;
  st.row = nr;
  st.w00 = (1.0 - s) * (1.0 - t);
  st.w01 = s * (1.0 - t);
  st.w10 = (1.0 - s) * t;
  st.w11 = s * t;
  st.valid = true;
  return st;
}

inline double interp_pd_stencil(const std::vector<double> &field,
                                const PdStencil2D &st)
{
  if (!st.valid || field.empty()) return 0.0;
  const double *f = field.data();
  const int c00 = st.c00;
  const int c10 = c00 + st.row;
  return st.w00 * f[c00]     + st.w01 * f[c00 + 1]
       + st.w10 * f[c10]     + st.w11 * f[c10 + 1];
}

}  // namespace


#define MAXSTUCK 20
#define EPSPARAM 1.0e-7
#define MAXLINE 16384

#define CORE_GROUP_NAME "CORE"
#define BIG 1.0e20

// max value (bytes) for global_mem_limit = 2000 MiB = 2097152000
// kept safely below MAXSMALLINT (INT_MAX = 2147483647) so that the buffer-size
// arithmetic in restart I/O (e.g. "max_size += 128") cannot overflow a 32-bit int

#define MEMLIMIT_MAX (2000*1024*1024)

// either set ID or PROC/INDEX, set other to -1

//#define MOVE_DEBUG 1              // un-comment to debug one particle
#define MOVE_DEBUG_ID 308143534  // particle ID
#define MOVE_DEBUG_PROC -1        // owning proc
#define MOVE_DEBUG_INDEX -1   // particle index on owning proc
#define MOVE_DEBUG_STEP 4107    // timestep

/* ---------------------------------------------------------------------- */

Update::Update(SPARTA *sparta) : Pointers(sparta)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  ntimestep = 0;
  runflag = 0;
  firststep = laststep = 0;
  beginstep = endstep = 0;
  first_update = 0;

  rcbflag = 0;
  rcblo[0] = rcblo[1] = rcblo[2] = 0.0;
  rcbhi[0] = rcbhi[1] = rcbhi[2] = 0.0;

  time = 0.0;
  time_last_update = 0;

  unit_style = NULL;
  set_units("si");

  fnum = 1.0;
  nrho = 1.0;
  vstream[0] = vstream[1] = vstream[2] = 0.0;
  temp_thermal = 273.15;
  optmove_flag = 0;
  move_flag = 1;
  fstyle = NOFIELD;
  fieldID = NULL;
  efstyle = NOFIELD;
  efieldID = NULL;
  efieldfreq = 0;

  bfstyle = NOFIELD;
  bfieldID = NULL;
  bfieldfreq = 0;

  maxmigrate = 0;
  mlist = NULL;

  nglist_compute = nslist_compute = nblist_compute = 0;
  glist_compute = slist_compute = blist_compute = NULL;
  glist_active = slist_active = blist_active = NULL;

  ndlist_surfcollide  = 0;
  dlist_surfcollide = NULL;

  ranmaster = new RanMars(sparta);

  reorder_period = 0;
  global_mem_limit = 0;
  mem_limit_grid_flag = 0;

  copymode = 0;

  // All pusher state lives in class Pusher (src/OPENEDGE/pusher.{h,cpp}).
  pusher = new Pusher(sparta);

  cd_flag = 0;
  cd_nmax = 0;
  dx_cd = NULL;

  early_exit_requested = 0;

  psi_reflect_flag = 0;
  psi_reflect_action = 0;
  psi_reflect_threshold = 1.0;
  psi_nw = psi_nh = 0;
  psi_axis = psi_bry = 0.0;
  psi_r_grid = NULL;
  psi_z_grid = NULL;
  psi_rz = NULL;

  sheath_flag = 0;
  sheath_geom_cid = NULL;
  sheath_geom_cidx = -1;
  sheath_mD_amu = 2.01410177811;
  sheath_dmax = 0.0;
  sheath_kick = 0;
  sheath_boundary = 0;
  sheath_waveform_attr = NULL;
  sheath_waveform_custom = -1;
  sheath_frequency_hz = 0.0;
  sheath_paid_custom = -1;
  sheath_bank_custom = -1;
  sheath_phiprev_custom = -1;
  tally_pweight = 1.0;

  plasma_cache_flag = 0;
  pcache_need_mask = 0;
  pcache_nevery = 1;
  phi_track = 0;
  phi_custom = -1;
  pc_te_custom = pc_ti_custom = pc_ne_custom = pc_ni_custom = -1;
  pc_vpar_custom = -1;
  pc_bx_custom = pc_by_custom = pc_bz_custom = -1;
  pc_ex_custom = pc_ey_custom = pc_ez_custom = -1;
  pc_grad_ne_r_custom = pc_grad_ne_z_custom = -1;
  pc_grad_te_r_custom = pc_grad_te_z_custom = -1;
  pc_grad_ti_r_custom = pc_grad_ti_z_custom = -1;

}

/* ---------------------------------------------------------------------- */

Update::~Update()
{
  if (copymode) return;

  delete [] unit_style;
  delete [] fieldID;
  memory->destroy(mlist);

  delete [] glist_compute;
  delete [] slist_compute;
  delete [] blist_compute;

  delete [] glist_active;
  delete [] slist_active;
  delete [] blist_active;
  delete [] dlist_surfcollide;
  delete [] sheath_geom_cid;
  delete [] sheath_waveform_attr;
  delete pusher;
  memory->destroy(dx_cd);
  // psi_r_grid, psi_z_grid, psi_rz are owned by fix_reflect_psi, not freed here
  delete ranmaster;
}

/* ---------------------------------------------------------------------- */

void Update::set_units(const char *style)
{
  // physical constants from:
  // http://physics.nist.gov/cuu/Constants/Table/allascii.txt

  if (strcmp(style,"cgs") == 0) {
    boltz = 1.380649e-16;
    mvv2e = 1.0;
    dt = 1.0;

  } else if (strcmp(style,"si") == 0) {
    boltz = 1.380649e-23;
    mvv2e = 1.0;
    dt = 1.0;
    echarge = 1.60217646e-19;
    ev2kelvin = 11604.505;
    proton_mass = 1.6726219e-27;
    epsilon_0 = 8.854187817e-12;
    electron_mass = 9.10938215e-31;
    ANGSTROM = 1e-10;
    hbar = 1.0545718e-34;
    joule2ev = 6.242e18;
    c = 299792458.0;



  } else error->all(FLERR,"Illegal units command");

  delete [] unit_style;
  int n = strlen(style) + 1;
  unit_style = new char[n];
  strcpy(unit_style,style);
}

/* ---------------------------------------------------------------------- */

void Update::init()
{
  // init the Update class if performing a run, else just return
  // only set first_update if a run is being performed

  if (runflag == 0) return;
  first_update = 1;

  if (optmove_flag) {
    if (!grid->uniform)
      error->all(FLERR,"Cannot use optimized move with non-uniform grid");
    else if (surf->exist)
      error->all(FLERR,"Cannot use optimized move when surfaces are defined");
    else {
      for (int ifix = 0; ifix < modify->nfix; ifix++) {
        if (strstr(modify->fix[ifix]->style,"adapt") != NULL)
          error->all(FLERR,"Cannot use optimized move with fix adapt");
      }
    }
  }

  // choose the appropriate move method

  if (domain->dimension == 3) {
    if (surf->exist)
      moveptr = &Update::move<3,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<3,0,1>;
      else moveptr = &Update::move<3,0,0>;
    }
  } else if (domain->axisymmetric) {
    if (surf->exist)
      moveptr = &Update::move<1,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<1,0,1>;
      else moveptr = &Update::move<1,0,0>;
    }
  } else if (domain->dimension == 2) {
    if (surf->exist)
      moveptr = &Update::move<2,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<2,0,1>;
      else moveptr = &Update::move<2,0,0>;
    }
  }

  // checks on external field options

  if (fstyle == CFIELD) {
    if (domain->dimension == 2 && field[2] != 0.0)
      error->all(FLERR,"External field in z not allowed for 2d");
    if (domain->axisymmetric && field[1] != 0.0)
      error->all(FLERR,
                 "External field in y not allowed for axisymmetric model");
  } else if (fstyle == PFIELD) {
    ifieldfix = modify->find_fix(fieldID);
    if (ifieldfix < 0) error->all(FLERR,"External field fix ID not found");
    if (!modify->fix[ifieldfix]->per_particle_field)
      error->all(FLERR,"External field fix does not compute necessary field");
  } else if (fstyle == GFIELD) {
    ifieldfix = modify->find_fix(fieldID);
    if (ifieldfix < 0) error->all(FLERR,"External field fix ID not found");
    if (!modify->fix[ifieldfix]->per_grid_field)
      error->all(FLERR,"External field fix does not compute necessary field");
  }
  // checks options for external electric field only particle perturbation
  // similar to above fstyle checks
  eperturbflag = 0;
  if (efstyle == PFIELD) {
    efieldfix = modify->find_fix(efieldID);        // <-- NO 'int' here
    if (efieldfix < 0) error->all(FLERR,"External electric field fix ID not found");
    if (!modify->fix[efieldfix]->per_particle_field)
      error->all(FLERR,"External electric field fix does not compute necessary field");
    efield_active = modify->fix[efieldfix]->field_active;  // packed columns
    eperturbflag = 1;
  }
    // add GFIELD now
    if (efstyle == GFIELD) {
    efieldfix = modify->find_fix(efieldID);        // <-- NO 'int' here
    if (efieldfix < 0) error->all(FLERR,"External electric field fix ID not found");
    if (!modify->fix[efieldfix]->per_grid_field)
      error->all(FLERR,"External electric field fix does not compute necessary field");
    efield_active = modify->fix[efieldfix]->field_active;  // packed columns
    eperturbflag = 1;
  }
  // checks options for external magnetic field only particle perturbation
  bperturbflag = 0;
  if (bfstyle == PFIELD) {
    bfieldfix = modify->find_fix(bfieldID);        
    if (bfieldfix < 0) error->all(FLERR,"External magnetic field fix ID not found");
    if (!modify->fix[bfieldfix]->per_particle_field)
      error->all(FLERR,"External magnetic field fix does not compute necessary field");
    bfield_active = modify->fix[bfieldfix]->field_active;  // packed columns
    bperturbflag = 1;
  }
  if (bfstyle == GFIELD) {
    bfieldfix = modify->find_fix(bfieldID);     
    if (bfieldfix < 0) error->all(FLERR,"External magnetic field fix ID not found");
    if (!modify->fix[bfieldfix]->per_grid_field)
      error->all(FLERR,"External magnetic field fix does not compute necessary field");
    bfield_active = modify->fix[bfieldfix]->field_active;  // packed columns
    bperturbflag = 1;
  }

  // Resolve the pusher's skip mixture (dust grains bypass the pusher)
  pusher->resolve_skip_species();

  // Resolve per-particle sheath geometry compute and plasma provider IDs.
  // The geometry compute is also needed by the pusher's boris_near shell,
  // which must work with the sheath off (Stage A bare-handoff tests).
  if (sheath_geom_cid &&
      (sheath_flag || pusher->pusher_boris_near > 0.0 ||
       pusher->pusher_gc_wall_flux)) {
    sheath_geom_cidx = modify->find_compute(sheath_geom_cid);
    if (sheath_geom_cidx < 0)
      error->all(FLERR,"global sheath: geometry compute ID not found");
    if (!modify->compute[sheath_geom_cidx]->per_grid_flag)
      error->all(FLERR,"global sheath: geometry compute must be per-grid");
    // fail CLOSED: group filtering and the sheath cache both require the
    // nearest_surf/grid type — a failed downstream dynamic_cast must not
    // silently turn the sheath permissive
    if (!dynamic_cast<ComputeNearestSurfGrid *>(
            modify->compute[sheath_geom_cidx]))
      error->all(FLERR,
                 "global sheath: geometry compute must be style nearest_surf/grid");
  }
  if (sheath_flag) {
    pusher->pusher_plasma_cidx = modify->find_compute(pusher->pusher_plasma_cid);
    pusher->pusher_plasma_fidx = -1;
    if (pusher->pusher_plasma_cidx >= 0) {
      if (!modify->compute[pusher->pusher_plasma_cidx]->per_grid_flag)
        error->all(FLERR,"global sheath: plasma compute must be per-grid");
    } else {
      pusher->pusher_plasma_fidx = modify->find_fix(pusher->pusher_plasma_cid);
      if (pusher->pusher_plasma_fidx < 0)
        error->all(FLERR,"global sheath: plasma provider ID not found");
      auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher->pusher_plasma_fidx]);
      if (!pd)
        error->all(FLERR,
                   "global sheath: plasma fix provider must be style background");
    }
  }

  // Resolve Boris point-query B-field compute
  if (pusher->pusher_plasma_cid) {
    pusher->pusher_plasma_cidx = modify->find_compute(pusher->pusher_plasma_cid);
    pusher->pusher_plasma_fidx = -1;
    if (pusher->pusher_plasma_cidx >= 0) {
      if (!modify->compute[pusher->pusher_plasma_cidx]->per_grid_flag)
        error->all(FLERR,"global bfield_compute: compute must be per-grid");
      if (comm->me == 0 && screen)
        fprintf(screen,
                "  boris: bfield_compute '%s' bound to compute (per-grid)\n",
                pusher->pusher_plasma_cid);
    } else {
      pusher->pusher_plasma_fidx = modify->find_fix(pusher->pusher_plasma_cid);
      if (pusher->pusher_plasma_fidx < 0)
        error->all(FLERR,"global bfield_compute: provider ID not found");
      auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher->pusher_plasma_fidx]);
      if (!pd)
        error->all(FLERR,
                   "global bfield_compute: fix provider must be style background");
      if (comm->me == 0 && screen)
        fprintf(screen,
                "  boris: bfield_compute '%s' bound to fix background "
                "(has_bfield=%d, mesh_tri_b=%zu)\n",
                pusher->pusher_plasma_cid, pd->has_bfield, pd->mesh_tri_br.size());
    }
  }

  // Pusher-specific init (GCA plasma compute resolution + persistent
  // guiding-center custom particle attributes). Body in pusher.cpp.
  pusher->init();

  // A tile waveform is currently a target-local, zero-thickness boundary
  // potential.  Spread its owned values so nearest-surface indices in the
  // mover can address local and ghost elements uniformly.
  sheath_waveform_custom = -1;
  if (sheath_waveform_attr) {
    if (!sheath_boundary)
      error->all(FLERR,
        "global pusher sheath waveform currently requires sheath boundary");
    sheath_waveform_custom = surf->find_custom(sheath_waveform_attr);
    if (sheath_waveform_custom < 0)
      error->all(FLERR,"global pusher sheath waveform attribute not found");
    if (surf->etype[sheath_waveform_custom] != 1 ||
        surf->esize[sheath_waveform_custom] != 3)
      error->all(FLERR,
        "global pusher sheath waveform attribute must be "
        "DOUBLE[3] = [Vdc,Vrf_peak,phase_rad]");
    surf->spread_custom(sheath_waveform_custom);
  }

  // Enable the spatial-sheath per-wall-element coefficient cache when the
  // sheath E-field is applied in the pusher (spatial mode, not kick) and
  // the plasma is a fix background. The background can only change at run
  // init (FixBackground::reload fires from init only), and the cache is
  // cleared right here each init — so the per-element coefficients are
  // invariant for the duration of any run segment, static or not. Any
  // other plasma source keeps the per-particle path.
  pusher->sheath_cache_enabled = 0;
  pusher->sheath_cache.clear();
  if (sheath_flag && !sheath_kick && pusher->pusher_plasma_fidx >= 0) {
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[pusher->pusher_plasma_fidx]);
    if (pd) pusher->sheath_cache_enabled = 1;
  }
  // Escape hatch to force the per-particle sheath path (A/B validation
  // and benchmarking): set OE_NO_SHEATH_CACHE in the environment.
  if (getenv("OE_NO_SHEATH_CACHE")) pusher->sheath_cache_enabled = 0;

  // Boundary mode needs a persistent per-particle "paid" flag so the escape
  // deceleration fires once per sheath transit (see the barrier impulse in
  // push_boris_2d). Register it as an int custom particle attribute.
  sheath_paid_custom = -1;
  if (sheath_boundary) {
    sheath_paid_custom = particle->find_custom((char *) "sheath_paid");
    if (sheath_paid_custom < 0)
      // type 0 = INT (enum{INT,DOUBLE}), size 0 = per-particle scalar.
      sheath_paid_custom = particle->add_custom((char *) "sheath_paid", 0, 0);
  }

  // Spatial mode: per-particle net-energy ledger. The potential impulse is
  // conservative within a move, but the nearest wall element (and its phi
  // profile) can change between moves, leaving a small circulation; cap the
  // lifetime net gain at Z e phi_tot, the most a sheath can give an ion.
  sheath_bank_custom = -1;
  // A/B diagnostic switch: OE_SHEATH_NO_LEDGER disables the spatial-mode
  // bank/phiprev customs on BOTH the CPU and Kokkos paths (the movers
  // handle absent vectors: geometric phi_old, no lifetime cap).
  const int sheath_no_ledger = getenv("OE_SHEATH_NO_LEDGER") != nullptr;
  if (sheath_no_ledger && comm->me == 0 && screen)
    fprintf(screen,"OE_SHEATH_NO_LEDGER: spatial-sheath bank/phiprev customs disabled\n");
  if (!sheath_no_ledger && sheath_flag && !sheath_kick && !sheath_boundary) {
    sheath_bank_custom = particle->find_custom((char *) "sheath_bank");
    if (sheath_bank_custom < 0)
      sheath_bank_custom = particle->add_custom((char *) "sheath_bank", 1, 0);
  }

  // Spatial mode: per-particle phi reference so a reference-element switch
  // between moves is charged as work instead of re-seeding the potential
  // for free (the pump that filled the bank cap for band-dwelling ions).
  // Stored as phi+1 V; 0 = unset (newborn / newly ionized).
  sheath_phiprev_custom = -1;
  if (!sheath_no_ledger && sheath_flag && !sheath_kick && !sheath_boundary) {
    sheath_phiprev_custom = particle->find_custom((char *) "sheath_phiprev");
    if (sheath_phiprev_custom < 0)
      sheath_phiprev_custom = particle->add_custom((char *) "sheath_phiprev", 1, 0);
  }

  // Register per-particle plasma cache vectors.
  // Active when any plasma provider is available (sheath, GCA, or Boris B query).
  {
    int plasma_cidx = -1;
    int plasma_fidx = -1;
    if (sheath_flag && (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0)) {
      plasma_cidx = pusher->pusher_plasma_cidx;
      plasma_fidx = pusher->pusher_plasma_fidx;
    } else if (pusher->pusher_mode == Pusher::PUSHER_HYBRID &&
               (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0)) {
      plasma_cidx = pusher->pusher_plasma_cidx;
      plasma_fidx = pusher->pusher_plasma_fidx;
    } else if (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0) {
      plasma_cidx = pusher->pusher_plasma_cidx;
      plasma_fidx = pusher->pusher_plasma_fidx;
    }

    if (plasma_cidx >= 0 || plasma_fidx >= 0) {
      const int custom_double = 1;
      auto reg = [&](int &idx, const char *name) {
        if (idx < 0) {
          idx = particle->find_custom((char *) name);
          if (idx < 0) idx = particle->add_custom((char *) name, custom_double, 0);
        }
      };
      reg(pc_te_custom,   "pc_te");
      reg(pc_ti_custom,   "pc_ti");
      reg(pc_ne_custom,   "pc_ne");
      reg(pc_ni_custom,   "pc_ni");
      reg(pc_vpar_custom, "pc_vpar");
      reg(pc_bx_custom,   "pc_bx");
      reg(pc_by_custom,   "pc_by");
      reg(pc_bz_custom,   "pc_bz");
      reg(pc_ex_custom,   "pc_ex");
      reg(pc_ey_custom,   "pc_ey");
      reg(pc_ez_custom,   "pc_ez");
      reg(pc_grad_ne_r_custom, "pc_grad_ne_r");
      reg(pc_grad_ne_z_custom, "pc_grad_ne_z");
      reg(pc_grad_te_r_custom, "pc_grad_te_r");
      reg(pc_grad_te_z_custom, "pc_grad_te_z");
      reg(pc_grad_ti_r_custom, "pc_grad_ti_r");
      reg(pc_grad_ti_z_custom, "pc_grad_ti_z");
      plasma_cache_flag = 1;
    }
  }

  // Resolve which cache slots are actually consumed this run by scanning
  // active fix styles. Any consumer not on this list will fall back to the
  // full cache below (set in the unrecognized-style branch).
  pcache_need_mask = 0;
  pcache_nevery = 1;
  if (plasma_cache_flag) {
    int recognized = 0;
    // auto-derive the cache refresh cadence: min nevery over the fixes
    // that actually read the cache; anything unrecognized (or the sheath)
    // forces every-step refresh.
    int cad = 0;                 // 0 = no consumer seen yet
    auto note_cad = [&cad](int n) {
      if (n < 1) n = 1;
      cad = (cad == 0) ? n : (n < cad ? n : cad);
    };
    for (int ifix = 0; ifix < modify->nfix; ifix++) {
      const char *s = modify->fix[ifix]->style;
      if (strcmp(s,"volume/chem/adas") == 0 || strcmp(s,"volume/chem/adas/kk") == 0) {
        // volume/chem/adas always reads Te+Ne; Ti/Vpar/Bfield only matter for the
        // charge-exchange (EXCHANGE) channel. Skip those bits when the
        // reactions file has no CX entry — common for pure ionization runs.
        pcache_need_mask |= PCACHE_TE | PCACHE_NE;
        FixVolumeChemAdas *fchem =
            dynamic_cast<FixVolumeChemAdas *>(modify->fix[ifix]);
        if (fchem && fchem->needs_cx_fields()) {
          pcache_need_mask |= PCACHE_TI | PCACHE_VPAR | PCACHE_BFIELD;
        }
        note_cad(modify->fix[ifix]->nevery);
        recognized = 1;
      } else if (strcmp(s,"force/thermal") == 0) {
        // In background mode the fix interpolates from FixBackground
        // directly and does not read pcache; skip the writes entirely.
        FixForceThermal *ftf =
            dynamic_cast<FixForceThermal *>(modify->fix[ifix]);
        if (!ftf || ftf->needs_pcache()) {
          pcache_need_mask |= PCACHE_BFIELD | PCACHE_GRAD_TE | PCACHE_GRAD_TI;
          note_cad(modify->fix[ifix]->nevery);
        }
        recognized = 1;
      } else if (strcmp(s,"coulomb/binary") == 0 ||
                 strcmp(s,"coulomb/binary/kk") == 0 ||
                 strcmp(s,"coulomb/background") == 0 ||
                 strcmp(s,"coulomb/background/kk") == 0) {
        // Same background-bypass pattern as thermal_force.
        FixCoulombBase *fcb =
            dynamic_cast<FixCoulombBase *>(modify->fix[ifix]);
        if (!fcb || fcb->needs_pcache()) {
          pcache_need_mask |= PCACHE_TE | PCACHE_NE | PCACHE_TI | PCACHE_NI |
                              PCACHE_VPAR | PCACHE_BFIELD;
          note_cad(modify->fix[ifix]->nevery);
        }
        recognized = 1;
      } else if (strcmp(s,"cross_field_diffusion") == 0) {
        // Same background-bypass pattern. NE/GRAD_NE only matter when
        // gradient_pinch is configured (needs_grad_ne()).
        FixCrossFieldDiffusion *fcd =
            dynamic_cast<FixCrossFieldDiffusion *>(modify->fix[ifix]);
        if (!fcd || fcd->needs_pcache()) {
          pcache_need_mask |= PCACHE_BFIELD;
          if (fcd && fcd->needs_grad_ne()) {
            pcache_need_mask |= PCACHE_NE | PCACHE_GRAD_NE;
          }
          note_cad(modify->fix[ifix]->nevery);
        }
        recognized = 1;
      } else if (strcmp(s,"efield/particle") == 0) {
        pcache_need_mask |= PCACHE_EFIELD;
        note_cad(modify->fix[ifix]->nevery);
        recognized = 1;
      }
    }
    // Sheath Boltzmann ne correction (inside cache_plasma_particles) reads
    // te, ti, ne locally and needs Bmag; B is queried locally so does not
    // require the PCACHE_BFIELD write slot.
    if (sheath_flag && sheath_geom_cidx >= 0) {
      pcache_need_mask |= PCACHE_TE | PCACHE_NE | PCACHE_TI;
      note_cad(1);
    }
    // Backward compat: if no recognized consumer was found but the cache
    // is enabled (some user-defined fix may read pc_* via particle vars),
    // fall back to filling everything to avoid silently starving a reader.
    if (!recognized) {
      pcache_need_mask = PCACHE_ALL;
      note_cad(1);
    }
    pcache_nevery = (cad == 0) ? 1 : cad;
    if (comm->me == 0 && pcache_nevery > 1) {
      char msg[128];
      snprintf(msg, sizeof(msg),
               "pcache: auto refresh cadence every %d steps "
               "(slowest cache consumer)\n", pcache_nevery);
      if (screen)  fputs(msg, screen);
      if (logfile) fputs(msg, logfile);
    }
  }

  // moveperturb method is set if external field perturbs particle motion
  moveperturb = NULL;

  if (fstyle == CFIELD) {
    if (domain->dimension == 2) moveperturb = &Update::field2d;
    if (domain->dimension == 3) moveperturb = &Update::field3d;
  } else if (fstyle == PFIELD) {
    moveperturb = &Update::field_per_particle;
    field_active = modify->fix[ifieldfix]->field_active;
  } else if (fstyle == GFIELD) {
    moveperturb = &Update::field_per_grid;
    field_active = modify->fix[ifieldfix]->field_active;
  }

  if (moveperturb) perturbflag = 1;
  else perturbflag = 0;

}

/* ---------------------------------------------------------------------- */

void Update::setup()
{
  // initialize counters in case stats outputs them
  // initialize running stats before each run

  ntouch_one = ncomm_one = 0;
  nboundary_one = nexit_one = 0;
  nscheck_one = nscollide_one = 0;
  surf->nreact_one = 0;

  first_running_step = update->ntimestep;
  niterate_running = 0;
  nmove_running = ntouch_running = ncomm_running = 0;
  nboundary_running = nexit_running = 0;
  nscheck_running = nscollide_running = 0;
  surf->nreact_running = 0;
  nstuck = naxibad = 0;

  collide_react = collide_react_setup();
  tallyflag = tally_setup();
  dynamic = dynamic_setup();

  modify->setup();
  if (dynamic) dynamic_update();
  output->setup(1);

  
}

/* ---------------------------------------------------------------------- */

void Update::run(int nsteps)
{
  int n_start_of_step = modify->n_start_of_step;
  int n_end_of_step = modify->n_end_of_step;

  // external per grid cell field
  // only evaluate once at beginning of run b/c time-independent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq == 0)
    modify->fix[ifieldfix]->compute_field();

    // external per grid cell electric field
  if (efstyle == GFIELD && efieldfreq == 0)
    modify->fix[efieldfix]->compute_field();

    // external per grid cell magnetic field
  if (bfstyle == GFIELD && bfieldfreq == 0)
    modify->fix[bfieldfix]->compute_field();

    // external per grid cell electron thermal gradient field
  // cellweightflag = 1 if grid-based particle weighting is ON

  int cellweightflag = 0;
  if (grid->cellweightflag) cellweightflag = 1;

  // loop over timesteps

  for (int i = 0; i < nsteps; i++) {

    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep++;

    if (collide_react) collide_react_reset();
    if (tallyflag) tally_set(ntimestep);
    if (dynamic) dynamic_update();


    timer->stamp();

    // start of step fixes

    if (n_start_of_step) {
      modify->start_of_step();
      timer->stamp(TIME_MODIFY);
    }

    // cache plasma fields at particle positions (one query per particle).
    // Gated by pcache_nevery so decks whose only consumer (e.g. fix
    // volume/chem/adas at nevery=10) doesn't need it every step can skip the
    // population on the off steps. When pcache_nevery > 1, consumers
    // read stale values between refreshes; tradeoff is fine for plasma
    // quantities whose spatial scale (~mm) exceeds particle displacement
    // over N steps (~10s of um at typical dt).
    if (plasma_cache_flag &&
        (pcache_nevery <= 1 || ntimestep % pcache_nevery == 0)) {
      cache_plasma_particles();
      timer->stamp(TIME_PCACHE);
    }

    // move particles (skip when global move no)

    if (move_flag) {
      if (cellweightflag) particle->pre_weight();
      (this->*moveptr)();
      timer->stamp(TIME_MOVE);

      // communicate particles

      comm->migrate_particles(nmigrate,mlist);
      if (cellweightflag) particle->post_weight();
      timer->stamp(TIME_COMM);
    }

    // Early-exit on empty domain — disabled. Was added for fix
    // volume/chem/adas Mode A exhaustion but kills emit-only decks
    // (fix droplet/emit with nevery > 1) at step 1 before the first
    // emission can fire. Re-enable behind a flag if Mode A needs it.
    // {
    //   bigint nlocal = particle->nlocal;
    //   bigint nglobal;
    //   MPI_Allreduce(&nlocal, &nglobal, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    //   if (nglobal == 0 || early_exit_requested) {
    //     output->next = ntimestep;
    //     output->write(ntimestep);
    //     update->nsteps = i;
    //     early_exit_requested = 0;
    //     break;
    //   }
    // }

    // sort particles by grid cell if collisions are enabled
    // also sort if reordering is requested this step, since reordering
    //   requires the particles first be sorted
    // reorder() must be called here, not from within sort(), so that it
    //   only acts on the main timestep loop and not other sort() callers

    int reorder_flag = (reorder_period &&
                        ntimestep % reorder_period == 0);

    if (collide || reorder_flag) {
      particle->sort();
      if (reorder_flag) particle->reorder();
      timer->stamp(TIME_SORT);
    }

    if (collide) {
      collide->collisions();
      timer->stamp(TIME_COLLIDE);
    }


    if (collide_react) collide_react_update();

    // diagnostic fixes

    if (n_end_of_step) {
      modify->end_of_step();
      timer->stamp(TIME_MODIFY);
    }

    // all output

    if (ntimestep == output->next) {
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
  }

  modify->post_run();
}

/* ----------------------------------------------------------------------
   Cache plasma fields at every particle position using custom vectors.
   Called once per timestep before the move.  All consumers (Boris sheath,
   ADAS chemistry, Nanbu collisions) read from these cached values.
------------------------------------------------------------------------- */

void Update::cache_plasma_particles()
{
  // Resolve the plasma compute used for point-sampled particle caches.
  int plasma_cidx = -1;
  int plasma_fidx = -1;
  if (sheath_flag && (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0)) {
    plasma_cidx = pusher->pusher_plasma_cidx;
    plasma_fidx = pusher->pusher_plasma_fidx;
  } else if (pusher->pusher_mode == Pusher::PUSHER_HYBRID && pusher->pusher_plasma_cidx >= 0) {
    plasma_cidx = pusher->pusher_plasma_cidx;
  } else if (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0) {
    plasma_cidx = pusher->pusher_plasma_cidx;
    plasma_fidx = pusher->pusher_plasma_fidx;
  }
  if (plasma_cidx < 0 && plasma_fidx < 0) return;

  ComputePlasmaFields *cp = nullptr;
  FixBackground *pd = nullptr;
  if (plasma_cidx >= 0) {
    Compute *c_base = modify->compute[plasma_cidx];
    cp = dynamic_cast<ComputePlasmaFields *>(c_base);
    if (!cp) return;
    if (!(c_base->invoked_flag & 16)) {  // INVOKED_PER_GRID = 16
      c_base->compute_per_grid();
      c_base->invoked_flag |= 16;
    }
  } else {
    pd = dynamic_cast<FixBackground *>(modify->fix[plasma_fidx]);
    if (!pd) return;
  }

  // resolve per-particle custom vectors
  // Guard: bail out if custom indices are invalid, vectors are NULL,
  // or no local particles on this rank
  if (particle->nlocal == 0) return;
  if (pc_te_custom < 0 || pc_ti_custom < 0 || pc_ne_custom < 0 ||
      pc_ni_custom < 0 || pc_vpar_custom < 0 ||
      pc_bx_custom < 0 || pc_by_custom < 0 || pc_bz_custom < 0 ||
      pc_ex_custom < 0 || pc_ey_custom < 0 || pc_ez_custom < 0 ||
      pc_grad_ne_r_custom < 0 || pc_grad_ne_z_custom < 0 ||
      pc_grad_te_r_custom < 0 || pc_grad_te_z_custom < 0 ||
      pc_grad_ti_r_custom < 0 || pc_grad_ti_z_custom < 0) return;
  if (particle->ewhich[pc_te_custom] < 0) return;

  double *te_vec   = particle->edvec[particle->ewhich[pc_te_custom]];
  double *ti_vec   = particle->edvec[particle->ewhich[pc_ti_custom]];
  double *ne_vec   = particle->edvec[particle->ewhich[pc_ne_custom]];
  double *ni_vec   = particle->edvec[particle->ewhich[pc_ni_custom]];
  double *vpar_vec = particle->edvec[particle->ewhich[pc_vpar_custom]];
  double *bx_vec   = particle->edvec[particle->ewhich[pc_bx_custom]];
  double *by_vec   = particle->edvec[particle->ewhich[pc_by_custom]];
  double *bz_vec   = particle->edvec[particle->ewhich[pc_bz_custom]];
  double *ex_vec   = particle->edvec[particle->ewhich[pc_ex_custom]];
  double *ey_vec   = particle->edvec[particle->ewhich[pc_ey_custom]];
  double *ez_vec   = particle->edvec[particle->ewhich[pc_ez_custom]];
  double *gne_r_vec = particle->edvec[particle->ewhich[pc_grad_ne_r_custom]];
  double *gne_z_vec = particle->edvec[particle->ewhich[pc_grad_ne_z_custom]];
  double *gte_r_vec = particle->edvec[particle->ewhich[pc_grad_te_r_custom]];
  double *gte_z_vec = particle->edvec[particle->ewhich[pc_grad_te_z_custom]];
  double *gti_r_vec = particle->edvec[particle->ewhich[pc_grad_ti_r_custom]];
  double *gti_z_vec = particle->edvec[particle->ewhich[pc_grad_ti_z_custom]];
  if (!te_vec || !ti_vec || !ne_vec || !ni_vec ||
      !vpar_vec || !bx_vec || !by_vec || !bz_vec ||
      !ex_vec || !ey_vec || !ez_vec ||
      !gne_r_vec || !gne_z_vec || !gte_r_vec || !gte_z_vec ||
      !gti_r_vec || !gti_z_vec) return;

  // Sheath geometry compute (for Boltzmann ne correction)
  ComputeNearestSurfGrid *csg = nullptr;
  if (sheath_flag && sheath_geom_cidx >= 0) {
    Compute *cg = modify->compute[sheath_geom_cidx];
    csg = dynamic_cast<ComputeNearestSurfGrid *>(cg);
  }

  Particle::OnePart *particles = particle->particles;
  Grid::ChildCell *cells = grid->cells;
  const int nlocal = particle->nlocal;
  const int dim = domain->dimension;

  const int mask = pcache_need_mask;
  const bool need_plasma =
      (mask & (PCACHE_TE | PCACHE_NE | PCACHE_TI | PCACHE_NI | PCACHE_VPAR |
               PCACHE_GRAD_NE | PCACHE_GRAD_TE | PCACHE_GRAD_TI |
               PCACHE_EFIELD)) ||
      (csg != nullptr);
  const bool need_bfield =
      (mask & (PCACHE_BFIELD | PCACHE_EFIELD)) || (csg != nullptr);
  const bool write_b = (mask & PCACHE_BFIELD) != 0;
  const bool write_e = (mask & PCACHE_EFIELD) != 0;

  // Pre-compute grad_ne finite-difference offsets for the pd path
  // (used only when PCACHE_GRAD_NE is set and we are on the fix path).
  double pd_dR = 0.0, pd_dZ = 0.0;
  if (pd && pd->nr >= 2 && pd->nz >= 2) {
    pd_dR = std::max(1.0e-9, 0.5 * std::fabs(pd->rvals[1] - pd->rvals[0]));
    pd_dZ = std::max(1.0e-9, 0.5 * std::fabs(pd->zvals[1] - pd->zvals[0]));
  }

  // Cell-indexed mesh cache: build once per (grid_changed | reload) and
  // share across all particles on this rank. Dominant pcache-loop saving
  // when the mesh is loaded — replaces a per-particle triangle spatial
  // search with an O(1) warm-start by particles[i].icell; the final
  // lookup is exact at the particle position either way (validated by
  // the probe_background gates).
  const bool use_cell_mesh_cache = (pd && pd->has_mesh);
  if (use_cell_mesh_cache) {
    // Invalidate when the grid changes. Two stamps catch the common cases:
    //  - nlocal differs → adapt added/removed cells
    //  - cells[0].id differs at same nlocal → RCB reshuffle
    const int nloc = grid->nlocal;
    const bigint first_id =
        (nloc > 0 && grid->cells) ? grid->cells[0].id : -1;
    if (static_cast<int>(pd->cell_mesh_cell.size()) != nloc ||
        pd->cell_mesh_stamp_n != nloc ||
        pd->cell_mesh_stamp_id != first_id) {
      pd->build_cell_mesh_index();
    }
  }
  const int cmc_size =
      use_cell_mesh_cache ? static_cast<int>(pd->cell_mesh_cell.size()) : 0;
  const int *cmc =
      use_cell_mesh_cache && cmc_size > 0 ? pd->cell_mesh_cell.data() : nullptr;

  for (int i = 0; i < nlocal; i++) {
    const double *x = particles[i].x;
    const int icell_p = particles[i].icell;

    PlasmaFileParams pf{};
    MagneticFieldFileDataParams bf{};

    if (cp) {
      // ---- compute plasma/fields path: existing point queries ----
      if (need_plasma) pf = cp->query_plasma_at_point(x);
      if (need_bfield) bf = cp->query_bfield_at_point(x);
    } else if (pd) {
      // ---- fix background path: shared bilinear stencil ----
      double R, Z;
      xyz_to_rz(x, dim, domain->axisymmetric, R, Z);
      const PdStencil2D st = make_pd_stencil(pd, R, Z);
      int mesh_cell;
      if (pd->has_mesh && need_plasma) {
        // exact hinted lookup at the particle position (no extrapolation)
        mesh_cell = pd->mesh_cell_for(R, Z, icell_p, i);
      } else {
        mesh_cell = -1;
      }
      if (need_plasma) {
        if (mesh_cell >= 0) {
          if ((mask & PCACHE_TE) && mesh_cell < static_cast<int>(pd->mesh_te.size()))
            pf.temp_e = pd->mesh_te[mesh_cell];
          if ((mask & PCACHE_NE) && mesh_cell < static_cast<int>(pd->mesh_ne.size()))
            pf.dens_e = pd->mesh_ne[mesh_cell];
          if ((mask & PCACHE_TI) && mesh_cell < static_cast<int>(pd->mesh_ti.size()))
            pf.temp_i = pd->mesh_ti[mesh_cell];
          if ((mask & PCACHE_NI) && mesh_cell < static_cast<int>(pd->mesh_ni.size()))
            pf.dens_i = pd->mesh_ni[mesh_cell];
          if ((mask & PCACHE_VPAR) && mesh_cell < static_cast<int>(pd->mesh_upar.size()))
            pf.parr_flow = pd->mesh_upar[mesh_cell];
        } else {
          if (mask & PCACHE_TE)   pf.temp_e   = interp_pd_stencil(pd->temp_e, st);
          if (mask & PCACHE_NE)   pf.dens_e   = interp_pd_stencil(pd->dens_e, st);
          if (mask & PCACHE_TI)   pf.temp_i   = interp_pd_stencil(pd->temp_i, st);
          if (mask & PCACHE_NI)   pf.dens_i   = interp_pd_stencil(pd->dens_i, st);
          if (mask & PCACHE_VPAR) pf.parr_flow = interp_pd_stencil(pd->parr_flow, st);
        }
        if (mask & PCACHE_GRAD_TE) {
          pf.grad_temp_e_r = interp_pd_stencil(pd->grad_te_r, st);
          pf.grad_temp_e_z = interp_pd_stencil(pd->grad_te_z, st);
        }
        if (mask & PCACHE_GRAD_TI) {
          pf.grad_temp_i_r = interp_pd_stencil(pd->grad_ti_r, st);
          pf.grad_temp_i_z = interp_pd_stencil(pd->grad_ti_z, st);
        }
        if (mask & PCACHE_GRAD_NE) {
          // finite-difference grad_ne. In mesh mode this reuses the
          // mesh-first interp2D() path; in grid mode it keeps the stencil fast path.
          const double fR_p = (mesh_cell >= 0)
            ? pd->interp2D(pd->dens_e, R + pd_dR, Z)
            : interp_pd_stencil(pd->dens_e, make_pd_stencil(pd, R + pd_dR, Z));
          const double fR_m = (mesh_cell >= 0)
            ? pd->interp2D(pd->dens_e, R - pd_dR, Z)
            : interp_pd_stencil(pd->dens_e, make_pd_stencil(pd, R - pd_dR, Z));
          const double fZ_p = (mesh_cell >= 0)
            ? pd->interp2D(pd->dens_e, R, Z + pd_dZ)
            : interp_pd_stencil(pd->dens_e, make_pd_stencil(pd, R, Z + pd_dZ));
          const double fZ_m = (mesh_cell >= 0)
            ? pd->interp2D(pd->dens_e, R, Z - pd_dZ)
            : interp_pd_stencil(pd->dens_e, make_pd_stencil(pd, R, Z - pd_dZ));
          pf.grad_dens_e_r = (fR_p - fR_m) / (2.0 * pd_dR);
          pf.grad_dens_e_z = (fZ_p - fZ_m) / (2.0 * pd_dZ);
        }
        if (write_e && !pd->epar.empty()) {
          pf.epar = interp_pd_stencil(pd->epar, st);
        }
      }
      if (need_bfield && pd->has_bfield) {
        // Route through bfield_at() so mesh-native B (mesh_tri_b*) is
        // picked up on mesh-only plasma.h5 runs; the stencil path only
        // hits the empty regular-grid arrays and would return zero.
        pd->bfield_at(R, Z, bf.br, bf.bz, bf.bt, particles[i].icell, i);
        bf.Bmag = std::sqrt(bf.br*bf.br + bf.bt*bf.bt + bf.bz*bf.bz);
      }
    }

    if (mask & PCACHE_TE)   te_vec[i]   = pf.temp_e;
    if (mask & PCACHE_TI)   ti_vec[i]   = pf.temp_i;
    if (mask & PCACHE_NE)   ne_vec[i]   = pf.dens_e;
    if (mask & PCACHE_NI)   ni_vec[i]   = pf.dens_i;
    if (mask & PCACHE_VPAR) vpar_vec[i] = pf.parr_flow;
    if (mask & PCACHE_GRAD_NE) {
      gne_r_vec[i] = pf.grad_dens_e_r;
      gne_z_vec[i] = pf.grad_dens_e_z;
    }
    if (mask & PCACHE_GRAD_TE) {
      gte_r_vec[i] = pf.grad_temp_e_r;
      gte_z_vec[i] = pf.grad_temp_e_z;
    }
    if (mask & PCACHE_GRAD_TI) {
      gti_r_vec[i] = pf.grad_temp_i_r;
      gti_z_vec[i] = pf.grad_temp_i_z;
    }

    // B-field decomposition into Cartesian-or-axisymmetric components.
    // bf was sampled above (cp or pd path); only enter this block when a
    // consumer needs B (cached B/E slots, sheath Boltzmann correction).
    double bx = 0.0, by = 0.0, bz = 0.0;
    if (need_bfield) {
      // Store B and background E at particle position using the same component
      // mapping as compute plasma/fields: 2D -> (Bx,By,Bz)=(Br,Bz,Bt).
      // Azimuth about the column axis, not the domain origin.
      const double bcol_x0 = cp ? cp->plasma_data.column_x0
                          : (pd ? pd->column_x0 : 0.0);
      const double bcol_y0 = cp ? cp->plasma_data.column_y0
                          : (pd ? pd->column_y0 : 0.0);
      const double rx = x[0] - bcol_x0, ry = x[1] - bcol_y0;
      const double rmag = std::sqrt(rx*rx + ry*ry);
      double ex = 0.0, ey = 0.0, ez = 0.0;
      const double Bmag = std::sqrt(bf.br*bf.br + bf.bt*bf.bt + bf.bz*bf.bz);
      // E: background mesh vector E (presheath) when available, else the
      // epar projection along B
      double Er_c = 0.0, Ez_c = 0.0, Et_c = 0.0;
      bool have_e = false;
      if (write_e) {
        // pd is null on the compute plasma/fields path; E then comes from
        // the epar projection below (cp fills pf.epar from its mesh E)
        have_e = pd && pd->query_efield_at_point(x, Er_c, Ez_c, Et_c, icell_p, i);
        if (!have_e && Bmag > 1.0e-30 && pf.epar != 0.0) {
          Er_c = pf.epar * bf.br / Bmag;
          Et_c = pf.epar * bf.bt / Bmag;
          Ez_c = pf.epar * bf.bz / Bmag;
          have_e = true;
        }
      }
      if (rmag > 1.0e-20 && dim == 3) {
        const double cphi = rx / rmag, sphi = ry / rmag;
        bx = bf.br * cphi - bf.bt * sphi;
        by = bf.br * sphi + bf.bt * cphi;
        if (have_e) {
          ex = Er_c * cphi - Et_c * sphi;
          ey = Er_c * sphi + Et_c * cphi;
          ez = Ez_c;
        }
      } else {
        bx = bf.br;
        by = (dim == 3) ? 0.0 : bf.bz;
        if (have_e) {
          ex = Er_c;
          ey = (dim == 3) ? 0.0 : Ez_c;
          ez = (dim == 3) ? Ez_c : Et_c;
        }
      }
      bz = (dim == 3) ? bf.bz : bf.bt;
      if (write_b) {
        bx_vec[i] = bx;
        by_vec[i] = by;
        bz_vec[i] = bz;
      }
      if (write_e) {
        ex_vec[i] = ex;
        ey_vec[i] = ey;
        ez_vec[i] = ez;
      }
    }

    // Boltzmann ne correction: ne_local = ne_upstream * exp(-phi/Te)
    // where phi = sheath potential drop at particle distance from wall
    // Safety net: bound icell before touching cells[]. A particle appended
    // after the last sort can survive a rebalance holding a pre-balance
    // index; the emitting fixes clear Particle::sorted to prevent that.
    if (csg && pf.temp_e > 0.0 && pf.dens_e > 0.0 &&
        particles[i].icell >= 0 &&
        particles[i].icell < grid->nlocal + grid->nghost) {
      int icell = particles[i].icell;
      int gcell = icell;
      if (cells[icell].nsplit <= 0 && cells[icell].isplit >= 0)
        gcell = grid->sinfo[cells[icell].isplit].icell;

      if (gcell >= 0 && gcell < csg->nglocal) {
        int midx = csg->midx_grid[gcell];

        // refine to nearest surface at particle position (same as pusher)
        Grid::ChildCell *pc = &cells[gcell];
        if (pc->nsurf > 0) {
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
          // get surface normal and reference point
          double sh_nx, sh_ny, sh_nz;
          double sref[3];
          if (dim == 2) {
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

          const double d_particle = std::fabs(
            (x[0]-sref[0])*sh_nx + (x[1]-sref[1])*sh_ny + (x[2]-sref[2])*sh_nz);

          const double te = pf.temp_e;
          const double ti = pf.temp_i;
          const double ne = pf.dens_e;
          const double bmag = std::sqrt(bx*bx + by*by + bz*bz);

          double alpha_deg = 90.0;
          if (bmag > 0.0) {
            double bvec[3] = {bx, by, bz};
            double nvec[3] = {sh_nx, sh_ny, sh_nz};
            SheathModels::ChoduraMetrics cm =
              SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
            alpha_deg = cm.alpha_deg;
          }

          const double d_max = sheath_auto_dmax(te, ti, ne, bmag, alpha_deg,
                                                sheath_mD_amu, sheath_dmax);
          if (d_particle > 0.0 && d_particle < d_max) {
            SheathModels::BorodkinaSheathResult sr =
              SheathModels::coulette_manfredi_sheath_at_distance(
                d_particle, te, ti, ne, bmag,
                alpha_deg, sheath_mD_amu, 0.0);

            // Boltzmann: ne_local = ne * exp(-phi/Te), phi = esheath_eV (positive)
            if (sr.esheath_eV > 0.0 && te > 0.0) {
              ne_vec[i] = ne * std::exp(-sr.esheath_eV / te);
            }
          }
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   advect particles thru grid
   DIM = 2/3 for 2d/3d, 1 for 2d axisymmetric
   SURF = 0/1 for no surfs or surfs
   use multiple iterations of move/comm if necessary
------------------------------------------------------------------------- */

template < int DIM, int SURF, int OPT > void Update::move()
{
  bool hitflag;
  int m,icell,icell_original,nmask,outface,bflag,nflag,pflag,itmp;
  int side,minside,minsurf,nsurf,cflag,isurf,exclude,stuck_iterate;
  int pstart,pstop,entryexit,any_entryexit,reaction;
  surfint *csurfs;
  cellint *neigh;
  double dtremain,frac,newfrac,param,minparam,rnew,dtsurf,tc,tmp;
  double xnew[3],xhold[3],xc[3],vc[3],minxc[3],minvc[3];
  double *x,*v,*lo,*hi;
  double Lx,Ly,Lz,dx,dy,dz;
  double *boxlo, *boxhi;
  Grid::ParentCell *pcell;
  Surf::Tri *tri;
  Surf::Line *line;
  Particle::OnePart iorig;
  Particle::OnePart *particles;
  Particle::OnePart *ipart,*jpart;
  Particle::Species* species = particle->species;


  if (OPT) {
    boxlo = domain->boxlo;
    boxhi = domain->boxhi;
    Lx = boxhi[0] - boxlo[0];
    Ly = boxhi[1] - boxlo[1];
    Lz = boxhi[2] - boxlo[2];
    dx = Lx/grid->unx;
    dy = Ly/grid->uny;
    dz = Lz/grid->unz;
  }

  // for 2d and axisymmetry only
  // xnew,xc passed to geometry routines which use or set z component

  if (DIM < 3) xnew[2] = xc[2] = 0.0;

  // extend migration list if necessary

  int nlocal = particle->nlocal;
  int maxlocal = particle->maxlocal;

  if (nlocal > maxmigrate) {
    maxmigrate = maxlocal;
    memory->destroy(mlist);
    memory->create(mlist,maxmigrate,"particle:mlist");
  }

  // counters

  niterate = 0;
  ntouch_one = ncomm_one = 0;
  nboundary_one = nexit_one = 0;
  nscheck_one = nscollide_one = 0;
  surf->nreact_one = 0;

  // Reset spatial-sheath diagnostics for this step (see print below).
  pusher->sheath_diag_nactive = 0;
  pusher->sheath_diag_nengage = 0;
  pusher->sheath_diag_emax = 0.0;
  pusher->sheath_diag_esum = 0.0;
  pusher->sheath_diag_nreflect = 0;
  pusher->sheath_diag_nescape = 0;

  // move/migrate iterations

  Grid::ChildCell *cells = grid->cells;
  Grid::ParentCell *pcells = grid->pcells;
  Surf::Tri *tris = surf->tris;
  Surf::Line *lines = surf->lines;
  Grid::ChildInfo *cinfo = grid->cinfo;

  double dt = update->dt;

  // Optional built-in per-particle first-hit tracking.
  // Attributes are created by Particle ctor and dumped via p_hit_flag/p_hit_surf_id.
  const int hit_flag_index = particle->find_custom((char *) "hit_flag");
  const int hit_surf_index = particle->find_custom((char *) "hit_surf_id");

  // external per particle field
  // fix calculates field acting on all owned particles

  if (fstyle == PFIELD) modify->fix[ifieldfix]->compute_field();

  // external per grid cell field
  // evaluate once every fieldfreq steps b/c time-dependent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ifieldfix]->compute_field();

  // per-particle E
  if (efstyle == PFIELD) modify->fix[efieldfix]->compute_field();

  // per-particle B 
  if (bfstyle == PFIELD) modify->fix[bfieldfix]->compute_field();

  // per-grid E
  if (efstyle == GFIELD && efieldfreq && ((ntimestep-1) % efieldfreq == 0))
    modify->fix[efieldfix]->compute_field();

  // per-grid B
  if (bfstyle == GFIELD && bfieldfreq && ((ntimestep-1) % bfieldfreq == 0))
    modify->fix[bfieldfix]->compute_field();

  // Pre-compute sheath geometry and plasma BEFORE the particle loop.
  // Geometry is static (surfaces don't move) — compute once, reuse forever.
  // Plasma may update if coupled to a solver; for analytic profiles it's also static.
  if (sheath_flag && sheath_geom_cidx >= 0 &&
      (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0)) {
    Compute *cg = modify->compute[sheath_geom_cidx];
    if (cg->invoked_per_grid < 0) cg->compute_per_grid();  // only first time
    if (pusher->pusher_plasma_cidx >= 0) {
      Compute *cp = modify->compute[pusher->pusher_plasma_cidx];
      if (cp->invoked_per_grid < 0) cp->compute_per_grid();   // only first time
    }
  }

  // Per-particle charge override from fix droplet/charge (custom DOUBLE
  // vector "particulate_charge"). Cache only the ewhich INDEX, not the array
  // pointer: add_particle (mid-move migration receives) can reallocate
  // edvec, and a pointer captured here would dangle — intermittent MPI-only
  // bus errors in move. Same policy as pweight below.
  const int dq_idx = particle->find_custom((char *) "particulate_charge");
  const int dq_ewhich = (dq_idx >= 0) ? particle->ewhich[dq_idx] : -1;

  // pweight custom (fix particle/weight) -> stamped into tally_pweight
  // before each surf_tally batch so pweight-aware surf computes can weight
  // the incident particle even when it is absorbed (ip=NULL). Refreshed in
  // the loop since add_particle can reallocate edvec mid-move.
  const int pw_idx = particle->find_custom((char *) "pweight");
  const int pw_ewhich = (pw_idx >= 0) ? particle->ewhich[pw_idx] : -1;

  // Sheath geometry compute: fix adapt reallocates per-grid computes
  // (grid->notify_changed), zeroing midx_grid and invalidating its static
  // cache. Re-evaluate before this move reads it, else the sheath model
  // silently switches off after the first grid adaptation. The call
  // no-ops whenever the cached geometry is still valid.
  // (also needed by the pusher's boris_near shell with the sheath off)
  if ((sheath_flag || pusher->pusher_boris_near > 0.0 ||
       pusher->pusher_gc_wall_flux) &&
      sheath_geom_cidx >= 0) {
    Compute *cg_sh = modify->compute[sheath_geom_cidx];
    if (cg_sh) cg_sh->compute_per_grid();
  }

  // one or more loops over particles
  // first iteration = all my particles
  // subsequent iterations = received particles

  while (1) {

    niterate++;
    particles = particle->particles;
    nmigrate = 0;
    entryexit = 0;

    if (niterate == 1) {
      pstart = 0;
      pstop = nlocal;
    }

    for (int i = pstart; i < pstop; i++) {
      pflag = particles[i].flag;

      // received from another proc and move is done
      // if first iteration, PDONE is from a previous step,
      //   set pflag to PKEEP so move the particle on this step
      // else do nothing

      if (pflag == PDONE) {
        pflag = particles[i].flag = PKEEP;
        if (niterate > 1) continue;
      }
    
      x = particles[i].x;
      v = particles[i].v;
      exclude = -1;

      // cross-field diffusion kick folded into v for this step (see PKEEP
      // block below); stripped again once the move completes so the random
      // kick does not accumulate as velocity-space heating.
      double vkick0 = 0.0, vkick1 = 0.0, vkick2 = 0.0;
      int has_kick = 0;

      double mass = particles[i].mass;
      double charge = species[particles[i].ispecies].charge;
      if (dq_ewhich >= 0) {
        const double qd = particle->edvec[dq_ewhich][i];
        if (qd != 0.0) charge = qd;
      }
      
      // apply moveperturb() to PKEEP and PINSERT since are computing xnew
      // not to PENTRY,PEXIT since are just re-computing xnew of sender
      // set xnew[2] to linear move for axisymmetry, will be remapped later
      // let pflag = PEXIT persist to check during axisymmetric cell crossing
      // 

      if (pflag == PKEEP) {
        dtremain = dt;
        if (DIM == 1 || DIM == 2)
        {
          if (pusher->pusher_mode == Pusher::PUSHER_HYBRID ||
              pusher->pusher_mode == Pusher::PUSHER_GCA)
            pusher->push_hybrid_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher->push_boris_2d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3)
        {
          if (pusher->pusher_mode == Pusher::PUSHER_HYBRID ||
              pusher->pusher_mode == Pusher::PUSHER_GCA)
            pusher->push_hybrid_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher->push_boris_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PINSERT) {
        dtremain = dt;
        if (DIM == 1 || DIM == 2) {
          if (pusher->pusher_mode == Pusher::PUSHER_HYBRID ||
              pusher->pusher_mode == Pusher::PUSHER_GCA)
            pusher->push_hybrid_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher->push_boris_2d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3) {
          if (pusher->pusher_mode == Pusher::PUSHER_HYBRID ||
              pusher->pusher_mode == Pusher::PUSHER_GCA)
            pusher->push_hybrid_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher->push_boris_3d(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PENTRY) {
        // printf("We are in PENTRY move\n");
        icell = particles[i].icell;
        if (cells[icell].nsplit > 1) {
          if (DIM == 3 && SURF) icell = split3d(icell,x);
          if (DIM < 3 && SURF) icell = split2d(icell,x);
          particles[i].icell = icell;
        }
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      } else if (pflag == PEXIT) {
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      } else if (pflag >= PSURF) {
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
        if (pflag > PSURF) exclude = pflag - PSURF - 1;
      }

      // Apply cross-field diffusion displacement (if active).
      // PKEEP only: fix cross_field_diffusion fills dx_cd at START_OF_STEP
      // for the then-current nlocal particles. PINSERT particles are
      // created mid-step by emission fixes and can live at indices beyond
      // that nlocal, so dx_cd[i] may be past the end of the buffer. They
      // get their first kick next step. Not applied on re-entries either
      // (PENTRY/PEXIT/PSURF), since the kick is already in v and xnew.
      //
      // The kick goes into v as well as xnew so the traced chord
      // xnew = x + dtremain*v stays exact. The axi crossing/remap tests
      // (axi_horizontal_line, axi_line_intersect, axi_remap) parameterize
      // the trajectory by v, and surface collisions see v — with the kick
      // only in xnew, axi tracing misses the diffusive displacement
      // (INTERIOR verdict -> remap outside cell -> naxibad) and walls
      // never see it. The kick is stripped from v again at the first
      // velocity-transforming event (surf/boundary collide, psi reflect,
      // mid-move migration) or at post_move_bookkeeping, whichever comes
      // first.

      // i < cd_nmax: particles created by end-of-step fixes (evaporate,
      // volume/chem) AFTER fix cross_field_diffusion filled the buffer are
      // PKEEP by the next move but can live at indices past the buffer —
      // reading dx_cd[i] there is an out-of-bounds row-pointer deref. They
      // get their first kick next step, same policy as PINSERT.
      if (cd_flag && pflag == PKEEP && i < cd_nmax) {
        vkick0 = dx_cd[i][0] / dtremain;
        vkick1 = dx_cd[i][1] / dtremain;
        v[0] += vkick0;
        v[1] += vkick1;
        xnew[0] += dx_cd[i][0];
        xnew[1] += dx_cd[i][1];
        if (DIM == 3) {
          vkick2 = dx_cd[i][2] / dtremain;
          v[2] += vkick2;
          xnew[2] += dx_cd[i][2];
        }
        has_kick = 1;
        // GCA-state particles: shift the STORED guiding center by the
        // same displacement — otherwise the pusher restores the pre-kick
        // GC next step and the diffusion silently reverts. This is a
        // position operator, not a velocity collision (no-op for
        // Boris-mode/invalid particles).
        {
          double dgc[3] = {dx_cd[i][0], dx_cd[i][1],
                           (DIM == 3) ? dx_cd[i][2] : 0.0};
          pusher->apply_gc_displacement(i, dgc);
        }
      }

      // Psi-based core boundary: check if xnew is inside the core
      // (psi_norm < threshold). If so, reflect the radial component of
      // both displacement and velocity so the particle bounces outward.
      // This check is done here (after Boris push, before surface tracing)
      // so that the reflected trajectory goes through proper surface checks.
      if (psi_reflect_flag && (pflag == PKEEP || pflag == PINSERT)) {
        double Rnew, Znew;
        // psi grid is independent of plasma column; no plasma context here,
        // default 0,0 (psi reflection is a tokamak-only feature anyway).
        OpenEdge::sparta_to_RZ(xnew, domain->dimension, domain->axisymmetric,
                                Rnew, Znew, 0.0, 0.0);

        // Bilinear interpolation of psi_norm at xnew
        double psi_n = 1.0;
        if (psi_nw > 1 && psi_nh > 1 && psi_rz) {
          double Rc = Rnew;
          double Zc = Znew;
          if (Rc < psi_r_grid[0]) Rc = psi_r_grid[0];
          if (Rc > psi_r_grid[psi_nw-1]) Rc = psi_r_grid[psi_nw-1];
          if (Zc < psi_z_grid[0]) Zc = psi_z_grid[0];
          if (Zc > psi_z_grid[psi_nh-1]) Zc = psi_z_grid[psi_nh-1];

          auto bracket_index = [](const double *grid, int n, double x) {
            if (x <= grid[0]) return 0;
            if (x >= grid[n-1]) return n - 2;
            const double *it = std::upper_bound(grid, grid + n, x);
            int idx = static_cast<int>(it - grid) - 1;
            if (idx < 0) idx = 0;
            if (idx > n - 2) idx = n - 2;
            return idx;
          };

          int ii = bracket_index(psi_r_grid, psi_nw, Rc);
          int jj = bracket_index(psi_z_grid, psi_nh, Zc);
          double dr = psi_r_grid[ii+1] - psi_r_grid[ii];
          double dz = psi_z_grid[jj+1] - psi_z_grid[jj];
          if (fabs(dr) > 1e-30 && fabs(dz) > 1e-30) {
            double t = (Rc - psi_r_grid[ii]) / dr;
            double u = (Zc - psi_z_grid[jj]) / dz;
            if (t < 0.0) t = 0.0; if (t > 1.0) t = 1.0;
            if (u < 0.0) u = 0.0; if (u > 1.0) u = 1.0;

            double psi_val = (1-t)*(1-u)*psi_rz[jj*psi_nw+ii]
                           + t*(1-u)*psi_rz[jj*psi_nw+ii+1]
                           + (1-t)*u*psi_rz[(jj+1)*psi_nw+ii]
                           + t*u*psi_rz[(jj+1)*psi_nw+ii+1];
            double dpsi = psi_bry - psi_axis;
            if (fabs(dpsi) > 1e-30) psi_n = (psi_val - psi_axis) / dpsi;
          }
        }

        if (psi_n < psi_reflect_threshold) {
          // Crossing into the core ends the diffusion step: strip the
          // cross-field kick before the move is rejected / v_R flipped,
          // so the kick cannot leak into v past the reflection.
          if (has_kick) {
            v[0] -= vkick0;
            v[1] -= vkick1;
            if (DIM == 3) v[2] -= vkick2;
            has_kick = 0;
          }

          if (psi_reflect_action == 1) {
            // Absorb: mark for deletion and let the normal post-move
            // bookkeeping queue the particle for removal.
            particles[i].flag = PDISCARD;
            icell = particles[i].icell;
            goto post_move_bookkeeping;
          }

          // Reflect: reject the move, particle stays at current position
          xnew[0] = x[0];
          xnew[1] = x[1];
          if (DIM == 3) xnew[2] = x[2];

          // Reverse radial velocity so particle moves outward next step.
          // Slot for v_R depends on domain layout:
          //   2D Cart (legacy):  x=R,y=Z  -> v_R = v[0]
          //   2D axi:            x=Z,y=R  -> v_R = v[1]
          //   3D Cart:           R = sqrt(x^2+y^2) -> project v[0..1] onto R
          if (DIM == 3) {
            double R0 = sqrt(x[0]*x[0] + x[1]*x[1]);
            if (R0 > 1e-10) {
              double cphi = x[0] / R0;
              double sphi = x[1] / R0;
              double vr = particles[i].v[0]*cphi + particles[i].v[1]*sphi;
              double vp = -particles[i].v[0]*sphi + particles[i].v[1]*cphi;
              vr = -vr;
              particles[i].v[0] = vr*cphi - vp*sphi;
              particles[i].v[1] = vr*sphi + vp*cphi;
            }
          } else if (domain->axisymmetric) {
            particles[i].v[1] = -particles[i].v[1];
          } else {
            particles[i].v[0] = -particles[i].v[0];
          }
        }
      }

      // optimized move

      if (OPT) {
        int optmove = 1;

        if (xnew[0] < boxlo[0] || xnew[0] > boxhi[0])
          optmove = 0;

        if (xnew[1] < boxlo[1] || xnew[1] > boxhi[1])
          optmove = 0;

        if (DIM == 3) {
          if (xnew[2] < boxlo[2] || xnew[2] > boxhi[2])
            optmove = 0;
        }

        if (optmove) {
          const int ip = static_cast<int>((xnew[0] - boxlo[0])/dx);
          const int jp = static_cast<int>((xnew[1] - boxlo[1])/dy);
          int kp = 0;
          if (DIM == 3) kp = static_cast<int>((xnew[2] - boxlo[2])/dz);

          int cellIdx = (kp*grid->uny + jp)*grid->unx + ip + 1;

          // particle outside ghost grid halo must use standard move

          Grid::MyHash::iterator hashptr = grid->hash->find(cellIdx);
          if (hashptr != grid->hash->end()) {

            int icell = hashptr->second;

            // reset particle cell and coordinates

            particles[i].icell = icell;
            particles[i].flag = PKEEP;
            x[0] = xnew[0];
            x[1] = xnew[1];
            x[2] = xnew[2];

            if (cells[icell].proc != me) {
              mlist[nmigrate++] = i;
              particles[i].flag = PDONE;
              ncomm_one++;
            }

            // move complete on the fast path: strip the diffusion kick
            if (has_kick) {
              particles[i].v[0] -= vkick0;
              particles[i].v[1] -= vkick1;
              if (DIM == 3) particles[i].v[2] -= vkick2;
            }

            continue;
          }
        }
      }

      particles[i].flag = PKEEP;
      icell = particles[i].icell;
      lo = cells[icell].lo;
      hi = cells[icell].hi;
      neigh = cells[icell].neigh;
      nmask = cells[icell].nmask;
      stuck_iterate = 0;
      ntouch_one++;

      // advect one particle from cell to cell and thru surf collides til done

      //int iterate = 0;

      while (1) {

#ifdef MOVE_DEBUG
        if (DIM == 3) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g %g: xnew %g %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g %g: hi %g %g %g: DTR %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                   icell,cells[icell].id,
                   lo[0],lo[1],lo[2],hi[0],hi[1],hi[2],dtremain);
        }
        if (DIM == 2) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],xnew[0],xnew[1],
                   icell,cells[icell].id,
                   lo[0],lo[1],hi[0],hi[1],dtremain);
        }
        if (DIM == 1) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                   icell,cells[icell].id,
                   lo[0],lo[1],hi[0],hi[1],dtremain);
        }
#endif

        // check if particle crosses any cell face
        // frac = fraction of move completed before hitting cell face
        // this section should be as efficient as possible,
        //   since most particles won't do anything else
        // axisymmetric y cell face crossings:
        //   these faces are curved cylindrical shells
        //   axi_horizontal_line() checks for intersection of
        //     straight-line y,z move with circle in y,z
        //   always check move against lower y face
        //     except when particle starts on face and
        //     PEXIT is set (just received) or particle is moving downward in y
        //   only check move against upper y face
        //     if remapped final y position (rnew) is within cell,
        //     or except when particle starts on face and
        //     PEXIT is set (just received) or particle is moving upward in y
        //   unset pflag so not checked again for this particle

        outface = INTERIOR;
        frac = 1.0;

        if (xnew[0] < lo[0]) {
          frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
          outface = XLO;
        } else if (xnew[0] >= hi[0]) {
          frac = (hi[0]-x[0]) / (xnew[0]-x[0]);
          outface = XHI;
        }

        if (DIM != 1) {
          if (xnew[1] < lo[1]) {
            newfrac = (lo[1]-x[1]) / (xnew[1]-x[1]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = YLO;
            }
          } else if (xnew[1] >= hi[1]) {
            newfrac = (hi[1]-x[1]) / (xnew[1]-x[1]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = YHI;
            }
          }
        }

        if (DIM == 1) {
          if (x[1] == lo[1] && (pflag == PEXIT || v[1] < 0.0)) {
            frac = 0.0;
            outface = YLO;
          } else if (Geometry::
                     axi_horizontal_line(dtremain,x,v,lo[1],itmp,tc,tmp)) {
            newfrac = tc/dtremain;
            if (newfrac < frac) {
              frac = newfrac;
              outface = YLO;
            }
          }

          if (x[1] == hi[1] && (pflag == PEXIT || v[1] > 0.0)) {
            frac = 0.0;
            outface = YHI;
          } else {
            rnew = sqrt(xnew[1]*xnew[1] + xnew[2]*xnew[2]);
            if (rnew >= hi[1]) {
              if (Geometry::
                  axi_horizontal_line(dtremain,x,v,hi[1],itmp,tc,tmp)) {
                newfrac = tc/dtremain;
                if (newfrac < frac) {
                  frac = newfrac;
                  outface = YHI;
                }
              }
            }
          }

          pflag = 0;
        }

        if (DIM == 3) {
          if (xnew[2] < lo[2]) {
            newfrac = (lo[2]-x[2]) / (xnew[2]-x[2]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = ZLO;
            }
          } else if (xnew[2] >= hi[2]) {
            newfrac = (hi[2]-x[2]) / (xnew[2]-x[2]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = ZHI;
            }
          }
        }

        //if (iterate == 10) exit(1);
        //iterate++;

#ifdef MOVE_DEBUG
        if (ntimestep == MOVE_DEBUG_STEP &&
            (MOVE_DEBUG_ID == particles[i].id ||
             (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX))) {
          if (outface != INTERIOR)
            printf("  OUTFACE %d out: %d %d, frac %g\n",
                   outface,grid->neigh_decode(nmask,outface),
                   neigh[outface],frac);
          else
            printf("  INTERIOR %d %d\n",outface,INTERIOR);
        }
#endif

        // START of code specific to surfaces

        if (SURF) {

          // skip surf checks if particle flagged as EXITing this cell
          // then unset pflag so not checked again for this particle

          nsurf = cells[icell].nsurf;
          if (pflag == PEXIT) {
            nsurf = 0;
            pflag = 0;
          }
          nscheck_one += nsurf;

          if (nsurf) {

            // particle crosses cell face, reset xnew exactly on face of cell
            // so surface check occurs only for particle path within grid cell
            // xhold = saved xnew so can restore below if no surf collision

            if (outface != INTERIOR) {
              xhold[0] = xnew[0];
              xhold[1] = xnew[1];
              if (DIM != 2) xhold[2] = xnew[2];

              xnew[0] = x[0] + frac*(xnew[0]-x[0]);
              xnew[1] = x[1] + frac*(xnew[1]-x[1]);
              if (DIM != 2) xnew[2] = x[2] + frac*(xnew[2]-x[2]);

              if (outface == XLO) xnew[0] = lo[0];
              else if (outface == XHI) xnew[0] = hi[0];
              else if (outface == YLO) xnew[1] = lo[1];
              else if (outface == YHI) xnew[1] = hi[1];
              else if (outface == ZLO) xnew[2] = lo[2];
              else if (outface == ZHI) xnew[2] = hi[2];
            }

            // for axisymmetric, dtsurf = time that particle stays in cell
            // used as arg to axi_line_intersect()

            if (DIM == 1) {
              if (outface == INTERIOR) dtsurf = dtremain;
              else dtsurf = dtremain * frac;
            }

            // check for collisions with triangles or lines in cell
            // find 1st surface hit via minparam
            // skip collisions with previous surf, but not for axisymmetric
            // not considered collision if 2 params are tied and one INSIDE surf
            // if collision occurs, perform collision with surface model
            // reset x,v,xnew,dtremain and continue single particle trajectory

            cflag = 0;
            minparam = 2.0;
            csurfs = cells[icell].csurfs;

            for (m = 0; m < nsurf; m++) {
              isurf = csurfs[m];

              if (DIM > 1) {
                if (isurf == exclude) continue;
              }
              if (DIM == 3) {
                tri = &tris[isurf];
                hitflag = Geometry::
                  line_tri_intersect(x,xnew,tri->p1,tri->p2,tri->p3,
                                     tri->norm,xc,param,side);
              }
              if (DIM == 2) {
                line = &lines[isurf];
                hitflag = Geometry::
                  line_line_intersect(x,xnew,line->p1,line->p2,
                                      line->norm,xc,param,side);
              }
              if (DIM == 1) {
                line = &lines[isurf];
                hitflag = Geometry::
                  axi_line_intersect(dtsurf,x,v,outface,lo,hi,line->p1,line->p2,
                                     line->norm,exclude == isurf,
                                     xc,vc,param,side);
              }

#ifdef MOVE_DEBUG
              if (DIM == 3) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE: %d %d %d %d: "
                         "P1 %g %g %g: P2 %g %g %g: "
                         "T1 %g %g %g: T2 %g %g %g: T3 %g %g %g: "
                         "TN %g %g %g: XC %g %g %g: "
                         "Param %g: Side %d\n",
                         MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                         tri->p1[0],tri->p1[1],tri->p1[2],
                         tri->p2[0],tri->p2[1],tri->p2[2],
                         tri->p3[0],tri->p3[1],tri->p3[2],
                         tri->norm[0],tri->norm[1],tri->norm[2],
                         xc[0],xc[1],xc[2],param,side);
              }
              if (DIM == 2) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE: %d %d %d %d: P1 %g %g: P2 %g %g: "
                         "L1 %g %g: L2 %g %g: LN %g %g: XC %g %g: "
                         "Param %g: Side %d\n",
                         MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],xnew[0],xnew[1],
                         line->p1[0],line->p1[1],line->p2[0],line->p2[1],
                         line->norm[0],line->norm[1],
                         xc[0],xc[1],param,side);
              }
              if (DIM == 1) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE %d %ld: %d %d %d %d: P1 %g %g: P2 %g %g: "
                         "L1 %g %g: L2 %g %g: LN %g %g: XC %g %g: "
                         "VC %g %g %g: Param %g: Side %d\n",
                         hitflag,ntimestep,MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],
                         xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                         line->p1[0],line->p1[1],line->p2[0],line->p2[1],
                         line->norm[0],line->norm[1],
                         xc[0],xc[1],vc[0],vc[1],vc[2],param,side);
                double edge1[3],edge2[3],xfinal[3],cross[3];
                MathExtra::sub3(line->p2,line->p1,edge1);
                MathExtra::sub3(x,line->p1,edge2);
                MathExtra::cross3(edge2,edge1,cross);
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    MOVE_DEBUG_ID == particles[i].id)
                  printf("CROSSSTART %g %g %g\n",cross[0],cross[1],cross[2]);
                xfinal[0] = xnew[0];
                xfinal[1] = sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]);
                xfinal[2] = 0.0;
                MathExtra::sub3(xfinal,line->p1,edge2);
                MathExtra::cross3(edge2,edge1,cross);
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    MOVE_DEBUG_ID == particles[i].id)
                  printf("CROSSFINAL %g %g %g\n",cross[0],cross[1],cross[2]);
              }
#endif

              if (hitflag && param < minparam && side == OUTSIDE) {
                cflag = 1;
                minparam = param;
                minside = side;
                minsurf = isurf;
                minxc[0] = xc[0];
                minxc[1] = xc[1];
                if (DIM == 3) minxc[2] = xc[2];
                if (DIM == 1) {
                  minvc[1] = vc[1];
                  minvc[2] = vc[2];
                }
              }

            } // END of for loop over surfs

            // tri/line = surf that particle hit first

            if (cflag) {
              if (DIM == 3) tri = &tris[minsurf];
              if (DIM != 3) line = &lines[minsurf];

              // set x to collision point
              // if axisymmetric, set v to remapped velocity at collision pt

              x[0] = minxc[0];
              x[1] = minxc[1];
              if (DIM == 3) x[2] = minxc[2];
              if (DIM == 1) {
                v[1] = minvc[1];
                v[2] = minvc[2];
              }

              // The diffusion step ends at the wall: strip the cross-field
              // kick from v BEFORE the sheath kick and the collision model,
              // so PWI physics (incident energy/angle, sputtering) and the
              // reflected velocity never see the phantom kick velocity
              // (dx_cd/dt can dwarf the thermal speed). The kick carried
              // the particle to the wall — that diffusive flux is the
              // physical part; the remaining chord after the bounce is
              // traced with the clean velocity.
              //
              // The chord reached this surf only because of the kick, so
              // the stripped velocity may no longer point at it. Reflecting
              // a non-incident velocity would aim it THROUGH the wall, and
              // the exclude guard below would let it escape the domain
              // (leak). Treat that case as a graze: no collision physics,
              // no tallies; the particle continues from the wall point
              // with its own velocity, which already carries it back into
              // the domain.
              if (has_kick) {
                v[0] -= vkick0;
                v[1] -= vkick1;
                if (DIM == 3) v[2] -= vkick2;
                has_kick = 0;

                const double *nrm = (DIM == 3) ? tri->norm : line->norm;
                if (v[0]*nrm[0] + v[1]*nrm[1] + v[2]*nrm[2] >= 0.0) {
                  dtremain *= 1.0 - minparam*frac;
                  if (minparam == 0.0) stuck_iterate++;
                  else stuck_iterate = 0;
                  if (stuck_iterate >= MAXSTUCK) {
                    particles[i].flag = PDISCARD;
                    nstuck++;
                    break;
                  }
                  xnew[0] = x[0] + dtremain*v[0];
                  xnew[1] = x[1] + dtremain*v[1];
                  if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
                  exclude = minsurf;
                  continue;
                }
              }

              // perform surface collision using surface collision model
              // surface chemistry may destroy particle or create new one
              // must update particle's icell to current icell so that
              //   if jpart is created, it will be added to correct cell
              // if jpart, add new particle to this iteration via pstop++
              // tally surface collision stats if requested using iorig

              ipart = &particles[i];
              ipart->icell = icell;
              dtremain *= 1.0 - minparam*frac;

              // Record the first explicit surface hit for this particle ID.
              if (hit_flag_index >= 0 && hit_surf_index >= 0) {
                int *hit_flag_vec = particle->eivec[particle->ewhich[hit_flag_index]];
                int *hit_surf_vec = particle->eivec[particle->ewhich[hit_surf_index]];
                if (hit_flag_vec && hit_surf_vec && hit_flag_vec[i] == 0) {
                  hit_flag_vec[i] = 1;
                  surfint sid = (DIM == 3) ? tris[minsurf].id : lines[minsurf].id;
                  hit_surf_vec[i] = static_cast<int>(sid);
                }
              }

              // A1 axi: a GCA particle reaches the wall on its GC chord —
              // swap in the flux-weighted physical gyro velocity at first
              // passage before the sheath kick and collide (3D samples
              // pre-step in the pusher; the axi retrace contract
              // xnew = x + dtremain*v forbids carrying it in v there)
              if (DIM != 3 && pusher && pusher->pusher_gc_wall_flux)
                pusher->materialize_impact_velocity(i, icell, minsurf, v);

              // --- Sheath kick: apply sheath energy as velocity boost at wall ---
              // Inbound sheath impact-energy boost: sets the wall impact
              // energy for sputtering. Applied in both kick mode and
              // boundary mode (boundary mode adds the outbound barrier in
              // the pusher; this is its inbound half).
              // Material walls only: periodic (toroidal) caps are virtual
              // boundaries with no sheath, and kicking on every transit
              // pumps the cap-normal velocity without bound.
              const int kick_isc = (DIM == 3) ? tri->isc : line->isc;
              bool kick_material_wall = (kick_isc < 0) ||
                  (strcmp(surf->sc[kick_isc]->style, "toroidal") != 0);
              // the sheath lives ONLY on surfaces in the sheath geometry
              // group (the actually-intersected face decides): an escape
              // collector or core boundary carries no wall sheath
              if (kick_material_wall && sheath_geom_cidx >= 0) {
                auto *csg_k = dynamic_cast<ComputeNearestSurfGrid *>(
                    modify->compute[sheath_geom_cidx]);
                if (csg_k) {
                  const int smask_k = (DIM == 3) ? tri->mask : line->mask;
                  if (!(smask_k & csg_k->sgroupbit)) kick_material_wall = false;
                }
              }
              if (kick_material_wall &&
                  (sheath_kick || sheath_boundary) && sheath_flag &&
                  sheath_geom_cidx >= 0 &&
                  (pusher->pusher_plasma_cidx >= 0 || pusher->pusher_plasma_fidx >= 0)) {
                // Get surface normal (outward, toward plasma)
                const double *snorm = (DIM == 3) ? tri->norm : line->norm;

                // Plasma conditions at particle position (point query)
                ComputePlasmaFields *cp = nullptr;
                FixBackground *pd = nullptr;
                if (pusher->pusher_plasma_cidx >= 0) {
                  Compute *cp_base = modify->compute[pusher->pusher_plasma_cidx];
                  cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
                } else if (pusher->pusher_plasma_fidx >= 0) {
                  pd = dynamic_cast<FixBackground *>(modify->fix[pusher->pusher_plasma_fidx]);
                }
                if (cp || pd) {
                  // unified evaluator first (CM base + RF waveform at the
                  // exact collision time); inline floating-potential
                  // fallback for non-cached plasma sources
                  const double phi_uni =
                      pusher->sheath_phi_wall(minsurf, dt - dtremain);
                  PlasmaFileParams sk_pf = cp ? cp->query_plasma_at_point(x)
                                              : query_plasma_from_fix(pd, x, DIM == 2 ? 2 : 3, domain->axisymmetric, icell);
                  const double sk_te = sk_pf.temp_e;
                  const double sk_ti = sk_pf.temp_i;
                  if (phi_uni >= 0.0 || sk_te > 0.0) {
                    constexpr double QE_kick = 1.602176634e-19;
                    constexpr double ME_kick = 9.1093837015e-31;
                    constexpr double AMU_kick = 1.66053906660e-27;
                    constexpr double PI_kick = 3.14159265358979323846;

                    const double mD_kg = sheath_mD_amu * AMU_kick;

                    // Floating potential: phi = 0.5*ln(mD/(2*pi*me)/(1+Ti/Te)) * Te (eV)
                    const double ti_ratio =
                      (sk_te > 0.0 && sk_ti > 0.0) ? (sk_ti / sk_te) : 0.0;
                    const double phi_float_mult = (sk_te > 0.0)
                      ? 0.5 * std::log(mD_kg / (2.0 * PI_kick * ME_kick) / (1.0 + ti_ratio))
                      : 0.0;
                    const double phi_eV = (phi_uni >= 0.0)
                      ? phi_uni
                      : std::max(phi_float_mult, 0.0) * sk_te;

                    // Particle charge and mass (from species table, not particle struct)
                    const int isp = particles[i].ispecies;
                    const double Z = std::abs(particle->species[isp].charge);
                    const double pmass = particle->species[isp].mass;

                    if (Z > 0.0 && pmass > 0.0 && phi_eV > 0.0) {
                      // Normal component of velocity (toward wall = negative v·n)
                      const double vdotn = v[0]*snorm[0] + v[1]*snorm[1] + v[2]*snorm[2];
                      // v·n < 0 means particle moving toward wall (against outward normal)
                      const double vn_toward = -vdotn;  // positive = toward wall

                      // New normal speed: sqrt(vn^2 + 2*Z*e*phi/m)
                      const double dE_J = Z * QE_kick * phi_eV;
                      const double vn_new = std::sqrt(vn_toward * vn_toward + 2.0 * dE_J / pmass);
                      const double dv = vn_new - vn_toward;  // always >= 0

                      // Add kick toward wall (against outward normal)
                      v[0] -= dv * snorm[0];
                      v[1] -= dv * snorm[1];
                      v[2] -= dv * snorm[2];
                    }
                  }
                }
              }

              if (nsurf_tally)
                memcpy(&iorig,&particles[i],sizeof(Particle::OnePart));

              const int nlocal_precollide = particle->nlocal;

              if (DIM == 3)
                jpart = surf->sc[tri->isc]->
                  collide(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction);
              if (DIM != 3)
                jpart = surf->sc[line->isc]->
                  collide(ipart,dtremain,minsurf,line->norm,line->isr,reaction);

              // re-fetch base unconditionally: a surf reaction may have
              // added particles (e.g. additive sputter emission) and
              // realloced particle->particles even when jpart is NULL
              particles = particle->particles;
              x = particles[i].x;
              v = particles[i].v;

              if (jpart) {
                jpart->flag = PSURF + 1 + minsurf;
                jpart->dtremain = dtremain;
                jpart->weight = particles[i].weight;
                pstop++;
              }

              // additive reaction products (sputter emission) must fly
              // like jpart: born mid-move (possibly in a ghost cell), they
              // need the move's completion paths to settle ownership.
              // Never append them to mlist directly — compress_migrate
              // requires ascending indices.
              for (int inew = nlocal_precollide; inew < particle->nlocal;
                   inew++) {
                if (particles[inew].flag != PKEEP) continue;  // skips jpart
                particles[inew].flag = PSURF + 1 + minsurf;
                particles[inew].dtremain = dtremain;
                particles[inew].weight = particles[i].weight;
                pstop++;
              }

              if (nsurf_tally) {
                // incident macroparticle weight for pweight-aware surf
                // computes (edvec refreshed post-collide reallocation).
                tally_pweight = (pw_ewhich >= 0)
                  ? particle->edvec[pw_ewhich][i] : 1.0;
                for (m = 0; m < nsurf_tally; m++)
                      slist_active[m]->surf_tally(dtremain,minsurf,icell,reaction,
                                                                    &iorig,ipart,jpart);

              // ---- DEBUG: post-collision wall-side audit ----------------
              // Flag events where surf_collide / surf_react leaves the
              // particle behind the wall normal (likely cause of visible
              // leaks in axi runs with full chemistry). Rate-limited;
              // remove once the underlying surface model is fixed.
              {
                static int wall_leak_count = 0;
                constexpr int wall_leak_max  = 200;
                constexpr double dxn_thresh  = -1.0e-10;  // 0.1 nm behind wall
                if (ipart && wall_leak_count < wall_leak_max) {
                  const double *snorm = (DIM == 3) ? tri->norm : line->norm;
                  double sref0, sref1, sref2;
                  if (DIM == 3) {
                    sref0 = (tri->p1[0]+tri->p2[0]+tri->p3[0])/3.0;
                    sref1 = (tri->p1[1]+tri->p2[1]+tri->p3[1])/3.0;
                    sref2 = (tri->p1[2]+tri->p2[2]+tri->p3[2])/3.0;
                  } else {
                    sref0 = 0.5*(line->p1[0]+line->p2[0]);
                    sref1 = 0.5*(line->p1[1]+line->p2[1]);
                    sref2 = 0.0;
                  }
                  const double dxn =
                    (x[0]-sref0)*snorm[0] + (x[1]-sref1)*snorm[1] +
                    (DIM == 3 ? (x[2]-sref2)*snorm[2] : 0.0);
                  const double vdotn =
                    v[0]*snorm[0] + v[1]*snorm[1] +
                    (DIM == 3 ? v[2]*snorm[2] : 0.0);
                  if (dxn < dxn_thresh) {
                    FILE *fp = screen ? screen : stdout;
                    fprintf(fp,
                      "[wall-leak] step=" BIGINT_FORMAT " proc=%d pid=%d "
                      "surf=%d dx.n=%.3e v.n=%.3e jpart=%d\n",
                      ntimestep, me, particles[i].id, minsurf,
                      dxn, vdotn, jpart ? 1 : 0);
                    wall_leak_count++;
                    if (wall_leak_count == wall_leak_max)
                      fprintf(fp, "[wall-leak] suppressing further events "
                                  "on proc %d (cap=%d)\n", me, wall_leak_max);
                  }
                }
              }
              // ---- END DEBUG -------------------------------------------
              }

              // stuck_iterate = consecutive iterations particle is immobile

              if (minparam == 0.0) stuck_iterate++;
              else stuck_iterate = 0;

              // reset post-bounce xnew

              xnew[0] = x[0] + dtremain*v[0];
              xnew[1] = x[1] + dtremain*v[1];
              if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];

              // check if surf_collide teleported particle outside current cell
              // (occurs with toroidal periodic boundary collision model)

              if (DIM == 3 &&
                  (x[0] < lo[0] || x[0] > hi[0] ||
                   x[1] < lo[1] || x[1] > hi[1] ||
                   x[2] < lo[2] || x[2] > hi[2])) {
                int newcell = grid->id_find_child(0,0,
                                  domain->boxlo,domain->boxhi,x);
                if (newcell >= 0) {
                  if (cells[newcell].proc == me) {
                    if (DIM == 3 && SURF) {
                      if (cells[newcell].nsplit > 1 &&
                          cells[newcell].nsurf >= 0)
                        newcell = split3d(newcell,x);
                    }
                    icell = newcell;
                    lo = cells[icell].lo;
                    hi = cells[icell].hi;
                    neigh = cells[icell].neigh;
                    nmask = cells[icell].nmask;
                    exclude = -1;
                    nscollide_one++;
                    continue;
                  } else {
                    icell = newcell;
                    particles[i].icell = icell;
                    particles[i].flag = PEXIT;
                    particles[i].dtremain = dtremain;
                    entryexit = 1;
                    nscollide_one++;
                    break;
                  }
                } else {
                  particles[i].flag = PDISCARD;
                  nscollide_one++;
                  break;
                }
              }

              exclude = minsurf;
              nscollide_one++;

#ifdef MOVE_DEBUG
              if (DIM == 3) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g %g: %g %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                         minparam,frac,dtremain);
              }
              if (DIM == 2) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g: %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],xnew[0],xnew[1],
                         minparam,frac,dtremain);
              }
              if (DIM == 1) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g: %g %g: vel %g %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],
                         xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                         v[0],v[1],v[2],
                         minparam,frac,dtremain);
              }
#endif

              // if ipart = NULL, particle discarded due to surface chem
              // else if particle not stuck, continue advection while loop
              // if stuck, mark for DISCARD, and drop out of SURF code

              if (ipart == NULL) particles[i].flag = PDISCARD;
              else if (stuck_iterate < MAXSTUCK) continue;
              else {
                particles[i].flag = PDISCARD;
                nstuck++;
              }

            } // END of cflag if section that performed collision

            // no collision, so restore saved xnew if changed it above

            if (outface != INTERIOR) {
              xnew[0] = xhold[0];
              xnew[1] = xhold[1];
              if (DIM != 2) xnew[2] = xhold[2];
            }

          } // END of if test for any surfs in this cell
        } // END of code specific to surfaces

        // break from advection loop if discarding particle

        if (particles[i].flag == PDISCARD) break;

        // no cell crossing and no surface collision
        // set final particle position to xnew, then break from advection loop
        // for axisymmetry, must first remap linear xnew and v
        // for axisymmetry, check if final particle position is within cell
        //   can be rare epsilon round-off cases where particle ends up outside
        //     of final cell curved surf when move logic thinks it is inside
        //   example is when Geom::axi_horizontal_line() says no crossing of cell edge
        //     but axi_remap() puts particle outside the cell
        //   in this case, just DISCARD particle and tally it to naxibad
        // if migrating to another proc,
        //   flag as PDONE so new proc won't move it more on this step

        if (outface == INTERIOR) {
          if (DIM == 1) axi_remap(xnew,v,
              (phi_track && phi_custom >= 0)
                ? &particle->edvec[particle->ewhich[phi_custom]][i] : NULL);
          x[0] = xnew[0];
          x[1] = xnew[1];
          if (DIM == 3) x[2] = xnew[2];
          if (DIM == 1) {
            if (x[1] < lo[1] || x[1] > hi[1]) {
              // Particle ended outside its current cell after axi_remap.
              // Try to rehome via id_find_child instead of silently
              // discarding: most of these are charged ions whose Boris
              // gyromotion crossed a cell boundary the linear cell-cross
              // check missed.
              int newcell = grid->id_find_child(0,0,
                                domain->boxlo,domain->boxhi,x);
              int rehomed = 0;
              if (newcell >= 0) {
                if (SURF && cells[newcell].nsplit > 1 &&
                    cells[newcell].nsurf >= 0) {
                  newcell = split2d(newcell,x);
                }
                // Accept only if the new cell is on the FLUID side of the
                // wall (volume > 0); a vacuum cell means the particle
                // slipped through wall.surf during a partial Boris step.
                if (newcell >= 0 && cinfo[newcell].volume > 0.0) {
                  icell = newcell;
                  if (cells[newcell].proc != me)
                    particles[i].flag = PDONE;
                  rehomed = 1;
                }
              }
              // Genuine escape: reflect specularly off the exited cell
              // boundary (approximates the wall hit the linear check
              // missed; conserves mass). naxibad now counts recoveries.
              if (!rehomed) {
                if (x[1] < lo[1]) {
                  x[1] = lo[1];
                  if (v[1] < 0.0) v[1] = -v[1];
                } else if (x[1] > hi[1]) {
                  x[1] = hi[1];
                  if (v[1] > 0.0) v[1] = -v[1];
                }
                naxibad++;
              }
              break;
            }
          }
          if (cells[icell].proc != me) particles[i].flag = PDONE;
          break;
        }

        // particle crosses cell face
        // decrement dtremain in case particle is passed to another proc
        // for axisymmetry, must then remap linear x and v
        // reset particle x to be exactly on cell face
        // for axisymmetry, must reset xnew for next iteration since v changed

        dtremain *= 1.0-frac;
        exclude = -1;

        x[0] += frac * (xnew[0]-x[0]);
        x[1] += frac * (xnew[1]-x[1]);
        if (DIM != 2) x[2] += frac * (xnew[2]-x[2]);
        if (DIM == 1) axi_remap(x,v,
            (phi_track && phi_custom >= 0)
              ? &particle->edvec[particle->ewhich[phi_custom]][i] : NULL);

        if (outface == XLO) x[0] = lo[0];
        else if (outface == XHI) x[0] = hi[0];
        else if (outface == YLO) x[1] = lo[1];
        else if (outface == YHI) x[1] = hi[1];
        else if (outface == ZLO) x[2] = lo[2];
        else if (outface == ZHI) x[2] = hi[2];

        if (DIM == 1) {
          xnew[0] = x[0] + dtremain*v[0];
          xnew[1] = x[1] + dtremain*v[1];
          xnew[2] = x[2] + dtremain*v[2];
        }

        // nflag = type of neighbor cell: child, parent, unknown, boundary
        // if parent, use id_find_child to identify child cell
        //   result can be -1 for unknown cell, occurs when:
        //   (a) particle hits face of ghost child cell
        //   (b) the ghost cell extends beyond ghost halo
        //   (c) cell on other side of face is a parent
        //   (d) its child, which the particle is in, is entirely beyond my halo
        // if new cell is child and surfs exist, check if a split cell

        nflag = grid->neigh_decode(nmask,outface);
        icell_original = icell;

        if (nflag == NCHILD) {
          icell = neigh[outface];
          if (DIM == 3 && SURF) {
            if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
              icell = split3d(icell,x);
          }
          if (DIM < 3 && SURF) {
            if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
              icell = split2d(icell,x);
          }
        } else if (nflag == NPARENT) {
          pcell = &pcells[neigh[outface]];
          icell = grid->id_find_child(pcell->id,cells[icell].level,
                                      pcell->lo,pcell->hi,x);
          if (icell >= 0) {
            if (DIM == 3 && SURF) {
              if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                icell = split3d(icell,x);
            }
            if (DIM < 3 && SURF) {
              if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                icell = split2d(icell,x);
            }
          }
        } else if (nflag == NUNKNOWN) icell = -1;

        // neighbor cell is global boundary
        // tally boundary stats if requested using iorig
        // collide() updates x,v,xnew as needed due to boundary interaction
        //   may also update dtremain (piston BC)
        // for axisymmetric, must recalculate xnew since v may have changed
        // surface chemistry may destroy particle or create new one
        // if jpart, add new particle to this iteration via pstop++
        // OUTFLOW: exit with particle flag = PDISCARD
        // PERIODIC: new cell via same logic as above for child/parent/unknown
        // OTHER: reflected particle stays in same grid cell

        else {
          ipart = &particles[i];

          // Diffusion step ends at a domain boundary too: strip the
          // cross-field kick before the boundary model transforms or
          // tallies v (same rationale as the surface-collision strip).
          if (has_kick) {
            v[0] -= vkick0;
            v[1] -= vkick1;
            if (DIM == 3) v[2] -= vkick2;
            has_kick = 0;
          }

          if (nboundary_tally)
            memcpy(&iorig,&particles[i],sizeof(Particle::OnePart));

          bflag = domain->collide(ipart,outface,icell,xnew,dtremain,
                                  jpart,reaction);

          if (jpart) {
            particles = particle->particles;
            x = particles[i].x;
            v = particles[i].v;
          }

          if (nboundary_tally)
            for (m = 0; m < nboundary_tally; m++)
              blist_active[m]->
                boundary_tally(dtremain,outface,bflag,reaction,&iorig,ipart,jpart);

          if (DIM == 1) {
            xnew[0] = x[0] + dtremain*v[0];
            xnew[1] = x[1] + dtremain*v[1];
            xnew[2] = x[2] + dtremain*v[2];
          }

          if (bflag == OUTFLOW) {
            particles[i].flag = PDISCARD;
            nexit_one++;
            break;

          } else if (bflag == PERIODIC) {
            if (nflag == NPBCHILD) {
              icell = neigh[outface];
              if (DIM == 3 && SURF) {
                if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                  icell = split3d(icell,x);
              }
              if (DIM < 3 && SURF) {
                if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                  icell = split2d(icell,x);
              }
            } else if (nflag == NPBPARENT) {
              pcell = &pcells[neigh[outface]];
              icell = grid->id_find_child(pcell->id,cells[icell].level,
                                          pcell->lo,pcell->hi,x);
              if (icell >= 0) {
                if (DIM == 3 && SURF) {
                  if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                    icell = split3d(icell,x);
                }
                if (DIM < 3 && SURF) {
                  if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                    icell = split2d(icell,x);
                }
              } else domain->uncollide(outface,x);
            } else if (nflag == NPBUNKNOWN) {
              icell = -1;
              domain->uncollide(outface,x);
            }

          } else if (bflag == SURFACE) {
            if (ipart == NULL) {
              particles[i].flag = PDISCARD;
              break;
            } else if (jpart) {
              jpart->flag = PSURF;
              jpart->dtremain = dtremain;
              jpart->weight = particles[i].weight;
              pstop++;
            }
            nboundary_one++;
            ntouch_one--;    // decrement here since will increment below

          } else {
            nboundary_one++;
            ntouch_one--;    // decrement here since will increment below
          }
        }

        // neighbor cell is unknown
        // reset icell to original icell which must be a ghost cell
        // exit with particle flag = PEXIT, so receiver can identify neighbor

        if (icell < 0) {
          icell = icell_original;
          particles[i].flag = PEXIT;
          particles[i].dtremain = dtremain;
          entryexit = 1;
          // strip the cross-field kick before migrating mid-move: the
          // receiver rebuilds xnew = x + dtremain*v and cannot strip
          // later, so the kick must not leave this rank inside v. Only
          // the untraversed remainder of the kick displacement is lost.
          if (has_kick) {
            v[0] -= vkick0;
            v[1] -= vkick1;
            if (DIM == 3) v[2] -= vkick2;
            has_kick = 0;
          }
          break;
        }

        // if nsurf < 0, new cell is EMPTY ghost
        // exit with particle flag = PENTRY, so receiver can continue move

        if (cells[icell].nsurf < 0) {
          particles[i].flag = PENTRY;
          particles[i].dtremain = dtremain;
          entryexit = 1;
          // same mid-move migration strip as the PEXIT case above
          if (has_kick) {
            v[0] -= vkick0;
            v[1] -= vkick1;
            if (DIM == 3) v[2] -= vkick2;
            has_kick = 0;
          }
          break;
        }

        // move particle into new grid cell for next stage of move

        lo = cells[icell].lo;
        hi = cells[icell].hi;
        neigh = cells[icell].neigh;
        nmask = cells[icell].nmask;
        ntouch_one++;
      }

      // END of while loop over advection of single particle

#ifdef MOVE_DEBUG
      if (ntimestep == MOVE_DEBUG_STEP &&
          (MOVE_DEBUG_ID == particles[i].id ||
           (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
        printf("MOVE DONE %d %d %d: %g %g %g: DTR %g\n",
               MOVE_DEBUG_INDEX,particles[i].flag,icell,
               x[0],x[1],x[2],dtremain);
#endif

      // move is complete, or as much as can be done on this proc
      // update particle's grid cell
      // if particle flag set, add particle to migrate list
      // if discarding, migration will delete particle

post_move_bookkeeping:

      // Strip the cross-field diffusion kick from v now that this step's
      // move is complete: the kick models a position random walk, not
      // heating, so it must not persist in the velocity. has_kick is only
      // still set here if the particle saw no surf/boundary/psi event and
      // no mid-move migration (those paths strip earlier), so this
      // subtraction is exact up to the tiny axi_remap rotation of v
      // accumulated within the step.
      if (has_kick &&
          (particles[i].flag == PKEEP || particles[i].flag == PDONE)) {
        particles[i].v[0] -= vkick0;
        particles[i].v[1] -= vkick1;
        if (DIM == 3) particles[i].v[2] -= vkick2;
      }

      particles[i].icell = icell;

      // safety net: a particle may not end the move in a foreign-owned
      // cell; flag stragglers PDONE (mirrors the INTERIOR owner check)
      if (particles[i].flag == PKEEP && icell >= 0 &&
          cells[icell].proc != me)
        particles[i].flag = PDONE;

      if (particles[i].flag != PKEEP) {
        mlist[nmigrate++] = i;
        if (particles[i].flag != PDISCARD) {
          if (cells[icell].proc == me) {
            char str[128];
            sprintf(str,
                    "Particle %d on proc %d being sent to self "
                    "on step " BIGINT_FORMAT,
                    i,me,update->ntimestep);
            error->one(FLERR,str);
          }
          ncomm_one++;
        }
      }
    }

    // END of pstart/pstop loop advecting all particles

    // if gridcut >= 0.0, check if another iteration of move is required
    // only the case if some particle flag = PENTRY/PEXIT
    //   in which case perform particle migration
    // if not, move is done and final particle comm will occur in run()
    // if iterating, reset pstart/pstop and extend migration list if necessary

    if (grid->cutoff < 0.0) break;

    timer->stamp(TIME_MOVE);
    MPI_Allreduce(&entryexit,&any_entryexit,1,MPI_INT,MPI_MAX,world);
    timer->stamp();

    if (any_entryexit) {
      timer->stamp(TIME_MOVE);
      pstart = comm->migrate_particles(nmigrate,mlist);
      timer->stamp(TIME_COMM);
      pstop = particle->nlocal;
      if (pstop-pstart > maxmigrate) {
        maxmigrate = pstop-pstart;
        memory->destroy(mlist);
        memory->create(mlist,maxmigrate,"particle:mlist");
      }
    } else break;

    // END of single move/migrate iteration

  }

  // END of all move/migrate iterations

  // Spatial-sheath diagnostic: reduce the per-step engagement counters and
  // field magnitudes across ranks and print on rank 0, so a run can confirm
  // the sheath E is non-zero and that particles actually enter its band.
  // Gated on the pusher `dump yes` flag + cadence to stay quiet by default.
  if (sheath_flag && !sheath_kick && pusher->pusher_dump_flag &&
      (ntimestep % pusher->pusher_dump_every == 0)) {
    long loc[4] = {pusher->sheath_diag_nactive, pusher->sheath_diag_nengage,
                   pusher->sheath_diag_nreflect, pusher->sheath_diag_nescape};
    long glob[4] = {0, 0, 0, 0};
    double emax_loc = pusher->sheath_diag_emax, emax_glob = 0.0;
    double esum_loc = pusher->sheath_diag_esum, esum_glob = 0.0;
    MPI_Reduce(loc, glob, 4, MPI_LONG, MPI_SUM, 0, world);
    MPI_Reduce(&emax_loc, &emax_glob, 1, MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(&esum_loc, &esum_glob, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    if (comm->me == 0) {
      const double emean = glob[1] > 0 ? esum_glob / glob[1] : 0.0;
      FILE *fp = screen ? screen : logfile;
      if (fp) {
        if (sheath_boundary)
          // Boundary mode: the spatial E is off by design; the sheath acts
          // through the potential-barrier impulse (reflect = prompt redep).
          fprintf(fp, "  sheath step " BIGINT_FORMAT " [boundary]: "
                  "near-wall=%ld  barrier reflect=%ld escape=%ld\n",
                  ntimestep, glob[0], glob[2], glob[3]);
        else
          // Spatial mode: report the per-subcycle E-field seen by particles.
          fprintf(fp, "  sheath step " BIGINT_FORMAT " [spatial]: "
                  "near-wall=%ld engaged=%ld turnrefl=%ld "
                  "|E_sheath| mean=%.3e max=%.3e V/m\n",
                  ntimestep, glob[0], glob[1], glob[2], emean, emax_glob);
      }
    }

  }

  particle->sorted = 0;

  // accumulate running totals

  niterate_running += niterate;
  nmove_running += particle->nlocal;
  ntouch_running += ntouch_one;
  ncomm_running += ncomm_one;
  nboundary_running += nboundary_one;
  nexit_running += nexit_one;
  nscheck_running += nscheck_one;
  nscollide_running += nscollide_one;
  surf->nreact_running += surf->nreact_one;
}

/* ----------------------------------------------------------------------
   calculate motion perturbation for a single particle I
     due to external per particle field
   array in fix[ifieldfix] stores per particle perturbations for x and v
------------------------------------------------------------------------- */

void Update::field_per_particle(int i, int icell, double dt, double *x, double *v)
{
  double dtsq = 0.5*dt*dt;
  double **array = modify->fix[ifieldfix]->array_particle;

  int icol = 0;
  if (field_active[0]) {
    x[0] += dtsq*array[i][icol];
    v[0] += dt*array[i][icol];
    icol++;
  }
  if (field_active[1]) {
    x[1] += dtsq*array[i][icol];
    v[1] += dt*array[i][icol];
    icol++;
  }
  if (field_active[2]) {
    x[2] += dtsq*array[i][icol];
    v[2] += dt*array[i][icol];
    icol++;
  }
};

/* ----------------------------------------------------------------------
   calculate motion perturbation for a single particle I in grid cell Icell
     due to external per grid cell field
   array in fix[ifieldfix] stores per grid cell perturbations for x and v
------------------------------------------------------------------------- */

void Update::field_per_grid(int i, int icell, double dt, double *x, double *v)
{
  double dtsq = 0.5*dt*dt;
  double **array = modify->fix[ifieldfix]->array_grid;

  int icol = 0;
  if (field_active[0]) {
    x[0] += dtsq*array[icell][icol];
    v[0] += dt*array[icell][icol];
    icol++;
  }
  if (field_active[1]) {
    x[1] += dtsq*array[icell][icol];
    v[1] += dt*array[icell][icol];
    icol++;
  }
  if (field_active[2]) {
    x[2] += dtsq*array[icell][icol];
    v[2] += dt*array[icell][icol];
    icol++;
  }
};


/* ----------------------------------------------------------------------
   particle is entering split parent icell at x
   determine which split child cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

int Update::split3d(int icell, double *x)
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Tri *tri;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Surf::Tri *tris = surf->tris;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider tris that are mapped via csplits to a split cell
  //   unmapped tris only touch cell surf at xnew
  //   another mapped tri should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particles starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = cells[icell].nsurf;
  surfint *csurfs = cells[icell].csurfs;
  int isplit = cells[icell].isplit;
  int *csplits = sinfo[isplit].csplits;
  double *xnew = sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;

  for (m = 0; m < nsurf; m++) {
    if (csplits[m] < 0) continue;
    isurf = csurfs[m];
    tri = &tris[isurf];
    hitflag = Geometry::
      line_tri_intersect(x,xnew,tri->p1,tri->p2,tri->p3,
                         tri->norm,xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  if (!cflag) return sinfo[isplit].csubs[sinfo[isplit].xsub];
  int index = csplits[minsurfindex];
  return sinfo[isplit].csubs[index];
}

/* ----------------------------------------------------------------------
   particle is entering split ICELL at X
   determine which split sub-cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

int Update::split2d(int icell, double *x)
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Line *line;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Surf::Line *lines = surf->lines;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider lines that are mapped via csplits to a split cell
  //   unmapped lines only touch cell surf at xnew
  //   another mapped line should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particle starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = cells[icell].nsurf;
  surfint *csurfs = cells[icell].csurfs;
  int isplit = cells[icell].isplit;
  int *csplits = sinfo[isplit].csplits;
  double *xnew = sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;
  for (m = 0; m < nsurf; m++) {
    if (csplits[m] < 0) continue;
    isurf = csurfs[m];
    line = &lines[isurf];
    hitflag = Geometry::
      line_line_intersect(x,xnew,line->p1,line->p2,line->norm,xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  if (!cflag) return sinfo[isplit].csubs[sinfo[isplit].xsub];
  int index = csplits[minsurfindex];
  return sinfo[isplit].csubs[index];
}

/* ----------------------------------------------------------------------
   check if any surface collision or reaction models are defined
   return 1 if there are any, 0 if not
------------------------------------------------------------------------- */

int Update::collide_react_setup()
{
  nsc = surf->nsc;
  sc = surf->sc;
  nsr = surf->nsr;
  sr = surf->sr;

  if (nsc || nsr) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   zero counters for tallying surface collisions/reactions
   done at start of each timestep
   done within individual SurfCollide and SurfReact instances
------------------------------------------------------------------------- */

void Update::collide_react_reset()
{
  for (int i = 0; i < nsc; i++) sc[i]->tally_reset();
  for (int i = 0; i < nsr; i++) sr[i]->tally_reset();
}

/* ----------------------------------------------------------------------
   update cumulative counters for tallying surface collisions/reactions
   done at end of each timestep
   done within individual SurfCollide and SurfReact instances
------------------------------------------------------------------------- */

void Update::collide_react_update()
{
  for (int i = 0; i < nsc; i++) sc[i]->tally_update();
  for (int i = 0; i < nsr; i++) sr[i]->tally_update();
}

/* ----------------------------------------------------------------------
   setup lists of all computes that are potentially called when events occur
     gas/gas collisions or reactions
     gas/surf collisions or reactions
     gas/boundary collisions or reactions
   return 1 if there are any, 0 if not
------------------------------------------------------------------------- */

int Update::tally_setup()
{
  delete [] glist_compute;
  delete [] slist_compute;
  delete [] blist_compute;

  delete [] glist_active;
  delete [] slist_active;
  delete [] blist_active;

  glist_compute = slist_compute = blist_compute = NULL;

  nglist_compute = nslist_compute = nblist_compute = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->gas_tally_flag) nglist_compute++;
    if (modify->compute[i]->surf_tally_flag) nslist_compute++;
    if (modify->compute[i]->boundary_tally_flag) nblist_compute++;
  }

  if (nglist_compute) glist_compute = new Compute*[nglist_compute];
  if (nslist_compute) slist_compute = new Compute*[nslist_compute];
  if (nblist_compute) blist_compute = new Compute*[nblist_compute];

  if (nglist_compute) glist_active = new Compute*[nglist_compute];
  if (nslist_compute) slist_active = new Compute*[nslist_compute];
  if (nblist_compute) blist_active = new Compute*[nblist_compute];

  nglist_compute = nslist_compute = nblist_compute = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->gas_tally_flag)
      glist_compute[nglist_compute++] = modify->compute[i];
    if (modify->compute[i]->surf_tally_flag)
      slist_compute[nslist_compute++] = modify->compute[i];
    if (modify->compute[i]->boundary_tally_flag)
      blist_compute[nblist_compute++] = modify->compute[i];
  }

  if (nglist_compute || nslist_compute || nblist_compute) return 1;
  ngas_tally = nsurf_tally = nboundary_tally = 0;
  return 0;
}

/* ----------------------------------------------------------------------
   set list computes that will called on current timestep when events occur
   ngas_tally = # of gas computes to be called on this step
   nsurf_tally = # of surface computes to be called on this step
   nboundary_tally = # of boundary computes to be called on this step
   also clear accumulators in computes which are invoked on this step
------------------------------------------------------------------------- */

void Update::tally_set(bigint ntimestep)
{
  int i;

  ngas_tally = 0;
  if (nglist_compute) {
    for (i = 0; i < nglist_compute; i++)
      if (glist_compute[i]->matchstep(ntimestep)) {
        glist_active[ngas_tally++] = glist_compute[i];
        glist_compute[i]->clear();
      }
  }

  nsurf_tally = 0;
  if (nslist_compute) {
    for (i = 0; i < nslist_compute; i++)
      if (slist_compute[i]->matchstep(ntimestep)) {
        slist_active[nsurf_tally++] = slist_compute[i];
        slist_compute[i]->clear();
      }
  }

  nboundary_tally = 0;
  if (nblist_compute) {
    for (i = 0; i < nblist_compute; i++)
      if (blist_compute[i]->matchstep(ntimestep)) {
        blist_active[nboundary_tally++] = blist_compute[i];
        blist_compute[i]->clear();
      }
  }
}

/* ----------------------------------------------------------------------
   make list of classes that reset dynamic parameters
   currently only surf collision models
------------------------------------------------------------------------- */

int Update::dynamic_setup()
{
  delete [] dlist_surfcollide;
  dlist_surfcollide = NULL;

  ndlist_surfcollide = 0;
  for (int i = 0; i < surf->nsc; i++)
    if (surf->sc[i]->dynamicflag) ndlist_surfcollide++;

  if (ndlist_surfcollide)
    dlist_surfcollide = new SurfCollide*[ndlist_surfcollide];

  ndlist_surfcollide = 0;
  for (int i = 0; i < surf->nsc; i++)
    if (surf->sc[i]->dynamicflag)
      dlist_surfcollide[ndlist_surfcollide++] = surf->sc[i];

  if (ndlist_surfcollide) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   invoke class methods that reset dynamic parameters
------------------------------------------------------------------------- */

void Update::dynamic_update()
{
  if (ndlist_surfcollide) {
    for (int i = 0; i < ndlist_surfcollide; i++)
      dlist_surfcollide[i]->dynamic();
  }
}

/* ----------------------------------------------------------------------
   set global properites via global command in input script
------------------------------------------------------------------------- */

void Update::global(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal global command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"fnum") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      fnum = input->numeric(FLERR,arg[iarg+1]);
      if (fnum <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"move") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) move_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) move_flag = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"optmove") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) optmove_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) optmove_flag = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nrho") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      nrho = input->numeric(FLERR,arg[iarg+1]);
      if (nrho <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"vstream") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal global command");
      vstream[0] = input->numeric(FLERR,arg[iarg+1]);
      vstream[1] = input->numeric(FLERR,arg[iarg+2]);
      vstream[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      temp_thermal = input->numeric(FLERR,arg[iarg+1]);
      if (temp_thermal <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"field") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"none") == 0) {
        fstyle = NOFIELD;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"constant") == 0) {
        if (iarg+6 > narg) error->all(FLERR,"Illegal global field command");
        fstyle = CFIELD;
        double fmag = input->numeric(FLERR,arg[iarg+2]);
        field[0] = input->numeric(FLERR,arg[iarg+3]);
        field[1] = input->numeric(FLERR,arg[iarg+4]);
        field[2] = input->numeric(FLERR,arg[iarg+5]);
        if (fmag <= 0.0) error->all(FLERR,"Illegal global field command");
        if (field[0] == 0.0 && field[1] == 0.0 && field[2] == 0.0)
          error->all(FLERR,"Illegal global field command");
        MathExtra::snorm3(fmag,field);
        iarg += 6;
      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal global field command");
        delete [] fieldID;
        fstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        fieldID = new char[n];
        strcpy(fieldID,arg[iarg+2]);
        iarg += 3;
      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global field command");
        delete [] fieldID;
        fstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        fieldID = new char[n];
        strcpy(fieldID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (fieldfreq < 0) error->all(FLERR,"Illegal global field command");
        iarg += 4;
      } else error->all(FLERR,"Illegal global field command");

    } 
      // --------------- E field ----------------
    else if (strcmp(arg[iarg],"efield") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");

      if (strcmp(arg[iarg+1],"none") == 0) {
        efstyle = NOFIELD;
        iarg += 2;

      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        efieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (efieldfreq < 0) error->all(FLERR,"Illegal global e field command");
        iarg += 4;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        efieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (efieldfreq < 0) error->all(FLERR,"Illegal global e field command");
        iarg += 4;

      } else error->all(FLERR,"Illegal global e field command");
    }

    // --------------- B field ----------------
    else if (strcmp(arg[iarg],"bfield") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");

      if (strcmp(arg[iarg+1],"none") == 0) {
        bfstyle = NOFIELD;
        iarg += 2;

      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        bfieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (bfieldfreq < 0) error->all(FLERR,"Illegal global b field command");
        iarg += 4;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        bfieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (bfieldfreq < 0) error->all(FLERR,"Illegal global b field command");
        iarg += 4;

      } else error->all(FLERR,"Illegal global b field command");


    }
    else if (strcmp(arg[iarg],"surfs") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      surf->global(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"surfgrid") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global surfgrid when surfaces already exist");
      if (strcmp(arg[iarg+1],"auto") == 0) grid->surfgrid_algorithm = PERAUTO;
      else if (strcmp(arg[iarg+1],"percell") == 0)
        grid->surfgrid_algorithm = PERCELL;
      else if (strcmp(arg[iarg+1],"persurf") == 0)
        grid->surfgrid_algorithm = PERSURF;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"surfmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global surfmax when surfaces already exist");
      grid->maxsurfpercell = atoi(arg[iarg+1]);
      if (grid->maxsurfpercell <= 0) error->all(FLERR,"Illegal global command");
      // reallocate paged data structs for variable-length surf info
      grid->allocate_surf_arrays();
      iarg += 2;
    } else if (strcmp(arg[iarg],"splitmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global splitmax when surfaces already exist");
      grid->maxsplitpercell = atoi(arg[iarg+1]);
      if (grid->maxsplitpercell <= 0) error->all(FLERR,"Illegal global command");
      // reallocate paged data structs for variable-length cell info
      grid->allocate_surf_arrays();
      iarg += 2;
    } else if (strcmp(arg[iarg],"gridcut") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      grid->cutoff = input->numeric(FLERR,arg[iarg+1]);
      if (grid->cutoff < 0.0 && grid->cutoff != -1.0)
        error->all(FLERR,"Illegal global command");
      // force ghost info to be regenerated with new cutoff
      grid->remove_ghosts();
      iarg += 2;
    } else if (strcmp(arg[iarg],"weight") == 0) {
      // for now assume just one arg after "cell"
      // may need to generalize later
      if (iarg+3 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"cell") == 0) grid->weight(1,&arg[iarg+2]);
      else error->all(FLERR,"Illegal weight command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"comm/sort") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) comm->commsortflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) comm->commsortflag = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/style") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"neigh") == 0) comm->commpartstyle = 1;
      else if (strcmp(arg[iarg+1],"all") == 0) comm->commpartstyle = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"surftally") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"auto") == 0) surf->tally_comm = TALLYAUTO;
      else if (strcmp(arg[iarg+1],"reduce") == 0) surf->tally_comm = TALLYREDUCE;
      else if (strcmp(arg[iarg+1],"rvous") == 0) surf->tally_comm = TALLYRVOUS;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"particle/reorder") == 0) {
      reorder_period = input->inumeric(FLERR,arg[iarg+1]);
      if (reorder_period < 0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    }
    // OpenEdge additions
    else if (strcmp(arg[iarg], "phi_track") == 0) {
      // unwrapped toroidal angle for mover-advected particles in axi:
      // per-remap accumulation of dphi = atan2(z, y) into the
      // "phi_unwrap" custom (DUSTT-style 2D3V + phi trajectories;
      // synthetic-camera projection without a 3D mesh). Pure-GCA ions
      // bypass the remap — their toroidal motion lives in p_gca_z.
      if (iarg + 1 >= narg)
        error->all(FLERR, "Illegal global phi_track command");
      if (strcmp(arg[iarg + 1], "yes") == 0) phi_track = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) phi_track = 0;
      else error->all(FLERR, "Illegal global phi_track command");
      if (phi_track) {
        const int custom_double = 1;   // Particle::add_custom type code
        phi_custom = particle->find_custom((char *) "phi_unwrap");
        if (phi_custom < 0)
          phi_custom = particle->add_custom((char *) "phi_unwrap",
                                            custom_double, 0);
      }
      iarg += 2;



    // Charged-particle pusher (Boris full-orbit or Boris/GCA hybrid) +
    // optional sheath overlay. Single hierarchical keyword:
    //
    //   global pusher mode boris|hybrid|gca
    //                 [subcycles N]
    //                 [plasma <ID>]            (compute plasma/fields or fix background)
    //                 [gca_switch <factor>]
    //                 [boris_near <m>]        (force Boris when |dist to sheath_geom surf| < m; 0 = off)
    //                 [gca_integrator rk2|rk4|simple]
    //                   (rk4 = 4 full-RHS stages and the backward-compatible
    //                    default; rk2 = 2 full-RHS midpoint stages; simple =
    //                    reduced 1-stage update without curvature/curl(b))
    //                 [dump yes|no] [dump_every N]
    //                 [bad_dt_check yes|no] [bad_dt_limit <max>]
    //                 [sheath off|kick|spatial
    //                         [geom <nearest_surf/grid-ID>]
    //                         [mD_amu <amu>]]
    //
    // Sheath dmax / pot_mult / model are auto: dmax = max(5*L_MPS, 10*lambdaD);
    // pot_mult = 0 -> Bohm-Stangeby floating wall; model is the combined
    // Coulette-Manfredi (close to wall) + Borodkina tail (s > 60 lambdaD).
    } else if (strcmp(arg[iarg], "pusher") == 0) {
      pusher->global_keyword(narg, arg, iarg);

    } else if (strcmp(arg[iarg],"mem/limit") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"grid") == 0) mem_limit_grid_flag = 1;
      else {
        double factor = input->numeric(FLERR,arg[iarg+1]);
        bigint global_mem_limit_big = static_cast<bigint> (factor * 1024*1024);
        if (global_mem_limit_big < 0) error->all(FLERR,"Illegal global command");
        if (global_mem_limit_big > MEMLIMIT_MAX)
          error->all(FLERR,"Global mem/limit setting cannot exceed 2GB");
        global_mem_limit = global_mem_limit_big;
      }
      iarg += 2;
    } else error->all(FLERR,"Illegal global command");
  }
}

/* ----------------------------------------------------------------------
   reset timestep as called from input script
------------------------------------------------------------------------- */

void Update::reset_timestep(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal reset_timestep command");
  bigint newstep = ATOBIGINT(arg[0]);
  reset_timestep(newstep);
}

/* ----------------------------------------------------------------------
   reset timestep
   set atimestep to new timestep, so future update_time() calls will be correct
   trigger reset of timestep for output and for fixes that require it
   do not allow any timestep-dependent fixes to be defined
   reset eflag/vflag global so nothing will think eng/virial are current
   reset invoked flags of computes,
     so nothing will think they are current between runs
   clear timestep list of computes that store future invocation times
   called from rerun command and input script (indirectly)
------------------------------------------------------------------------- */

void Update::reset_timestep(bigint newstep)
{
  ntimestep = newstep;
  if (ntimestep < 0) error->all(FLERR,"Timestep must be >= 0");
  if (ntimestep > MAXBIGINT) error->all(FLERR,"Too big a timestep");

  output->reset_timestep(ntimestep);

  for (int i = 0; i < modify->nfix; i++) {
    if (modify->fix[i]->time_depend)
      error->all(FLERR,
                 "Cannot reset timestep with a time-dependent fix defined");
    //modify->fix[i]->reset_timestep(ntimestep);
  }

  for (int i = 0; i < modify->ncompute; i++) {
    modify->compute[i]->invoked_scalar = -1;
    modify->compute[i]->invoked_vector = -1;
    modify->compute[i]->invoked_array = -1;
    modify->compute[i]->invoked_per_particle = -1;
    modify->compute[i]->invoked_per_grid = -1;
    modify->compute[i]->invoked_per_surf = -1;
    modify->compute[i]->invoked_per_tally = -1;
  }

  for (int i = 0; i < modify->ncompute; i++)
    if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();
}

/* ----------------------------------------------------------------------
   get mem/limit based on grid memory
------------------------------------------------------------------------- */

void Update::set_mem_limit_grid(int gnlocal)
{
  if (gnlocal == 0) gnlocal = grid->nlocal;

  bigint global_mem_limit_big = static_cast<bigint> (gnlocal*sizeof(Grid::ChildCell));

  // cap at 2 GB rather than erroring out so large grids can still be handled

  if (global_mem_limit_big > MEMLIMIT_MAX)
    global_mem_limit_big = MEMLIMIT_MAX;

  global_mem_limit = global_mem_limit_big;
}

/* ----------------------------------------------------------------------
   get mem/limit based on grid memory
------------------------------------------------------------------------- */

int Update::have_mem_limit()
{
  if (mem_limit_grid_flag)
    set_mem_limit_grid();

  int mem_limit_flag = 0;

  if (global_mem_limit > 0 || (mem_limit_grid_flag && !grid->nlocal))
    mem_limit_flag = 1;

  return mem_limit_flag;
}
