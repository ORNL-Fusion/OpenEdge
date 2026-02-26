/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_coll_background.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "math.h"
#include "react_bird.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "random_knuth.h"
#include "math_const.h"
#include <filesystem>
#include "math_extra.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include "compute.h"
#include "variable.h"
#include "random_mars.h"      // <-- add this
#include "update.h"

namespace fs = std::filesystem;
using namespace SPARTA_NS;
#define INVOKED_PER_GRID 16

/* ---------------------------------------------------------------------- */

FixCollBackground::FixCollBackground(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{

 // Syntax:
  // fix ID coll/background <nevery> <A_bg> <Z_bg>
  //      plasma <TeSrc> <TiSrc> <NiSrc> <VparSrc>
  //      bfield <BrSrc> <BtSrc> <BzSrc>

  if (narg < 14)
    error->all(FLERR,"Illegal fix coll/background (need: nevery A_bg Z_bg plasma Te Ti Ni Vpar bfield Br Bt Bz)");

  int iarg = 2;
  nevery        = input->inumeric(FLERR,arg[iarg++]);
  A_background  = input->numeric(FLERR,arg[iarg++]);  // atomic mass (amu)
  Z_background  = input->inumeric(FLERR,arg[iarg++]); // charge state (integer)

  // expect "plasma"
  if (strcmp(arg[iarg++],"plasma") != 0)
    error->all(FLERR,"fix coll/background: missing 'plasma' keyword");

  auto dupstr = [](const char* s)->char* { auto *p=new char[strlen(s)+1]; strcpy(p,s); return p; };

  // tiny parser for a token: either VAR name or c_ID[idx]
  auto parse_src = [&](const char *tok, CollGridSrc &dst, const char *label){
    if (!tok || !*tok) {
      char msg[128]; snprintf(msg,sizeof(msg),"fix coll/background: empty token for %s",label);
      error->all(FLERR,msg);
    }
    if (strncmp(tok,"c_",2)==0) {
      dst.kind = COLL_SRC_COMP;
      const char *name = tok + 2;
      const char *lb   = strchr(name,'[');
      const char *rb   = (lb ? strrchr(name,']') : nullptr);
      if (!lb || !rb || rb <= lb+1)
        error->all(FLERR,"fix coll/background: use c_ID[idx] for compute sources");
      const int idlen = lb - name;
      dst.cid = new char[idlen+1];
      strncpy(dst.cid, name, idlen);
      dst.cid[idlen] = '\0';
      dst.col = atoi(lb+1);         // 1-based
      if (dst.col <= 0)
        error->all(FLERR,"fix coll/background: compute column must be >=1");
    } else {
      dst.kind  = COLL_SRC_VAR;
      dst.vname = dupstr(tok);
    }
  };

  // plasma sources
  parse_src(arg[iarg++], srcTe,   "Te");
  parse_src(arg[iarg++], srcTi,   "Ti");
  parse_src(arg[iarg++], srcNi,   "Ni");
  parse_src(arg[iarg++], srcVpar, "Vpar");

  // expect "bfield"
  if (strcmp(arg[iarg++],"bfield") != 0)
    error->all(FLERR,"fix coll/background: missing 'bfield' keyword");

  // B sources
  parse_src(arg[iarg++], srcBr, "Br");
  parse_src(arg[iarg++], srcBt, "Bt");
  parse_src(arg[iarg++], srcBz, "Bz");

  use_grid_plasma = (srcTe.kind==COLL_SRC_VAR || srcTi.kind==COLL_SRC_VAR ||
                     srcNi.kind==COLL_SRC_VAR || srcVpar.kind==COLL_SRC_VAR);
  use_grid_bfield = (srcBr.kind==COLL_SRC_VAR || srcBt.kind==COLL_SRC_VAR || srcBz.kind==COLL_SRC_VAR);

  // defaults
  maxgrid_plasma = maxgrid_b = 0;
  plasma_grid = b_grid = nullptr;
}
    
/* ---------------------------------------------------------------------- */

FixCollBackground::~FixCollBackground()
{
  if (copymode) return;
  delete rng;
  rng = nullptr;

}

/* ---------------------------------------------------------------------- */

int FixCollBackground::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixCollBackground::init()
{

    // Create RNG once, seeded from ranmaster (same pattern as AdaptGrid::setup)
  if (!rng) {
    rng = new RanKnuth(update->ranmaster->uniform());
    const double seed = update->ranmaster->uniform();
    rng->reset(seed, comm->me, 100);
  }

  
    // --- resolve VAR sources (grid variables) ---
  auto bind_var = [&](CollGridSrc &S, const char *label){
    if (S.kind != COLL_SRC_VAR) return;
    if (!S.vname || !*S.vname) {
      char msg[160];
      snprintf(msg,sizeof(msg),
               "fix coll/background: empty name for %s grid variable", label);
      error->all(FLERR,msg);
    }
    S.varid = input->variable->find(S.vname);
    if (S.varid < 0 || !input->variable->grid_style(S.varid)) {
      char msg[200];
      snprintf(msg,sizeof(msg),
        "fix coll/background: %s ('%s') must be a grid-style variable", label, S.vname);
      error->all(FLERR,msg);
    }
  };
  bind_var(srcTe,"Te");
  bind_var(srcTi,"Ti");
  bind_var(srcNi,"Ni");
  bind_var(srcVpar,"Vpar");
  bind_var(srcBr,"Br");
  bind_var(srcBt,"Bt");
  bind_var(srcBz,"Bz");

  // --- allocate VAR buffers if needed ---
  if (use_grid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "coll/background:plasma_grid");
    if (grid->nlocal)
      memset(&plasma_grid[0][0], 0, sizeof(double)*grid->nlocal*4);
  }
  if (use_grid_bfield) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "coll/background:b_grid");
    if (grid->nlocal)
      memset(&b_grid[0][0], 0, sizeof(double)*grid->nlocal*3);
  }

  // --- resolve COMPUTE sources (per-grid arrays) ---
  auto bind_compute = [&](CollGridSrc &S, const char *label){
    if (S.kind != COLL_SRC_COMP) return;
    if (!S.cid || !*S.cid) {
      char msg[160];
      snprintf(msg,sizeof(msg),
               "fix coll/background: empty compute ID for %s", label);
      error->all(FLERR,msg);
    }
    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg,sizeof(msg),
               "fix coll/background: compute '%s' for %s not found", S.cid, label);
      error->all(FLERR,msg);
    }
    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR,"fix coll/background: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR,"fix coll/background: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[200];
      snprintf(msg,sizeof(msg),
               "fix coll/background: column %d for compute '%s' (%s) out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR,msg);
    }
  };
  bind_compute(srcTe,"Te");
  bind_compute(srcTi,"Ti");
  bind_compute(srcNi,"Ni");
  bind_compute(srcVpar,"Vpar");
  bind_compute(srcBr,"Br");
  bind_compute(srcBt,"Bt");
  bind_compute(srcBz,"Bz");
}



/* ---------------------------------------------------------------------- */

void FixCollBackground::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  if (!particle->sorted) particle->sort();
  end_of_step_no_average();
}


/* ----------------------------------------------------------------------
   current thermal temperature is calculated on a per-cell basis
---------------------------------------------------------------------- */
void FixCollBackground::end_of_step_no_average()
{
  // Refresh grid-sourced inputs (if any)
// Refresh grid VAR buffers if requested
if (use_grid_plasma) compute_plasma_grid();
if (use_grid_bfield) compute_bfield_grid();

// Refresh all COMP sources exactly once this step
refresh_compute_src(srcTe);   refresh_compute_src(srcTi);
refresh_compute_src(srcNi);   refresh_compute_src(srcVpar);
refresh_compute_src(srcBr);   refresh_compute_src(srcBt);
refresh_compute_src(srcBz);


  // Cheap early-outs
  if (grid->nlocal == 0) return;
  if (particle->nlocal == 0) return;   // if your Particle has nlocal; remove if not available

  Particle::OnePart  * const particles = particle->particles;
  Particle::Species  * const species   = particle->species;  // kept for symmetry; not used here
  int                * const next      = particle->next;
  Grid::ChildInfo    * const cinfo     = grid->cinfo;
  const int nglocal = grid->nlocal;

  for (int icell = 0; icell < nglocal; ++icell) {
    if (cinfo[icell].count == 0) continue;

    int ip = cinfo[icell].first;   // no outer 'ip' → no shadowing
    while (ip >= 0) {
      backgroundCollisions(&particles[ip]);
      ip = next[ip];
    }
  }
}


/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixCollBackground::memory_usage()
{
  double bytes = 0.0;
  if (plasma_grid) bytes += (size_t)maxgrid_plasma * 4 * sizeof(double);
  if (b_grid)      bytes += (size_t)maxgrid_b       * 3 * sizeof(double);
  return bytes;
}
/* ---------------------------------------------------------------------- */


void FixCollBackground::backgroundCollisions(Particle::OnePart *ip) {
  
    // printf("Background collisions\n");
    Particle::Species *species = particle->species;
    int icell = ip->icell;
    // if (icell < 0 || icell >= grid->nlocal) return;

    int isp = ip->ispecies;
    size_t charge = int(species[isp].charge);

    // size_t charge = int(species[isp].charge);//
    // if charge is zero, return
    if (charge == 0) {
        return;
    }

    double mass = species[isp].molwt;
    double *v = ip->v;
    double dt = update->dt;

  //     auto read_src = [&](const CollGridSrc& S, int col_in_plasma, int col_in_b)->double {
  //   if (S.kind == COLL_SRC_COMP) return fetch_compute_cell_value(S, icell);
  //   if (S.kind == COLL_SRC_VAR)  return (col_in_plasma>=0) ? plasma_grid[icell][col_in_plasma]
  //                                                       : b_grid[icell][col_in_b];
  //   return 0.0;
  // };
  auto read_src = [&](const CollGridSrc& S, int col_in_plasma, int col_in_b)->double {
  if (S.kind == COLL_SRC_COMP) {
    // O(1): just a pointer read now
    return (S.arr_cache && S.src_index >= 0) ? S.arr_cache[icell][S.src_index] : 0.0;
  }
  if (S.kind == COLL_SRC_VAR) {
    return (col_in_plasma >= 0) ? plasma_grid[icell][col_in_plasma]
                                : b_grid[icell][col_in_b];
  }
  return 0.0;
};



  // Plasma (units you expect in old code: Te, Ti in eV; Ni in m^-3; Vpar in m/s)
  const double te_eV     = std::max(read_src(srcTe,   0,-1), 0.0);
  const double ti_eV     = std::max(read_src(srcTi,   1,-1), 0.0);
  const double dens    = std::max(read_src(srcNi,   2,-1), 0.0);
  const double vpar_flow = read_src(srcVpar, 3,-1);          // already in m/s, do NOT scale

  // Magnetic field (in Tesla; Br,Bt,Bz)
  const double B_R = read_src(srcBr,-1,0);
  const double B_T = read_src(srcBt,-1,1);
  const double B_Z = read_src(srcBz,-1,2);

  // bail if no B
  double B[3] = {B_R, B_T, B_Z};
  double Bnorm = std::sqrt(B_R*B_R + B_T*B_T + B_Z*B_Z);
  if (Bnorm < 1e-12) return;

  // Build parallel flow vector
  double flowVelocity[3] = { vpar_flow * B_R / Bnorm,
                            vpar_flow * B_T / Bnorm,
                            vpar_flow * B_Z / Bnorm };

  
      // Get plasma data
      // generate random numbers
      // double r1 = static_cast<double>(random()) / static_cast<double>(RAND_MAX);
      // double r2 = static_cast<double>(random()) / static_cast<double>(RAND_MAX);
      // double r3 = static_cast<double>(random()) / static_cast<double>(RAND_MAX);
      const double r1  = rng->uniform();   // [0,1)
      const double r2  = rng->uniform();
      const double r3 = rng->uniform();


      double n1 = std::max(0.0, std::min(r1, 1.0));
      double n2 = std::max(0.0, std::min(r2, 1.0));
      double xsi = std::max(0.0,std::min(r3, 1.0));
  

      double relativeVelocity[3];
      for(int i=0; i<3; i++) {
          relativeVelocity[i] = v[i] - flowVelocity[i];
      }
      double velocityRelativeNorm = std::sqrt(std::inner_product(relativeVelocity, relativeVelocity+3, relativeVelocity, 0.0));
      // print realtive velocities
      // getSlowDownFrequencies
      double nu_friction, nu_deflection, nu_parallel, nu_energy;
      getSlowDownFrequencies(nu_friction, nu_deflection, nu_parallel, nu_energy, v, charge, mass, te_eV, ti_eV,  dens);
      // getSlowDownDirections2
      double parallel_direction[3], perp_direction1[3], perp_direction2[3];
        getSlowDownDirections2(parallel_direction, perp_direction1, perp_direction2, v[0], v[1], v[2]);
      double coeff_par = n1 * std::sqrt(2.0 * nu_parallel * dt);
      double cosXsi = std::min(std::max(cos(2.0 * M_PI * xsi), -1.0), 1.0);
      double coeff_perp1 = cosXsi * std::sqrt(nu_deflection * dt * 0.5);
      double coeff_perp2 = sin(2.0 * M_PI * xsi) * std::sqrt(nu_deflection * dt * 0.5);
      double nuEdt = std::min(std::max(nu_energy * dt, -1.0), 1.0);
      double vx_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[0] + std::abs(n2)*(coeff_perp1 * perp_direction1[0] + coeff_perp2 * perp_direction2[0])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[0];
      double vy_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[1] + std::abs(n2)*(coeff_perp1 * perp_direction1[1] + coeff_perp2 * perp_direction2[1])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[1];  
      double vz_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[2] + std::abs(n2)*(coeff_perp1 * perp_direction1[2] + coeff_perp2 * perp_direction2[2])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[2];
     
      v[0] = vx_relative + flowVelocity[0];
      v[1] = vy_relative + flowVelocity[1];
      v[2] = vz_relative + flowVelocity[2];
      // update velocity
      ip->v[0] = v[0];
      ip->v[1] = v[1];
      ip->v[2] = v[2];
  }


void FixCollBackground::getSlowDownFrequencies(double& nu_friction, double& nu_deflection, 
  double& nu_parallel, double& nu_energy, 
  double *v, double charge, double amu, double te_eV, double ti_eV, double density) {

     if (te_eV <= 0.0 || density <= 0.0) {
    nu_friction = nu_deflection = nu_parallel = nu_energy = 0.0;
    return;
  }

    const double vx = v[0];
    const double vy = v[1];
    const double vz = v[2];
    const double flowVelocity[3]= {0.0, 0.0, 0.0};
    const double relativeVelocity[3] = {vx - flowVelocity[0], vy - flowVelocity[1], vz - flowVelocity[2]};
    const double velocityNorm = std::sqrt(relativeVelocity[0]*relativeVelocity[0] + 
                                          relativeVelocity[1]*relativeVelocity[1] + 
                                          relativeVelocity[2]*relativeVelocity[2]);

    const double background_Z = 1.0; // Deuterium
    const double background_mass = 2.0; // Deuterium
    const double Q = 1.60217662e-19;

    const double lam_d = std::sqrt(update->epsilon_0 * te_eV / (density * background_Z * background_Z * Q));
    const double lam = 12.0 * M_PI * density * lam_d * lam_d * lam_d / charge;
    double lam_gitr = 12.0 * M_PI * density * std::pow(lam_d,3)/charge;
    double gam_electron_background = 0.238762895 * charge * charge * std::log(lam) / (amu * amu);
    double gam_ion_background = 0.238762895 * charge * charge * background_Z * background_Z * std::log(lam) / (amu * amu);
    if(gam_electron_background < 0.0) gam_electron_background=0.0;
    if(gam_ion_background < 0.0) gam_ion_background=0.0;
    const double a_ion = background_mass * update->proton_mass / (2 * ti_eV * Q);
    const double a_electron = update->electron_mass / (2 * te_eV * Q);
    const double xx = velocityNorm * velocityNorm * a_ion;
    // printf("xx %g\n", xx);
    const double psi_prime = 2.0 * std::sqrt(xx/M_PI) * std::exp(-xx);
    // printf("psi_prime %g\n", psi_prime);
    const double psi_psiprime = std::erf(std::sqrt(xx));
    // printf("psi_psiprime %g\n", psi_psiprime);
    const double psi = psi_psiprime - psi_prime;
    const double xx_e = velocityNorm * velocityNorm * a_electron;
    const double psi_prime_e = 1.128379 * std::sqrt(xx_e);
    const double psi_e = 0.75225278 * std::pow(xx_e,1.5);
    const double psi_psiprime_e = psi_e + psi_prime_e;
    const double nu_0_i = gam_electron_background * density / std::pow(velocityNorm, 3);
    const double nu_0_e = gam_ion_background * density / std::pow(velocityNorm, 3);
    nu_friction = (1 + amu / background_mass) * psi * nu_0_i;
    nu_deflection = 2 * (psi_psiprime - psi / (2 * xx)) * nu_0_i;
    nu_parallel = psi / xx * nu_0_i;
    double factor = 2 * (amu / background_mass * psi - psi_prime) ;
    nu_energy = 2 * (amu / background_mass * psi - psi_prime) * nu_0_i;

}
void FixCollBackground::getSlowDownDirections2(double parallel_direction[], double perp_direction1[], 
  double perp_direction2[], double vx, double vy, double vz) {
  double v = std::sqrt(vx * vx + vy * vy + vz * vz);

  // Handle the case where v == 0.0
  if (v == 0.0) {
    v = 1.0;
    vz = 1.0;
    vx = 0.0;
    vy = 0.0;
  }
  double ez1 = vx / v;
  double ez2 = vy / v;
  double ez3 = vz / v;
    
  // Compute the first perpendicular direction
  double ex1 = ez2;
  double ex2 = -ez1;
  double ex3 = 0.0;

  // Handle edge case for particles moving purely in z-direction
  double exnorm = std::sqrt(ex1 * ex1 + ex2 * ex2);
  if (std::abs(exnorm) < 1.0e-12) {
    ex1 = -ez3;
    ex2 = 0.0;
    ex3 = ez1;
  }
  // Normalize ex
  exnorm = std::sqrt(ex1 * ex1 + ex2 * ex2 + ex3 * ex3);
  ex1 /= exnorm;
  ex2 /= exnorm;
  ex3 /= exnorm;

  // Compute the second perpendicular direction
  double ey1 = ez2 * ex3 - ez3 * ex2;
  double ey2 = ez3 * ex1 - ez1 * ex3;
  double ey3 = ez1 * ex2 - ez2 * ex1;
  // Populate output arrays
  parallel_direction[0] = ez1; 
  parallel_direction[1] = ez2;
  parallel_direction[2] = ez3;

  perp_direction1[0] = ex1; 
  perp_direction1[1] = ex2;
  perp_direction1[2] = ex3;

  perp_direction2[0] = ey1; 
  perp_direction2[1] = ey2;
  perp_direction2[2] = ey3;
}

// Compute grid variables into our local buffers (only those requested)
void FixCollBackground::compute_plasma_grid() {
  if (!use_grid_plasma || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "coll/background:plasma_grid");
  }
  // zero only the rows we will actually write
  memset(&plasma_grid[0][0], 0, sizeof(double)*grid->nlocal*4);

  const int stride = 4;
  if (srcTe.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcTe.varid,   &plasma_grid[0][0], stride, 0);
  if (srcTi.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcTi.varid,   &plasma_grid[0][1], stride, 0);
  if (srcNi.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcNi.varid,   &plasma_grid[0][2], stride, 0);
  if (srcVpar.kind == COLL_SRC_VAR) input->variable->compute_grid(srcVpar.varid, &plasma_grid[0][3], stride, 0);
}

void FixCollBackground::compute_bfield_grid() {
  if (!use_grid_bfield || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_b) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "coll/background:b_grid");
  }
  memset(&b_grid[0][0], 0, sizeof(double)*grid->nlocal*3);

  const int stride = 3;
  if (srcBr.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBr.varid, &b_grid[0][0], stride, 0);
  if (srcBt.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBt.varid, &b_grid[0][1], stride, 0);
  if (srcBz.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBz.varid, &b_grid[0][2], stride, 0);
}

// Ensure compute_per_grid ran and read 1-based column c_ID[col]
double FixCollBackground::fetch_compute_cell_value(const CollGridSrc& S, int icell)
{
  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr;
  int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap <= 0 || !arr) return 0.0;             // defensive: unmapped or no data

  const int src = cols ? cols[0] : (S.col - 1);  // fallback: 1:1 layout
  return arr[icell][src];
}

inline void FixCollBackground::refresh_compute_src(CollGridSrc &S) {
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr; int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap <= 0 || !arr) { S.arr_cache=nullptr; S.src_index=-1; S.cache_ts=update->ntimestep; return; }

  S.arr_cache  = arr;
  S.src_index  = cols ? cols[0] : (S.col - 1);
  S.cache_ts   = update->ntimestep;
}
