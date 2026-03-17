/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix thermal_force: Braginskii thermal forces on impurity ions.

    Per-particle parallel acceleration:
      a_par = (beta_i * Z^2 * e * grad_par(Ti)
             + alpha_e * Z^2 * e * grad_par(Te)) / m_Z

    Applied as leapfrog half-kicks at START_OF_STEP and END_OF_STEP,
    consistent with the Boris integrator.

    B-field is read in SPARTA coordinate order (bx, by, bz):
      2D: bx=B_R, by=B_Z, bz=B_toroidal
      3D: bx=B_x, by=B_y, bz=B_z (Cartesian)
    Temperature gradients are always cylindrical (grad_T_R, grad_T_Z).
------------------------------------------------------------------------- */

#include "fix_thermal_force.h"

#include <cmath>
#include <cstring>

#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "modify.h"
#include "particle.h"
#include "update.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16

/* ---------------------------------------------------------------------- */

FixThermalForce::FixThermalForce(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  have_ion_thermal_(0),
  have_elec_thermal_(0),
  beta_i_(2.6),
  alpha_e_(0.71)
{
  // fix ID thermal_force Nevery bfield BxSRC BySRC BzSRC [keywords...]

  if (narg < 7)
    error->all(FLERR,
      "Illegal fix thermal_force command "
      "(need: Nevery bfield BxSRC BySRC BzSRC)");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg++], "bfield") != 0)
    error->all(FLERR, "fix thermal_force: missing 'bfield' keyword");

  parse_compute_src(arg[iarg++], srcBx_, "Bx");
  parse_compute_src(arg[iarg++], srcBy_, "By");
  parse_compute_src(arg[iarg++], srcBz_, "Bz");

  // optional keywords
  while (iarg < narg) {

    if (strcmp(arg[iarg], "ion_thermal") == 0) {
      iarg++;
      if (iarg + 2 > narg)
        error->all(FLERR,
          "fix thermal_force ion_thermal: need gradTiR_SRC gradTiZ_SRC");
      parse_compute_src(arg[iarg++], srcGradTiR_, "gradTiR");
      parse_compute_src(arg[iarg++], srcGradTiZ_, "gradTiZ");
      have_ion_thermal_ = 1;

      // optional coeff keyword
      if (iarg + 1 < narg && strcmp(arg[iarg], "coeff") == 0) {
        iarg++;
        beta_i_ = input->numeric(FLERR, arg[iarg++]);
      }

    } else if (strcmp(arg[iarg], "elec_thermal") == 0) {
      iarg++;
      if (iarg + 2 > narg)
        error->all(FLERR,
          "fix thermal_force elec_thermal: need gradTeR_SRC gradTeZ_SRC");
      parse_compute_src(arg[iarg++], srcGradTeR_, "gradTeR");
      parse_compute_src(arg[iarg++], srcGradTeZ_, "gradTeZ");
      have_elec_thermal_ = 1;

      // optional coeff keyword
      if (iarg + 1 < narg && strcmp(arg[iarg], "coeff") == 0) {
        iarg++;
        alpha_e_ = input->numeric(FLERR, arg[iarg++]);
      }

    } else {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix thermal_force: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (!have_ion_thermal_ && !have_elec_thermal_)
    error->all(FLERR,
      "fix thermal_force: at least one of ion_thermal or "
      "elec_thermal must be specified");
}

/* ---------------------------------------------------------------------- */

FixThermalForce::~FixThermalForce()
{
  if (copymode) return;
  delete[] srcBx_.cid;
  delete[] srcBy_.cid;
  delete[] srcBz_.cid;
  if (have_ion_thermal_) {
    delete[] srcGradTiR_.cid;
    delete[] srcGradTiZ_.cid;
  }
  if (have_elec_thermal_) {
    delete[] srcGradTeR_.cid;
    delete[] srcGradTeZ_.cid;
  }
}

/* ---------------------------------------------------------------------- */

int FixThermalForce::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::init()
{
  // resolve compute sources
  auto bind = [&](CollGridSrc &S, const char *label) {
    if (S.kind != COLL_SRC_COMP) return;
    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix thermal_force: compute '%s' for %s not found",
               S.cid, label);
      error->all(FLERR, msg);
    }
    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR, "fix thermal_force: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR, "fix thermal_force: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix thermal_force: column %d for compute '%s' (%s) "
               "out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR, msg);
    }
  };

  bind(srcBx_, "Bx");
  bind(srcBy_, "By");
  bind(srcBz_, "Bz");

  if (have_ion_thermal_) {
    bind(srcGradTiR_, "gradTiR");
    bind(srcGradTiZ_, "gradTiZ");
  }
  if (have_elec_thermal_) {
    bind(srcGradTeR_, "gradTeR");
    bind(srcGradTeZ_, "gradTeZ");
  }
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // refresh caches once per timestep
  refresh_compute_src(srcBx_);
  refresh_compute_src(srcBy_);
  refresh_compute_src(srcBz_);

  if (have_ion_thermal_) {
    refresh_compute_src(srcGradTiR_);
    refresh_compute_src(srcGradTiZ_);
  }
  if (have_elec_thermal_) {
    refresh_compute_src(srcGradTeR_);
    refresh_compute_src(srcGradTeZ_);
  }

  kick_half(0.5 * update->dt);
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  kick_half(0.5 * update->dt);
}

/* ----------------------------------------------------------------------
   Apply half-kick from thermal forces.

   Per-particle parallel acceleration:
     a_par = (beta_i * Z^2 * QE * grad_par_Ti
            + alpha_e * Z^2 * QE * grad_par_Te) / m_Z

   B-field is in SPARTA coordinate order:
     2D: B[0]=B_R, B[1]=B_Z, B[2]=B_toroidal (= bx, by, bz output)
     3D: B[0]=B_x, B[1]=B_y, B[2]=B_z (Cartesian)

   Temperature gradients are always cylindrical (grad_T_R, grad_T_Z).

   For the parallel gradient, we need bhat in cylindrical frame:
     2D: bhat_R = B[0]/|B|, bhat_Z = B[1]/|B|
     3D: bhat_R_cyl = (bhat_x*cos_phi + bhat_y*sin_phi)
         bhat_Z_cyl = bhat_z

   Velocity kick is always in SPARTA coordinates (matches bhat order).
------------------------------------------------------------------------- */

void FixThermalForce::kick_half(double dt_half)
{
  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  const int nlocal = particle->nlocal;
  const int dim    = domain->dimension;
  const double QE  = update->echarge;   // elementary charge [C]

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];
    const int isp = p.ispecies;
    const double Z = species[isp].charge;
    if (Z == 0.0) continue;  // skip neutrals

    const int icell = p.icell;
    const double m_Z = species[isp].mass;  // kg
    if (m_Z <= 0.0) continue;

    // read B in SPARTA coordinate order from per-grid compute
    const double B0 = read_cell_src(srcBx_, icell);
    const double B1 = read_cell_src(srcBy_, icell);
    const double B2 = read_cell_src(srcBz_, icell);
    const double Bmag = std::sqrt(B0*B0 + B1*B1 + B2*B2);
    if (Bmag < 1.0e-20) continue;

    const double inv_Bmag = 1.0 / Bmag;

    // bhat in SPARTA coordinates (for velocity kick)
    const double bhat0 = B0 * inv_Bmag;
    const double bhat1 = B1 * inv_Bmag;
    const double bhat2 = B2 * inv_Bmag;

    // bhat in cylindrical frame (for gradient dot product)
    // gradients are always (grad_T_R, grad_T_Z) with no toroidal component
    double bhat_R_cyl, bhat_Z_cyl;
    if (dim == 2) {
      // 2D: B[0]=B_R, B[1]=B_Z directly
      bhat_R_cyl = bhat0;
      bhat_Z_cyl = bhat1;
    } else {
      // 3D: convert Cartesian bhat to cylindrical components
      const double rx = p.x[0];
      const double ry = p.x[1];
      const double R = std::sqrt(rx*rx + ry*ry);
      if (R > 1.0e-20) {
        const double cphi = rx / R;
        const double sphi = ry / R;
        bhat_R_cyl = bhat0 * cphi + bhat1 * sphi;
      } else {
        bhat_R_cyl = bhat0;
      }
      bhat_Z_cyl = bhat2;
    }

    // accumulate parallel acceleration
    double a_par = 0.0;
    const double Z2 = Z * Z;

    if (have_ion_thermal_) {
      const double gTiR = read_cell_src(srcGradTiR_, icell);  // eV/m
      const double gTiZ = read_cell_src(srcGradTiZ_, icell);  // eV/m
      const double grad_par_Ti = gTiR * bhat_R_cyl + gTiZ * bhat_Z_cyl;
      a_par += beta_i_ * Z2 * QE * grad_par_Ti / m_Z;
    }

    if (have_elec_thermal_) {
      const double gTeR = read_cell_src(srcGradTeR_, icell);  // eV/m
      const double gTeZ = read_cell_src(srcGradTeZ_, icell);  // eV/m
      const double grad_par_Te = gTeR * bhat_R_cyl + gTeZ * bhat_Z_cyl;
      a_par += alpha_e_ * Z2 * QE * grad_par_Te / m_Z;
    }

    if (a_par == 0.0) continue;

    // apply half-kick along bhat in SPARTA coordinates
    p.v[0] += a_par * bhat0 * dt_half;
    p.v[1] += a_par * bhat1 * dt_half;
    p.v[2] += a_par * bhat2 * dt_half;
  }
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::parse_compute_src(const char *tok, CollGridSrc &dst,
                                        const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "fix thermal_force: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) != 0)
    error->all(FLERR,
      "fix thermal_force: source must be a compute (c_ID[col])");

  dst.kind = COLL_SRC_COMP;
  const char *name = tok + 2;
  const char *lb   = strchr(name, '[');
  const char *rb   = lb ? strrchr(name, ']') : nullptr;

  if (!lb || !rb || rb <= lb + 1)
    error->all(FLERR,
      "fix thermal_force: use c_ID[col] syntax for compute sources");

  int idlen = static_cast<int>(lb - name);
  dst.cid = new char[idlen + 1];
  strncpy(dst.cid, name, idlen);
  dst.cid[idlen] = '\0';
  dst.col = atoi(lb + 1);   // 1-based

  if (dst.col <= 0)
    error->all(FLERR,
      "fix thermal_force: compute column must be >= 1");
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  // tally-style mapped access (e.g. compute grid)
  double **arr = nullptr;
  int    *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts  = update->ntimestep;
    return;
  }

  // standard per-grid array (e.g. compute plasma/fields)
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  S.arr_cache = nullptr;
  S.src_index = -1;
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double FixThermalForce::read_cell_src(const CollGridSrc &S, int icell) const
{
  if (S.kind != COLL_SRC_COMP) return 0.0;
  if (!S.arr_cache || S.src_index < 0) return 0.0;
  return S.arr_cache[icell][S.src_index];
}
