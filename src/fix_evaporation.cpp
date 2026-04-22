/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    Built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "fix_evaporation.h"
#include "fix_plasma_data.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "input.h"
#include "modify.h"
#include "fix.h"
#include "math_const.h"
#include "domain.h"
#include "openedge_geom.h"
#include "mixture.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixEvap::FixEvap(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  heatflux_scale(1.0),
  rocket_eta(0.0),
  pd_(nullptr)
{
  // fix ID evaporation Nevery MIXTURE plasma_data PD [keywords...]
  if (narg < 6)
    error->all(FLERR,
      "Illegal fix evaporation command "
      "(need: Nevery MIXTURE plasma_data PD)");

  nevery = atoi(arg[2]);
  imix   = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Fix evaporation: unknown mixture ID");

  if (strcmp(arg[4], "plasma_data") != 0)
    error->all(FLERR,
      "Fix evaporation: argument 5 must be 'plasma_data'");
  plasma_fix_id_ = std::string(arg[5]);

  set_mass = set_temp = set_radius = -1.0;

  int i = 6;
  while (i < narg) {
    if (strcmp(arg[i], "mass") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'mass'");
      set_mass = atof(arg[i+1]); i += 2;
    } else if (strcmp(arg[i], "temp") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'temp'");
      set_temp = atof(arg[i+1]); i += 2;
    } else if (strcmp(arg[i], "radius") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'radius'");
      set_radius = atof(arg[i+1]); i += 2;
    } else if (strcmp(arg[i], "heatflux/scale") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heatflux/scale'");
      heatflux_scale = atof(arg[i+1]);
      if (!std::isfinite(heatflux_scale) || heatflux_scale < 0.0)
        error->all(FLERR,"Fix evaporation: heatflux/scale must be finite and >= 0");
      i += 2;
    } else if (strcmp(arg[i], "rocket_eta") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'rocket_eta'");
      rocket_eta = atof(arg[i+1]);
      if (rocket_eta < 0.0 || rocket_eta > 1.0)
        error->all(FLERR,"Fix evaporation: rocket_eta must be in [0,1]");
      i += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Fix evaporation: unknown keyword '%s'", arg[i]);
      error->all(FLERR, msg);
    }
  }
}

/* ---------------------------------------------------------------------- */

FixEvap::~FixEvap() {}

/* ---------------------------------------------------------------------- */

int FixEvap::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEvap::init()
{
  if (domain->dimension != 2)
    error->all(FLERR,"Fix evaporation: only 2D geometry supported");

  const int ifix = modify->find_fix(plasma_fix_id_.c_str());
  if (ifix < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "Fix evaporation: plasma_data fix '%s' not found",
             plasma_fix_id_.c_str());
    error->all(FLERR, msg);
  }
  pd_ = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
  if (!pd_)
    error->all(FLERR,
      "Fix evaporation: plasma_data fix must be style plasma/data");
  pd_->init();
}

/* ---------------------------------------------------------------------- */

void FixEvap::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

void FixEvap::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

double FixEvap::memory_usage() { return 0.0; }

/* ---------------------------------------------------------------------- */

void FixEvap::evap_half(double dt_half)
{
  if ((update->ntimestep % nevery) != 0) return;

  Particle::OnePart *parts = particle->particles;
  const int nlocal = particle->nlocal;
  int *s2g = particle->mixture[imix]->species2group;
  int ndeleted = 0;

  for (int ip = 0; ip < nlocal; ip++) {
    const int is = parts[ip].ispecies;
    const int ig = s2g[is];
    if (ig < 0) continue;

    if (set_mass   > 0.0 && parts[ip].mass   <= 0.0) parts[ip].mass   = set_mass;
    if (set_radius > 0.0 && parts[ip].radius <= 0.0) parts[ip].radius = set_radius;
    if (set_temp   > 0.0 && parts[ip].temp   <= 0.0) parts[ip].temp   = set_temp;

    droplet_evaporation_model(&parts[ip], dt_half);

    const double m0_ref = (set_mass > 0.0) ? set_mass : particle->species[is].mass;
    const double m_cut  = 0.1 * m0_ref;
    if (parts[ip].mass > 0.0 && m0_ref > 0.0 && parts[ip].mass <= m_cut) {
      parts[ip].mass   = 0.0;
      parts[ip].radius = 0.0;
      parts[ip].temp   = 0.0;
      parts[ip].icell  = -1;
      ndeleted++;
    }
  }

  if (ndeleted > 0) particle->compress_rebalance();
}

/* ----------------------------------------------------------------------
   Sergey's droplet evaporation model:
     dR/dt = -(m_atom / rho) * Gevap
     dT/dt = (3 / (rho * Cp * R)) * (Qs - Gevap * DHm / N_A)
   where Qs = sqrt(q_par^2 + q_perp^2) at the droplet position, pulled
   from fix plasma/data. Rocket force uses -grad(Te) as the recoil axis.
------------------------------------------------------------------------- */
void FixEvap::droplet_evaporation_model(Particle::OnePart *ip,
                                        const double dt_half)
{
  const double AM   = 1.53e-26;      // Li atom mass [kg]
  const double Rho  = 534.0;         // kg/m^3
  const double Cp   = 4200.0;        // J/kg-K
  const double DHm  = 3.158e+03;     // J/mol
  const double AN   = 6.022e+23;     // 1/mol
  const double DT   = dt_half;

  const double mass   = (ip->mass   > 0.0) ? ip->mass
                        : particle->species[ip->ispecies].mass;
  const double radius = (ip->radius > 0.0) ? ip->radius
                        : std::pow((3.0*mass)/(4.0*MY_PI*Rho), 1.0/3.0);
  const double TK     = (ip->temp   > 0.0) ? ip->temp : 300.0;

  // R/Z at particle position (handles 2D Cart, 2D axi, 3D Cart via helper).
  double R = 0.0, Z = 0.0;
  OpenEdge::sparta_to_RZ(ip->x, domain->dimension, domain->axisymmetric, R, Z);

  // Heat-flux vector at droplet position. When plasma.h5 carries q_par /
  // q_perp (mesh-level or regular grid), interp2D routes through the
  // appropriate path inside fix plasma/data. Otherwise read the default.
  double q_par  = pd_->default_q_par;
  double q_perp = pd_->default_q_perp;
  if (pd_->has_qheatflux) {
    if (!pd_->mesh_q_par.empty() || !pd_->q_par.empty()) {
      q_par  = !pd_->mesh_q_par.empty()
               ? pd_->interp2D(pd_->mesh_q_par,  R, Z, ip->icell)
               : pd_->interp2D(pd_->q_par,       R, Z, ip->icell);
    }
    if (!pd_->mesh_q_perp.empty() || !pd_->q_perp.empty()) {
      q_perp = !pd_->mesh_q_perp.empty()
               ? pd_->interp2D(pd_->mesh_q_perp, R, Z, ip->icell)
               : pd_->interp2D(pd_->q_perp,      R, Z, ip->icell);
    }
  }
  // |q_plasma| from the stored components. A sphere in a uniform
  // directional flux intercepts q·πR² = (1/4)·q·(4πR²), so the
  // surface-averaged flux driving the energy balance is |q|/4. The
  // model below applies Qs across the full 4πR² surface (3/(ρ·Cp·R)
  // factor), so bake the 1/4 in here. heatflux_scale stays as a user
  // knob for calibration.
  double Qs = 0.25 * std::sqrt(q_par*q_par + q_perp*q_perp);
  if (!std::isfinite(Qs) || Qs < 0.0) Qs = 0.0;
  Qs *= heatflux_scale;

  // grad_Te at particle position (for rocket-force direction).
  double gTeR = 0.0, gTeZ = 0.0;
  if (rocket_eta > 0.0) {
    if (!pd_->mesh_grad_te_r.empty())
      gTeR = pd_->interp2D(pd_->mesh_grad_te_r, R, Z, ip->icell);
    else if (!pd_->grad_te_r.empty())
      gTeR = pd_->interp2D(pd_->grad_te_r, R, Z, ip->icell);
    if (!pd_->mesh_grad_te_z.empty())
      gTeZ = pd_->interp2D(pd_->mesh_grad_te_z, R, Z, ip->icell);
    else if (!pd_->grad_te_z.empty())
      gTeZ = pd_->interp2D(pd_->grad_te_z, R, Z, ip->icell);
  }

  if (Qs <= 0.0) {
    ip->radius = radius;
    ip->temp   = TK;
    ip->mass   = mass;
    return;
  }

  // Antoine + Hertz-Knudsen flux.
  const double a1 = 5.055, b1 = -8023.0, xm1 = 6.939;
  const double vpres1 = 760.0 * std::pow(10.0, a1 + b1/TK);          // mmHg
  const double Gevap_atoms = 1.0e4 * 3.513e22 * vpres1 / std::sqrt(xm1 * TK);

  const double dRdt = -AM * Gevap_atoms / Rho;
  const double HF   = Qs - Gevap_atoms * (DHm / AN);
  const double r_safe = (radius > 1.0e-20) ? radius : 1.0e-20;
  const double dTdt = (3.0 / (Rho * Cp * r_safe)) * HF;

  const double R_new   = std::max(0.0, radius + dRdt * DT);
  const double T_new   = TK + dTdt * DT;
  const double m_new   = (R_new > 0.0)
                         ? (Rho * (4.0/3.0) * MY_PI * R_new*R_new*R_new)
                         : 0.0;
  if (T_new < 0.0)
    error->one(FLERR,"Fix evaporation: particle temperature dropped below 0 K");

  if (rocket_eta > 0.0 && m_new > 0.0 && Gevap_atoms > 0.0) {
    const double grad_mag = std::sqrt(gTeR*gTeR + gTeZ*gTeZ);
    if (std::isfinite(grad_mag) && grad_mag > 0.0) {
      const double kB = 1.380649e-23;
      const double v_thermal = std::sqrt(8.0 * kB * TK / (MY_PI * AM));
      const double area = 4.0 * MY_PI * radius * radius;
      const double dmdt = area * Gevap_atoms * AM;
      const double a_mag = rocket_eta * dmdt * v_thermal / m_new;
      const double nr = -gTeR / grad_mag;
      const double nz = -gTeZ / grad_mag;

      double phi = 0.0;
      if (domain->dimension == 3) phi = std::atan2(ip->x[1], ip->x[0]);
      double dvx, dvy, dvz;
      OpenEdge::RZphi_force_to_sparta(a_mag * nr, a_mag * nz, 0.0,
                                       domain->dimension, domain->axisymmetric,
                                       phi, dvx, dvy, dvz);
      ip->v[0] += dvx * DT;
      ip->v[1] += dvy * DT;
      ip->v[2] += dvz * DT;
    }
  }

  ip->radius = R_new;
  ip->temp   = T_new;
  ip->mass   = m_new;
}
