/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    Built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "fix_droplet_evaporate.h"
#include "fix_background.h"
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
#include "random_knuth.h"
#include "random_mars.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

using namespace SPARTA_NS;
using namespace MathConst;

namespace {
  // Sample a Poisson-distributed integer with mean lam.
  // Knuth direct method for small lam; Gaussian approx for large lam.
  inline int sample_poisson(double lam, SPARTA_NS::RanKnuth *rng) {
    if (lam <= 0.0) return 0;
    if (lam < 30.0) {
      const double L = std::exp(-lam);
      int k = 0;
      double p = 1.0;
      while (true) {
        ++k;
        p *= rng->uniform();
        if (p <= L) return k - 1;
      }
    }
    // Gaussian approximation: Box-Muller.
    const double u1 = std::max(rng->uniform(), 1.0e-300);
    const double u2 = rng->uniform();
    const double z  = std::sqrt(-2.0 * std::log(u1)) *
                      std::cos(2.0 * M_PI * u2);
    long long n = static_cast<long long>(std::floor(lam + std::sqrt(lam) * z + 0.5));
    if (n < 0) n = 0;
    return static_cast<int>(n);
  }
}

/* ---------------------------------------------------------------------- */

FixDropletEvaporate::FixDropletEvaporate(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  heatflux_scale(1.0),
  rocket_eta(0.0),
  evap_atoms_local_(0.0),
  pd_(nullptr),
  emit_imix(-1),
  random(nullptr)
{
  scalar_flag = 1;
  global_freq = 1;

  // fix ID evaporation Nevery MIXTURE background PD [keywords...]
  if (narg < 6)
    error->all(FLERR,
      "Illegal fix evaporation command "
      "(need: Nevery MIXTURE background PD)");

  nevery = atoi(arg[2]);
  imix   = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Fix evaporation: unknown mixture ID");

  if (strcmp(arg[4], "background") != 0)
    error->all(FLERR,
      "Fix evaporation: argument 5 must be 'background'");
  plasma_fix_id_ = std::string(arg[5]);

  int i = 6;
  while (i < narg) {
    if (strcmp(arg[i], "heatflux/scale") == 0) {
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
    } else if (strcmp(arg[i], "emit_into") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'emit_into'");
      emit_imix = particle->find_mixture(arg[i+1]);
      if (emit_imix < 0)
        error->all(FLERR,"Fix evaporation: unknown emit_into mixture ID");
      i += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Fix evaporation: unknown keyword '%s'", arg[i]);
      error->all(FLERR, msg);
    }
  }

  if (emit_imix >= 0) {
    random = new RanKnuth(update->ranmaster->uniform());
    double seed = comm->me + 1;
    random->reset(seed, comm->me, 100);
  }
}

/* ---------------------------------------------------------------------- */

FixDropletEvaporate::~FixDropletEvaporate()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixDropletEvaporate::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::init()
{
  if (domain->dimension != 2)
    error->all(FLERR,"Fix evaporation: only 2D geometry supported");

  const int ifix = modify->find_fix(plasma_fix_id_.c_str());
  if (ifix < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "Fix evaporation: background fix '%s' not found",
             plasma_fix_id_.c_str());
    error->all(FLERR, msg);
  }
  pd_ = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd_)
    error->all(FLERR,
      "Fix evaporation: background fix must be style background");
  pd_->init();
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

void FixDropletEvaporate::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

double FixDropletEvaporate::memory_usage() { return 0.0; }

double FixDropletEvaporate::compute_scalar()
{
  double global = 0.0;
  MPI_Allreduce(&evap_atoms_local_, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  return global;
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::evap_half(double dt_half)
{
  if ((update->ntimestep % nevery) != 0) return;

  // Snapshot nlocal before the loop so we don't process atoms spawned by
  // emit_into during this same call.
  const int nlocal = particle->nlocal;
  int *s2g = particle->mixture[imix]->species2group;
  int ndeleted = 0;

  for (int ip = 0; ip < nlocal; ip++) {
    // Refetch pointer each iter — emit_into may have realloc'd particles.
    Particle::OnePart *parts = particle->particles;
    const int is = parts[ip].ispecies;
    const int ig = s2g[is];
    if (ig < 0) continue;

    droplet_evaporation_model(ip, dt_half);

    parts = particle->particles;     // refresh in case spawn_evap reallocated
    const double m_cut = 0.1 * particle->species[is].mass;
    if (parts[ip].mass > 0.0 && parts[ip].mass <= m_cut) {
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
   from fix background. Rocket force uses -grad(Te) as the recoil axis.
------------------------------------------------------------------------- */
void FixDropletEvaporate::droplet_evaporation_model(int idrop,
                                        const double dt_half)
{
  const double AM   = 1.15225e-26;   // Li atom mass [kg] (6.94 amu)
  const double Rho  = 534.0;         // kg/m^3
  const double Cp   = 4200.0;        // J/kg-K
  const double DHm  = 1.47e+05;      // Li heat of vaporization [J/mol]
  const double AN   = 6.022e+23;     // 1/mol
  const double DT   = dt_half;

  // Snapshot droplet state. After spawn_evap_atoms() the Particle::particles
  // array may realloc — never read through ip after that. We refresh the
  // pointer at the end and write final state via index.
  Particle::OnePart *ip = &particle->particles[idrop];
  const double mass   = ip->mass;
  const double radius = ip->radius;
  const double TK     = ip->temp;
  const double xs[3]  = {ip->x[0], ip->x[1], ip->x[2]};
  const int icell_ip  = ip->icell;

  // R/Z at particle position (handles 2D Cart, 2D axi, 3D Cart via helper).
  double R = 0.0, Z = 0.0;
  OpenEdge::sparta_to_RZ(xs, domain->dimension, domain->axisymmetric, R, Z,
                         pd_->column_x0, pd_->column_y0);

  // Heat-flux vector at droplet position. When plasma.h5 carries q_par /
  // q_perp (mesh-level or regular grid), interp2D routes through the
  // appropriate path inside fix background. Otherwise read the default.
  double q_par  = pd_->default_q_par;
  double q_perp = pd_->default_q_perp;
  if (pd_->has_qheatflux) {
    // Always pass the regular-grid field handle: interp2D() routes to the
    // mesh-native counterpart via mesh_field_for() when plasma.h5 is
    // mesh-based. Passing the mesh_* array directly bypasses that routing
    // and falls through to the (empty) regular-grid path -> returns 0.
    if (!pd_->mesh_q_par.empty() || !pd_->q_par.empty())
      q_par  = pd_->interp2D(pd_->q_par,  R, Z, ip->icell);
    if (!pd_->mesh_q_perp.empty() || !pd_->q_perp.empty())
      q_perp = pd_->interp2D(pd_->q_perp, R, Z, ip->icell);
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
    if (!pd_->mesh_grad_te_r.empty() || !pd_->grad_te_r.empty())
      gTeR = pd_->interp2D(pd_->grad_te_r, R, Z, ip->icell);
    if (!pd_->mesh_grad_te_z.empty() || !pd_->grad_te_z.empty())
      gTeZ = pd_->interp2D(pd_->grad_te_z, R, Z, ip->icell);
  }

  if (Qs <= 0.0) {
    Particle::OnePart *ip_w = &particle->particles[idrop];
    ip_w->radius = radius;
    ip_w->temp   = TK;
    ip_w->mass   = mass;
    return;
  }

  // Antoine + Hertz-Knudsen flux.
  const double a1 = 5.055, b1 = -8023.0, xm1 = 6.939;
  const double vpres1 = 760.0 * std::pow(10.0, a1 + b1/TK);          // mmHg
  const double Gevap_atoms = 1.0e4 * 3.513e22 * vpres1 / std::sqrt(xm1 * TK);

  // Tally cumulative real Li atoms evaporated from this droplet over the
  // half-step. Each macro-particle = 1 real droplet (specwt=1 expected).
  evap_atoms_local_ += 4.0 * MY_PI * radius * radius * Gevap_atoms * DT;

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

  // Rocket-force kick (in-place velocity update via fresh pointer).
  if (rocket_eta > 0.0 && m_new > 0.0 && Gevap_atoms > 0.0) {
    const double grad_mag = std::sqrt(gTeR*gTeR + gTeZ*gTeZ);
    if (std::isfinite(grad_mag) && grad_mag > 0.0) {
      const double kB = 1.380649e-23;
      const double v_thermal = std::sqrt(8.0 * kB * TK / (MY_PI * AM));
      const double area0 = 4.0 * MY_PI * radius * radius;
      const double dmdt = area0 * Gevap_atoms * AM;
      const double a_mag = rocket_eta * dmdt * v_thermal / m_new;
      const double nr = -gTeR / grad_mag;
      const double nz = -gTeZ / grad_mag;

      double phi = 0.0;
      if (domain->dimension == 3) phi = std::atan2(xs[1], xs[0]);
      double dvx, dvy, dvz;
      OpenEdge::RZphi_force_to_sparta(a_mag * nr, a_mag * nz, 0.0,
                                       domain->dimension, domain->axisymmetric,
                                       phi, dvx, dvy, dvz);
      Particle::OnePart *ip_w = &particle->particles[idrop];
      ip_w->v[0] += dvx * DT;
      ip_w->v[1] += dvy * DT;
      ip_w->v[2] += dvz * DT;
    }
  }

  // Volumetric Li source: spawn evaporated atoms in the droplet's cell.
  // Must precede the final ip-> write (spawn may realloc the particle array).
  if (emit_imix >= 0 && Gevap_atoms > 0.0 && radius > 0.0) {
    const double area = 4.0 * MY_PI * radius * radius;
    spawn_evap_atoms(idrop, area, Gevap_atoms, TK, dt_half);
  }

  // Final state write through a fresh pointer (any spawn above may have
  // invalidated earlier pointers via Particle::particles realloc).
  Particle::OnePart *ip_w = &particle->particles[idrop];
  ip_w->radius = R_new;
  ip_w->temp   = T_new;
  ip_w->mass   = m_new;
}

/* ----------------------------------------------------------------------
   Spawn evaporated atoms in the droplet's cell. Called only when
   emit_imix >= 0. Lambda = area * Gevap_atoms * dt / fnum.
------------------------------------------------------------------------- */
void FixDropletEvaporate::spawn_evap_atoms(int idrop, double area,
                                            double Gevap_atoms, double TK,
                                            double dt_half)
{
  const double fnum = update->fnum;
  if (fnum <= 0.0) return;
  const double dN_phys = area * Gevap_atoms * dt_half;       // atoms / call
  const double lam     = dN_phys / fnum;                     // sim particles
  if (lam <= 0.0 || !std::isfinite(lam)) return;

  const int n_to_emit = sample_poisson(lam, random);
  if (n_to_emit == 0) return;

  // Pick species inside emit mixture by fraction; cumulative CDF.
  Mixture *mix = particle->mixture[emit_imix];
  const int    nsp_mix = mix->nspecies;
  const double *frac   = mix->fraction;
  const int    *spec   = mix->species;

  // Snapshot droplet position / cell — pointer may invalidate after first add.
  Particle::OnePart *ip_snap = &particle->particles[idrop];
  const double xs[3] = {ip_snap->x[0], ip_snap->x[1], ip_snap->x[2]};
  const int icell_ip = ip_snap->icell;
  if (icell_ip < 0) return;

  const double kB = 1.380649e-23;

  for (int k = 0; k < n_to_emit; ++k) {
    // Pick species via cumulative fraction.
    int isp = spec[0];
    if (nsp_mix > 1) {
      const double u = random->uniform();
      double cum = 0.0;
      for (int s = 0; s < nsp_mix; ++s) {
        cum += frac[s];
        if (u <= cum) { isp = spec[s]; break; }
      }
    }
    const double m_atom = particle->species[isp].mass;
    if (m_atom <= 0.0) continue;

    // 3D Maxwellian velocity at droplet T (sigma = sqrt(kT/m) per component).
    const double sigma = std::sqrt(kB * TK / m_atom);
    double u1 = std::max(random->uniform(), 1.0e-300);
    double u2 = random->uniform();
    double u3 = std::max(random->uniform(), 1.0e-300);
    double u4 = random->uniform();
    const double r12 = std::sqrt(-2.0 * std::log(u1));
    const double r34 = std::sqrt(-2.0 * std::log(u3));
    double v[3];
    v[0] = sigma * r12 * std::cos(2.0 * MY_PI * u2);
    v[1] = sigma * r12 * std::sin(2.0 * MY_PI * u2);
    v[2] = sigma * r34 * std::cos(2.0 * MY_PI * u4);

    double x[3] = {xs[0], xs[1], xs[2]};
    int newid = MAXSMALLINT * random->uniform();
    particle->add_particle(newid, isp, icell_ip, x, v, 0.0, 0.0);
  }
}
