/* ----------------------------------------------------------------------
   OpenEdge: fix droplet/charge — OML charging against background.
------------------------------------------------------------------------- */

#include "fix_droplet_charge.h"
#include "fix_background.h"
#include "grain_material.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "mixture.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
#include "input.h"
#include "modify.h"
#include "math_const.h"
#include "openedge_geom.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixDropletCharge::FixDropletCharge(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // fix ID droplet/charge Nevery background PD [keywords...]
  if (narg < 5)
    error->all(FLERR,
      "Illegal fix droplet/charge command "
      "(need: Nevery background PD)");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg++], "background") != 0)
    error->all(FLERR, "fix droplet/charge: argument 4 must be 'background'");
  plasma_fix_id_ = std::string(arg[iarg++]);

  while (iarg < narg) {
    if (strcmp(arg[iarg], "radius") == 0) {
      seed_radius = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "mass") == 0) {
      seed_mass = input->numeric(FLERR, arg[iarg+1]);
      if (seed_mass <= 0.0) error->all(FLERR, "fix droplet/charge: mass must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "ion_mass_amu") == 0) {
      ion_mass_amu = input->numeric(FLERR, arg[iarg+1]);
      if (ion_mass_amu <= 0.0)
        error->all(FLERR, "fix droplet/charge: ion_mass_amu must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "thermionic") == 0) {
      if      (strcmp(arg[iarg+1], "yes") == 0) thermionic_on = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) thermionic_on = 0;
      else error->all(FLERR, "fix droplet/charge: thermionic must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "richardson_A") == 0) {
      richardson_A = input->numeric(FLERR, arg[iarg+1]);
      if (richardson_A < 0.0)
        error->all(FLERR, "fix droplet/charge: richardson_A must be >= 0");
      ra_set_ = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "work_function_eV") == 0) {
      work_function_eV = input->numeric(FLERR, arg[iarg+1]);
      if (work_function_eV <= 0.0)
        error->all(FLERR, "fix droplet/charge: work_function_eV must be > 0");
      wf_set_ = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      seed_temp = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "mixture") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: missing mixture ID");
      imix = particle->find_mixture(arg[iarg+1]);
      if (imix < 0) error->all(FLERR, "fix droplet/charge: unknown mixture ID");
      iarg += 2;
    } else if (strcmp(arg[iarg], "material") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: missing material name");
      if (strlen(arg[iarg+1]) >= sizeof(mat_name_))
        error->all(FLERR, "fix droplet/charge: material name too long");
      strcpy(mat_name_, arg[iarg+1]);
      iarg += 2;
    } else {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix droplet/charge: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  // Register the per-particle "droplet_charge" custom DOUBLE vector here
  // (constructor) so it exists at dump-command parse time.
  qcustom = particle->find_custom((char *) "droplet_charge");
  if (qcustom < 0)
    qcustom = particle->add_custom((char *) "droplet_charge", 1, 0);
}

/* ---------------------------------------------------------------------- */

FixDropletCharge::~FixDropletCharge() {}

int FixDropletCharge::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::init()
{
  const int ifix = modify->find_fix(plasma_fix_id_.c_str());
  if (ifix < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "fix droplet/charge: background fix '%s' not found",
             plasma_fix_id_.c_str());
    error->all(FLERR, msg);
  }
  pd_ = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd_)
    error->all(FLERR,
      "fix droplet/charge: background fix must be style background");
  pd_->init();

  // Optional grain material: supplies thermionic work function and
  // Richardson constant unless the explicit keywords overrode them.
  if (mat_name_[0]) {
    mat_ = grain_material_find(mat_name_);
    if (!mat_)
      error->all(FLERR, "fix droplet/charge: unknown material");
    if (!wf_set_ && mat_->work_function_eV > 0.0)
      work_function_eV = mat_->work_function_eV;
    if (!ra_set_ && mat_->richardson_A > 0.0)
      richardson_A = mat_->richardson_A;
  }
  if (!random_) {
    random_ = new RanKnuth(update->ranmaster->uniform());
    random_->reset(comm->me + 1, comm->me, 100);
  }

  if (qcustom < 0)
    qcustom = particle->find_custom((char *) "droplet_charge");
}

void FixDropletCharge::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  apply_charge_update();
}

void FixDropletCharge::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  apply_charge_update();
}

double FixDropletCharge::memory_usage() { return 0.0; }

/* ---------------------------------------------------------------------- */

bool FixDropletCharge::solve_phi_oml(double Te_eV, double Ti_eV,
                                     double ne_m3, double ni_m3,
                                     double Td_K, double rd_m,
                                     double &phi_V) const
{
  if (!(Te_eV > 0.0) || !(Ti_eV > 0.0) || !(ne_m3 > 0.0) || !(ni_m3 > 0.0)
      || !(rd_m > 0.0))
    return false;

  const double qe = update->echarge;
  const double me = update->electron_mass;
  const double mi = ion_mass_amu * update->proton_mass;
  const double kB = update->boltz;

  const double area_coll = MY_PI * rd_m * rd_m;
  const double area_emit = 4.0 * MY_PI * rd_m * rd_m;

  const double Ce = qe * area_coll * ne_m3 *
                    std::sqrt(8.0 * (Te_eV * qe) / (MY_PI * me));
  const double Ci = qe * area_coll * ni_m3 *
                    std::sqrt(8.0 * (Ti_eV * qe) / (MY_PI * mi));

  auto thermionic_current = [&]() -> double {
    if (!thermionic_on) return 0.0;
    if (!(Td_K > 0.0) || richardson_A <= 0.0 || work_function_eV <= 0.0) return 0.0;
    const double eps0 = 8.8541878128e-12;
    const double W_inf_J = work_function_eV * qe;
    const double dW = (qe * qe) / (16.0 * MY_PI * eps0 * rd_m);
    const double W_rd = std::max(0.0, W_inf_J - dW);
    return area_emit * richardson_A * Td_K * Td_K
           * std::exp(-W_rd / (kB * Td_K));
  };
  const double Ith = thermionic_current();

  auto f = [&](double phi) -> double {
    const double Ie = -Ce * std::exp(phi / Te_eV);
    const double Ii =  Ci * (1.0 - phi / Ti_eV);
    return Ie + Ii + Ith;
  };

  double lo = -80.0 * std::max(Te_eV, Ti_eV);
  double hi =  20.0 * std::max(Te_eV, Ti_eV);
  double flo = f(lo);
  double fhi = f(hi);

  int expand = 0;
  while (std::isfinite(flo) && std::isfinite(fhi)
         && flo * fhi > 0.0 && expand < 12) {
    lo *= 2.0;
    hi *= 2.0;
    flo = f(lo);
    fhi = f(hi);
    ++expand;
  }
  if (!std::isfinite(flo) || !std::isfinite(fhi) || flo * fhi > 0.0)
    return false;

  double mid = 0.5 * (lo + hi);
  for (int it = 0; it < 100; ++it) {
    mid = 0.5 * (lo + hi);
    const double fm = f(mid);
    if (!std::isfinite(fm)) return false;
    if (std::fabs(fm) <= 1.0e-10 * std::max(1.0, std::fabs(Ci + Ith))) break;
    if (flo * fm <= 0.0) { hi = mid; fhi = fm; }
    else                 { lo = mid; flo = fm; }
    if (std::fabs(hi - lo) <= 1.0e-10 * std::max(1.0, std::fabs(mid))) break;
  }

  phi_V = mid;
  return std::isfinite(phi_V);
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::apply_charge_update()
{
  const double eps0 = 8.8541878128e-12;
  const double qe   = update->echarge;
  const int    dim  = domain->dimension;
  const int    axi  = domain->axisymmetric;

  auto *parts = particle->particles;
  const int nlocal = particle->nlocal;

  int *s2g = (imix >= 0) ? particle->mixture[imix]->species2group : nullptr;

  if (qcustom < 0) return;
  double *qvec = particle->edvec[particle->ewhich[qcustom]];

  for (int ip = 0; ip < nlocal; ++ip) {
    Particle::OnePart &p = parts[ip];
    if (s2g && s2g[p.ispecies] < 0) continue;

    if (p.mass   <= 0.0 && seed_mass   > 0.0) p.mass   = seed_mass;
    if (p.radius <= 0.0 && seed_radius > 0.0) p.radius = seed_radius;
    if (p.temp   <= 0.0 && seed_temp   > 0.0) p.temp   = seed_temp;

    const double rd   = (p.radius > 0.0) ? p.radius : seed_radius;
    const double Td_K = (p.temp   > 0.0) ? p.temp   : seed_temp;
    if (!(rd > 0.0)) continue;

    double R = 0.0, Z = 0.0;
    OpenEdge::sparta_to_RZ(p.x, dim, axi, R, Z,
                           pd_->column_x0, pd_->column_y0);
    const double Te = std::max(pd_->interp2D(pd_->temp_e, R, Z, p.icell), 0.0);
    const double Ti = std::max(pd_->interp2D(pd_->temp_i, R, Z, p.icell), 0.0);
    const double Ne = std::max(pd_->interp2D(pd_->dens_e, R, Z, p.icell), 0.0);
    const double Ni = std::max(pd_->interp2D(pd_->dens_i, R, Z, p.icell), 0.0);
    if (!(Te > 0.0) || !(Ti > 0.0) || !(Ne > 0.0) || !(Ni > 0.0)) continue;

    double phi_s = 0.0;
    if (!solve_phi_oml(Te, Ti, Ne, Ni, Td_K, rd, phi_s)) continue;

    const double qd_coulomb = 4.0 * MY_PI * eps0 * rd * phi_s;
    const double zd         = qd_coulomb / qe;
    if (!std::isfinite(zd)) continue;

    qvec[ip] = zd;

    // Electrostatic disruption (DUSTT): critical potential
    // phi* = beta * sqrt(F_t[dyne/cm^2]) * R_d[um] volts. Solids only
    // (tensile_Pa = 0 disables; e.g. liquid Li). Splits are deferred to
    // after the loop: add_particle reallocates edvec/particles.
    if (mat_ && mat_->tensile_Pa > 0.0) {
      const double phistar = breakup_beta_
        * std::sqrt(10.0 * mat_->tensile_Pa) * (rd * 1.0e6);
      if (std::fabs(phi_s) > phistar) split_list_.push_back(ip);
    }
  }

  if (!split_list_.empty()) {
    const int nw_custom = particle->find_custom((char *) "grain_nweight");
    const double gamma = std::pow(2.0, -1.0/3.0);
    for (int ip : split_list_) {
      Particle::OnePart &p = particle->particles[ip];
      if (!(p.radius > 0.0) || !(p.mass > 0.0)) continue;
      // fragment kinematics: two equal halves, gamma = 2^(-1/3) radius
      // (exactly mass-conserving), small isotropic separation kick
      double kick[3];
      const double th = 2.0 * MY_PI * random_->uniform();
      kick[0] = 0.1 * std::cos(th); kick[1] = 0.1 * std::sin(th); kick[2] = 0.0;
      double xnew[3] = {p.x[0], p.x[1], p.x[2]};
      double vnew[3] = {p.v[0] + kick[0], p.v[1] + kick[1], p.v[2] + kick[2]};
      const double m_half = 0.5 * p.mass;
      const double r_frag = gamma * p.radius;
      const double t_frag = p.temp;
      const int    isp    = p.ispecies;
      const int    icell  = p.icell;
      int newid = MAXSMALLINT * random_->uniform();
      particle->add_particle(newid, isp, icell, xnew, vnew, 0.0, 0.0);
      if (modify->n_update_custom) {
        double zv[3] = {0.0, 0.0, 0.0};
        modify->update_custom(particle->nlocal - 1, 0.0, 0.0, 0.0, zv);
      }
      // re-fetch everything after add_particle (reallocation)
      Particle::OnePart &pn = particle->particles[particle->nlocal - 1];
      Particle::OnePart &po = particle->particles[ip];
      pn.radius = r_frag; pn.mass = m_half; pn.temp = t_frag;
      po.radius = r_frag; po.mass = m_half;
      po.v[0] -= kick[0]; po.v[1] -= kick[1]; po.v[2] -= kick[2];
      double *qv = particle->edvec[particle->ewhich[qcustom]];
      qv[particle->nlocal - 1] = 0.5 * qv[ip];
      qv[ip] *= 0.5;
      if (nw_custom >= 0) {
        double *nwv = particle->edvec[particle->ewhich[nw_custom]];
        nwv[particle->nlocal - 1] = nwv[ip];
      }
      ++nbreak_;
    }
    split_list_.clear();
  }
}
