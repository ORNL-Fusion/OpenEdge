/* ----------------------------------------------------------------------
   OpenEdge: fix droplet/charge — OML charging against background.
------------------------------------------------------------------------- */

#include "fix_droplet_charge.h"
#include "fix_background.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
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
      iarg += 2;
    } else if (strcmp(arg[iarg], "work_function_eV") == 0) {
      work_function_eV = input->numeric(FLERR, arg[iarg+1]);
      if (work_function_eV <= 0.0)
        error->all(FLERR, "fix droplet/charge: work_function_eV must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      seed_temp = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix droplet/charge: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }
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

  // Per-species accumulated mean charge (avoids last-particle-wins behavior).
  std::vector<double>    zsum(particle->nspecies, 0.0);
  std::vector<long long> zcount(particle->nspecies, 0);

  for (int ip = 0; ip < nlocal; ++ip) {
    Particle::OnePart &p = parts[ip];

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

    const int is = p.ispecies;
    if (is >= 0 && is < particle->nspecies) {
      zsum[is]   += zd;
      zcount[is] += 1;
    }
  }

  // Aggregate mean species charges across ranks and write back.
  std::vector<double>    zsum_g(zsum.size(), 0.0);
  std::vector<long long> zcount_g(zcount.size(), 0);
  if (!zsum.empty()) {
    MPI_Allreduce(zsum.data(),   zsum_g.data(),
                  static_cast<int>(zsum.size()),   MPI_DOUBLE,     MPI_SUM, world);
    MPI_Allreduce(zcount.data(), zcount_g.data(),
                  static_cast<int>(zcount.size()), MPI_LONG_LONG,  MPI_SUM, world);
  }
  for (int is = 0; is < particle->nspecies; ++is) {
    const long long n = zcount_g[is];
    if (n > 0) particle->species[is].charge = zsum_g[is] / static_cast<double>(n);
  }
}
