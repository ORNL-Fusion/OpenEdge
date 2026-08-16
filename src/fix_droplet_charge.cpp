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
    } else if (strcmp(arg[iarg], "fixed_charge") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: fixed_charge <Zd>");
      fixed_zd_ = input->numeric(FLERR, arg[iarg+1]);
      fixed_mode_ = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "validity") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: validity warn|error|no");
      if      (strcmp(arg[iarg+1], "warn")  == 0) validity_mode = 1;
      else if (strcmp(arg[iarg+1], "error") == 0) validity_mode = 2;
      else if (strcmp(arg[iarg+1], "no")    == 0) validity_mode = 0;
      else error->all(FLERR, "fix droplet/charge: validity warn|error|no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "see") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: see yes|no");
      if      (strcmp(arg[iarg+1], "yes") == 0) see_on = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) see_on = 0;
      else error->all(FLERR, "fix droplet/charge: see must be yes or no");
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

  // Register the per-particle "particulate_charge" custom DOUBLE vector here
  // (constructor) so it exists at dump-command parse time.
  qcustom = particle->find_custom((char *) "particulate_charge");
  if (qcustom < 0)
    qcustom = particle->add_custom((char *) "particulate_charge", 1, 0);
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
  if (see_on && comm->me == 0)
    error->warning(FLERR, "fix droplet/charge: see yes is EXPERIMENTAL "
                   "(Sternglass electron-impact channel only; no IIEE/"
                   "backscattering, no positive-potential branch, no "
                   "heat-balance coupling)");
  if (!random_) {
    random_ = new RanKnuth(update->ranmaster->uniform());
    random_->reset(comm->me + 1, comm->me, 100);
  }

  if (qcustom < 0)
    qcustom = particle->find_custom((char *) "particulate_charge");
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
                                     double u_mach, double &phi_V) const
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

  // Secondary electron emission (see yes): Sternglass yield
  //   delta(E) = 7.4 delta_m (E/E_m) exp(-2 sqrt(E/E_m))
  // flux-averaged over the incident Maxwellian, <delta>(Te) =
  //   int delta(E) E e^{-E/Te} dE / Te^2  (64-pt trapezoid, 0..30 Te).
  // Emitted secondaries leave the grain: electron collection scales by
  // (1 - <delta>), clamped >= 0 (space-charge-limited regime is out of
  // scope, as is the positive-potential branch it can drive).
  double see_frac = 0.0;
  if (see_on && mat_ && mat_->see_delta_m > 0.0 && mat_->see_E_m_eV > 0.0) {
    const double dm = mat_->see_delta_m, em = mat_->see_E_m_eV;
    const int NQ = 256;
    const double emax = 30.0 * Te_eV, de = emax / NQ;
    double num = 0.0;
    for (int k = 0; k < NQ; k++) {
      const double E = (k + 0.5) * de;
      num += 7.4 * dm * (E / em) * std::exp(-2.0 * std::sqrt(E / em))
             * E * std::exp(-E / Te_eV) * de;
    }
    see_frac = std::min(num / (Te_eV * Te_eV), 1.0);
    if (see_frac > 0.95 && !see_sat_warned_) {
      see_sat_warned_ = 1;
      if (comm->me == 0)
        error->warning(FLERR, "fix droplet/charge: SEE yield near/above "
                       "unity — the negative-potential OML branch is "
                       "outside its validity range (positive-potential/"
                       "space-charge model not implemented)");
    }
  }

  // DUSTT flow-dependent OML ion collection [Pigarov PoP 12, 122508,
  // Eq. 4]: J_i = n_i sigma_d e v_Ti F_G(u, a), a = -phi/Ti,
  //   F_G = [u + 1/(2u) + a/u] erf(u) + exp(-u^2)/sqrt(pi)
  // (normalized so F_G(0,a) = 2(1+a)/sqrt(pi): the stationary limit
  // reproduces the classic Ci*(1 - phi/Ti) exactly). Below U_SMALL the
  // closed form cancels catastrophically — use the analytic limit.
  const double Ci_f = qe * area_coll * ni_m3 *
                      std::sqrt(2.0 * (Ti_eV * qe) / mi);
  auto f_gamma = [](double u, double a) -> double {
    const double sqrt_pi = std::sqrt(MY_PI);
    const double U_SMALL = 1.0e-3;
    if (u < U_SMALL) return 2.0 * (1.0 + a) / sqrt_pi;
    return (u + 0.5/u + a/u) * std::erf(u)
           + std::exp(-u*u) / sqrt_pi;
  };
  auto f = [&](double phi) -> double {
    const double Ie = -Ce * (1.0 - see_frac) * std::exp(phi / Te_eV);
    const double Ii =  Ci_f * std::max(f_gamma(u_mach, -phi / Ti_eV), 0.0);
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
    if (std::fabs(fm) <= 1.0e-10 * std::max(1.0, std::fabs(Ci_f + Ith))) break;
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

    // fixed_charge mode: stamp and skip the OML solve entirely (vacuum
    // force tests; charging-sensitivity studies)
    if (fixed_mode_) { qvec[ip] = fixed_zd_; continue; }

    double R = 0.0, Z = 0.0;
    OpenEdge::sparta_to_RZ(p.x, dim, axi, R, Z,
                           pd_->column_x0, pd_->column_y0);
    const double Te = std::max(pd_->interp2D(pd_->temp_e, R, Z, p.icell), 0.0);
    const double Ti = std::max(pd_->interp2D(pd_->temp_i, R, Z, p.icell), 0.0);
    const double Ne = std::max(pd_->interp2D(pd_->dens_e, R, Z, p.icell), 0.0);
    const double Ni = std::max(pd_->interp2D(pd_->dens_i, R, Z, p.icell), 0.0);
    if (!(Te > 0.0) || !(Ti > 0.0) || !(Ne > 0.0) || !(Ni > 0.0)) continue;

    // DUSTT Mach number for the flow-dependent ion current:
    // u = |V_i - v| / v_Ti with V_i the parallel flow along bhat
    double u_mach = 0.0;
    {
      const double Vpar = pd_->interp2D(pd_->parr_flow, R, Z, p.icell);
      double Br = 0.0, Bz = 0.0, Bt = 0.0;
      if (pd_->has_bfield || !pd_->mesh_tri_br.empty())
        pd_->bfield_at(R, Z, Br, Bz, Bt, p.icell, ip);
      const double Bn = std::sqrt(Br*Br + Bt*Bt + Bz*Bz);
      double up[3] = {0.0, 0.0, 0.0};
      if (Bn > 1.0e-12) {
        double phit = 0.0;
        if (dim == 3) phit = std::atan2(p.x[1], p.x[0]);
        OpenEdge::RZphi_force_to_sparta(Vpar*(Br/Bn), Vpar*(Bz/Bn),
                                        Vpar*(Bt/Bn), dim, axi, phit,
                                        up[0], up[1], up[2]);
      }
      const double vti = std::sqrt(2.0 * (Ti * qe) /
                                   (ion_mass_amu * update->proton_mass));
      if (vti > 0.0) {
        const double d0 = p.v[0]-up[0], d1 = p.v[1]-up[1], d2 = p.v[2]-up[2];
        u_mach = std::sqrt(d0*d0 + d1*d1 + d2*d2) / vti;
      }
    }
    double phi_s = 0.0;
    if (!solve_phi_oml(Te, Ti, Ne, Ni, Td_K, rd, u_mach, phi_s)) continue;

    const double qd_coulomb = 4.0 * MY_PI * eps0 * rd * phi_s;
    const double zd         = qd_coulomb / qe;
    if (!std::isfinite(zd)) continue;

    qvec[ip] = zd;

    // OML validity monitor (Referee 1: OML needs R_d << lambda_D,
    // rho_e). Tracks the worst R_d/lambda_D and R_d/rho_e seen; warn
    // fires once per run at R_d > lambda_D, error aborts. No silent
    // OML/probe blending — a probe-limited model is the planned upgrade.
    if (validity_mode > 0) {
      const double eps0v = 8.8541878128e-12;
      const double lamD = std::sqrt(eps0v * (Te * qe) / (Ne * qe * qe));
      const double rl = rd / lamD;
      if (rl > val_max_rl_) val_max_rl_ = rl;
      double Brv = 0.0, Bzv = 0.0, Btv = 0.0;
      if (pd_->has_bfield || !pd_->mesh_tri_br.empty())
        pd_->bfield_at(R, Z, Brv, Bzv, Btv, p.icell, ip);
      const double Bm = std::sqrt(Brv*Brv + Bzv*Bzv + Btv*Btv);
      if (Bm > 0.0) {
        const double rho_e = std::sqrt((Te * qe) * update->electron_mass)
                             / (qe * Bm);
        const double rr = rd / rho_e;
        if (rr > val_max_re_) val_max_re_ = rr;
      }
      if (rl > 1.0) {
        val_nviol_++;
        if (validity_mode == 2)
          error->one(FLERR, "fix droplet/charge: OML validity violated "
                     "(R_d > lambda_D; validity error requested)");
        if (!val_warned_ && comm->me == 0) {
          val_warned_ = 1;
          error->warning(FLERR, "fix droplet/charge: R_d > lambda_D — "
                         "OML charging is outside its validity range "
                         "(probe-limited model not yet implemented)");
        }
      }
    }

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
