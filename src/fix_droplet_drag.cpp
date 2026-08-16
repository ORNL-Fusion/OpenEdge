/* ----------------------------------------------------------------------
   OpenEdge: fix drag — Epstein / Coulomb drag on droplets.
   Background plasma at particle position pulled from fix background.
------------------------------------------------------------------------- */

#include "fix_droplet_drag.h"
#include "fix_background.h"
#include "grain_material.h"
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

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixDropletDrag::FixDropletDrag(SPARTA *sparta, int narg, char **arg)
    : Fix(sparta, narg, arg)
{
  // fix ID drag Nevery A_bg Z_bg background PD [keywords...]
  if (narg < 7)
    error->all(FLERR,
      "Illegal fix drag command "
      "(need: Nevery A_bg Z_bg background PD)");

  int iarg = 2;
  nevery       = input->inumeric(FLERR, arg[iarg++]);
  A_background = input->numeric (FLERR, arg[iarg++]);
  Z_background = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg++], "background") != 0)
    error->all(FLERR, "fix drag: argument 6 must be 'background'");
  plasma_fix_id_ = std::string(arg[iarg++]);

  while (iarg < narg) {
    if (strcmp(arg[iarg], "gravity") == 0) {
      if (iarg + 3 >= narg)
        error->all(FLERR, "fix drag: gravity requires gx gy gz");
      g_input_[0] = input->numeric(FLERR, arg[iarg+1]);
      g_input_[1] = input->numeric(FLERR, arg[iarg+2]);
      g_input_[2] = input->numeric(FLERR, arg[iarg+3]);
      use_gravity = true;
      iarg += 4;
    } else if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: model needs a value");
      if (strcmp(arg[iarg+1], "dustt2005") != 0 &&
          strcmp(arg[iarg+1], "dustt") != 0)
        error->all(FLERR,
          "fix drag: model must be 'dustt2005' (DUSTT collection+Coulomb "
          "kinetic drag, Pigarov et al. 2005)");
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/chi") == 0) {
      chi_coulomb = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/delta") == 0) {
      delta_ite = input->numeric(FLERR, arg[iarg+1]);
      if (delta_ite <= 0.0) error->all(FLERR, "fix drag: coulomb/delta must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/lnlambda") == 0) {
      ln_lambda_coulomb = input->numeric(FLERR, arg[iarg+1]);
      if (ln_lambda_coulomb < 0.0)
        error->all(FLERR, "fix drag: coulomb/lnlambda must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "mass") == 0) {
      seed_mass = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "radius") == 0) {
      seed_radius = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      seed_temp = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "mixture") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing mixture ID");
      imix = particle->find_mixture(arg[iarg+1]);
      if (imix < 0) error->all(FLERR, "fix drag: unknown mixture ID");
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/self") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: coulomb/self yes|no");
      if      (strcmp(arg[iarg+1], "yes") == 0) self_consistent_ = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) self_consistent_ = 0;
      else error->all(FLERR, "fix drag: coulomb/self must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "neutrals") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: neutrals yes|no");
      if      (strcmp(arg[iarg+1], "yes") == 0) neutrals_on_ = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) neutrals_on_ = 0;
      else error->all(FLERR, "fix drag: neutrals must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "efield") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: efield yes|no");
      if      (strcmp(arg[iarg+1], "yes") == 0) efield_on_ = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) efield_on_ = 0;
      else error->all(FLERR, "fix drag: efield must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "material") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing material name");
      if (strlen(arg[iarg+1]) >= sizeof(mat_name_))
        error->all(FLERR, "fix drag: material name too long");
      strcpy(mat_name_, arg[iarg+1]);
      iarg += 2;
    } else {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix drag: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }
}

/* ---------------------------------------------------------------------- */

FixDropletDrag::~FixDropletDrag() {}

int FixDropletDrag::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDropletDrag::init()
{
  if (use_gravity) {
    for (int k = 0; k < modify->nfix; ++k)
      if (strcmp(modify->fix[k]->style, "gravity") == 0)
        error->all(FLERR,
          "fix drag: do not combine 'fix gravity' with 'gravity' inside fix drag");
  }

  const int ifix = modify->find_fix(plasma_fix_id_.c_str());
  if (ifix < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "fix drag: background fix '%s' not found",
             plasma_fix_id_.c_str());
    error->all(FLERR, msg);
  }
  pd_ = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd_)
    error->all(FLERR, "fix drag: background fix must be style background");

  dq_custom_ = particle->find_custom((char *) "particulate_charge");
  if (self_consistent_ && dq_custom_ < 0 && comm->me == 0)
    error->warning(FLERR, "fix drag: coulomb/self without fix grain/charge - "
                   "chi falls back to 0 (collection drag only)");

  // Optional grain material: overrides the legacy hardcoded Li density
  // used in the Epstein frequency (rho_d = 534 kg/m^3).
  if (mat_name_[0]) {
    const GrainMaterial *mat = grain_material_find(mat_name_);
    if (!mat || mat->rho <= 0.0)
      error->all(FLERR, "fix drag: unknown material or material has no rho");
    rho_d = mat->rho;
  }
  pd_->init();
}

/* ---------------------------------------------------------------------- */

void FixDropletDrag::start_of_step()
{
  if (update->ntimestep % nevery) return;
  kick_half(0.5 * update->dt);
}

void FixDropletDrag::end_of_step()
{
  if (update->ntimestep % nevery) return;
  kick_half(0.5 * update->dt);
}

/* ---------------------------------------------------------------------- */

void FixDropletDrag::kick_half(double dt_half)
{
  const int nlocal = particle->nlocal;
  Particle::OnePart * const parts = particle->particles;

  const double mi       = A_background * update->proton_mass;
  const int    dim      = domain->dimension;
  const int    axisym   = domain->axisymmetric;

  int *s2g = (imix >= 0) ? particle->mixture[imix]->species2group : nullptr;

  for (int ip = 0; ip < nlocal; ++ip) {
    Particle::OnePart &p = parts[ip];
    if (s2g && s2g[p.ispecies] < 0) continue;

    if (seed_mass   > 0.0 && p.mass   <= 0.0) p.mass   = seed_mass;
    if (seed_radius > 0.0 && p.radius <= 0.0) p.radius = seed_radius;
    if (seed_temp   > 0.0 && p.temp   <= 0.0) p.temp   = seed_temp;

    // R/Z at particle for plasma queries.
    double R = 0.0, Z = 0.0;
    OpenEdge::sparta_to_RZ(p.x, dim, axisym, R, Z);

    const double Ti_eV = std::max(pd_->interp2D(pd_->temp_i, R, Z, p.icell), 0.0);
    const double Ni    = std::max(pd_->interp2D(pd_->dens_i, R, Z, p.icell), 0.0);
    const double Vpar  =          pd_->interp2D(pd_->parr_flow, R, Z, p.icell);
    double Br = 0.0, Bz = 0.0, Bt = 0.0;
    if (pd_->has_bfield || !pd_->mesh_tri_br.empty())
      pd_->bfield_at(R, Z, Br, Bz, Bt, p.icell, ip);

    const double rd = p.radius;
    double nuE = 0.0;
    double upar[3] = {0.0, 0.0, 0.0};

    if (rd > 0.0 && Ni > 0.0 && Ti_eV > 0.0) {
      // DUSTT ion-drag base rate: F_Epstein = m_i n_i v_Ti (V_i - v) sigma_d
      // with v_Ti = sqrt(2 T_i / m_i) [Pigarov PoP 12, 122508, Eq. 16]
      nuE = ion_drag_nu(Ni, Ti_eV, rd);

      const double Bn = std::sqrt(Br*Br + Bt*Bt + Bz*Bz);
      if (Bn > 1.0e-12) {
        // Parallel flow vector in cylindrical (R, Z, t), then to SPARTA slots.
        const double Vr = Vpar * (Br / Bn);
        const double Vz = Vpar * (Bz / Bn);
        const double Vt = Vpar * (Bt / Bn);
        double phi = 0.0;
        if (dim == 3) phi = std::atan2(p.x[1], p.x[0]);
        OpenEdge::RZphi_force_to_sparta(Vr, Vz, Vt, dim, axisym, phi,
                                         upar[0], upar[1], upar[2]);
      }

      if (nuE > 0.0) {
        // u = |V_i - v| / v_Ti (DUSTT Mach number, v_Ti = sqrt(2T_i/m_i))
        const double vth_i = std::sqrt(2.0 * (Ti_eV * update->echarge) / mi);
        if (vth_i > 0.0) {
          const double dv0 = p.v[0] - upar[0];
          const double dv1 = p.v[1] - upar[1];
          const double dv2 = p.v[2] - upar[2];
          const double u   = std::sqrt(dv0*dv0 + dv1*dv1 + dv2*dv2) / vth_i;
          if (!self_consistent_) {
            nuE *= coulomb_multiplier(u);
          } else {
            // Per-particle chi/delta/lnLambda from the OML charge and the
            // local plasma (DUSTT ion-drag closure, Hutchinson-fit lnL).
            const double QE   = update->echarge;
            const double EPS0 = 8.8541878128e-12;
            const double Te_eV = std::max(pd_->interp2D(pd_->temp_e, R, Z, p.icell), 0.0);
            const double ne    = std::max(pd_->interp2D(pd_->dens_e, R, Z, p.icell), 0.0);
            double chi = 0.0;
            if (dq_custom_ >= 0 && Te_eV > 0.0) {
              const int ew = particle->ewhich[dq_custom_];
              if (ew >= 0) {
                const double Zd   = particle->edvec[ew][ip];
                const double r_c  = std::max(rd, 1.0e-9);
                const double phiV = Zd * QE / (4.0 * MY_PI * EPS0 * r_c);
                if (std::isfinite(phiV) && phiV < 0.0)
                  chi = std::min(-phiV / Te_eV, 20.0);
              }
            }
            const double delta = (Te_eV > 0.0)
              ? std::max(Ti_eV / Te_eV, 1.0e-6) : 1.0;
            double lnlam = 0.0;
            if (ne > 0.0 && Te_eV > 0.0 && chi > 0.0) {
              const double TiJ  = Ti_eV * QE;
              const double TeJ  = Te_eV * QE;
              const double mve2 = TiJ * (3.0 + 2.0*u*u);
              const double b90  = rd * chi * TeJ / mve2;
              const double lamD = std::sqrt(EPS0 * TeJ / (ne * QE * QE));
              const double lams = lamD / std::sqrt(1.0 + 3.0*TeJ/mve2);
              const double eta  = 1.0 + (rd/lams)
                                  * (1.0 + std::sqrt(Te_eV/(6.0*Ti_eV)));
              lnlam = 0.5 * std::log((b90*b90 + (eta*lams)*(eta*lams))
                                     / (b90*b90 + rd*rd));
              if (!(lnlam > 0.0)) lnlam = 0.0;
            }
            nuE *= coulomb_multiplier(u, chi, delta, lnlam);
          }
        } else nuE = 0.0;
      }
    }

    // DUSTT neutral friction [Eq. 17]: stationary Maxwellian neutrals,
    //   F_n = zeta_n(s) m_n n_n v_Tn (V_n - v) sigma_d,  V_n = 0,
    //   zeta_n = {[1+s^2-(2s)^-2]erf(s) + [s+(2s)^-1]e^{-s^2}/sqrt(pi)}/s
    // zeta_n(0) = 8/(3 sqrt(pi)) (Epstein specular limit), -> s ram.
    // Auto-off when the background carries no neutral data.
    double nuN = 0.0;
    if (neutrals_on_ && rd > 0.0) {
      const double Nn = std::max(pd_->interp2D(pd_->dens_n, R, Z, p.icell), 0.0);
      const double Tn = std::max(pd_->interp2D(pd_->temp_n, R, Z, p.icell), 0.0);
      if (Nn > 0.0 && Tn > 0.0 && rho_d > 0.0) {
        const double vtn = std::sqrt(2.0 * (Tn * update->echarge) / mi);
        const double s = std::sqrt(p.v[0]*p.v[0] + p.v[1]*p.v[1] +
                                   p.v[2]*p.v[2]) / vtn;
        const double ZN0 = 8.0 / (3.0 * std::sqrt(MY_PI));
        double zn;
        if (s < 1.0e-3) zn = ZN0;
        else {
          const double e2 = std::exp(-s*s), er = std::erf(s);
          zn = ((1.0 + s*s - 0.25/(s*s)) * er
                + (s + 0.5/s) * e2 / std::sqrt(MY_PI)) / s;
        }
        nuN = zn * 0.75 * (Nn * mi * vtn) / (rho_d * rd);
      }
    }

    // Gravity — convert Cartesian user-input to SPARTA slot order via helper.
    double g0 = 0.0, g1 = 0.0, g2 = 0.0;
    if (use_gravity) {
      if (dim == 2 && !axisym) {
        // 2D Cart: x=R, y=Z, z=phi
        g0 = g_input_[0]; g1 = g_input_[1]; g2 = g_input_[2];
      } else if (dim == 2 && axisym) {
        // True axi: x=Z, y=R, z=phi. User inputs gravity in Cartesian R/Z/phi,
        // so g_input_[0]=gR, g_input_[1]=gZ, g_input_[2]=gphi → slots (gZ, gR, gphi).
        g0 = g_input_[1]; g1 = g_input_[0]; g2 = g_input_[2];
      } else {
        // 3D Cartesian
        g0 = g_input_[0]; g1 = g_input_[1]; g2 = g_input_[2];
      }
    }

    // Electric force (DUSTT F_E = Z_d e E): folded into the effective
    // constant acceleration so the exact exponential integrator handles
    // drag + gravity + qE together. F_E/F_g ~ 1/R_d^2 — negligible at
    // 50 um, dominant for ablated-down sub-um grains.
    if (efield_on_ && dq_custom_ >= 0 && p.mass > 0.0) {
      const int ew = particle->ewhich[dq_custom_];
      const double zd = (ew >= 0) ? particle->edvec[ew][ip] : 0.0;
      if (zd != 0.0) {
        double ER = 0.0, EZ = 0.0, Et = 0.0;
        if (pd_->query_efield_at_point(p.x, ER, EZ, Et, p.icell, ip)) {
          const double qom = zd * update->echarge / p.mass;
          double phi = 0.0;
          if (dim == 3) phi = std::atan2(p.x[1], p.x[0]);
          double aE0, aE1, aE2;
          OpenEdge::RZphi_force_to_sparta(qom*ER, qom*EZ, qom*Et,
                                          dim, axisym, phi, aE0, aE1, aE2);
          g0 += aE0; g1 += aE1; g2 += aE2;
        }
      }
    }

    // Combined ion+neutral relaxation: dv/dt = -(nu_i+nu_n)(v - u_eff)
    // + g with u_eff = nu_i u_i / (nu_i + nu_n) (neutrals drag toward
    // rest) — one exact exponential integrator for both channels.
    const double nuT = ((nuE > 0.0 && std::isfinite(nuE)) ? nuE : 0.0)
                     + ((nuN > 0.0 && std::isfinite(nuN)) ? nuN : 0.0);
    if (nuT > 0.0) {
      const double wi  = (nuE > 0.0 && std::isfinite(nuE)) ? nuE / nuT : 0.0;
      const double ue0 = wi * upar[0], ue1 = wi * upar[1], ue2 = wi * upar[2];
      const double s   = nuT * dt_half;
      const double ex  = (std::fabs(s) < 1.0e-8)
                         ? (1.0 - s + 0.5*s*s)
                         : std::exp(-s);
      const double inv = 1.0 / nuT;
      p.v[0] = ue0 + (p.v[0] - ue0 - g0*inv)*ex + g0*inv;
      p.v[1] = ue1 + (p.v[1] - ue1 - g1*inv)*ex + g1*inv;
      p.v[2] = ue2 + (p.v[2] - ue2 - g2*inv)*ex + g2*inv;
    } else if (g0 != 0.0 || g1 != 0.0 || g2 != 0.0) {
      // no drag: plain kick from the accumulated constant acceleration
      // (gravity AND electric — a charged grain in a plasma-free region
      // must still feel Z_d e E)
      p.v[0] += g0 * dt_half;
      p.v[1] += g1 * dt_half;
      p.v[2] += g2 * dt_half;
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixDropletDrag::ion_drag_nu(double Ni, double Ti_eV, double rd_m) const
{
  // nu = F_Epstein/(M_d |V_i - v|) = m_i n_i v_Ti sigma_d / M_d
  //    = (3/4) rho_g v_Ti / (rho_d R_d),  v_Ti = sqrt(2 T_i / m_i)
  if (Ni <= 0.0 || Ti_eV <= 0.0 || rd_m <= 0.0 || rho_d <= 0.0) return 0.0;
  const double mi    = A_background * update->proton_mass;
  const double vti   = std::sqrt(2.0 * (Ti_eV * update->echarge) / mi);
  const double rho_g = Ni * mi;
  return 0.75 * (rho_g * vti) / (rho_d * rd_m);
}

double FixDropletDrag::coulomb_multiplier(double u) const
{
  return coulomb_multiplier(u, chi_coulomb, delta_ite, ln_lambda_coulomb);
}

double FixDropletDrag::coulomb_multiplier(double u, double chi, double delta,
                                          double lnlam) const
{
  const double sqrt_pi    = std::sqrt(MY_PI);
  const double chi_over_d = chi / std::max(delta, 1.0e-12);
  // subsonic branch: below U_SMALL the closed form cancels to O(u^3)
  // against O(u) terms (catastrophic near machine epsilon) — use the
  // analytic limits xi_coll -> (5+4a)/(3 sqrt(pi)), Y(u)/u -> 2/(3 sqrt(pi))
  const double U_SMALL = 1.0e-3;
  if (u < U_SMALL)
    return (5.0 + 4.0*chi_over_d) / (3.0*sqrt_pi)
         + 2.0*chi_over_d*chi_over_d*lnlam * 2.0/(3.0*sqrt_pi);
  const double ueff       = u;
  const double e2         = std::exp(-ueff * ueff);
  const double erf_u      = std::erf(ueff);

  // Collection: F_coll = F_Ep/(2u^3 sqrt(pi)) {u(2u^2+1+2chi/d)e^{-u^2}
  //   + sqrt(pi)[4u^4 + 2u^2 - 1 - 2(1-2u^2)chi/d] erf(u)/2}
  const double u2 = ueff * ueff;
  const double cp = 1.0 / (2.0 * u2 * ueff * sqrt_pi);
  const double ca = ueff * (2.0*u2 + 1.0 + 2.0*chi_over_d) * e2;
  const double cb = 0.5 * sqrt_pi *
                    (4.0*u2*u2 + 2.0*u2 - 1.0
                     - 2.0 * (1.0 - 2.0*u2) * chi_over_d) * erf_u;
  const double xi_coll = cp * (ca + cb);

  // Orbit: F_orb = 2 F_Ep (chi/d)^2 lnLambda Y(u)/u, Chandrasekhar
  //   Y(u) = [erf(u) - 2u e^{-u^2}/sqrt(pi)] / (2u^2)
  const double Y      = (erf_u - (2.0 * ueff / sqrt_pi) * e2) / (2.0 * u2);
  const double xi_orb = 2.0 * chi_over_d * chi_over_d * lnlam
                        * (Y / ueff);

  const double xi = xi_coll + xi_orb;
  if (!std::isfinite(xi)) return 0.0;
  return std::max(0.0, xi);
}
