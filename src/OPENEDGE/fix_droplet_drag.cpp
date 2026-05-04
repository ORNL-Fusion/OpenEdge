/* ----------------------------------------------------------------------
   OpenEdge: fix drag — Epstein / Coulomb drag on droplets.
   Background plasma at particle position pulled from fix background.
------------------------------------------------------------------------- */

#include "fix_droplet_drag.h"
#include "fix_background.h"
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
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for 'model'");
      if      (strcmp(arg[iarg+1], "epstein") == 0) drag_model = DRAG_EPSTEIN;
      else if (strcmp(arg[iarg+1], "coulomb") == 0) drag_model = DRAG_COULOMB;
      else error->all(FLERR, "fix drag: model must be 'epstein' or 'coulomb'");
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
  const bool   coulomb  = (drag_model == DRAG_COULOMB);
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
    OpenEdge::sparta_to_RZ(p.x, dim, axisym, R, Z,
                           pd_->column_x0, pd_->column_y0);

    const double Ti_eV = std::max(pd_->interp2D(pd_->temp_i, R, Z, p.icell), 0.0);
    const double Ni    = std::max(pd_->interp2D(pd_->dens_i, R, Z, p.icell), 0.0);
    const double Vpar  =          pd_->interp2D(pd_->parr_flow, R, Z, p.icell);
    double Br = 0.0, Bz = 0.0, Bt = 0.0;
    if (pd_->has_bfield || !pd_->mesh_tri_br.empty())
      pd_->bfield_at(R, Z, Br, Bz, Bt, p.icell);

    const double rd = p.radius;
    double nuE = 0.0;
    double upar[3] = {0.0, 0.0, 0.0};

    if (rd > 0.0 && Ni > 0.0 && Ti_eV > 0.0) {
      nuE = epstein_nu(Ni, Ti_eV, rd);

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

      if (coulomb && nuE > 0.0) {
        const double vth_i = std::sqrt(8.0 * (Ti_eV * update->echarge)
                                       / (MY_PI * mi));
        if (vth_i > 0.0) {
          const double dv0 = p.v[0] - upar[0];
          const double dv1 = p.v[1] - upar[1];
          const double dv2 = p.v[2] - upar[2];
          const double u   = std::sqrt(dv0*dv0 + dv1*dv1 + dv2*dv2) / vth_i;
          nuE *= coulomb_multiplier(u);
        } else nuE = 0.0;
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

    if (nuE > 0.0 && std::isfinite(nuE)) {
      const double s   = nuE * dt_half;
      const double ex  = (std::fabs(s) < 1.0e-8)
                         ? (1.0 - s + 0.5*s*s)
                         : std::exp(-s);
      const double inv = 1.0 / nuE;
      p.v[0] = upar[0] + (p.v[0] - upar[0] - g0*inv)*ex + g0*inv;
      p.v[1] = upar[1] + (p.v[1] - upar[1] - g1*inv)*ex + g1*inv;
      p.v[2] = upar[2] + (p.v[2] - upar[2] - g2*inv)*ex + g2*inv;
    } else if (use_gravity) {
      p.v[0] += g0 * dt_half;
      p.v[1] += g1 * dt_half;
      p.v[2] += g2 * dt_half;
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixDropletDrag::epstein_nu(double Ni, double Ti_eV, double rd_m) const
{
  if (Ni <= 0.0 || Ti_eV <= 0.0 || rd_m <= 0.0 || rho_d <= 0.0) return 0.0;
  const double mi    = A_background * update->proton_mass;
  const double vth   = std::sqrt(8.0 * (Ti_eV * update->echarge) / (MY_PI * mi));
  const double rho_g = Ni * mi;
  return alpha_E * (rho_g * vth) / (rho_d * rd_m);
}

double FixDropletDrag::coulomb_multiplier(double u) const
{
  const double sqrt_pi    = std::sqrt(MY_PI);
  const double ueff       = std::max(u, 1.0e-8);
  const double chi_over_d = chi_coulomb / std::max(delta_ite, 1.0e-12);
  const double e2         = std::exp(-ueff * ueff);
  const double erf_u      = std::erf(ueff);

  const double cp = 1.0 / (2.0 * ueff * ueff * ueff * sqrt_pi);
  const double ca = ueff * (2.0*ueff*ueff + 1.0 + 2.0*chi_over_d) * e2;
  const double cb = 0.5 * sqrt_pi *
                    (4.0 * std::pow(ueff, 4) - 1.0
                     - 2.0 * (1.0 - 2.0*ueff*ueff) * chi_over_d) * erf_u;
  const double xi_coll = cp * (ca + cb);

  const double Y      = erf_u - (2.0 * ueff / sqrt_pi) * e2;
  const double xi_orb = 2.0 * chi_over_d * chi_over_d * ln_lambda_coulomb
                        * (Y / ueff);

  const double xi = xi_coll + xi_orb;
  if (!std::isfinite(xi)) return 0.0;
  return std::max(0.0, xi);
}
