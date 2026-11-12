/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    fix ID neutral/collide/background Nevery background <pdID>
        [ion     sigma <m^2> gref <m/s> omega <w> mass_amu <m>]
        [neutral sigma <m^2> gref <m/s> omega <w> mass_amu <m>]

    Elastic scattering of kinetic NEUTRALS off the fluid background:
    ion channel partner = main ion (D+): dens_i, temp_i, parallel flow;
    neutral channel partner = fluid-neutral D: dens_n, temp_n (zero
    flow). sigma(g) = sigma_ref (g_ref/g)^(2w-1); w=0.5 hard sphere,
    w=1 Langevin. Guidance for W in D plasma:
      ion     sigma 5.2e-19 gref 1e4 omega 1.0 mass_amu 2.014
        (Langevin k_L ~ 5e-15 m^3/s for alpha'_W ~ 1e-29 m^3)
      neutral sigma 3.0e-19 gref 1e4 omega 0.5 mass_amu 2.014
        (hard sphere from combined vdW radii)
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "fix_neutral_collide_background.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "particle.h"
#include "comm.h"
#include "fix_background.h"
#include "openedge_geom.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <cmath>

using namespace SPARTA_NS;
using namespace MathConst;


/* ---------------------------------------------------------------------- */

FixNeutralCollideBackground::FixNeutralCollideBackground(SPARTA *sparta,
    int narg, char **arg) : Fix(sparta, narg, arg)
{
  if (narg < 5)
    error->all(FLERR,"Illegal fix neutral/collide/background command");

  nevery = atoi(arg[2]);
  if (nevery <= 0)
    error->all(FLERR,"fix neutral/collide/background: Nevery must be > 0");

  if (strcmp(arg[3],"background") != 0)
    error->all(FLERR,"fix neutral/collide/background: expected 'background <ID>'");
  int n = strlen(arg[4]) + 1;
  pd_id = new char[n];
  strcpy(pd_id, arg[4]);
  pd = nullptr;

  int iarg = 5;
  while (iarg < narg) {
    Channel *c = nullptr;
    if (strcmp(arg[iarg],"ion") == 0) c = &chan_ion;
    else if (strcmp(arg[iarg],"neutral") == 0) c = &chan_neut;
    else error->all(FLERR,"fix neutral/collide/background: unknown channel "
                          "(use ion|neutral)");
    c->active = 1;
    c->mass_b = 2.014 * PhysConst::AMU;          // default: deuterium partner
    iarg++;
    while (iarg + 1 < narg) {
      if (strcmp(arg[iarg],"sigma") == 0)         c->sigma_ref = atof(arg[iarg+1]);
      else if (strcmp(arg[iarg],"gref") == 0)     c->g_ref = atof(arg[iarg+1]);
      else if (strcmp(arg[iarg],"omega") == 0)    c->omega = atof(arg[iarg+1]);
      else if (strcmp(arg[iarg],"mass_amu") == 0) c->mass_b = atof(arg[iarg+1]) * PhysConst::AMU;
      else break;
      iarg += 2;
    }
    if (c->sigma_ref <= 0.0)
      error->all(FLERR,"fix neutral/collide/background: channel needs sigma > 0");
    if (c->g_ref <= 0.0 || c->omega < 0.5 || c->omega > 1.5)
      error->all(FLERR,"fix neutral/collide/background: need gref > 0 and "
                       "0.5 <= omega <= 1.5");
  }
  if (!chan_ion.active && !chan_neut.active)
    error->all(FLERR,"fix neutral/collide/background: no channel given");

  rng = new RanKnuth(update->ranmaster->uniform());
  rng->reset(update->ranmaster->uniform(), comm->me, 100);   // distinct stream per rank

  ncollide = 0;
  mass_p_current = 0.0;
  scalar_flag = 1;
  global_freq = 1;
}

/* ---------------------------------------------------------------------- */

FixNeutralCollideBackground::~FixNeutralCollideBackground()
{
  delete [] pd_id;
  delete rng;
}

/* ---------------------------------------------------------------------- */

int FixNeutralCollideBackground::setmask()
{
  return END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixNeutralCollideBackground::init()
{
  int ifix = modify->find_fix(pd_id);
  if (ifix < 0)
    error->all(FLERR,"fix neutral/collide/background: background fix not found");
  pd = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd)
    error->all(FLERR,"fix neutral/collide/background: ID is not a background fix");

  dim = domain->dimension;
  axi = domain->axisymmetric;

  if (chan_neut.active && !pd->has_neutral_dens() && comm->me == 0 && screen)
    fprintf(screen, "WARNING: fix neutral/collide/background: neutral channel "
                    "requested but plasma file has no mesh/dens_n; channel "
                    "will be inert\n");
}

/* ----------------------------------------------------------------------
   one BGK-style elastic collision attempt of particle velocity v
   against a virtual Maxwellian partner (density nb [m^-3], temperature
   Tb_eV, drift ub). Modifies v in place on collision.
------------------------------------------------------------------------- */

void FixNeutralCollideBackground::collide_channel(const Channel &c,
    double *v, double nb, double Tb_eV, const double ub[3])
{
  if (nb <= 0.0 || Tb_eV <= 0.0) return;

  const double vth = std::sqrt(Tb_eV * PhysConst::QE / c.mass_b);

  // partner velocity ~ drifting Maxwellian
  double vb[3];
  for (int k = 0; k < 3; k += 2) {
    const double u1 = std::max(rng->uniform(), 1e-30);
    const double u2 = rng->uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    vb[k] = ub[k] + vth * r * std::cos(MY_2PI * u2);
    if (k + 1 < 3) vb[k+1] = ub[k+1] + vth * r * std::sin(MY_2PI * u2);
  }

  double g[3] = {v[0]-vb[0], v[1]-vb[1], v[2]-vb[2]};
  const double gmag = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
  if (gmag < 1.0e-12) return;

  // VSS power law sigma(g); nu = nb * sigma * g
  const double sigma = c.sigma_ref *
      std::pow(c.g_ref / gmag, 2.0 * c.omega - 1.0);
  const double nu = nb * sigma * gmag;
  const double P = -std::expm1(-nu * nevery * update->dt);
  if (rng->uniform() >= P) return;

  // isotropic COM scattering (HS / VSS alpha=1); the partner is a bath
  // (not updated -- trace-impurity assumption, momentum sink into the
  // background fluid)
  const double costh = 1.0 - 2.0 * rng->uniform();
  const double sinth = std::sqrt(std::max(0.0, 1.0 - costh*costh));
  const double phi = MY_2PI * rng->uniform();
  double gnew[3] = {gmag * sinth * std::cos(phi),
                    gmag * sinth * std::sin(phi),
                    gmag * costh};

  // v' = v_com + (m_b/(m_p+m_b)) * g_new
  const double mp = mass_p_current;   // set per-particle in end_of_step
  const double mb = c.mass_b;
  const double inv = 1.0 / (mp + mb);
  double vcom[3];
  for (int k = 0; k < 3; k++)
    vcom[k] = (mp * v[k] + mb * vb[k]) * inv;
  for (int k = 0; k < 3; k++)
    v[k] = vcom[k] + mb * inv * gnew[k];

  ncollide++;
}

/* ---------------------------------------------------------------------- */

void FixNeutralCollideBackground::end_of_step()
{
  if (update->ntimestep % nevery) return;

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species = particle->species;
  const int nlocal = particle->nlocal;

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];
    if (species[p.ispecies].charge != 0.0) continue;   // neutrals only

    double R = 0.0, Z = 0.0;
    OpenEdge::sparta_to_RZ(p.x, dim, axi, R, Z,
                           pd->column_x0, pd->column_y0);
    mass_p_current = species[p.ispecies].mass;

    if (chan_ion.active) {
      const double ni = pd->dens_i.empty()
          ? pd->interp2D(pd->dens_e, R, Z, p.icell)
          : pd->interp2D(pd->dens_i, R, Z, p.icell);
      const double Ti = pd->temp_i.empty()
          ? pd->interp2D(pd->temp_e, R, Z, p.icell)
          : pd->interp2D(pd->temp_i, R, Z, p.icell);
      double ub[3] = {0.0, 0.0, 0.0};
      if (!pd->parr_flow.empty() && pd->has_bfield) {
        const double upar = pd->interp2D(pd->parr_flow, R, Z, p.icell);
        double Br = 0.0, Bz = 0.0, Bt = 0.0;
        pd->bfield_at(R, Z, Br, Bz, Bt, p.icell, ip);
        const double Bmag = std::sqrt(Br*Br + Bz*Bz + Bt*Bt);
        if (Bmag > 1e-30) {
          // (R,t,Z) unit b -> sparta frame via the same convention as
          // the pusher: for the wedge/cart 3D case x=R-ish local frame;
          // project with sparta_to_RZ's inverse is overkill for a
          // drift term -- use the toroidal-dominant approximation
          double bx, by, bz;
          OpenEdge::RZphi_force_to_sparta(Br/Bmag, Bz/Bmag, Bt/Bmag,
              dim, axi, std::atan2(p.x[1] - pd->column_y0,
                                   p.x[0] - pd->column_x0),
              bx, by, bz);
          ub[0] = upar * bx; ub[1] = upar * by; ub[2] = upar * bz;
        }
      }
      collide_channel(chan_ion, p.v, ni, Ti, ub);
    }

    if (chan_neut.active && pd->has_neutral_dens()) {
      const double nn = pd->interp2D(pd->dens_n, R, Z, p.icell);
      const double Tn = pd->temp_n.empty()
          ? pd->interp2D(pd->temp_i, R, Z, p.icell)
          : pd->interp2D(pd->temp_n, R, Z, p.icell);
      const double ub[3] = {0.0, 0.0, 0.0};
      collide_channel(chan_neut, p.v, nn, Tn, ub);
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixNeutralCollideBackground::compute_scalar()
{
  double all = 0.0;
  double mine = static_cast<double>(ncollide);
  MPI_Allreduce(&mine, &all, 1, MPI_DOUBLE, MPI_SUM, world);
  return all;
}
