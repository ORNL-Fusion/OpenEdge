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
#include "fix_evap.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "math.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "math_const.h"
#include "math_extra.h"
#include <cmath>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixEvap::FixEvap(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal fix temp/rescale command");
    nevery = atoi(arg[2]);

}

/* ---------------------------------------------------------------------- */

FixEvap::~FixEvap()
{
  if (copymode) return;

}

/* ---------------------------------------------------------------------- */

int FixEvap::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEvap::init()
{

}

/* ---------------------------------------------------------------------- */

void FixEvap::end_of_step()
{
  if (!particle->sorted) particle->sort();
  end_of_step_no_average();
}

/* ----------------------------------------------------------------------
   current thermal temperature is calculated on a per-cell basis
---------------------------------------------------------------------- */

void FixEvap::end_of_step_no_average()
{

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species = particle->species;
  int *next = particle->next;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  int ip;
  for (int icell = 0; icell < nglocal; icell++) {
    if (cinfo[icell].count == 0) continue;
  
    int ip = cinfo[icell].first;
    while (ip >= 0) {
      droplet_evaporation_model(&particles[ip]);
      ip = next[ip];
    }
  }
  
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixEvap::memory_usage()
{
  double bytes = 0.0;
  bytes += maxgrid*3 * sizeof(double);    // vcom
  return bytes;
}

/* ----------------------------------------------------------------------
   droplet evaporation model
   Adapted from Sergey's code
   ---------------------------------------------------- ------------------ */
   void FixEvap::droplet_evaporation_model(Particle::OnePart *ip) {
    // --- Physical constants ---
    const double AM = 1.53E-26;         // Li atom mass [kg]
    const double Rho = 534.0;           // Li density [kg/m³]
    const double Cp = 4200.0;           // Li specific heat [J/kg-K]
    const double DH = 3.158E+03;        // Latent heat [J/mol]
    const double AN = 6.022E+23;        // Avogadro's number
    const double DT = update->dt; // Time step [s]

    // --- Get particle properties ---
    int icell = ip->icell;
    int isp = ip->ispecies;
    double mass = particle->species[isp].mass;
    double charge = particle->species[isp].charge;
    double T = ip->temp;                // Celsius
    double TK = T + 273.15;             // Kelvin
    double droplet_mass = mass;

    // --- Compute radius from mass ---
    double R = pow((3.0 * droplet_mass) / (4.0 * M_PI * Rho), 1.0 / 3.0);

    // --- Get background plasma data ---
    const PlasmaDataParams& plasma_data = update->plasma_data_map[icell];
    double temp_e = plasma_data.temp_e;
    double temp_i = std::max(0.0, plasma_data.temp_i);
    double dens_e = plasma_data.dens_e;
    double dens_i = std::max(0.0, plasma_data.dens_i);
    double v_parr = plasma_data.parr_flow;

    // --- Compute thermal velocities ---
    double Te_J = temp_e * 1.602e-19;
    double Ti_J = temp_i * 1.602e-19;
    double vth_e = sqrt(8.0 * Te_J / (M_PI * update->electron_mass));
    double vth_i = sqrt(8.0 * Ti_J / (M_PI * update->proton_mass * 2.0));

    // --- Compute heat fluxes ---
    double q_e = 2.5 * dens_e * Te_J * vth_e;
    double q_i = 2.5 * dens_i * Ti_J * vth_i;
    double q_conv = 0.5 * dens_i * update->proton_mass * 2.0 * pow(v_parr, 3);
    double Qs = q_e; // + q_i + 0.*q_conv;

    // --- Vapor pressure from Antoine equation ---
    const double a1 = 5.055;
    const double b1 = -8023.0;
    const double xm1 = 6.939;
    double vpres1 = 760.0 * pow(10.0, (a1 + b1 / TK));

    // --- Evaporation flux and radius change ---
    double Gevap = 1.0e4 * 3.513e22 * vpres1 / sqrt(xm1 * TK);
    double dRdt = -AM * Gevap / Rho;

    // --- Heat flux and temperature change ---
    double HF = Qs - (Gevap * DH / AN);
    double dTdt = (3.0 / (Rho * Cp)) * HF;

    // --- Update radius and temperature ---
    double R_new = std::max(0.0, R + dRdt * DT);
    double T_new = std::max(0.0, T + dTdt * DT);

    // --- Update mass based on new radius ---
    double dm_dt = Gevap * AM * 4.0 * M_PI * R_new * R_new;
    double mass_new = std::max(0.0, droplet_mass - dm_dt * DT);
    double radius_new = pow((3.0 * mass_new) / (4.0 * M_PI * Rho), 1.0 / 3.0);

    // --- Store updated values ---
    ip->mass = mass_new;
    ip->radius = radius_new;
    ip->temp = T_new;
}
