/* ----------------------------------------------------------------------
   OpenEdge: aggregate Ion Energy-Angle Distribution (IEAD)

   Accumulates a single 2-D histogram (energy x angle) across all
   surfaces in a group.  Each bin stores the weighted particle count.

   Syntax:
     compute ID iead surf_group mix_ID Emax NbinsE Amax NbinsA

   Output: global array  (rows = bins_a, cols = bins_e)
   Use with fix ave/time for periodic / end-of-run dumps:

     fix fIEAD ave/time all 1 1 ${Ndump} c_cIEAD mode array &
         ave running file output/iead.dat title1 "IEAD: rows=angle cols=energy"

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
------------------------------------------------------------------------- */

#include "compute_iead.h"

#include <cstring>
#include <cmath>

#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "grid.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeIEAD::ComputeIEAD(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  // compute ID iead surf_group mix_ID Emax NbE Amax NbA
  if (narg < 8) error->all(FLERR,"Illegal compute iead command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute iead surface group ID does not exist");
  groupbit = surf->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Compute iead mixture ID does not exist");

  emax = atof(arg[4]);
  bins_e = atoi(arg[5]);
  amax = atof(arg[6]);
  bins_a = atoi(arg[7]);
  if (emax <= 0.0 || amax <= 0.0 || bins_e <= 0 || bins_a <= 0)
    error->all(FLERR,"Illegal compute iead command");

  de = emax / bins_e;
  da = amax / bins_a;

  // Global array output: rows = bins_a, cols = bins_e
  array_flag = 1;
  size_array_rows = bins_a;
  size_array_cols = bins_e;

  // Register for surface-tally callbacks
  surf_tally_flag = 1;
  timeflag = 1;

  int ntot = bins_a * bins_e;
  hist_local = new double[ntot]();
  hist_global = new double[ntot]();

  // Build 2D pointer array into hist_global for array_array
  hist_2d = new double*[bins_a];
  for (int ia = 0; ia < bins_a; ia++)
    hist_2d[ia] = &hist_global[ia * bins_e];
  array = hist_2d;

  last_reduced = -1;
  dim = domain->dimension;
  lines = NULL;
  tris = NULL;
  weightflag = 0;
}

/* ---------------------------------------------------------------------- */

ComputeIEAD::~ComputeIEAD()
{
  delete [] hist_local;
  delete [] hist_global;
  delete [] hist_2d;
}

/* ---------------------------------------------------------------------- */

void ComputeIEAD::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute iead when surfs do not exist");

  if (grid->cellweightflag) weightflag = 1;
  else weightflag = 0;

  lines = surf->lines;
  tris = surf->tris;
}

/* ---------------------------------------------------------------------- */

void ComputeIEAD::clear()
{
  // Intentionally do NOT clear hist_local.
  // We accumulate over the entire run so the histogram grows.
  // If the user wants per-interval data, they can diff snapshots
  // or use fix ave/time with ave one.
  lines = surf->lines;
  tris = surf->tris;
}

/* ---------------------------------------------------------------------- */

void ComputeIEAD::surf_tally(double /*dtremain*/, int isurf, int /*icell*/,
                              int /*reaction*/,
                              Particle::OnePart *iorig,
                              Particle::OnePart * /*ip*/,
                              Particle::OnePart * /*jp*/)
{
  if (!iorig) return;

  // Check surface group membership
  if (dim == 2) {
    if (!(lines[isurf].mask & groupbit)) return;
  } else {
    if (!(tris[isurf].mask & groupbit)) return;
  }

  // Check species mixture membership
  int origspecies = iorig->ispecies;
  int igroup = particle->mixture[imix]->species2group[origspecies];
  if (igroup < 0) return;

  // Surface normal
  double *norm;
  if (dim == 2) norm = lines[isurf].norm;
  else norm = tris[isurf].norm;

  double *v = iorig->v;
  double vx = v[0], vy = v[1], vz = v[2];
  double v2 = vx*vx + vy*vy + vz*vz;
  if (v2 <= 0.0) return;

  // Particle mass
  double mass = (iorig->mass > 0.0) ? iorig->mass
                : particle->species[origspecies].mass;
  if (mass <= 0.0) return;

  // Particle weight
  double weight = 1.0;
  if (weightflag) weight = iorig->weight;

  // Kinetic energy in eV
  double e_ev = 0.5 * update->mvv2e * mass * v2;
  if (strcmp(update->unit_style,"si") == 0) e_ev *= update->joule2ev;
  if (e_ev < 0.0 || e_ev >= emax) return;

  // Incidence angle from surface normal
  double nx = norm[0], ny = norm[1];
  double nz = (dim == 3) ? norm[2] : 0.0;
  double n2 = nx*nx + ny*ny + nz*nz;
  if (n2 <= 0.0) return;

  double cosang = (vx*nx + vy*ny + vz*nz) / sqrt(v2 * n2);
  if (cosang < -1.0) cosang = -1.0;
  if (cosang >  1.0) cosang =  1.0;
  double ang = acos(fabs(cosang)) * 180.0 / MathConst::MY_PI;
  if (ang < 0.0 || ang >= amax) return;

  // Bin indices
  int ie = static_cast<int>(e_ev / de);
  int ia = static_cast<int>(ang / da);
  if (ie >= bins_e) ie = bins_e - 1;
  if (ia >= bins_a) ia = bins_a - 1;
  if (ie < 0 || ia < 0) return;

  hist_local[ia * bins_e + ie] += weight;
}

/* ---------------------------------------------------------------------- */

void ComputeIEAD::compute_array()
{
  invoked_array = update->ntimestep;

  // MPI reduce: sum local histograms across all ranks
  int ntot = bins_a * bins_e;
  MPI_Allreduce(hist_local, hist_global, ntot, MPI_DOUBLE, MPI_SUM, world);

  // array_array already points to hist_2d → hist_global
}

/* ---------------------------------------------------------------------- */

int ComputeIEAD::tallyinfo(surfint *&ptr)
{
  ptr = NULL;
  return 0;
}

/* ---------------------------------------------------------------------- */

bigint ComputeIEAD::memory_usage()
{
  bigint bytes = 0;
  bigint ntot = static_cast<bigint>(bins_a) * bins_e;
  bytes += 2 * ntot * sizeof(double);       // hist_local + hist_global
  bytes += bins_a * sizeof(double *);        // hist_2d
  return bytes;
}
