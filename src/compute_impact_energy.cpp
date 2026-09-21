/* ----------------------------------------------------------------------
    OpenEdge: compute impact/energy — incident-KE histogram at the wall.
    See compute_impact_energy.h.
------------------------------------------------------------------------- */

#include "compute_impact_energy.h"

#include "math.h"
#include "string.h"

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "mixture.h"
#include "particle.h"
#include "surf.h"
#include "update.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeImpactEnergy::ComputeImpactEnergy(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal compute impact/energy command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute impact/energy surf group does not exist");
  groupbit = surf->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Compute impact/energy mixture does not exist");

  nbins = atoi(arg[4]);
  emin  = atof(arg[5]);
  emax  = atof(arg[6]);
  if (nbins < 1) error->all(FLERR,"Compute impact/energy nbins must be >= 1");
  if (emin <= 0.0 || emax <= emin)
    error->all(FLERR,"Compute impact/energy needs 0 < Emin < Emax");

  logscale = 1;
  if (narg > 7) {
    if (strcmp(arg[7],"lin") == 0) logscale = 0;
    else if (strcmp(arg[7],"log") == 0) logscale = 1;
    else error->all(FLERR,"Compute impact/energy scale must be log or lin");
  }

  if (logscale) {
    logemin = log10(emin);
    invlogdE = nbins / (log10(emax) - logemin);
  } else {
    invdE = nbins / (emax - emin);
  }

  // global vector output: per-bin incident real-atom counts
  surf_tally_flag = 1;
  timeflag = 1;
  vector_flag = 1;
  size_vector = nbins;

  memory->create(vector, nbins, "impact/energy:vector");
  memory->create(hist, nbins, "impact/energy:hist");
  for (int i = 0; i < nbins; i++) vector[i] = hist[i] = 0.0;

  dim = domain->dimension;
}

/* ---------------------------------------------------------------------- */

ComputeImpactEnergy::~ComputeImpactEnergy()
{
  if (copy || copymode) return;
  memory->destroy(vector);
  memory->destroy(hist);
}

/* ---------------------------------------------------------------------- */

void ComputeImpactEnergy::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute impact/energy when surfs do not exist");

  // announce the bin edges once so the histogram can be reconstructed in post
  if (comm->me == 0) {
    const char *s = logscale ? "log" : "lin";
    if (screen) fprintf(screen,
      "compute impact/energy: %d %s bins, %g..%g eV (edges: %s-spaced)\n",
      nbins, s, emin, emax, s);
    if (logfile) fprintf(logfile,
      "compute impact/energy: %d %s bins, %g..%g eV (edges: %s-spaced)\n",
      nbins, s, emin, emax, s);
  }
}

/* ----------------------------------------------------------------------
   tally one particle-surf collision: bin the incident impact energy
------------------------------------------------------------------------- */

void ComputeImpactEnergy::surf_tally(double /*dtremain*/, int isurf,
                                     int /*icell*/, int /*reaction*/,
                                     Particle::OnePart *iorig,
                                     Particle::OnePart * /*ip*/,
                                     Particle::OnePart * /*jp*/)
{
  if (!iorig) return;                       // incidence only (skip emission)

  // surf group filter
  if (dim == 2) { if (!(surf->lines[isurf].mask & groupbit)) return; }
  else          { if (!(surf->tris[isurf].mask & groupbit)) return; }

  // species/mixture filter
  int isp = iorig->ispecies;
  if (particle->mixture[imix]->species2group[isp] < 0) return;

  // incident kinetic energy [eV]. iorig->v already carries the sheath boost.
  double *v = iorig->v;
  double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  double E = 0.5 * particle->species[isp].mass * v2
             * update->mvv2e * update->joule2ev;
  if (E <= 0.0) return;

  int b;
  if (logscale) b = (int) ((log10(E) - logemin) * invlogdE);
  else          b = (int) ((E - emin) * invdE);
  if (b < 0) b = 0;
  if (b >= nbins) b = nbins - 1;

  // pweight-weighted: the mover stamps the incident pweight into tally_pweight
  // before the surf_tally batch (defaults to 1.0 without fix particle/weight).
  hist[b] += update->tally_pweight;
}

/* ---------------------------------------------------------------------- */

void ComputeImpactEnergy::clear()
{
  for (int i = 0; i < nbins; i++) hist[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void ComputeImpactEnergy::compute_vector()
{
  invoked_vector = update->ntimestep;
  MPI_Allreduce(hist, vector, nbins, MPI_DOUBLE, MPI_SUM, world);
}
