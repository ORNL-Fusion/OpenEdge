/* ----------------------------------------------------------------------
   OpenEdge:
   Per-surface bulk evaporation flux. Antoine + Hertz-Knudsen.
------------------------------------------------------------------------- 
   UNITS NOTE: this compute expects the bound Tsurf custom in degrees
   CELSIUS (+273.15 is applied internally); compute surface/physical/
   sputter expects KELVIN for its tsurf custom - do not bind one shared
   attribute to both without converting.
*/

#include "compute_surface_chemical_evaporation.h"

#include <cstring>
#include <cstdlib>

#include "surf.h"
#include "domain.h"
#include "input.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "liquid_metal_strip.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeSurfaceChemicalEvaporation::ComputeSurfaceChemicalEvaporation(
    SPARTA *sparta, int narg, char **arg) : Compute(sparta, narg, arg)
{
  if (narg < 6)
    error->all(FLERR,
      "compute surface/chemical/evaporation: "
      "Usage: ID surfgroup tsurf <surf-custom-name> [sticking VAL]");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,
      "compute surface/chemical/evaporation: surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  tsurf_name = nullptr;
  sticking = 1.0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "tsurf") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute surface/chemical/evaporation: tsurf needs a value");
      const int n = strlen(arg[iarg+1]) + 1;
      delete [] tsurf_name;
      tsurf_name = new char[n];
      strcpy(tsurf_name, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "sticking") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute surface/chemical/evaporation: sticking needs a value");
      sticking = input->numeric(FLERR, arg[iarg+1]);
      if (sticking <= 0.0)
        error->all(FLERR, "compute surface/chemical/evaporation: sticking must be > 0");
      iarg += 2;
    } else {
      error->all(FLERR, "compute surface/chemical/evaporation: unknown keyword");
    }
  }

  if (!tsurf_name)
    error->all(FLERR,
      "compute surface/chemical/evaporation: tsurf <surf-custom-name> required");

  per_surf_flag = 1;
  size_per_surf_cols = 0;   // vector output
  vector_surf = nullptr;
  nslocal = 0;
  tsurf_index = -1;
}

/* ---------------------------------------------------------------------- */

ComputeSurfaceChemicalEvaporation::~ComputeSurfaceChemicalEvaporation()
{
  if (copymode) return;
  delete [] tsurf_name;
  memory->destroy(vector_surf);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalEvaporation::init()
{
  tsurf_index = surf->find_custom(tsurf_name);
  if (tsurf_index < 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
      "compute surface/chemical/evaporation: surf custom '%s' not found "
      "(register it with fix surface/state/liquid_metal or similar)",
      tsurf_name);
    error->all(FLERR, msg);
  }
  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalEvaporation::reallocate()
{
  const int n = surf->nown;
  if (n == nslocal && vector_surf) return;
  memory->destroy(vector_surf);
  nslocal = n;
  memory->create(vector_surf, nslocal, "surface/chemical/evaporation:vector");
  for (int i = 0; i < nslocal; i++) vector_surf[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalEvaporation::compute_per_surf()
{
  invoked_per_surf = update->ntimestep;
  reallocate();

  const int dim = domain->dimension;
  const int distributed = surf->distributed;
  Surf::Line *lines = distributed ? surf->mylines : surf->lines;
  Surf::Tri  *tris  = distributed ? surf->mytris  : surf->tris;
  const int me = comm->me, nprocs = comm->nprocs;

  double *tvec = surf->edvec[surf->ewhich[tsurf_index]];

  for (int i = 0; i < nslocal; i++) {
    const int m = distributed ? i : me + i * nprocs;
    int mask;
    if (dim == 2) mask = lines[m].mask;
    else          mask = tris[m].mask;

    if (!(mask & groupbit)) {
      vector_surf[i] = 0.0;
      continue;
    }

    // Tsurf is stored in degrees C (matches fix surface/state/liquid_metal).
    vector_surf[i] = LiquidMetal::li_evap_flux(tvec[i], sticking);
  }
}
