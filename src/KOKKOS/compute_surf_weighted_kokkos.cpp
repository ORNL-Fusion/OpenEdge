/* ----------------------------------------------------------------------
    OpenEdge: compute surf/weighted/kk (see compute_surf_weighted_kokkos.h)
------------------------------------------------------------------------- */

#include "compute_surf_weighted_kokkos.h"
#include "particle_kokkos.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeSurfWeightedKokkos::ComputeSurfWeightedKokkos(SPARTA *sparta, int narg, char **arg) :
  ComputeSurfKokkos(sparta, narg, arg)
{
  // same restriction as the CPU class: count/flux keywords only
  for (int i = 0; i < nvalue; i++)
    if (which[i] > MFLUXIN)
      error->all(FLERR,"compute surf/weighted supports only num, numwt, nflux, "
                 "nflux_incident, mflux, mflux_incident");
  weighted = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfWeightedKokkos::init()
{
  ComputeSurfKokkos::init();
  const int pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0)
    error->all(FLERR,"compute surf/weighted requires the pweight custom "
               "(add fix particle/weight)");
  pw_ewhich = particle->ewhich[pweight_index];
}
