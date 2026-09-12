/* ----------------------------------------------------------------------
    OpenEdge: compute surf/weighted/kk — device twin of compute surf/weighted
    (pweight-aware, incidence-only per-surf count/flux tallies). The tally
    itself lives in ComputeSurfKokkos::surf_tally_weighted_kk because the
    Kokkos mover works on a memcpy copy of the base class.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surf/weighted/kk,ComputeSurfWeightedKokkos)

#else

#ifndef SPARTA_COMPUTE_SURF_WEIGHTED_KOKKOS_H
#define SPARTA_COMPUTE_SURF_WEIGHTED_KOKKOS_H

#include "compute_surf_kokkos.h"

namespace SPARTA_NS {

class ComputeSurfWeightedKokkos : public ComputeSurfKokkos {
 public:
  ComputeSurfWeightedKokkos(class SPARTA *, int, char **);
  void init();
};

}

#endif
#endif
