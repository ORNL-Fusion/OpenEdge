/* ----------------------------------------------------------------------
   OpenEdge: Nanbu Coulomb collisions — Kokkos wrapper.
   Runs CPU-side end_of_step (host), syncs particle velocity changes back.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coll/nanbu/kk,FixCollNanbuKokkos)

#else

#ifndef SPARTA_FIX_COLL_NANBU_KOKKOS_H
#define SPARTA_FIX_COLL_NANBU_KOKKOS_H

#include "fix_coll_nanbu.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class FixCollNanbuKokkos : public FixCollNanbu {
 public:
  FixCollNanbuKokkos(class SPARTA *, int, char **);
  ~FixCollNanbuKokkos();
  void end_of_step();
};

}

#endif
#endif
