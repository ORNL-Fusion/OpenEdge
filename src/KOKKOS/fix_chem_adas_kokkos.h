/* ----------------------------------------------------------------------
   OpenEdge: ADAS ionization/recombination — Kokkos wrapper.
   Runs CPU-side end_of_step (host), syncs particle species changes back.
   Ensures compatibility with -sf kk builds.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(chem/adas/kk,FixChemAdasKokkos)

#else

#ifndef SPARTA_FIX_CHEM_ADAS_KOKKOS_H
#define SPARTA_FIX_CHEM_ADAS_KOKKOS_H

#include "fix_chem_adas.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class FixChemAdasKokkos : public FixChemAdas {
 public:
  FixChemAdasKokkos(class SPARTA *, int, char **);
  ~FixChemAdasKokkos();
  void end_of_step();
};

}

#endif
#endif
