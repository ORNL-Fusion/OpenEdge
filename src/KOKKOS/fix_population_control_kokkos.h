/* ----------------------------------------------------------------------
    OpenEdge:
    fix population/control/kk: the resampling runs on the host (base class)
    once every Nevery steps. Splitting a marker creates particles, which is
    simple and safe on the host; the price is one particle round trip per
    pass (ModifyKokkos syncs to the host before the call and marks the host
    copy modified after it, per the masks set in the constructor).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(population/control/kk,FixPopulationControlKokkos)

#else

#ifndef SPARTA_FIX_POPULATION_CONTROL_KOKKOS_H
#define SPARTA_FIX_POPULATION_CONTROL_KOKKOS_H

#include "fix_population_control.h"

namespace SPARTA_NS {

class FixPopulationControlKokkos : public FixPopulationControl {
 public:
  FixPopulationControlKokkos(class SPARTA *, int, char **);
  void end_of_step() override;
};

}

#endif
#endif
