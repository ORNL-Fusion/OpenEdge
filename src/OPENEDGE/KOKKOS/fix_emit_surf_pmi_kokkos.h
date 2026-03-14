/* ----------------------------------------------------------------------
   OpenEdge: PMI surface emission — Kokkos wrapper.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(emit/surf/pmi/kk,FixEmitSurfPmiKokkos)

#else

#ifndef SPARTA_FIX_EMIT_SURF_PMI_KOKKOS_H
#define SPARTA_FIX_EMIT_SURF_PMI_KOKKOS_H

#include "fix_emit_surf_pmi.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class FixEmitSurfPmiKokkos : public FixEmitSurfPmi {
 public:
  FixEmitSurfPmiKokkos(class SPARTA *, int, char **);
  ~FixEmitSurfPmiKokkos();
  void perform_task();
};

}

#endif
#endif
