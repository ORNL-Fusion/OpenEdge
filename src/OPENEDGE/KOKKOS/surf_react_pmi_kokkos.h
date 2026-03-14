/* ----------------------------------------------------------------------
   OpenEdge: PMI surface reaction — Kokkos wrapper.
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(pmi/kk,SurfReactPMIKokkos)

#else

#ifndef SPARTA_SURF_REACT_PMI_KOKKOS_H
#define SPARTA_SURF_REACT_PMI_KOKKOS_H

#include "surf_react_pmi.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class SurfReactPMIKokkos : public SurfReactPMI {
 public:
  SurfReactPMIKokkos(class SPARTA *, int, char **);
  ~SurfReactPMIKokkos();
};

}

#endif
#endif
