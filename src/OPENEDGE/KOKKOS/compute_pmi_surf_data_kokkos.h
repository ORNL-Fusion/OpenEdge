/* ----------------------------------------------------------------------
   OpenEdge: PMI surface data compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(pmi/surf/data/kk,ComputePMISurfDataKokkos)

#else

#ifndef SPARTA_COMPUTE_PMI_SURF_DATA_KOKKOS_H
#define SPARTA_COMPUTE_PMI_SURF_DATA_KOKKOS_H

#include "compute_pmi_surf_data.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class ComputePMISurfDataKokkos : public ComputePMISurfData, public KokkosBase {
 public:
  ComputePMISurfDataKokkos(class SPARTA *, int, char **);
  ~ComputePMISurfDataKokkos();
  void compute_per_surf();
};

}

#endif
#endif
