/* ----------------------------------------------------------------------
   OpenEdge: PMI surface data compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surface/physical/sputter/kk,ComputeSurfacePhysicalSputterKokkos)

#else

#ifndef SPARTA_COMPUTE_SURFACE_PHYSICAL_SPUTTER_KOKKOS_H
#define SPARTA_COMPUTE_SURFACE_PHYSICAL_SPUTTER_KOKKOS_H

#include "compute_surface_physical_sputter.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class ComputeSurfacePhysicalSputterKokkos : public ComputeSurfacePhysicalSputter, public KokkosBase {
 public:
  ComputeSurfacePhysicalSputterKokkos(class SPARTA *, int, char **);
  ~ComputeSurfacePhysicalSputterKokkos();
  void compute_per_surf();
};

}

#endif
#endif
