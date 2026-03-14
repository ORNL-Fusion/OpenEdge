/* ----------------------------------------------------------------------
   OpenEdge: sheath geometry per grid cell — Kokkos wrapper.
   Computes on host (once, cached), exposes d_array_grid for device access.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(sheath/geometry/grid/kk,ComputeSheathGeometryGridKokkos)

#else

#ifndef SPARTA_COMPUTE_SHEATH_GEOMETRY_GRID_KOKKOS_H
#define SPARTA_COMPUTE_SHEATH_GEOMETRY_GRID_KOKKOS_H

#include "compute_sheath_geometry_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class ComputeSheathGeometryGridKokkos : public ComputeSheathGeometryGrid, public KokkosBase {
 public:
  ComputeSheathGeometryGridKokkos(class SPARTA *, int, char **);
  ~ComputeSheathGeometryGridKokkos();
  void compute_per_grid();

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  DAT::tdual_int_1d k_midx_grid;
  DAT::t_int_1d d_midx_grid_kk;
  int maxgrid_kk;
  void sync_to_device();
};

}

#endif
#endif
