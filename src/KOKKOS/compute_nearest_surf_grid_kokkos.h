/* ----------------------------------------------------------------------
   OpenEdge: nearest wall surface per grid cell — Kokkos wrapper.
   Computes on host (once, cached), exposes d_array_grid for device access.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nearest_surf/grid/kk,ComputeNearestSurfGridKokkos)

#else

#ifndef SPARTA_COMPUTE_NEAREST_SURF_GRID_KOKKOS_H
#define SPARTA_COMPUTE_NEAREST_SURF_GRID_KOKKOS_H

#include "compute_nearest_surf_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class ComputeNearestSurfGridKokkos : public ComputeNearestSurfGrid, public KokkosBase {
 public:
  ComputeNearestSurfGridKokkos(class SPARTA *, int, char **);
  ~ComputeNearestSurfGridKokkos();
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
