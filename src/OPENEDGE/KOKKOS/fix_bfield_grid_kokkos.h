/* ----------------------------------------------------------------------
   OpenEdge: grid magnetic field fix — Kokkos version.
   Inherits from FixBfieldGrid, adds KokkosBase for device-accessible views.

   After compute_field() fills array_grid on host, the result is
   deep_copied to d_array_grid so the Boris kernel can read B on device.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(bfield/grid/kk,FixBfieldGridKokkos)

#else

#ifndef SPARTA_FIX_BFIELD_GRID_KOKKOS_H
#define SPARTA_FIX_BFIELD_GRID_KOKKOS_H

#include "fix_bfield_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixBfieldGridKokkos : public FixBfieldGrid, public KokkosBase {
 public:
  FixBfieldGridKokkos(class SPARTA *, int, char **);
  ~FixBfieldGridKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
};

}

#endif
#endif
