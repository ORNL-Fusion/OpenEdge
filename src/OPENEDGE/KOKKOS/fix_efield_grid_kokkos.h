/* ----------------------------------------------------------------------
   OpenEdge: grid electric field fix — Kokkos version.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(efield/grid/kk,FixEfieldGridKokkos)

#else

#ifndef SPARTA_FIX_EFIELD_GRID_KOKKOS_H
#define SPARTA_FIX_EFIELD_GRID_KOKKOS_H

#include "fix_efield_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixEfieldGridKokkos : public FixEfieldGrid, public KokkosBase {
 public:
  FixEfieldGridKokkos(class SPARTA *, int, char **);
  ~FixEfieldGridKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
};

}

#endif
#endif
