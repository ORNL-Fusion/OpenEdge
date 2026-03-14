/* ----------------------------------------------------------------------
   OpenEdge: generic grid field fix — Kokkos version.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(field/grid/kk,FixFieldGridKokkos)

#else

#ifndef SPARTA_FIX_FIELD_GRID_KOKKOS_H
#define SPARTA_FIX_FIELD_GRID_KOKKOS_H

#include "fix_field_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixFieldGridKokkos : public FixFieldGrid, public KokkosBase {
 public:
  FixFieldGridKokkos(class SPARTA *, int, char **);
  ~FixFieldGridKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
};

}

#endif
#endif
