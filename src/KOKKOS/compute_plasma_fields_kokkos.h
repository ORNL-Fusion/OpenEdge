/* ----------------------------------------------------------------------
   OpenEdge: plasma fields compute — Kokkos wrapper.
   Computes on host (HDF5 read + bilinear interp), exposes d_array_grid
   for device access by Boris kernel and other device-side consumers.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(plasma/fields/kk,ComputePlasmaFieldsKokkos)

#else

#ifndef SPARTA_COMPUTE_PLASMA_FIELDS_KOKKOS_H
#define SPARTA_COMPUTE_PLASMA_FIELDS_KOKKOS_H

#include "compute_plasma_fields.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class ComputePlasmaFieldsKokkos : public ComputePlasmaFields, public KokkosBase {
 public:
  ComputePlasmaFieldsKokkos(class SPARTA *, int, char **);
  ~ComputePlasmaFieldsKokkos();
  void compute_per_grid();

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
  void sync_to_device();
};

}

#endif
#endif
