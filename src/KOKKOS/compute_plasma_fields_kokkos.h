/* ----------------------------------------------------------------------
   OpenEdge: plasma fields compute — Kokkos wrapper.

   Computes on host (HDF5 read + bilinear interp), exposes:
   - d_array_grid: per-cell column data (legacy cell-center B reads)
   - device-resident equilibrium psi map: r, z, psi, btf, rtf
       used by EquilibriumKokkos::query_bfield_at_point() in pusher_kokkos.h
       for smooth B at particle position (Phase A of GPU pusher port).

   The equilibrium views mirror ComputePlasmaFields::equ_data. They are
   filled at init time by sync_equilibrium_to_device(), called from
   compute_per_grid() (the host already populates equ_data there).
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

  // Device-resident equilibrium psi map. Filled by sync_equilibrium_to_device()
  // when ComputePlasmaFields::has_equilibrium is set on the host. Views remain
  // empty (extent 0) when no equilibrium is loaded; callers should check
  // d_has_equilibrium before invoking the point-query.
  Kokkos::View<double*,  DeviceType> d_equ_r;            // [jm]
  Kokkos::View<double*,  DeviceType> d_equ_z;            // [km]
  Kokkos::View<double**, DeviceType> d_equ_psi;          // [km][jm]
  double d_equ_btf = 0.0;
  double d_equ_rtf = 0.0;
  int    d_equ_jm = 0;
  int    d_equ_km = 0;
  int    d_has_equilibrium = 0;

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
  void sync_to_device();
  void sync_equilibrium_to_device();
};

}

#endif
#endif
