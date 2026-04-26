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

  // Phase B: device-resident mesh triangulation B (per-triangle vertex-avg)
  // for SOLPS / SOLEDGE3X plasmas where B is carried on /mesh/vtx_b*.
  // Spatial hash is flattened to CSR (offset+entries) for device lookup.
  Kokkos::View<double*, DeviceType> d_mesh_vtx_r;        // [nvtx]
  Kokkos::View<double*, DeviceType> d_mesh_vtx_z;        // [nvtx]
  Kokkos::View<int*,    DeviceType> d_mesh_tri;          // [ntri*3] vertex idx
  Kokkos::View<double*, DeviceType> d_mesh_tri_br;       // [ntri]
  Kokkos::View<double*, DeviceType> d_mesh_tri_bz;       // [ntri]
  Kokkos::View<double*, DeviceType> d_mesh_tri_bt;       // [ntri]
  Kokkos::View<double*, DeviceType> d_mesh_tri_rmin;     // [ntri] bbox
  Kokkos::View<double*, DeviceType> d_mesh_tri_rmax;
  Kokkos::View<double*, DeviceType> d_mesh_tri_zmin;
  Kokkos::View<double*, DeviceType> d_mesh_tri_zmax;
  Kokkos::View<int*,    DeviceType> d_hash_offset;       // [nbins+1] CSR offsets
  Kokkos::View<int*,    DeviceType> d_hash_entries;      // flat tri indices
  double d_mesh_hash_rmin = 0.0, d_mesh_hash_zmin = 0.0;
  double d_mesh_hash_dr   = 1.0, d_mesh_hash_dz   = 1.0;
  int    d_mesh_hash_nr   = 0,   d_mesh_hash_nz   = 0;
  int    d_mesh_ntri      = 0;
  int    d_has_mesh_b     = 0;

 private:
  DAT::tdual_float_2d_lr k_array_grid;
  int maxgrid_kk;
  void sync_to_device();
  void sync_equilibrium_to_device();
  void sync_mesh_to_device();
};

}

#endif
#endif
