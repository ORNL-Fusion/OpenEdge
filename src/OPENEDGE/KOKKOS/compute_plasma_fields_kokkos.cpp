/* ----------------------------------------------------------------------
   OpenEdge: plasma fields compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "compute_plasma_fields_kokkos.h"
#include "grid.h"
#include "memory_kokkos.h"

using namespace SPARTA_NS;

ComputePlasmaFieldsKokkos::ComputePlasmaFieldsKokkos(
    SPARTA *sparta, int narg, char **arg) :
  ComputePlasmaFields(sparta, narg, arg)
{
  kokkos_flag = 1;
  maxgrid_kk = 0;
}

ComputePlasmaFieldsKokkos::~ComputePlasmaFieldsKokkos()
{
  if (copymode) return;
}

void ComputePlasmaFieldsKokkos::compute_per_grid()
{
  ComputePlasmaFields::compute_per_grid();
  sync_to_device();
  sync_equilibrium_to_device();   // idempotent; only first call allocates
}

/* ----------------------------------------------------------------------
   Mirror ComputePlasmaFields::equ_data onto device-resident Views so
   the GPU pusher can run bilinear B interpolation at particle position.
   Idempotent: re-running with the same dimensions is a no-op besides
   the deep_copy. has_equilibrium flag is mirrored to d_has_equilibrium
   so callers can skip the kernel branch when no equilibrium is loaded.
------------------------------------------------------------------------- */
void ComputePlasmaFieldsKokkos::sync_equilibrium_to_device()
{
  if (!has_equilibrium || equ_data.jm < 3 || equ_data.km < 3) {
    d_has_equilibrium = 0;
    return;
  }

  const int jm = equ_data.jm;
  const int km = equ_data.km;

  if (d_equ_jm != jm || d_equ_km != km) {
    d_equ_r   = Kokkos::View<double*,  DeviceType>("equ_r",  jm);
    d_equ_z   = Kokkos::View<double*,  DeviceType>("equ_z",  km);
    d_equ_psi = Kokkos::View<double**, DeviceType>("equ_psi", km, jm);
    d_equ_jm = jm;
    d_equ_km = km;
  }

  auto h_r   = Kokkos::create_mirror_view(d_equ_r);
  auto h_z   = Kokkos::create_mirror_view(d_equ_z);
  auto h_psi = Kokkos::create_mirror_view(d_equ_psi);
  for (int j = 0; j < jm; j++) h_r(j) = equ_data.r[j];
  for (int k = 0; k < km; k++) h_z(k) = equ_data.z[k];
  for (int k = 0; k < km; k++)
    for (int j = 0; j < jm; j++)
      h_psi(k, j) = equ_data.psi[k][j];
  Kokkos::deep_copy(d_equ_r,   h_r);
  Kokkos::deep_copy(d_equ_z,   h_z);
  Kokkos::deep_copy(d_equ_psi, h_psi);

  d_equ_btf = equ_data.btf;
  d_equ_rtf = equ_data.rtf;
  d_has_equilibrium = 1;
}

void ComputePlasmaFieldsKokkos::sync_to_device()
{
  int ng = grid->nlocal;
  if (ng <= 0) return;

  int nc = size_per_grid_cols;
  if (nc == 0) nc = 1;

  // Guard: host arrays may not be allocated yet
  if (size_per_grid_cols == 0 && !vector_grid) return;
  if (size_per_grid_cols > 0 && !array_grid) return;

  if (ng > maxgrid_kk || maxgrid_kk == 0) {
    maxgrid_kk = grid->maxlocal;
    k_array_grid = DAT::tdual_float_2d_lr("plasma_fields:array", maxgrid_kk, nc);
    d_array_grid = k_array_grid.d_view;
  }

  auto h_arr = k_array_grid.h_view;

  if (size_per_grid_cols == 0) {
    for (int i = 0; i < ng; i++)
      h_arr(i, 0) = vector_grid[i];
  } else {
    for (int i = 0; i < ng; i++)
      for (int j = 0; j < size_per_grid_cols; j++)
        h_arr(i, j) = array_grid[i][j];
  }

  k_array_grid.modify_host();
  k_array_grid.sync_device();
  d_array_grid = k_array_grid.d_view;
}
