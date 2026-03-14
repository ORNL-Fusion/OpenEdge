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
