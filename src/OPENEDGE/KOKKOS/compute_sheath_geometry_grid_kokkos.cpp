/* ----------------------------------------------------------------------
   OpenEdge: sheath geometry per grid cell — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "compute_sheath_geometry_grid_kokkos.h"
#include "grid.h"
#include "memory_kokkos.h"

using namespace SPARTA_NS;

ComputeSheathGeometryGridKokkos::ComputeSheathGeometryGridKokkos(
    SPARTA *sparta, int narg, char **arg) :
  ComputeSheathGeometryGrid(sparta, narg, arg)
{
  kokkos_flag = 1;
  maxgrid_kk = 0;
}

ComputeSheathGeometryGridKokkos::~ComputeSheathGeometryGridKokkos()
{
  if (copymode) return;
}

void ComputeSheathGeometryGridKokkos::compute_per_grid()
{
  ComputeSheathGeometryGrid::compute_per_grid();
  sync_to_device();
}

void ComputeSheathGeometryGridKokkos::sync_to_device()
{
  int ng = grid->nlocal;
  if (ng <= 0) return;
  if (size_per_grid_cols == 0 && !vector_grid) return;
  if (size_per_grid_cols > 0 && !array_grid) return;
  if (!midx_grid) return;

  int nc = size_per_grid_cols;
  if (nc == 0) nc = 1;  // vector_grid case

  if (ng > maxgrid_kk || maxgrid_kk == 0) {
    maxgrid_kk = grid->maxlocal;
    k_array_grid = DAT::tdual_float_2d_lr("sheath_geom:array", maxgrid_kk, nc);
    k_midx_grid = DAT::tdual_int_1d("sheath_geom:midx", maxgrid_kk);
    d_array_grid = k_array_grid.d_view;
    d_midx_grid_kk = k_midx_grid.d_view;
  }

  auto h_arr = k_array_grid.h_view;
  auto h_midx = k_midx_grid.h_view;

  if (size_per_grid_cols == 0) {
    for (int i = 0; i < ng; i++) {
      h_arr(i, 0) = vector_grid[i];
      h_midx(i) = midx_grid[i];
    }
  } else {
    for (int i = 0; i < ng; i++) {
      for (int j = 0; j < size_per_grid_cols; j++)
        h_arr(i, j) = array_grid[i][j];
      h_midx(i) = midx_grid[i];
    }
  }

  k_array_grid.modify_host();
  k_array_grid.sync_device();
  k_midx_grid.modify_host();
  k_midx_grid.sync_device();
  d_array_grid = k_array_grid.d_view;
  d_midx_grid_kk = k_midx_grid.d_view;
}
