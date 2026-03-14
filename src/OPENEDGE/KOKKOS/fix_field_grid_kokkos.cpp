/* ----------------------------------------------------------------------
   OpenEdge: generic grid field fix — Kokkos version.
------------------------------------------------------------------------- */

#include "fix_field_grid_kokkos.h"
#include "grid.h"
#include "memory_kokkos.h"
#include "error.h"

using namespace SPARTA_NS;

FixFieldGridKokkos::FixFieldGridKokkos(SPARTA *sparta, int narg, char **arg) :
  FixFieldGrid(sparta, narg, arg)
{
  kokkos_flag = 1;
  maxgrid_kk = 0;
}

FixFieldGridKokkos::~FixFieldGridKokkos()
{
  if (copymode) return;
}

void FixFieldGridKokkos::init()
{
  FixFieldGrid::init();
  int ng = grid->maxlocal;
  int nc = size_per_grid_cols;
  if (ng > maxgrid_kk || maxgrid_kk == 0) {
    maxgrid_kk = ng;
    k_array_grid = DAT::tdual_float_2d_lr("field_grid:array", maxgrid_kk, nc);
    d_array_grid = k_array_grid.d_view;
  }
}

void FixFieldGridKokkos::compute_field()
{
  FixFieldGrid::compute_field();
  if (!grid->nlocal) return;

  int ng = grid->nlocal;
  int nc = size_per_grid_cols;
  if (ng > maxgrid_kk) {
    maxgrid_kk = grid->maxlocal;
    k_array_grid = DAT::tdual_float_2d_lr("field_grid:array", maxgrid_kk, nc);
    d_array_grid = k_array_grid.d_view;
  }

  auto h_array = k_array_grid.h_view;
  for (int i = 0; i < ng; i++)
    for (int j = 0; j < nc; j++)
      h_array(i, j) = array_grid[i][j];

  k_array_grid.modify_host();
  k_array_grid.sync_device();
  d_array_grid = k_array_grid.d_view;
}
