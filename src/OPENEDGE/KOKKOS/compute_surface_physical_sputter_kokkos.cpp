/* ----------------------------------------------------------------------
   OpenEdge: PMI surface data compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "compute_surface_physical_sputter_kokkos.h"

using namespace SPARTA_NS;

ComputeSurfacePhysicalSputterKokkos::ComputeSurfacePhysicalSputterKokkos(
    SPARTA *sparta, int narg, char **arg) :
  ComputeSurfacePhysicalSputter(sparta, narg, arg)
{
  kokkos_flag = 1;
}

ComputeSurfacePhysicalSputterKokkos::~ComputeSurfacePhysicalSputterKokkos()
{
  if (copymode) return;
}

void ComputeSurfacePhysicalSputterKokkos::compute_per_surf()
{
  ComputeSurfacePhysicalSputter::compute_per_surf();
}
