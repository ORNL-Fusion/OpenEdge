/* ----------------------------------------------------------------------
   OpenEdge: PMI surface data compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "compute_pmi_surf_data_kokkos.h"

using namespace SPARTA_NS;

ComputePMISurfDataKokkos::ComputePMISurfDataKokkos(
    SPARTA *sparta, int narg, char **arg) :
  ComputePMISurfData(sparta, narg, arg)
{
  kokkos_flag = 1;
}

ComputePMISurfDataKokkos::~ComputePMISurfDataKokkos()
{
  if (copymode) return;
}

void ComputePMISurfDataKokkos::compute_per_surf()
{
  ComputePMISurfData::compute_per_surf();
}
