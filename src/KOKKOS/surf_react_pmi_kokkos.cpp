/* ----------------------------------------------------------------------
   OpenEdge: PMI surface reaction — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "surf_react_pmi_kokkos.h"

using namespace SPARTA_NS;

SurfReactPMIKokkos::SurfReactPMIKokkos(SPARTA *sparta, int narg, char **arg) :
  SurfReactPMI(sparta, narg, arg)
{
  kokkosable = 1;
}

SurfReactPMIKokkos::~SurfReactPMIKokkos()
{
  if (copymode) return;
}
