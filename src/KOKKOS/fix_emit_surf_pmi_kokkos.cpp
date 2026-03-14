/* ----------------------------------------------------------------------
   OpenEdge: PMI surface emission — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "fix_emit_surf_pmi_kokkos.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

FixEmitSurfPmiKokkos::FixEmitSurfPmiKokkos(SPARTA *sparta, int narg, char **arg) :
  FixEmitSurfPmi(sparta, narg, arg)
{
  kokkos_flag = 1;
}

FixEmitSurfPmiKokkos::~FixEmitSurfPmiKokkos()
{
  if (copymode) return;
}

void FixEmitSurfPmiKokkos::perform_task()
{
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);

  FixEmitSurfPmi::perform_task();

  particle_kk->modify(Host, PARTICLE_MASK);
}
