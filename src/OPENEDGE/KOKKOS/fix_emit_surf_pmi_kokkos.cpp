/* ----------------------------------------------------------------------
   OpenEdge: PMI surface emission — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "fix_emit_surf_pmi_kokkos.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"
#include "kokkos.h"

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
  // Emit creates new particles via add_particle() → grow().
  // With Kokkos, ParticleKokkos::grow_custom() must resize DualViews.
  // Force host sync and let ParticleKokkos handle allocation.
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Host, ALL_MASK);

  // Temporarily enable prewrap so particle growth uses host-safe path
  int save_prewrap = sparta->kokkos->prewrap;
  sparta->kokkos->prewrap = 1;

  FixEmitSurfPmi::perform_task();

  sparta->kokkos->prewrap = save_prewrap;

  particle_kk->modify(Host, ALL_MASK);
  particle_kk->sorted_kk = 0;
}
