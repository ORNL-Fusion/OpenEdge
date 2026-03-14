/* ----------------------------------------------------------------------
   OpenEdge: Nanbu Coulomb collisions — Kokkos wrapper.
   Runs the CPU Nanbu logic then syncs particle data back to device.
------------------------------------------------------------------------- */

#include "fix_coll_nanbu_kokkos.h"
#include "particle_kokkos.h"

using namespace SPARTA_NS;

FixCollNanbuKokkos::FixCollNanbuKokkos(SPARTA *sparta, int narg, char **arg) :
  FixCollNanbu(sparta, narg, arg)
{
  kokkos_flag = 1;
}

FixCollNanbuKokkos::~FixCollNanbuKokkos()
{
  if (copymode) return;
}

void FixCollNanbuKokkos::end_of_step()
{
  // Sync particles to host for CPU-side Nanbu collisions
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);

  // Run CPU Nanbu (modifies particle velocities)
  FixCollNanbu::end_of_step();

  // Mark particles modified on host
  particle_kk->modify(Host, PARTICLE_MASK);
}
