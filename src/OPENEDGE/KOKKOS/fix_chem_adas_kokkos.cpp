/* ----------------------------------------------------------------------
   OpenEdge: ADAS ionization/recombination — Kokkos wrapper.
   Runs the CPU ADAS logic then syncs particle data back to device.
------------------------------------------------------------------------- */

#include "fix_chem_adas_kokkos.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

FixChemAdasKokkos::FixChemAdasKokkos(SPARTA *sparta, int narg, char **arg) :
  FixChemAdas(sparta, narg, arg)
{
  kokkos_flag = 1;
}

FixChemAdasKokkos::~FixChemAdasKokkos()
{
  if (copymode) return;
}

void FixChemAdasKokkos::end_of_step()
{
  // Sync particles to host for CPU-side ADAS processing
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);

  // Run CPU ADAS (modifies particle species via ip->ispecies)
  FixChemAdas::end_of_step();

  // Mark particles modified on host so next device sync picks up species changes
  particle_kk->modify(Host, PARTICLE_MASK);
}
