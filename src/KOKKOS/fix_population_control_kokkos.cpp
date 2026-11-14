/* ----------------------------------------------------------------------
   fix population/control/kk — see header.
------------------------------------------------------------------------- */

#include "fix_population_control_kokkos.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixPopulationControlKokkos::FixPopulationControlKokkos(SPARTA *sparta, int narg, char **arg) :
  FixPopulationControl(sparta, narg, arg)
{
  // host fix: ModifyKokkos brings particles and customs to the host before
  // end_of_step and marks the host copy modified after it
  kokkos_flag = 0;
  execution_space = Host;
  datamask_read = PARTICLE_MASK | SPECIES_MASK | CUSTOM_MASK;
  datamask_modify = PARTICLE_MASK | CUSTOM_MASK;
}

/* ---------------------------------------------------------------------- */

void FixPopulationControlKokkos::end_of_step()
{
  FixPopulationControl::end_of_step();
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;      // the device cell lists no longer match
}
