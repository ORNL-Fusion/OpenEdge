/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/binary -- binary kinetic-kinetic Coulomb pairs.
------------------------------------------------------------------------- */

#include "fix_coulomb_binary.h"
#include "sparta_masks.h"
#include "error.h"

using namespace SPARTA_NS;

FixCoulombBinary::FixCoulombBinary(SPARTA *sparta, int narg, char **arg) :
  FixCoulombBase(sparta, narg, arg)
{
  do_binary_ = 1;
  have_background_ = 0;
  // Kokkos host-fix sandwich: customs may be read (custom plasma sources) but
  // only velocities are written
  datamask_read = PARTICLE_MASK | SPECIES_MASK | CUSTOM_MASK;
  datamask_modify = PARTICLE_MASK;
  if (iarg_after_common_ < narg)
    error->all(FLERR,
      "fix coulomb/binary: extra arguments after plasma block "
      "(use fix coulomb/background for fluid-background mode)");
}
