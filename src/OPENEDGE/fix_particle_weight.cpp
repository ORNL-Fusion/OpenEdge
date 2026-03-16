/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#include "string.h"
#include "fix_particle_weight.h"
#include "particle.h"
#include "update.h"
#include "error.h"

using namespace SPARTA_NS;

enum{INT,DOUBLE};                      // several files

/* ---------------------------------------------------------------------- */

FixParticleWeight::FixParticleWeight(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 2) error->all(FLERR,"Illegal fix particle/weight command");

  // check if pweight custom attribute already exists (e.g. from restart)
  // if not, create it as a per-particle double vector

  pweight_index = particle->find_custom((char *) "pweight");

  if (pweight_index < 0)
    pweight_index = particle->add_custom((char *) "pweight",DOUBLE,0);

  pweight_ewhich = particle->ewhich[pweight_index];
}

/* ---------------------------------------------------------------------- */

FixParticleWeight::~FixParticleWeight()
{
  if (copy || copymode) return;
  particle->remove_custom(pweight_index);
}

/* ---------------------------------------------------------------------- */

int FixParticleWeight::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixParticleWeight::init()
{
  // refresh ewhich in case custom arrays were reallocated
  pweight_ewhich = particle->ewhich[pweight_index];
}
