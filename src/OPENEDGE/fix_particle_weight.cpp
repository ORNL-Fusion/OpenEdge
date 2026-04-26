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

  // Default pweight = 1.0 for any particle whose weight is still zero.
  // particle->add_particle() zeros all custom attributes, so an unweighted
  // emitter (fix emit/face etc.) leaves pweight at 0 — without this hook
  // compute grid/weighted nrho_w would tally zero on those particles.
  // fix surface/emit/source overrides with a real per-particle weight, so
  // this default never clobbers a non-zero value.
  flag_update_custom = 1;
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

/* ---------------------------------------------------------------------- */

void FixParticleWeight::setup()
{
  // Initialize pweight = 1.0 for all existing particles whose weight is
  // still zero (e.g. those created by `create_particles` before the fix
  // was active). update_custom() handles particles emitted later.
  pweight_ewhich = particle->ewhich[pweight_index];
  double *pweight_dvec = particle->edvec[pweight_ewhich];
  int nlocal = particle->nlocal;
  for (int i = 0; i < nlocal; i++)
    if (pweight_dvec[i] == 0.0) pweight_dvec[i] = 1.0;
}

/* ---------------------------------------------------------------------- */

void FixParticleWeight::update_custom(int index, double, double, double,
                                      double *)
{
  // Set default weight on newly-emitted particles. Skip if a previous
  // writer (e.g. fix surface/emit/source) already set a non-zero value.
  pweight_ewhich = particle->ewhich[pweight_index];
  double *pweight_dvec = particle->edvec[pweight_ewhich];
  if (pweight_dvec[index] == 0.0) pweight_dvec[index] = 1.0;
}
