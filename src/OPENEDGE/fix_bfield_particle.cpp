/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_bfield_particle.h"
#include "particle.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixBfieldParticle::FixBfieldParticle(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 5) error->all(FLERR,"Illegal fix field/particle command");

  int ncols = 0;

  if (strcmp(arg[2],"NULL") == 0) bxstr = NULL;
  else {
    int n = strlen(arg[2]) + 1;
    bxstr = new char[n];
    strcpy(bxstr,arg[2]);
    ncols++;
  }
  if (strcmp(arg[3],"NULL") == 0) bystr = NULL;
  else {
    int n = strlen(arg[3]) + 1;
    bystr = new char[n];
    strcpy(bystr,arg[3]);
    ncols++;
  }
  if (strcmp(arg[4],"NULL") == 0) bzstr = NULL;
  else {
    int n = strlen(arg[4]) + 1;
    bzstr = new char[n];
    strcpy(bzstr,arg[4]);
    ncols++;
  }

  // fix settings

  per_particle_flag = 1;
  size_per_particle_cols = ncols;
  per_particle_freq = 1;
  per_particle_field = 1;

  field_active[0] = field_active[1] = field_active[2] = 0;
  if (bxstr) field_active[0] = 1;
  if (bystr) field_active[1] = 1;
  if (bzstr) field_active[2] = 1;

  // per-particle memory initialization

  maxparticle = 0;
  array_particle = NULL;
}

/* ---------------------------------------------------------------------- */

FixBfieldParticle::~FixBfieldParticle()
{
  delete [] bxstr;
  delete [] bystr;
  delete [] bzstr;

  memory->destroy(array_particle);
}

/* ---------------------------------------------------------------------- */

int FixBfieldParticle::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBfieldParticle::init()
{
  // check if all variables exist and are particle-style vars

  if (bxstr) {
    bxvar = input->variable->find(bxstr);
    if (bxvar < 0)
      error->all(FLERR,"Variable name for fix field/particle does not exist");
    if (!input->variable->particle_style(bxvar))
      error->all(FLERR,"Variable for fix field/particle is invalid style");
  }
  if (bystr) {
    byvar = input->variable->find(bystr);
    if (byvar < 0)
      error->all(FLERR,"Variable name for fix field/particle does not exist");
    if (!input->variable->particle_style(byvar))
      error->all(FLERR,"Variable for fix field/particle is invalid style");
  }
  if (bzstr) {
    bzvar = input->variable->find(bzstr);
    if (bzvar < 0)
      error->all(FLERR,"Variable name for fix field/particle does not exist");
    if (!input->variable->particle_style(bzvar))
      error->all(FLERR,"Variable for fix field/particle is invalid style");
  }

  // set initial particle values to zero in case dump is performed at step 0

  if (particle->nlocal > maxparticle) {
    maxparticle = particle->maxlocal;
    memory->destroy(array_particle);
    memory->create(array_particle,maxparticle,size_per_particle_cols,
                   "array_particle");
  }

  bigint nbytes = (bigint) particle->nlocal * size_per_particle_cols;
  if (nbytes) memset(&array_particle[0][0],0,nbytes*sizeof(double));
}

/* ---------------------------------------------------------------------- */

void FixBfieldParticle::compute_field()
{
  if (!particle->nlocal) return;

  // reallocate array_particle if necessary

  if (particle->nlocal > maxparticle) {
    maxparticle = particle->maxlocal;
    memory->destroy(array_particle);
    memory->create(array_particle,maxparticle,size_per_particle_cols,
                   "array_particle");
  }

  // evaluate each particle-style variable
  // results are put into strided array_particle

  int stride = size_per_particle_cols;
  int icol = 0;

  if (bxstr) {
    input->variable->compute_particle(bxvar,&array_particle[0][icol],stride,0);
    icol++;
  }

  if (bystr) {
    input->variable->compute_particle(byvar,&array_particle[0][icol],stride,0);
    icol++;
  }

  if (bzstr) {
    input->variable->compute_particle(bzvar,&array_particle[0][icol],stride,0);
    icol++;
  }
}