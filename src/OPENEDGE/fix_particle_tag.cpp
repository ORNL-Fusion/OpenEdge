/* ----------------------------------------------------------------------
   OpenEdge: fix particle/tag
     fix ID particle/tag <surf-group|none> [psi <psi_N> <reflect/psi-fix-ID> [nevery N]]
   Creates the per-particle int custom "transit" (0/1). The surf-group form
   is applied in Update::move at every surface hit (zero cost elsewhere);
   the psi form checks psi_N(x) < threshold every N steps.
------------------------------------------------------------------------- */

#include "string.h"
#include "fix_particle_tag.h"
#include "fix_reflect_psi.h"
#include "particle.h"
#include "surf.h"
#include "update.h"
#include "modify.h"
#include "input.h"
#include "error.h"

using namespace SPARTA_NS;

enum{INT,DOUBLE};                      // several files

/* ---------------------------------------------------------------------- */

FixParticleTag::FixParticleTag(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal fix particle/tag command");

  groupbit = 0;
  if (strcmp(arg[2],"none") != 0) {
    int igroup = surf->find_group(arg[2]);
    if (igroup < 0) error->all(FLERR,"fix particle/tag: surf group ID does not exist");
    groupbit = surf->bitmask[igroup];
  }

  psi_threshold = -1.0; psi_fix = nullptr; psi_fix_id = nullptr; nevery_psi = 10;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"psi") == 0) {
      if (iarg+2 >= narg) error->all(FLERR,"fix particle/tag psi needs psi_N and a reflect/psi fix ID");
      psi_threshold = input->numeric(FLERR,arg[iarg+1]);
      int n = strlen(arg[iarg+2]) + 1;
      psi_fix_id = new char[n];
      strcpy(psi_fix_id,arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"nevery") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"fix particle/tag nevery needs N");
      nevery_psi = input->inumeric(FLERR,arg[iarg+1]);
      if (nevery_psi < 1) error->all(FLERR,"fix particle/tag nevery must be >= 1");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix particle/tag command");
  }
  if (!groupbit && psi_threshold < 0.0)
    error->all(FLERR,"fix particle/tag: give a surf group or psi");

  // custom int vector, kept across restarts; add_particle() zeroes it
  tag_index = particle->find_custom((char *) "transit");
  if (tag_index < 0) tag_index = particle->add_custom((char *) "transit",INT,0);
}

/* ---------------------------------------------------------------------- */

FixParticleTag::~FixParticleTag()
{
  if (copy || copymode) return;
  delete [] psi_fix_id;
  if (update->tag_index == tag_index) {
    update->tag_index = -1;
    update->tag_groupbit = 0;
    update->tag_psi_thr = -1.0;
  }
  particle->remove_custom(tag_index);
}

/* ---------------------------------------------------------------------- */

int FixParticleTag::setmask()
{
  int mask = 0;
  // under Kokkos the mover applies the psi form on the device (UpdateKokkos::tag_psi_kokkos):
  // a host end_of_step would copy every particle off the device each call
  if (psi_threshold >= 0.0 && !update->tag_psi_device) mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixParticleTag::init()
{
  if (update->tag_index >= 0 && update->tag_index != tag_index)
    error->all(FLERR,"only one fix particle/tag is allowed");
  update->tag_index = tag_index;
  update->tag_groupbit = groupbit;
  update->tag_psi_thr = psi_threshold;
  update->tag_psi_nevery = nevery_psi;
  if (psi_fix_id) {
    int ifix = modify->find_fix(psi_fix_id);
    if (ifix < 0) error->all(FLERR,"fix particle/tag: reflect/psi fix ID not found");
    psi_fix = dynamic_cast<FixReflectPsi *>(modify->fix[ifix]);
    if (!psi_fix) error->all(FLERR,"fix particle/tag: psi fix is not reflect/psi");
  }
}

/* ---------------------------------------------------------------------- */

void FixParticleTag::end_of_step()
{
  if (!psi_fix || update->ntimestep % nevery_psi) return;
  int *transit = particle->eivec[particle->ewhich[tag_index]];
  Particle::OnePart *particles = particle->particles;
  const int nlocal = particle->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (transit[i]) continue;
    if (psi_fix->psi_norm_at_sparta(particles[i].x) < psi_threshold) transit[i] = 1;
  }
}
