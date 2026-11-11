/* ----------------------------------------------------------------------
   OpenEdge: fix particle/tag — per-particle "transit" flag, set once a
   marker hits a surface of a group (e.g. the core cut) or, optionally,
   enters psi_N below a threshold; read by surf_react surface/pwi ehist
   to split wall impacts into locally recycled and core-transited.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(particle/tag,FixParticleTag)

#else

#ifndef SPARTA_FIX_PARTICLE_TAG_H
#define SPARTA_FIX_PARTICLE_TAG_H

#include "fix.h"

namespace SPARTA_NS {

class FixParticleTag : public Fix {
 public:
  FixParticleTag(class SPARTA *, int, char **);
  ~FixParticleTag();
  int setmask();
  void init();
  void end_of_step();

  int tag_index;                 // custom "transit" (int vector)
  int groupbit;                  // surf group mask, 0 = none
  double psi_threshold;          // < 0 = off
  class FixReflectPsi *psi_fix;  // supplies psi_N(x) for the psi form
  int nevery_psi;

 private:
  char *psi_fix_id;
};

}

#endif
#endif
