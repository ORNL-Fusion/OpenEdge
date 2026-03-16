/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(particle/weight,FixParticleWeight)

#else

#ifndef SPARTA_FIX_PARTICLE_WEIGHT_H
#define SPARTA_FIX_PARTICLE_WEIGHT_H

#include "fix.h"

namespace SPARTA_NS {

class FixParticleWeight : public Fix {
 public:
  FixParticleWeight(class SPARTA *, int, char **);
  ~FixParticleWeight();
  int setmask();
  void init();

  int pweight_index;          // index into particle custom data structs

 private:
  int pweight_ewhich;         // index into edvec for fast access
};

}

#endif
#endif
