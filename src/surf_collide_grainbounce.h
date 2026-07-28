/* ----------------------------------------------------------------------
   OpenEdge: surf_collide grainbounce — grain-wall bounce/sticking
   (DUSTT-style restitution).

   surf_collide ID grainbounce pstick pvel pdiff [pmass V] [ptemp V]

     pstick = sticking probability per impact (stuck grains are removed)
     pvel   = velocity restitution |v'|/|v|
     pdiff  = diffuse fraction (else mirror reflection)
     pmass  = radius retention per bounce, R' = pmass * R  (default 1.0)
     ptemp  = temperature retention T' = ptemp * T          (default 1.0)

   Non-grain particles (radius <= 0) vanish, so the style can be applied
   to walls that also absorb vapor/ions.
------------------------------------------------------------------------- */

#ifdef SURF_COLLIDE_CLASS

SurfCollideStyle(grainbounce,SurfCollideGrainBounce)

#else

#ifndef SPARTA_SURF_COLLIDE_GRAINBOUNCE_H
#define SPARTA_SURF_COLLIDE_GRAINBOUNCE_H

#include "surf_collide.h"

namespace SPARTA_NS {

class SurfCollideGrainBounce : public SurfCollide {
 public:
  SurfCollideGrainBounce(class SPARTA *, int, char **);
  ~SurfCollideGrainBounce() override;
  Particle::OnePart *collide(Particle::OnePart *&, double &,
                             int, double *, int, int &) override;

 private:
  double pstick_, pvel_, pdiff_, pmass_, ptemp_;
  class RanKnuth *random_;
};

}

#endif
#endif
