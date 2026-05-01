/* ----------------------------------------------------------------------
   OpenEdge: surf_collide partial_recycle

   EIRENE-style partial recycling: per particle hitting the surface, with
   probability (1-R) the particle is absorbed (vanishes, counted as pumped
   throughput); with probability R it is diffusely re-emitted at twall,
   optionally as a different species (e.g. atomic D arriving recombines on
   the wall and returns as molecular D2).

   Syntax:
       surf_collide ID partial_recycle Twall acc R [return_species NAME]
                                                    [ke_emit Ub]
------------------------------------------------------------------------- */

#ifdef SURF_COLLIDE_CLASS

SurfCollideStyle(partial_recycle,SurfCollidePartialRecycle)

#else

#ifndef SPARTA_SURF_COLLIDE_PARTIAL_RECYCLE_H
#define SPARTA_SURF_COLLIDE_PARTIAL_RECYCLE_H

#include "surf_collide_diffuse.h"

namespace SPARTA_NS {

class SurfCollidePartialRecycle : public SurfCollideDiffuse {
 public:
  SurfCollidePartialRecycle(class SPARTA *, int, char **);
  SurfCollidePartialRecycle(class SPARTA *sparta) : SurfCollideDiffuse(sparta) {}
  virtual ~SurfCollidePartialRecycle();

  Particle::OnePart *collide(Particle::OnePart *&, double &,
                             int, double *, int, int &);

  // diagnostic counters
  bigint nvanish_local, nrecycle_local;

 protected:
  double r_recycle;                  // recycle probability in [0,1]
  int return_species;                // species index to emit as, -1 = same
  char *return_species_name;         // for deferred lookup at init()
};

}

#endif
#endif
