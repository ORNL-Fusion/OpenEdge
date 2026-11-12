/* ----------------------------------------------------------------------
   OpenEdge / SPARTA
   Toroidal periodic surface collision model for phi-face caps.
   Rotates particle position and velocity by +/-dphi around the Z axis.
------------------------------------------------------------------------- */

#ifdef SURF_COLLIDE_CLASS

SurfCollideStyle(toroidal,SurfCollideToroidal)

#else

#ifndef SPARTA_SURF_COLLIDE_TOROIDAL_H
#define SPARTA_SURF_COLLIDE_TOROIDAL_H

#include "surf_collide.h"
#include "particle.h"

namespace SPARTA_NS {

class SurfCollideToroidal : public SurfCollide {
 public:
  SurfCollideToroidal(class SPARTA *, int, char **);
  SurfCollideToroidal(class SPARTA *sparta) : SurfCollide(sparta) {} // needed for Kokkos
  virtual ~SurfCollideToroidal() {}
  Particle::OnePart *collide(Particle::OnePart *&, double &,
                             int, double *, int, int &);
  double rot_cos,rot_sin;                        // rotation applied by the last collide
  double wedge_angle() const { return dphi_rad; }
  double wedge_phi_min() const { return phi_mid - 0.5*dphi_rad; }

 protected:              // read by SurfCollideToroidalKokkos
  double dphi_rad;      // wedge angle in radians
  double cos_dphi;      // cos(dphi)
  double sin_dphi;      // sin(dphi)
  double phi_mid;       // midpoint angle = dphi/2 (phi_min assumed 0)
};

}

#endif
#endif
