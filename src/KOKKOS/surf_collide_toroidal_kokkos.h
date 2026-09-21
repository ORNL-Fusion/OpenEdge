/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.

   OpenEdge: Kokkos twin of surf_collide toroidal (periodic sector caps).
------------------------------------------------------------------------- */

#ifdef SURF_COLLIDE_CLASS

SurfCollideStyle(toroidal/kk,SurfCollideToroidalKokkos)

#else

#ifndef SPARTA_SURF_COLLIDE_TOROIDAL_KOKKOS_H
#define SPARTA_SURF_COLLIDE_TOROIDAL_KOKKOS_H

#include "surf_collide_toroidal.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class SurfCollideToroidalKokkos : public SurfCollideToroidal {
 public:
  SurfCollideToroidalKokkos(class SPARTA *, int, char **);
  SurfCollideToroidalKokkos(class SPARTA *);
  ~SurfCollideToroidalKokkos() {}
  void pre_collide();
  void post_collide();

 private:
  DAT::tdual_int_scalar k_nsingle;
  DAT::t_int_scalar d_nsingle;
  HAT::t_int_scalar h_nsingle;

 public:

  /* ----------------------------------------------------------------------
     particle collision with a phi-face cap surface (device twin of
     SurfCollideToroidal::collide). ip->x = collision point on the cap;
     decide which cap from atan2(y,x) vs the wedge midpoint and rotate x
     and v about the machine axis by +/-dphi onto the opposite cap.
     No reactions (allowreact = 0), no new particle. The caller (move
     kernel) relocates the teleported particle into its new grid cell,
     exactly as the CPU mover does after this model.
     The CPU model also invalidates hybrid/GCA guiding-center state; the
     GCA pusher does not exist under Kokkos (removed 2026-08-26).
  ------------------------------------------------------------------------- */

  template<int REACT, int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  Particle::OnePart* collide_kokkos(Particle::OnePart *&ip, double &,
                                    int, const double *, int, int &reaction,
                                    const DAT::t_int_scalar &, const DAT::t_int_scalar &) const
  {
    if (ATOMIC_REDUCTION == 0)
      d_nsingle()++;
    else
      Kokkos::atomic_inc(&d_nsingle());

    reaction = 0;

    double *x = ip->x;
    double *v = ip->v;

    const double phi = atan2(x[1],x[0]);
    const double cos_d = cos_dphi;
    const double sin_d = (phi < phi_mid) ? sin_dphi : -sin_dphi;

    const double xold = x[0], yold = x[1];
    x[0] = xold*cos_d - yold*sin_d;
    x[1] = xold*sin_d + yold*cos_d;

    const double vxold = v[0], vyold = v[1];
    v[0] = vxold*cos_d - vyold*sin_d;
    v[1] = vxold*sin_d + vyold*cos_d;

    return NULL;
  }
};

}

#endif
#endif
