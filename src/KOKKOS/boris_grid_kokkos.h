/* ----------------------------------------------------------------------
   OpenEdge Boris grid helper — Kokkos device-callable version.

   Device-callable equivalents of BorisGrid::push_velocity and
   BorisGrid::read_field_from_fix for use in the Kokkos particle
   move kernel.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifndef SPARTA_BORIS_GRID_KOKKOS_H
#define SPARTA_BORIS_GRID_KOKKOS_H

#include "kokkos_type.h"

namespace SPARTA_NS {
namespace BorisGridKokkos {

/* ---------------------------------------------------------------------- */
// Read 3-component field from a Kokkos 2D view (grid or particle indexed).
// d_arr(idx, col) where col is the column offset for the 3 active components.
/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void read_field(const DAT::t_float_2d_lr &d_arr,
                int idx, double out[3])
{
  out[0] = d_arr(idx, 0);
  out[1] = d_arr(idx, 1);
  out[2] = d_arr(idx, 2);
}

/* ---------------------------------------------------------------------- */
// Boris velocity push (leapfrog half-kick scheme).
//
// qm = charge * e / mass  [C/kg]
// dt = timestep [s]
// E[3] = electric field [V/m]
// B[3] = magnetic field [T]
// v[3] = velocity (modified in place) [m/s]
/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void push_velocity(double qm, double dt,
                   const double E[3], const double B[3],
                   double v[3])
{
  // Half E-kick
  double vminus[3] = {
    v[0] + qm * E[0] * 0.5 * dt,
    v[1] + qm * E[1] * 0.5 * dt,
    v[2] + qm * E[2] * 0.5 * dt
  };

  // Rotation vectors
  const double t[3] = {
    qm * B[0] * 0.5 * dt,
    qm * B[1] * 0.5 * dt,
    qm * B[2] * 0.5 * dt
  };
  const double t2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
  const double s[3] = {
    2.0 * t[0] / (1.0 + t2),
    2.0 * t[1] / (1.0 + t2),
    2.0 * t[2] / (1.0 + t2)
  };

  // vminus x t
  double vprime[3];
  vprime[0] = vminus[1]*t[2] - vminus[2]*t[1] + vminus[0];
  vprime[1] = vminus[2]*t[0] - vminus[0]*t[2] + vminus[1];
  vprime[2] = vminus[0]*t[1] - vminus[1]*t[0] + vminus[2];

  // vprime x s
  double vplus[3];
  vplus[0] = vprime[1]*s[2] - vprime[2]*s[1] + vminus[0];
  vplus[1] = vprime[2]*s[0] - vprime[0]*s[2] + vminus[1];
  vplus[2] = vprime[0]*s[1] - vprime[1]*s[0] + vminus[2];

  // Second half E-kick
  v[0] = vplus[0] + qm * E[0] * 0.5 * dt;
  v[1] = vplus[1] + qm * E[1] * 0.5 * dt;
  v[2] = vplus[2] + qm * E[2] * 0.5 * dt;
}

}  // namespace BorisGridKokkos
}  // namespace SPARTA_NS

#endif  // SPARTA_BORIS_GRID_KOKKOS_H
