/* ----------------------------------------------------------------------
   OpenEdge Boris grid helper
   Keeps Boris math and field extraction out of Update implementation.
------------------------------------------------------------------------- */

#ifndef SPARTA_BORIS_GRID_H
#define SPARTA_BORIS_GRID_H

#include "fix.h"
#include "math_extra.h"

namespace SPARTA_NS {
namespace BorisGrid {

inline void read_field_from_fix(Fix *fix, int use_grid, const int active[3],
                                int iparticle, int icell, double out[3])
{
  out[0] = out[1] = out[2] = 0.0;
  if (!fix) return;

  double **arr = use_grid ? fix->array_grid : fix->array_particle;
  if (!arr) return;

  const int idx = use_grid ? icell : iparticle;
  int col = 0;
  if (active[0]) out[0] = arr[idx][col++];
  if (active[1]) out[1] = arr[idx][col++];
  if (active[2]) out[2] = arr[idx][col++];
}

inline void push_velocity(double qm, double dt,
                          const double E[3], const double B[3],
                          double v[3])
{
  double vminus[3] = {
    v[0] + qm * E[0] * 0.5 * dt,
    v[1] + qm * E[1] * 0.5 * dt,
    v[2] + qm * E[2] * 0.5 * dt
  };

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

  double vprime[3], vplus[3];
  MathExtra::cross3(vminus, t, vprime);
  vprime[0] += vminus[0];
  vprime[1] += vminus[1];
  vprime[2] += vminus[2];

  MathExtra::cross3(vprime, s, vplus);
  vplus[0] += vminus[0];
  vplus[1] += vminus[1];
  vplus[2] += vminus[2];

  v[0] = vplus[0] + qm * E[0] * 0.5 * dt;
  v[1] = vplus[1] + qm * E[1] * 0.5 * dt;
  v[2] = vplus[2] + qm * E[2] * 0.5 * dt;
}

}  // namespace BorisGrid
}  // namespace SPARTA_NS

#endif  // SPARTA_BORIS_GRID_H
