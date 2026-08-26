#include "surface_incidence.h"

#include <cassert>
#include <cmath>

namespace {

bool close(double a, double b, double tol = 1.0e-13)
{
  return std::fabs(a-b) <= tol;
}

} // namespace

int main()
{
  using OpenEdge::SurfaceIncidenceMode;
  using OpenEdge::surface_incidence_sine;

  const double tilt = M_PI / 180.0;
  const double st = std::sin(tilt);
  const double ct = std::cos(tilt);

  // A purely toroidal field is invisible to the legacy poloidal model.
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::POLOIDAL,
                                      0.0, 1.0, 0.0,
                                      0.0, -st, ct, 1.0), 0.0));
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::FULL_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, -st, ct, 1.0), st));

  // Outward normals point into the plasma. Positive flow along +B hits the
  // -phi-tilted face and misses the +phi-tilted face; reversing u_parallel
  // swaps those results.
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::DIRECTED_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, -st, ct, 1.0), st));
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::DIRECTED_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, st, ct, 1.0), 0.0));
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::DIRECTED_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, -st, ct, -1.0), 0.0));
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::DIRECTED_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, st, ct, -1.0), st));

  // With no toroidal B or surface-normal component, full3d exactly reduces
  // to the legacy axisymmetric projection.
  const double legacy = surface_incidence_sine(SurfaceIncidenceMode::POLOIDAL,
                                                3.0, 0.0, 4.0,
                                                0.6, 0.0, 0.8, 1.0);
  const double full = surface_incidence_sine(SurfaceIncidenceMode::FULL_3D,
                                              3.0, 0.0, 4.0,
                                              0.6, 0.0, 0.8, 1.0);
  assert(close(legacy, full));

  // No signed flow direction means no one-sided directed source.
  assert(close(surface_incidence_sine(SurfaceIncidenceMode::DIRECTED_3D,
                                      0.0, 1.0, 0.0,
                                      0.0, -st, ct, 0.0), 0.0));
  return 0;
}
