/* ----------------------------------------------------------------------
   Magnetic-field incidence helpers for OpenEdge surface source models.
------------------------------------------------------------------------- */

#ifndef SPARTA_OPENEDGE_SURFACE_INCIDENCE_H
#define SPARTA_OPENEDGE_SURFACE_INCIDENCE_H

#include <algorithm>
#include <cmath>

namespace OpenEdge {

enum class SurfaceIncidenceMode {
  POLOIDAL = 0,   // legacy: |Br nr + Bz nz| / |B|
  FULL_3D = 1,    // geometry only: |B . n| / |B|
  DIRECTED_3D = 2 // one-sided: max(0, -sign(u_parallel) Bhat . n)
};

// Return sin(alpha), where alpha is the grazing angle measured from the
// surface plane.  `n` is the outward unit normal (toward the plasma), and
// signed u_parallel is positive along B.  DIRECTED_3D therefore admits only
// velocity directions that point into the material (v . n < 0).
inline double surface_incidence_sine(SurfaceIncidenceMode mode,
                                     double br, double bt, double bz,
                                     double nr, double nt, double nz,
                                     double u_parallel)
{
  const double bmag = std::sqrt(br*br + bt*bt + bz*bz);
  if (!(bmag > 0.0) || !std::isfinite(bmag)) return 0.0;

  double incidence = 0.0;
  if (mode == SurfaceIncidenceMode::POLOIDAL) {
    incidence = std::fabs(br*nr + bz*nz) / bmag;
  } else {
    const double bdotn = br*nr + bt*nt + bz*nz;
    if (mode == SurfaceIncidenceMode::FULL_3D) {
      incidence = std::fabs(bdotn) / bmag;
    } else {
      if (!std::isfinite(u_parallel) || u_parallel == 0.0) return 0.0;
      const double flow_sign = (u_parallel > 0.0) ? 1.0 : -1.0;
      incidence = -flow_sign * bdotn / bmag;
    }
  }

  if (!std::isfinite(incidence)) return 0.0;
  return std::min(1.0, std::max(0.0, incidence));
}

} // namespace OpenEdge

#endif
