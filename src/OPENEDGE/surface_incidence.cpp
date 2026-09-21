/* ----------------------------------------------------------------------
   Magnetic-field incidence helpers for OpenEdge surface source models.
------------------------------------------------------------------------- */

#include "surface_incidence.h"

#include <algorithm>
#include <cmath>

namespace OpenEdge {

double surface_incidence_sine(const SurfaceIncidenceMode mode,
                              const double br, const double bt,
                              const double bz, const double nr,
                              const double nt, const double nz,
                              const double u_parallel)
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
      const double flow_sign = u_parallel > 0.0 ? 1.0 : -1.0;
      incidence = -flow_sign * bdotn / bmag;
    }
  }

  if (!std::isfinite(incidence)) return 0.0;
  return std::min(1.0, std::max(0.0, incidence));
}

} // namespace OpenEdge
