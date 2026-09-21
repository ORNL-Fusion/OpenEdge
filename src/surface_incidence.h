/* ----------------------------------------------------------------------
   Magnetic-field incidence helpers for OpenEdge surface source models.
------------------------------------------------------------------------- */

#ifndef SPARTA_OPENEDGE_SURFACE_INCIDENCE_H
#define SPARTA_OPENEDGE_SURFACE_INCIDENCE_H

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
double surface_incidence_sine(SurfaceIncidenceMode mode,
                              double br, double bt, double bz,
                              double nr, double nt, double nz,
                              double u_parallel);

} // namespace OpenEdge

#endif
