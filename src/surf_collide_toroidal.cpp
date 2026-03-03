/* ----------------------------------------------------------------------
   OpenEdge / SPARTA
   Toroidal periodic surface collision model for phi-face caps.

   Usage: surf_collide ID toroidal dphi_deg [phi_min_deg]
     dphi_deg    = toroidal wedge angle in degrees (required)
     phi_min_deg = starting phi angle in degrees (optional, default 0)

   When a particle hits a phi-face cap, it is teleported to the
   opposite cap with position and velocity rotated by +/-dphi
   around the Z axis (tokamak symmetry axis).
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "surf_collide_toroidal.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfCollideToroidal::
SurfCollideToroidal(SPARTA *sparta, int narg, char **arg) :
  SurfCollide(sparta, narg, arg)
{
  if (narg < 3 || narg > 4)
    error->all(FLERR,"Illegal surf_collide toroidal command");

  double dphi_deg = atof(arg[2]);
  double phi_min_deg = 0.0;
  if (narg == 4) phi_min_deg = atof(arg[3]);

  dphi_rad = dphi_deg * M_PI / 180.0;
  cos_dphi = cos(dphi_rad);
  sin_dphi = sin(dphi_rad);

  double phi_min_rad = phi_min_deg * M_PI / 180.0;
  phi_mid = phi_min_rad + 0.5 * dphi_rad;

  allowreact = 0;
  transparent = 0;
}

/* ----------------------------------------------------------------------
   particle collision with a phi-face cap surface
   ip = particle with current x = collision pt, current v = incident v
   Determine which cap was hit from particle phi angle, then rotate
   position and velocity by +/-dphi to teleport to the opposite cap.
   return NULL (no new particle created)
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollideToroidal::
collide(Particle::OnePart *&ip, double &, int, double *, int, int &)
{
  double *x = ip->x;
  double *v = ip->v;

  // determine which cap: compare particle phi to midpoint of wedge
  double phi = atan2(x[1], x[0]);

  double cos_d, sin_d;
  if (phi < phi_mid) {
    // hit phi_min cap -> rotate by +dphi to phi_max
    cos_d = cos_dphi;
    sin_d = sin_dphi;
  } else {
    // hit phi_max cap -> rotate by -dphi to phi_min
    cos_d = cos_dphi;
    sin_d = -sin_dphi;
  }

  // rotate position in XY plane (Z unchanged)
  double xold = x[0];
  double yold = x[1];
  x[0] = xold * cos_d - yold * sin_d;
  x[1] = xold * sin_d + yold * cos_d;

  // rotate velocity in XY plane (vz unchanged)
  double vxold = v[0];
  double vyold = v[1];
  v[0] = vxold * cos_d - vyold * sin_d;
  v[1] = vxold * sin_d + vyold * cos_d;

  nsingle++;

  return NULL;
}
