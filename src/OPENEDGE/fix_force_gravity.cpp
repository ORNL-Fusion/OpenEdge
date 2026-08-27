/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Constant gravity via symmetric splitting:
   v <- v + (g * dt/2) at start_of_step
   mover/Boris runs
   v <- v + (g * dt/2) at end_of_step
------------------------------------------------------------------------- */

#include "fix_force_gravity.h"
#include "update.h"
#include "particle.h"
#include "domain.h"
#include "utils.h"
#include "error.h"
#include "comm.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cmath>

using namespace SPARTA_NS;


FixForceGravity::FixForceGravity(SPARTA *sparta, int narg, char **arg)
: Fix(sparta, narg, arg)
{
  // Expected: fix ID group-ID gravity g1 g2 g3
  if (narg < 6)     error->all(FLERR, "Illegal fix force/gravity: need group-ID and 3 components (g1 g2 g3)");
  
  auto parse_or_die = [&](const char *tok, const char *label) -> double {
    errno = 0;
    char *endp = nullptr;
    double v = std::strtod(tok, &endp);
    // reject empty, trailing junk, or errno set
    if (tok == endp || (endp && *endp != '\0') || errno != 0) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "Bad %s in fix gravity: '%s'", label, tok);
      error->all(FLERR, msg);
    }
    return v;
  };

  // components are at arg[3], arg[4], arg[5]
  g_[0] = parse_or_die(arg[3], "g1");
  g_[1] = parse_or_die(arg[4], "g2");
  g_[2] = parse_or_die(arg[5], "g3");
}

int FixForceGravity::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;  // +½ kick before mover
  mask |= END_OF_STEP;    // +½ kick after mover
  return mask;
}

void FixForceGravity::init()
{
}

void FixForceGravity::start_of_step()
{
  half_kick(0.5 * update->dt);
}

void FixForceGravity::end_of_step()
{
  half_kick(0.5 * update->dt);
  
}

void FixForceGravity::half_kick(double dt_half)
{
  const int nlocal = particle->nlocal;
  if (nlocal == 0) return;

  auto *const parts = particle->particles;

  const double gx = g_[0], gy = g_[1], gz = g_[2];

  // (gx, gy, gz) is interpreted in SPARTA slot order in every mode:
  //   2D Cartesian (legacy): x=R, y=Z, z=phi  -> set gx=g_R, gy=g_Z, gz=g_phi
  //   2D axisymmetric:       x=Z, y=R, z=phi  -> set gx=g_Z, gy=g_R, gz=g_phi
  //   3D Cartesian:          gx, gy, gz are Cartesian components
  // The user is responsible for picking the right slot. This matches the
  // bx/by/bz convention used everywhere else (CLAUDE.md "B-field sources
  // must be in SPARTA coordinate order"). The pre-existing axisymmetric
  // branch mixed slot conventions and was incorrect under SPARTA's true
  // axi mode (which keeps particles in the symmetry plane, so x[2]==0).
  for (int i = 0; i < nlocal; ++i) {
    double *v = parts[i].v;
    v[0] += gx * dt_half;
    v[1] += gy * dt_half;
    v[2] += gz * dt_half;
  }
}

double FixForceGravity::memory_usage()
{
  return 0.0;
}
