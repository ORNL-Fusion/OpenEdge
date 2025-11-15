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

#include "fix_gravity.h"
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


FixGravity::FixGravity(SPARTA *sparta, int narg, char **arg)
: Fix(sparta, narg, arg)
{
  // Expected: fix ID group-ID gravity g1 g2 g3
  if (narg < 6)     error->all(FLERR, "Illegal fix gravity: need group-ID and 3 components (g1 g2 g3)");
  
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

int FixGravity::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;  // +½ kick before mover
  mask |= END_OF_STEP;    // +½ kick after mover
  return mask;
}

void FixGravity::init()
{
}

void FixGravity::start_of_step()
{
  half_kick(0.5 * update->dt);
}

void FixGravity::end_of_step()
{
  half_kick(0.5 * update->dt);
  
}

void FixGravity::half_kick(double dt_half)
{
  const int nlocal = particle->nlocal;
  if (nlocal == 0) return;

  auto *const parts = particle->particles;

  const double gx = g_[0], gy = g_[1], gz = g_[2];

  for (int i = 0; i < nlocal; ++i) {
    double *v   = parts[i].v;     // stored as (v_r, v_z, v_phi)
    const double phi = parts[i].x[2];
    const double c = std::cos(phi), s = std::sin(phi);

    // rotate Cartesian g to cylindrical at this phi
    const double gr   =  gx*c + gy*s;
    const double gphi = -gx*s + gy*c;
    // gz stays gz

    v[0] += gr   * dt_half;   // v_r
    v[1] += gz   * dt_half;   // v_z  <<< what your plot uses
    v[2] += gphi * dt_half;   // v_phi
  }
}

double FixGravity::memory_usage()
{
  return 0.0;
}
