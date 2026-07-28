/* ----------------------------------------------------------------------
   OpenEdge: surf_collide grainbounce — see header for the model.
------------------------------------------------------------------------- */

#include "surf_collide_grainbounce.h"
#include "particle.h"
#include "update.h"
#include "comm.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "random_knuth.h"
#include "random_mars.h"

#include <cmath>
#include <cstring>

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

SurfCollideGrainBounce::SurfCollideGrainBounce(SPARTA *sparta, int narg,
                                               char **arg) :
  SurfCollide(sparta, narg, arg)
{
  if (narg < 5)
    error->all(FLERR,
      "surf_collide grainbounce: need pstick pvel pdiff [pmass V] [ptemp V]");

  pstick_ = input->numeric(FLERR, arg[2]);
  pvel_   = input->numeric(FLERR, arg[3]);
  pdiff_  = input->numeric(FLERR, arg[4]);
  pmass_  = 1.0;
  ptemp_  = 1.0;

  if (pstick_ < 0.0 || pstick_ > 1.0 || pdiff_ < 0.0 || pdiff_ > 1.0 ||
      pvel_ < 0.0 || pvel_ > 1.0)
    error->all(FLERR, "surf_collide grainbounce: probabilities must be in [0,1]");

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "pmass") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "grainbounce: pmass V");
      pmass_ = input->numeric(FLERR, arg[iarg+1]);
      if (pmass_ <= 0.0 || pmass_ > 1.0)
        error->all(FLERR, "grainbounce: pmass must be in (0,1]");
      iarg += 2;
    } else if (strcmp(arg[iarg], "ptemp") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "grainbounce: ptemp V");
      ptemp_ = input->numeric(FLERR, arg[iarg+1]);
      if (ptemp_ <= 0.0) error->all(FLERR, "grainbounce: ptemp must be > 0");
      iarg += 2;
    } else error->all(FLERR, "surf_collide grainbounce: unknown keyword");
  }

  random_ = new RanKnuth(update->ranmaster->uniform());
  random_->reset(comm->me + 1, comm->me, 100);
}

SurfCollideGrainBounce::~SurfCollideGrainBounce()
{
  delete random_;
}

/* ----------------------------------------------------------------------
   ip = incident particle. Grains (radius > 0): stick with prob pstick,
   else reflect (pdiff diffuse cosine-law : mirror) with velocity
   restitution pvel, radius retention pmass, temperature retention ptemp.
   Non-grain particles vanish.
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollideGrainBounce::
collide(Particle::OnePart *&ip, double &, int, double *norm, int, int &)
{
  nsingle++;

  if (!(ip->radius > 0.0) || random_->uniform() < pstick_) {
    ip = NULL;
    return NULL;
  }

  double *v = ip->v;
  const double dot = v[0]*norm[0] + v[1]*norm[1] + v[2]*norm[2];
  const double vmag_in = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  const double vmag = pvel_ * vmag_in;

  if (random_->uniform() < pdiff_) {
    // diffuse: cosine-law about the surface normal
    double t1[3] = {v[0] - dot*norm[0], v[1] - dot*norm[1], v[2] - dot*norm[2]};
    double t1m = std::sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    if (t1m < 1.0e-16) {
      // grazing-degenerate: build any tangent
      if (std::fabs(norm[0]) < 0.9) { t1[0] = 0.0; t1[1] = -norm[2]; t1[2] = norm[1]; }
      else                          { t1[0] = -norm[2]; t1[1] = 0.0; t1[2] = norm[0]; }
      t1m = std::sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    }
    for (int k = 0; k < 3; k++) t1[k] /= t1m;
    double t2[3] = {norm[1]*t1[2] - norm[2]*t1[1],
                    norm[2]*t1[0] - norm[0]*t1[2],
                    norm[0]*t1[1] - norm[1]*t1[0]};
    const double u1 = random_->uniform();
    const double ct = std::sqrt(u1);
    const double st = std::sqrt(1.0 - u1);
    const double ph = 2.0 * MY_PI * random_->uniform();
    for (int k = 0; k < 3; k++)
      v[k] = vmag * (ct*norm[k] + st*(std::cos(ph)*t1[k] + std::sin(ph)*t2[k]));
  } else {
    // mirror reflection with restitution
    for (int k = 0; k < 3; k++) v[k] = pvel_ * (v[k] - 2.0*dot*norm[k]);
  }

  if (pmass_ < 1.0) {
    ip->radius *= pmass_;
    ip->mass   *= pmass_ * pmass_ * pmass_;
  }
  if (ptemp_ != 1.0) ip->temp *= ptemp_;

  return NULL;
}
