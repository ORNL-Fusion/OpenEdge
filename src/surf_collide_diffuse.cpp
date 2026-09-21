/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include <cmath>
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "surf_collide_diffuse.h"
#include "surf.h"
#include "surf_react.h"
#include "input.h"
#include "variable.h"
#include "particle.h"
#include "domain.h"
#include "update.h"
#include "pusher.h"
#include "modify.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_const.h"
#include "math_extra.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{NUMERIC,CUSTOM,VARIABLE,VAREQUAL,VARSURF};   // surf_collide classes

/* ---------------------------------------------------------------------- */

SurfCollideDiffuse::SurfCollideDiffuse(SPARTA *sparta, int narg, char **arg) :
  SurfCollide(sparta, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal surf_collide diffuse command");

  parse_tsurf(arg[2]);

  acc = input->numeric(FLERR,arg[3]);
  if (acc < 0.0 || acc > 1.0)
    error->all(FLERR,"Illegal surf_collide diffuse command");

  // optional args

  tflag = rflag = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"temp/freq") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal surf_collide diffuse command");
      tfreq = atoi(arg[iarg+1]);
      if (tfreq <= 0) error->all(FLERR,"Illegal surf_collide diffuse command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"translate") == 0) {
      if (iarg+4 > narg)
        error->all(FLERR,"Illegal surf_collide diffuse command");
      tflag = 1;
      vx = atof(arg[iarg+1]);
      vy = atof(arg[iarg+2]);
      vz = atof(arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+7 > narg)
        error->all(FLERR,"Illegal surf_collide diffuse command");
      rflag = 1;
      px = atof(arg[iarg+1]);
      py = atof(arg[iarg+2]);
      pz = atof(arg[iarg+3]);
      wx = atof(arg[iarg+4]);
      wy = atof(arg[iarg+5]);
      wz = atof(arg[iarg+6]);

      if (domain->dimension == 2) {
        if (pz != 0.0)
          error->all(FLERR,"Surf_collide diffuse rotation invalid for 2d");
        if (!domain->axisymmetric && (wx != 0.0 || wy != 0.0))
          error->all(FLERR,"Surf_collide diffuse rotation invalid for 2d");
        if (domain->axisymmetric && (wy != 0.0 || wz != 0.0))
          error->all(FLERR,
                     "Surf_collide diffuse rotation invalid for 2d axisymmetric");
      }

      iarg += 7;

    } else error->all(FLERR,"Illegal surf_collide diffuse command");
  }

  if (tflag && rflag) error->all(FLERR,"Illegal surf_collide diffuse command");
  if (tflag || rflag) trflag = 1;
  else trflag = 0;

  vstream[0] = vstream[1] = vstream[2] = 0.0;

  // initialize RNG

  random = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);
}

/* ---------------------------------------------------------------------- */

SurfCollideDiffuse::~SurfCollideDiffuse()
{
  if (copy) return;

  delete random;
}

/* ---------------------------------------------------------------------- */

void SurfCollideDiffuse::init()
{
  SurfCollide::init();
  check_tsurf();
}

/* ----------------------------------------------------------------------
   particle collision with surface with optional chemistry
   ip = particle with current x = collision pt, current v = incident v
   isurf = index of surface element
   norm = surface normal unit vector
   isr = index of reaction model if >= 0, -1 for no chemistry
   ip = reset to NULL if destroyed by chemistry
   return jp = new particle if created by chemistry
   return reaction = index of reaction (1 to N) that took place, 0 = no reaction
   resets particle(s) to post-collision outward velocity
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollideDiffuse::
collide(Particle::OnePart *&ip, double &,
        int isurf, double *norm, int isr, int &reaction)
{
  nsingle++;

  // if surface chemistry defined, attempt reaction
  // reaction = 1 to N for which reaction took place, 0 for none
  // velreset = 1 if reaction reset post-collision velocity, else 0

  Particle::OnePart iorig;
  Particle::OnePart *jp = NULL;
  reaction = 0;
  int velreset = 0;

  if (isr >= 0) {
    if (modify->n_surf_react) memcpy(&iorig,ip,sizeof(Particle::OnePart));
    reaction = surf->sr[isr]->react(ip,isurf,norm,jp,velreset);
    if (reaction) surf->nreact_one++;
  }

  // set temperature of isurf if VARSURF or CUSTOM

  if (persurf_temperature) {
    tsurf = t_persurf[isurf];
    if (tsurf <= 0.0) error->one(FLERR,"Surf_collide tsurf <= 0.0");
  }

  // diffuse reflection for each particle
  // only if SurfReact did not already reset velocities
  // also both particles need to trigger any fixes
  //   to update per-particle properties which depend on
  //   temperature of the particle, e.g. fix vibmode and fix ambipolar

  if (ip) {
    if (!velreset) diffuse(ip,norm);
    if (modify->n_update_custom) {
      int i = ip - particle->particles;
      modify->update_custom(i,tsurf,tsurf,tsurf,vstream);
    }
  }
  if (jp) {
    if (!velreset) diffuse(jp,norm);
    if (modify->n_update_custom) {
      int j = jp - particle->particles;
      modify->update_custom(j,tsurf,tsurf,tsurf,vstream);
    }
  }

  // Invalidate persistent guiding-center (GCA) custom state for any
  // surviving/created particle: velocities changed here, so stored
  // v_par/mu/X are stale (mirror of the surf_collide_toroidal fix).
  if (update->pusher) {
    if (ip) update->pusher->invalidate_gc(ip - particle->particles,
                                          Pusher::GC_INVAL_BOUNDARY);
    if (jp) update->pusher->invalidate_gc(jp - particle->particles,
                                          Pusher::GC_INVAL_BOUNDARY);
  }

  // call any fixes with a surf_react() method
  // they may reset j to -1, e.g. fix ambipolar
  //   in which case newly created j is deleted

  if (reaction && modify->n_surf_react) {
    int i = -1;
    if (ip) i = ip - particle->particles;
    int j = -1;
    if (jp) j = jp - particle->particles;
    modify->surf_react(&iorig,i,j);
    if (jp && j < 0) {
      jp = NULL;
      particle->nlocal--;
    }
  }

  return jp;
}

/* ----------------------------------------------------------------------
   diffusive particle collision with surface
   p = particle with current x = collision pt, current v = incident v
   norm = surface normal unit vector
   resets particle(s) to post-collision outward velocity
------------------------------------------------------------------------- */

void SurfCollideDiffuse::diffuse(Particle::OnePart *p, double *norm)
{
  // specular reflection
  // reflect incident v around norm

  if (random->uniform() > acc) {
    MathExtra::reflect3(p->v,norm);
  // diffuse reflection
  // vrm = most probable speed of species, eqns (4.1) and (4.7)
  // vperp = velocity component perpendicular to surface along norm, eqn (12.3)
  // vtan12 = 2 velocity components tangential to surface
  // tangent1 = component of particle v tangential to surface,
  //   check if tangent1 = 0 (normal collision), set randomly
  // tangent2 = norm x tangent1 = orthogonal tangential direction
  // tangent12 are both unit vectors

  } else {
    double tangent1[3],tangent2[3];
    Particle::Species *species = particle->species;
    int ispecies = p->ispecies;

    double *v = p->v;
    double dot = MathExtra::dot3(v,norm);
    double vmag = MathExtra::len3(v);
    if (vmag == 0.0) return;

    // component of v tangential to the surface
    tangent1[0] = v[0] - dot * norm[0];
    tangent1[1] = v[1] - dot * norm[1];
    tangent1[2] = v[2] - dot * norm[2];

       // if tangent1 is zero (normal incidence), pick a random tangential vector
    if (MathExtra::lensq3(tangent1) == 0.0) {
      tangent1[0] = random->uniform();
      tangent1[1] = random->uniform();
      tangent1[2] = random->uniform();
      MathExtra::cross3(norm, tangent1, tangent1); // make it perpendicular
    }

    MathExtra::norm3(tangent1);
    MathExtra::cross3(norm, tangent1, tangent2);   // tangent2 = norm × tangent1

    // sample hemisphere: cos(theta) ∈ [0,1], phi ∈ [0,2π)
    double xi1 = random->uniform();
    double xi2 = random->uniform();

    double cosTheta = xi1;
    double sinTheta = sqrt(std::max(0.0, 1.0 - cosTheta*cosTheta));
    double phi      = MY_2PI * xi2;

    double cosPhi = cos(phi);
    double sinPhi = sin(phi);

 // outgoing direction in local basis
    double dir[3];
    dir[0] = sinTheta * cosPhi * tangent1[0] + sinTheta * sinPhi * tangent2[0] + cosTheta * norm[0];
    dir[1] = sinTheta * cosPhi * tangent1[1] + sinTheta * sinPhi * tangent2[1] + cosTheta * norm[1];
    dir[2] = sinTheta * cosPhi * tangent1[2] + sinTheta * sinPhi * tangent2[2] + cosTheta * norm[2];

      MathExtra::norm3(dir);

    // assign new velocity with same magnitude vmag
    v[0] = vmag * dir[0];
    v[1] = vmag * dir[1];
    v[2] = vmag * dir[2];


    p->erot = particle->erot(ispecies,tsurf,random);
    p->evib = particle->evib(ispecies,tsurf,random);
  }
}

/* ----------------------------------------------------------------------
   wrapper on diffuse() method to perform collision for a single particle
   pass in 2 coefficients to match command-line args for style diffuse
   flags, coeffs can be NULL
   called by SurfReactAdsorb
------------------------------------------------------------------------- */

void SurfCollideDiffuse::wrapper(Particle::OnePart *p, double *norm,
                                 int *flags, double *coeffs)
{
  if (coeffs) {
    tsurf = coeffs[0];
    acc = coeffs[1];
  }

  diffuse(p,norm);
}

/* ----------------------------------------------------------------------
   return flags and coeffs for this SurfCollide instance to caller
------------------------------------------------------------------------- */

void SurfCollideDiffuse::flags_and_coeffs(int *flags, double *coeffs)
{
  if (tmode != NUMERIC)
    error->all(FLERR,"Surf_collide diffuse with non-numeric Tsurf "
               "does not support external caller");

  coeffs[0] = tsurf;
  coeffs[1] = acc;
}


