/* ----------------------------------------------------------------------
   OpenEdge: surf_collide partial_recycle

   EIRENE-style partial recycling. Per particle hitting the surface, sample
   u ~ U(0,1):
       u < (1-R)  -> vanish (count toward pumped throughput)
       else       -> diffuse re-emit at twall, optionally as a different
                     species (e.g. D arriving -> D2 emitted, modelling
                     atom recombination on the wall).
------------------------------------------------------------------------- */

#include "string.h"
#include "surf_collide_partial_recycle.h"
#include "particle.h"
#include "input.h"
#include "comm.h"
#include "random_knuth.h"
#include "update.h"
#include "modify.h"
#include "surf.h"
#include "surf_react.h"
#include "error.h"

using namespace SPARTA_NS;

// Helper: truncate narg so the SurfCollideDiffuse base parser sees only
// (ID style Twall acc), letting it skip its optional keywords.  Our
// keywords (r, return_species) are then consumed by *our* constructor.
static int trim_narg_for_base(int narg)
{
  return (narg < 4) ? narg : 4;
}

/* ---------------------------------------------------------------------- */

SurfCollidePartialRecycle::SurfCollidePartialRecycle(SPARTA *sparta, int narg, char **arg) :
  SurfCollideDiffuse(sparta, trim_narg_for_base(narg), arg)
{
  // SurfCollideDiffuse parses arg[2]=Twall and arg[3]=acc.  Our additional
  // keywords (in any order):
  //   r <prob>                       (required for partial recycling)
  //   return_species <NAME>          (optional; default = same species)

  r_recycle = 1.0;                    // default = full recycle (R=1)
  return_species = -1;                // -1 means "same as incoming"
  return_species_name = NULL;
  nvanish_local = nrecycle_local = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"r") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_collide partial_recycle command: r missing value");
      r_recycle = input->numeric(FLERR,arg[iarg+1]);
      if (r_recycle < 0.0 || r_recycle > 1.0)
        error->all(FLERR,"surf_collide partial_recycle: r must be in [0,1]");
      iarg += 2;
    } else if (strcmp(arg[iarg],"return_species") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_collide partial_recycle command: return_species missing name");
      int n = strlen(arg[iarg+1]) + 1;
      delete [] return_species_name;
      return_species_name = new char[n];
      strcpy(return_species_name,arg[iarg+1]);
      iarg += 2;
    } else {
      // SurfCollideDiffuse already consumed everything it understood;
      // any leftover keyword we don't recognise is an error.
      error->all(FLERR,"Illegal surf_collide partial_recycle keyword");
    }
  }

  if (return_species_name) {
    return_species = particle->find_species(return_species_name);
    if (return_species < 0) {
      char msg[256];
      snprintf(msg,sizeof(msg),
               "surf_collide partial_recycle: return_species '%s' is not defined",
               return_species_name);
      error->all(FLERR,msg);
    }
  }

  if (comm->me == 0) {
    printf("surf_collide partial_recycle: twall=%.3g K, acc=%.3g, R=%.3g",
           tsurf, acc, r_recycle);
    if (return_species >= 0)
      printf(", return_species=%s (idx %d)\n", return_species_name, return_species);
    else
      printf(", return_species=same\n");
  }
}

/* ---------------------------------------------------------------------- */

SurfCollidePartialRecycle::~SurfCollidePartialRecycle()
{
  if (copy) return;
  delete [] return_species_name;
}

/* ----------------------------------------------------------------------
   particle collision with surface with optional chemistry.

   Per particle:
     u < (1 - R)       -> vanish (pumped)
     else, no surface  -> diffuse re-emission (optionally swap species)
       reaction
     reaction handled
       by surf_react   -> defer entirely to the base SurfCollideDiffuse
                          path (don't double-fire the partial-recycle dice
                          on a particle the reaction already consumed).
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollidePartialRecycle::
collide(Particle::OnePart *&ip, double &dtremain,
        int isurf, double *norm, int isr, int &reaction)
{
  // If a surface reaction model is attached, let the base class run first
  // — surface chemistry takes precedence. Only apply partial-recycle to
  // particles the reaction did NOT consume.
  if (isr >= 0)
    return SurfCollideDiffuse::collide(ip, dtremain, isurf, norm, isr, reaction);

  nsingle++;
  reaction = 0;

  const double u = random->uniform();
  if (u >= r_recycle) {
    // PUMPED: vanish, count, return NULL.
    ip = NULL;
    nvanish_local++;
    return NULL;
  }

  // RECYCLED: optionally swap species, then call base diffuse() to set
  // the post-collision velocity at twall.
  if (return_species >= 0)
    ip->ispecies = return_species;

  diffuse(ip, norm);

  if (modify->n_update_custom) {
    int i = ip - particle->particles;
    modify->update_custom(i, tsurf, tsurf, tsurf, vstream);
  }

  nrecycle_local++;
  return NULL;
}
