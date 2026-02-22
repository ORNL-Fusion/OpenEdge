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

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "surf_react_vanish.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "domain.h"
#include "surf.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfReactVanish::SurfReactVanish(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  me = comm->me;
  logflag = 0;
  logfp = NULL;
  logfile = NULL;

  if (narg < 3) error->all(FLERR,"Illegal surf_react vanish command");

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"log") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_react vanish command");
      if (strcmp(arg[iarg+1],"yes") == 0) logflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) logflag = 0;
      else error->all(FLERR,"Illegal surf_react vanish command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_react vanish command");
      delete [] logfile;
      int n = strlen(arg[iarg+1]) + 1;
      logfile = new char[n];
      strcpy(logfile,arg[iarg+1]);
      logflag = 1;
      iarg += 2;

    } else {
      error->all(FLERR,"Illegal surf_react vanish command");
    }
  }

  nlist = 1;
  tally_single = new int[nlist];
  tally_total = new int[nlist];
  tally_single_all = new int[nlist];
  tally_total_all = new int[nlist];
  size_vector = 2 + 2*nlist;
}

/* ---------------------------------------------------------------------- */

SurfReactVanish::~SurfReactVanish()
{
  if (copy) return;

  close_logfile();
  delete [] logfile;
}

/* ---------------------------------------------------------------------- */

int SurfReactVanish::react(Particle::OnePart *&ip, int isurf, double *,
                           Particle::OnePart *&, int &)
{
  if (ip == NULL) return 0;

  if (logflag) log_event(ip,isurf);

  nsingle++;
  tally_single[0]++;
  ip = NULL;
  return 1;
}

/* ---------------------------------------------------------------------- */

char *SurfReactVanish::reactionID(int)
{
  return (char *) "delete";
}

/* ---------------------------------------------------------------------- */

void SurfReactVanish::open_logfile()
{
  if (!logflag || logfp) return;

  const char *base = logfile ? logfile : "surf_react_vanish.csv";
  int nbase = strlen(base);
  char *filename = new char[nbase + 32];
  snprintf(filename,nbase+32,"%s.rank%d",base,me);

  logfp = fopen(filename,"w");
  delete [] filename;

  if (logfp == NULL)
    error->one(FLERR,"Cannot open surf_react vanish log file");

  fprintf(logfp,
          "timestep,rank,surf_index,surf_id,id,ispecies,x,y,z,vx,vy,vz,mass,radius,temp\n");
}

/* ---------------------------------------------------------------------- */

void SurfReactVanish::close_logfile()
{
  if (logfp) {
    fclose(logfp);
    logfp = NULL;
  }
}

/* ---------------------------------------------------------------------- */

void SurfReactVanish::log_event(Particle::OnePart *ip, int isurf)
{
  open_logfile();
  if (!logfp || ip == NULL) return;

  long long surf_id = -1;
  if (isurf >= 0) {
    if (domain->dimension == 2) surf_id = (long long) surf->lines[isurf].id;
    else surf_id = (long long) surf->tris[isurf].id;
  }

  fprintf(logfp,
          "%lld,%d,%d,%lld,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          (long long) update->ntimestep, me, isurf, surf_id,
          ip->id, ip->ispecies,
          ip->x[0], ip->x[1], ip->x[2],
          ip->v[0], ip->v[1], ip->v[2],
          ip->mass, ip->radius, ip->temp);
}
