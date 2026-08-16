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

#include "math.h"
#include "stdio.h"
#include "string.h"
#include "surf_collide_vanish.h"
#include "comm.h"
#include "domain.h"
#include "surf.h"
#include "update.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfCollideVanish::SurfCollideVanish(SPARTA *sparta, int narg, char **arg) :
  SurfCollide(sparta, narg, arg)
{
  me = comm->me;
  logflag = 0;
  tallyflag = 0;
  logfp = NULL;
  logfile = NULL;
  pweight_index = -1;
  wsum_cum = 0.0;

  if (narg < 2) error->all(FLERR,"Illegal surf_collide vanish command");

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"log") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_collide vanish command");
      if (strcmp(arg[iarg+1],"yes") == 0) logflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) logflag = 0;
      else error->all(FLERR,"Illegal surf_collide vanish command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_collide vanish command");
      delete [] logfile;
      int n = strlen(arg[iarg+1]) + 1;
      logfile = new char[n];
      strcpy(logfile,arg[iarg+1]);
      logflag = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg],"tally") == 0) {
      // per-surf counts + absorbed-weight sums only, no per-event log
      if (iarg + 1 >= narg)
        error->all(FLERR,"Illegal surf_collide vanish command");
      delete [] logfile;
      int n = strlen(arg[iarg+1]) + 1;
      logfile = new char[n];
      strcpy(logfile,arg[iarg+1]);
      tallyflag = 1;
      iarg += 2;

    } else {
      error->all(FLERR,"Illegal surf_collide vanish command");
    }
  }

  allowreact = 0;
}

/* ---------------------------------------------------------------------- */

SurfCollideVanish::~SurfCollideVanish()
{
  if (copy) return;
  write_counts_file();
  close_logfile();
  delete [] logfile;
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanish::init()
{
  SurfCollide::init();
  pweight_index = particle->find_custom((char *) "pweight");
}

/* ----------------------------------------------------------------------
   particle collision with surface with optional chemistry
   ip = particle with current x = collision pt, current v = incident v
   norm = surface normal unit vector
   isr = index of reaction model if >= 0, -1 for no chemistry
   simply return ip = NULL to delete particle
   return reaction = 0 = no reaction took place
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollideVanish::
collide(Particle::OnePart *&ip, double &, int isurf, double *normal, int, int &)
{
  nsingle++;

  // accumulate absorbed real-atom weight; pweight carries real atoms per
  // macroparticle (fix particle/weight), fallback fnum when absent/unset
  double pw = 0.0;
  if (pweight_index >= 0)
    pw = particle->edvec[particle->ewhich[pweight_index]][ip - particle->particles];
  if (pw == 0.0) pw = update->fnum;
  wsum_cum += pw;

  if (logflag || tallyflag) {
    long long surf_id = -1;
    if (isurf >= 0) {
      if (domain->dimension == 2) surf_id = (long long) surf->lines[isurf].id;
      else surf_id = (long long) surf->tris[isurf].id;
    }
    if (surf_id >= 0) {
      surf_hit_counts[surf_id]++;
      surf_wsums[surf_id] += pw;
    }
    // Capture vanish event with surface ID and local collision normal.
    if (logflag) log_event(ip,isurf,normal,pw);
  }

  ip = NULL;
  return NULL;
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanish::open_logfile()
{
  if (!logflag || logfp) return;

  const char *base = logfile ? logfile : "surf_collide_vanish.csv";
  int nbase = strlen(base);
  char *filename = new char[nbase + 32];
  snprintf(filename,nbase+32,"%s.rank%d",base,me);

  logfp = fopen(filename,"w");
  delete [] filename;

  if (logfp == NULL)
    error->one(FLERR,"Cannot open surf_collide vanish log file");

  fprintf(logfp,
          "timestep,rank,surf_index,surf_id,surf_hit_count,id,ispecies,x,y,z,vx,vy,vz,nx,ny,nz,mass,radius,temp,weight,dtremain,erot,evib,pweight,gca_mode,gca_valid,gca_chi\n");
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanish::close_logfile()
{
  if (logfp) {
    fclose(logfp);
    logfp = NULL;
  }
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanish::log_event(Particle::OnePart *ip, int isurf, double *normal,
                                  double pw)
{
  open_logfile();
  if (!logfp || ip == NULL) return;

  long long surf_id = -1;
  if (isurf >= 0) {
    if (domain->dimension == 2) surf_id = (long long) surf->lines[isurf].id;
    else surf_id = (long long) surf->tris[isurf].id;
  }

  long long count = 0;
  if (surf_id >= 0) count = surf_hit_counts[surf_id];

  // Orbit-state observability (hybrid/GCA pusher): proves which pusher
  // branch was active at the actual impact. -1 = attributes absent.
  double gmode = -1.0, gvalid = -1.0, gchi = -1.0;
  const int ic_mode = particle->find_custom((char *) "gca_mode");
  const int ic_valid = particle->find_custom((char *) "gca_valid");
  const int ic_chi = particle->find_custom((char *) "gca_chi");
  if (ic_mode >= 0 && ic_valid >= 0 && ic_chi >= 0) {
    const int idx = static_cast<int>(ip - particle->particles);
    gmode = particle->edvec[particle->ewhich[ic_mode]][idx];
    gvalid = particle->edvec[particle->ewhich[ic_valid]][idx];
    gchi = particle->edvec[particle->ewhich[ic_chi]][idx];
  }

  fprintf(logfp,
          "%lld,%d,%d,%lld,%lld,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%g,%g,%g\n",
          (long long) update->ntimestep, me, isurf, surf_id, count,
          ip->id, ip->ispecies,
          ip->x[0], ip->x[1], ip->x[2],
          ip->v[0], ip->v[1], ip->v[2],
          normal ? normal[0] : 0.0, normal ? normal[1] : 0.0, normal ? normal[2] : 0.0,
          ip->mass, ip->radius, ip->temp,
          ip->weight, ip->dtremain, ip->erot, ip->evib, pw,
          gmode, gvalid, gchi);
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanish::write_counts_file()
{
  if (!logflag && !tallyflag) return;

  const char *base = logfile ? logfile : "surf_collide_vanish.csv";
  int nbase = strlen(base);
  char *filename = new char[nbase + 48];
  snprintf(filename,nbase+48,"%s.counts.rank%d",base,me);

  FILE *fp = fopen(filename,"w");
  delete [] filename;
  if (!fp) return;

  fprintf(fp,"surf_id,hit_count,weight_sum\n");
  for (const auto &kv : surf_hit_counts)
    fprintf(fp,"%lld,%lld,%.17g\n",kv.first,kv.second,surf_wsums[kv.first]);

  fclose(fp);
}
