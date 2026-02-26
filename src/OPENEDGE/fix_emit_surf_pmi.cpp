/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_emit_surf_pmi.h"
#include "update.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "modify.h"
#include "cut2d.h"
#include "cut3d.h"
#include "input.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <cmath>

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};
enum{FLOW,CONSTANT};

#define DELTATASK 256

/* ---------------------------------------------------------------------- */

FixEmitSurfPmi::FixEmitSurfPmi(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix emit/surf/pmi command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf/pmi mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf/pmi group ID does not exist");
  groupbit = surf->bitmask[igroup];

  iflux = modify->find_compute(arg[4]);
  if (iflux < 0)
    error->all(FLERR,"Fix emit/surf/pmi compute ID does not exist");

  flux_index = 0;
  int iarg = 5;
  if (strcmp(arg[iarg],"flux_index") == 0) {
    if (iarg + 1 >= narg) error->all(FLERR,"Illegal fix emit/surf/pmi command");
    flux_index = atoi(arg[iarg+1]);
    if (flux_index <= 0)
      error->all(FLERR,"Fix emit/surf/pmi flux_index must be >= 1");
    iarg += 2;
  }

  np = 0;
  npmode = FLOW;
  npstr = NULL;
  normalflag = 0;

  options(narg-iarg,&arg[iarg]);

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf/pmi requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf/pmi not allowed for implicit surfaces");
  if (npmode == CONSTANT && perspecies)
    error->all(FLERR,"Cannot use fix emit/surf/pmi n > 0 with perspecies yes");

  tasks = NULL;
  ntask = ntaskmax = 0;

  dimension = domain->dimension;
  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta,domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixEmitSurfPmi::~FixEmitSurfPmi()
{
  if (copymode) return;

  delete [] npstr;

  for (int i = 0; i < ntaskmax; i++) {
    delete [] tasks[i].ntargetsp;
    delete [] tasks[i].vscale;
    delete [] tasks[i].path;
    delete [] tasks[i].fracarea;
  }
  memory->sfree(tasks);

  if (dimension == 3) delete cut3d;
  else delete cut2d;
}

/* ---------------------------------------------------------------------- */

void FixEmitSurfPmi::init()
{
  FixEmit::init();

  if (iflux < 0 || iflux >= modify->ncompute)
    error->all(FLERR,"Fix emit/surf/pmi compute ID no longer exists");

  Compute *c = modify->compute[iflux];
  if (!c->per_surf_flag)
    error->all(FLERR,"Fix emit/surf/pmi compute must provide per-surf values");

  fnum = update->fnum;

  nspecies = particle->mixture[imix]->nspecies;
  fraction = particle->mixture[imix]->fraction;
  cummulative = particle->mixture[imix]->cummulative;

  // magvstream = magnitude of mixture vstream vector
  // norm_vstream = unit vector in stream direction
  double *vstream = particle->mixture[imix]->vstream;
  magvstream = MathExtra::len3(vstream);
  norm_vstream[0] = vstream[0];
  norm_vstream[1] = vstream[1];
  norm_vstream[2] = vstream[2];
  if (norm_vstream[0] != 0.0 || norm_vstream[1] != 0.0 || norm_vstream[2] != 0.0)
    MathExtra::norm3(norm_vstream);

  if (perspecies) {
    for (int i = 0; i < ntask; i++) {
      delete [] tasks[i].ntargetsp;
      tasks[i].ntargetsp = new double[nspecies];
    }
  }

  grid_changed();
}

/* ---------------------------------------------------------------------- */

int FixEmitSurfPmi::local_isurf_index(surfint isurf) const
{
  if (surf->distributed) {
    if (isurf < 0 || isurf >= surf->nown) return -1;
    return static_cast<int>(isurf);
  }

  int me = comm->me;
  int nprocs = comm->nprocs;
  if ((isurf % nprocs) != me) return -1;
  int ilocal = static_cast<int>(isurf / nprocs);
  if (ilocal < 0 || ilocal >= surf->nown) return -1;
  return ilocal;
}

/* ---------------------------------------------------------------------- */

double FixEmitSurfPmi::flux_for_surface(surfint isurf)
{
  Compute *c = modify->compute[iflux];

  int ilocal = local_isurf_index(isurf);
  if (ilocal < 0) return 0.0;

  if (c->size_per_surf_cols == 0) {
    if (!c->vector_surf) return 0.0;
    return c->vector_surf[ilocal];
  }

  if (!c->array_surf) return 0.0;
  int icol = (flux_index > 0) ? flux_index-1 : 0;
  if (icol < 0 || icol >= c->size_per_surf_cols) return 0.0;
  return c->array_surf[ilocal][icol];
}

/* ---------------------------------------------------------------------- */

void FixEmitSurfPmi::grid_changed()
{
  create_tasks();

  // for mode CONSTANT, set per-task ntarget to area fraction
  if (npmode != FLOW) {
    double areasum_me = 0.0;
    for (int i = 0; i < ntask; i++) areasum_me += tasks[i].area;

    double areasum = 0.0;
    MPI_Allreduce(&areasum_me,&areasum,1,MPI_DOUBLE,MPI_SUM,world);

    if (areasum > 0.0) {
      for (int i = 0; i < ntask; i++)
        tasks[i].ntarget = tasks[i].area / areasum;
    } else {
      for (int i = 0; i < ntask; i++)
        tasks[i].ntarget = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   create task for one grid cell
   add them to tasks list and increment ntasks
------------------------------------------------------------------------- */

void FixEmitSurfPmi::create_task(int icell)
{
  int i,m,isurf,npoint,isplit,subcell;
  double indot,area,areaone;
  double *normal,*p1,*p2,*p3,*path;
  double cpath[36],delta[3],e1[3],e2[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  // no tasks if no surfs in cell
  if (cells[icell].nsurf == 0) return;
  // no tasks if cell is outside flow volume
  if (cinfo[icell].volume == 0.0) return;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  double nrho = particle->mixture[imix]->nrho;
  double *vstream = particle->mixture[imix]->vstream;
  double temp_thermal = particle->mixture[imix]->temp_thermal;

  double *lo = cells[icell].lo;
  double *hi = cells[icell].hi;
  surfint *csurfs = cells[icell].csurfs;
  int nsurf = cells[icell].nsurf;

  for (i = 0; i < nsurf; i++) {
    isurf = csurfs[i];

    if (dimension == 2) {
      if (!(lines[isurf].mask & groupbit)) continue;
    } else {
      if (!(tris[isurf].mask & groupbit)) continue;
    }

    if (ntask == ntaskmax) grow_task();

    tasks[ntask].icell = icell;
    tasks[ntask].isurf = isurf;
    if (cells[icell].nsplit == 1) tasks[ntask].pcell = icell;
    else {
      isplit = cells[icell].isplit;
      subcell = sinfo[isplit].csplits[i];
      tasks[ntask].pcell = sinfo[isplit].csubs[subcell];
    }

    if (dimension == 2) {
      normal = lines[isurf].norm;
      indot = magvstream;

      p1 = lines[isurf].p1;
      p2 = lines[isurf].p2;
      npoint = cut2d->clip_external(p1,p2,lo,hi,cpath);
      if (npoint < 2) continue;

      tasks[ntask].npoint = 2;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[6];
      path = tasks[ntask].path;
      path[0] = cpath[0];
      path[1] = cpath[1];
      path[2] = 0.0;
      path[3] = cpath[2];
      path[4] = cpath[3];
      path[5] = 0.0;

      if (domain->axisymmetric) {
        double sqrtarg = (path[1]-path[4])*(path[1]-path[4]) +
                         (path[0]-path[3])*(path[0]-path[3]);
        area = MY_PI * (path[1]+path[4]) * sqrt(sqrtarg);
      } else {
        MathExtra::sub3(&path[0],&path[3],delta);
        area = MathExtra::len3(delta);
      }
      tasks[ntask].area = area;

      tasks[ntask].tan1[0] = normal[1];
      tasks[ntask].tan1[1] = -normal[0];
      tasks[ntask].tan1[2] = 0.0;
      tasks[ntask].tan2[0] = 0.0;
      tasks[ntask].tan2[1] = 0.0;
      tasks[ntask].tan2[2] = 1.0;

    } else {
      normal = tris[isurf].norm;
      if (normalflag) indot = magvstream;
      else indot = vstream[0]*normal[0] + vstream[1]*normal[1] +
                   vstream[2]*normal[2];

      p1 = tris[isurf].p1;
      p2 = tris[isurf].p2;
      p3 = tris[isurf].p3;
      npoint = cut3d->clip_external(p1,p2,p3,lo,hi,cpath);
      if (npoint < 3) continue;

      tasks[ntask].npoint = npoint;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[npoint*3];
      path = tasks[ntask].path;
      memcpy(path,cpath,npoint*3*sizeof(double));
      delete [] tasks[ntask].fracarea;
      tasks[ntask].fracarea = new double[npoint-2];

      area = 0.0;
      p1 = &path[0];
      for (m = 0; m < npoint-2; m++) {
        p2 = &path[3*(m+1)];
        p3 = &path[3*(m+2)];
        MathExtra::sub3(p2,p1,e1);
        MathExtra::sub3(p3,p1,e2);
        MathExtra::cross3(e1,e2,delta);
        areaone = fabs(0.5*MathExtra::len3(delta));
        area += areaone;
        tasks[ntask].fracarea[m] = area;
      }
      tasks[ntask].area = area;
      for (m = 0; m < npoint-2; m++) tasks[ntask].fracarea[m] /= area;

      delta[0] = random->uniform();
      delta[1] = random->uniform();
      delta[2] = random->uniform();
      MathExtra::cross3(tris[isurf].norm,delta,tasks[ntask].tan1);
      MathExtra::norm3(tasks[ntask].tan1);
      MathExtra::cross3(tris[isurf].norm,tasks[ntask].tan1,tasks[ntask].tan2);
      MathExtra::norm3(tasks[ntask].tan2);
    }

    tasks[ntask].ntarget = 0.0;
    if (perspecies) {
      for (int isp = 0; isp < nspecies; isp++) tasks[ntask].ntargetsp[isp] = 0.0;
    }

    tasks[ntask].nrho = nrho;
    tasks[ntask].temp_thermal = temp_thermal;
    tasks[ntask].temp_rot = particle->mixture[imix]->temp_rot;
    tasks[ntask].temp_vib = particle->mixture[imix]->temp_vib;
    tasks[ntask].magvstream = magvstream;
    tasks[ntask].vstream[0] = vstream[0];
    tasks[ntask].vstream[1] = vstream[1];
    tasks[ntask].vstream[2] = vstream[2];

    ntask++;
  }
}

/* ----------------------------------------------------------------------
   insert particles in grid cells with emitting surface elements
------------------------------------------------------------------------- */

void FixEmitSurfPmi::perform_task()
{
  int i,m,n,pcell,isurf,ninsert,nactual,isp,ispecies,ntri,id;
  double indot,scosine,rn,ntarget,vr,alpha,beta;
  double beta_un,normalized_distbn_fn,theta,erot,evib;
  double vnmag,vamag,vbmag;
  double *normal,*p1,*p2,*p3,*atan,*btan,*vstream,*vscale;
  double x[3],v[3],e1[3],e2[3];
  Particle::OnePart *p;

  const double dt = update->dt;
  int *species = particle->mixture[imix]->species;

  // evaluate requested per-surf flux once per timestep
  Compute *c = modify->compute[iflux];
  c->compute_per_surf();

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;
  int nfix_update_custom = modify->n_update_custom;

  for (i = 0; i < ntask; i++) {
    pcell = tasks[i].pcell;
    isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"BAD surf index");

    if (dimension == 2) normal = lines[isurf].norm;
    else normal = tris[isurf].norm;
    atan = tasks[i].tan1;
    btan = tasks[i].tan2;

    temp_thermal = tasks[i].temp_thermal;
    temp_rot = tasks[i].temp_rot;
    temp_vib = tasks[i].temp_vib;
    magvstream = tasks[i].magvstream;
    vstream = tasks[i].vstream;
    vscale = particle->mixture[imix]->vscale;

    if (normalflag) indot = magvstream;
    else indot = vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];

    if (npmode == FLOW) {
      double flux = flux_for_surface(isurf);
      if (!std::isfinite(flux) || flux <= 0.0) continue;
      ntarget = flux * tasks[i].area * dt / fnum;
    } else {
      ntarget = np * tasks[i].ntarget;
    }
    if (ntarget <= 0.0) continue;

    if (perspecies) {
      for (isp = 0; isp < nspecies; isp++) {
        ispecies = species[isp];
        double ntarget_sp = ntarget * fraction[isp];
        ninsert = static_cast<int>(ntarget_sp + random->uniform());
        if (ninsert <= 0) continue;
        scosine = indot / vscale[isp];

        nactual = 0;
        for (m = 0; m < ninsert; m++) {
          if (dimension == 2) {
            rn = random->uniform();
            p1 = &tasks[i].path[0];
            p2 = &tasks[i].path[3];
            x[0] = p1[0] + rn * (p2[0]-p1[0]);
            x[1] = p1[1] + rn * (p2[1]-p1[1]);
            x[2] = 0.0;
          } else {
            rn = random->uniform();
            ntri = tasks[i].npoint - 2;
            for (n = 0; n < ntri; n++)
              if (rn < tasks[i].fracarea[n]) break;
            p1 = &tasks[i].path[0];
            p2 = &tasks[i].path[3*(n+1)];
            p3 = &tasks[i].path[3*(n+2)];
            MathExtra::sub3(p2,p1,e1);
            MathExtra::sub3(p3,p1,e2);
            alpha = random->uniform();
            beta = random->uniform();
            if (alpha+beta > 1.0) {
              alpha = 1.0 - alpha;
              beta = 1.0 - beta;
            }
            x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
            x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
            x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
          }

          if (region && !region->match(x)) continue;

          do {
            do beta_un = (6.0*random->uniform() - 3.0);
            while (beta_un + scosine < 0.0);
            normalized_distbn_fn = 2.0 * (beta_un + scosine) /
              (scosine + sqrt(scosine*scosine + 2.0)) *
              exp(0.5 + (0.5*scosine)*(scosine-sqrt(scosine*scosine + 2.0)) -
                  beta_un*beta_un);
          } while (normalized_distbn_fn < random->uniform());

          if (normalflag) vnmag = beta_un*vscale[isp] + magvstream;
          else vnmag = beta_un*vscale[isp] + indot;

          theta = MY_2PI * random->uniform();
          vr = vscale[isp] * sqrt(-log(random->uniform()));
          if (normalflag) {
            vamag = vr * sin(theta);
            vbmag = vr * cos(theta);
          } else {
            vamag = vr * sin(theta) + MathExtra::dot3(vstream,atan);
            vbmag = vr * cos(theta) + MathExtra::dot3(vstream,btan);
          }

          v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
          v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
          v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

          erot = particle->erot(ispecies,temp_rot,random);
          evib = particle->evib(ispecies,temp_vib,random);
          id = MAXSMALLINT*random->uniform();

          particle->add_particle(id,ispecies,pcell,x,v,erot,evib);
          nactual++;

          p = &particle->particles[particle->nlocal-1];
          p->flag = PSURF + 1 + isurf;
          p->dtremain = dt * random->uniform();

          if (nsurf_tally)
            for (int k = 0; k < nsurf_tally; k++)
              slist_active[k]->surf_tally(p->dtremain,isurf,pcell,0,NULL,p,NULL);

          if (nfix_update_custom)
            modify->update_custom(particle->nlocal-1,temp_thermal,
                                  temp_rot,temp_vib,vstream);
        }
        nsingle += nactual;
      }

    } else {
      ninsert = static_cast<int>(ntarget + random->uniform());
      if (ninsert <= 0) continue;

      nactual = 0;
      for (m = 0; m < ninsert; m++) {
        rn = random->uniform();
        isp = 0;
        while (cummulative[isp] < rn) isp++;
        ispecies = species[isp];
        scosine = indot / vscale[isp];

        if (dimension == 2) {
          rn = random->uniform();
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3];
          x[0] = p1[0] + rn * (p2[0]-p1[0]);
          x[1] = p1[1] + rn * (p2[1]-p1[1]);
          x[2] = 0.0;
        } else {
          rn = random->uniform();
          ntri = tasks[i].npoint - 2;
          for (n = 0; n < ntri; n++)
            if (rn < tasks[i].fracarea[n]) break;
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3*(n+1)];
          p3 = &tasks[i].path[3*(n+2)];
          MathExtra::sub3(p2,p1,e1);
          MathExtra::sub3(p3,p1,e2);
          alpha = random->uniform();
          beta = random->uniform();
          if (alpha+beta > 1.0) {
            alpha = 1.0 - alpha;
            beta = 1.0 - beta;
          }
          x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
          x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
          x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
        }

        if (region && !region->match(x)) continue;

        do {
          do beta_un = (6.0*random->uniform() - 3.0);
          while (beta_un + scosine < 0.0);
          normalized_distbn_fn = 2.0 * (beta_un + scosine) /
            (scosine + sqrt(scosine*scosine + 2.0)) *
            exp(0.5 + (0.5*scosine)*(scosine-sqrt(scosine*scosine + 2.0)) -
                beta_un*beta_un);
        } while (normalized_distbn_fn < random->uniform());

        if (normalflag) vnmag = beta_un*vscale[isp] + magvstream;
        else vnmag = beta_un*vscale[isp] + indot;

        theta = MY_2PI * random->uniform();
        vr = vscale[isp] * sqrt(-log(random->uniform()));
        if (normalflag) {
          vamag = vr * sin(theta);
          vbmag = vr * cos(theta);
        } else {
          vamag = vr * sin(theta) + MathExtra::dot3(vstream,atan);
          vbmag = vr * cos(theta) + MathExtra::dot3(vstream,btan);
        }

        v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
        v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
        v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

        erot = particle->erot(ispecies,temp_rot,random);
        evib = particle->evib(ispecies,temp_vib,random);
        id = MAXSMALLINT*random->uniform();

        particle->add_particle(id,ispecies,pcell,x,v,erot,evib);
        nactual++;

        p = &particle->particles[particle->nlocal-1];
        p->flag = PSURF + 1 + isurf;
        p->dtremain = dt * random->uniform();

        if (nsurf_tally)
          for (int k = 0; k < nsurf_tally; k++)
            slist_active[k]->surf_tally(p->dtremain,isurf,pcell,0,NULL,p,NULL);

        if (nfix_update_custom)
          modify->update_custom(particle->nlocal-1,temp_thermal,
                                temp_rot,temp_vib,vstream);
      }
      nsingle += nactual;
    }
  }
}

/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixEmitSurfPmi::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks,ntaskmax*sizeof(Task),
                                    "emit/surf/pmi:tasks");

  memset(&tasks[oldmax],0,(ntaskmax-oldmax)*sizeof(Task));

  for (int i = oldmax; i < ntaskmax; i++) {
    if (perspecies) tasks[i].ntargetsp = new double[nspecies];
    else tasks[i].ntargetsp = NULL;
    tasks[i].vscale = NULL;
    tasks[i].path = NULL;
    tasks[i].fracarea = NULL;
  }
}

/* ----------------------------------------------------------------------
   process keywords specific to this class
------------------------------------------------------------------------- */

int FixEmitSurfPmi::option(int narg, char **arg)
{
  if (strcmp(arg[0],"n") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf/pmi command");
    np = atoi(arg[1]);
    if (np <= 0) npmode = FLOW;
    else npmode = CONSTANT;
    return 2;
  }

  if (strcmp(arg[0],"normal") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf/pmi command");
    if (strcmp(arg[1],"yes") == 0) normalflag = 1;
    else if (strcmp(arg[1],"no") == 0) normalflag = 0;
    else error->all(FLERR,"Illegal fix emit/surf/pmi command");
    return 2;
  }

  error->all(FLERR,"Illegal fix emit/surf/pmi command");
  return 0;
}
