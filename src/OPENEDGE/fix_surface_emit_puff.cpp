/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_surface_emit_puff.h"
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
#include "surf_collide_vanish.h"
#include "input.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <cmath>
#include <vector>

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};
enum{FLOW,CONSTANT,SLAVE};

#define DELTATASK 256

/* ---------------------------------------------------------------------- */

FixSurfaceEmitPuff::FixSurfaceEmitPuff(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  // Emission tasks cache per-grid pcell indices; gridmigrate=1 puts this
  // fix in Modify::list_pergrid so Grid::notify_changed() invokes
  // grid_changed() -> create_tasks() after mid-run balance/adapt
  // migrations. Without it, stale task pcell values made post-balance
  // emissions insert particles with old (wrong or ghost/out-of-range)
  // cell indices -> silent mis-binning and downstream segfaults.
  gridmigrate = 1;

  // Usage: fix ID emit/surf/puff mixture group [keyword args]
  if (narg < 4) error->all(FLERR,"Illegal fix surface/emit/puff command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf/puff mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf/puff group ID does not exist");
  groupbit = surf->bitmask[igroup];

  np = 0;
  npmode = FLOW;
  npstr = NULL;
  normalflag = 0;
  stop_at_np = 0;
  stop_latched = 0;
  slave_R = slave_gamma = 0.0;
  slave_scstr = NULL;

  int iarg = 4;
  options(narg-iarg,&arg[iarg]);

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf/puff requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf/puff not allowed for implicit surfaces");
  if (npmode != FLOW && perspecies)
    error->all(FLERR,"Cannot use fix surface/emit/puff n > 0 or slave "
               "with perspecies yes");

  tasks = NULL;
  ntask = ntaskmax = 0;

  dimension = domain->dimension;
  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta,domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitPuff::~FixSurfaceEmitPuff()
{
  if (copymode) return;

  delete [] npstr;
  delete [] slave_scstr;

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

void FixSurfaceEmitPuff::init()
{
  FixEmit::init();

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

  // SLAVE mode: resolve vanish collide IDs and baseline their cumulative
  // absorbed weights (cum only advances during moves, so re-init is safe)
  if (npmode == SLAVE) {
    slave_sc.clear();
    char *copy = new char[strlen(slave_scstr)+1];
    strcpy(copy,slave_scstr);
    char *tok = strtok(copy,",");
    while (tok) {
      int isc = surf->find_collide(tok);
      if (isc < 0)
        error->all(FLERR,"Fix surface/emit/puff slave collide ID not found");
      SurfCollideVanish *scv = dynamic_cast<SurfCollideVanish *>(surf->sc[isc]);
      if (!scv)
        error->all(FLERR,"Fix surface/emit/puff slave collide is not style vanish");
      slave_sc.push_back(scv);
      tok = strtok(NULL,",");
    }
    delete [] copy;
    slave_lastread.resize(slave_sc.size());
    for (size_t k = 0; k < slave_sc.size(); k++)
      slave_lastread[k] = slave_sc[k]->escaped_weight();
  }

  grid_changed();
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitPuff::grid_changed()
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

void FixSurfaceEmitPuff::create_task(int icell)
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

void FixSurfaceEmitPuff::perform_task()
{
  // Emit-count cap: use FixEmit::ntotal (cumulative emitted) rather than
  // live particle count, so byproducts from chem/adas dissociation don't
  // inflate the count. Fast-path once latched: skip the allreduce entirely.
  // cap_remaining is the remaining global quota for this step; the inner
  // task loops clamp ninsert against it so final emitted count = stop_at_np
  // exactly (single-rank emit) or +/- (nprocs - 1) worst case.
  bigint cap_remaining = -1;  // -1 sentinel = no cap
  if (stop_at_np > 0) {
    if (stop_latched) return;
    bigint nt_local = ntotal;
    bigint nt_global;
    MPI_Allreduce(&nt_local, &nt_global, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    if (nt_global >= stop_at_np) {
      stop_latched = 1;
      return;
    }
    cap_remaining = stop_at_np - nt_global;
  }

  // SLAVE mode: total macroparticles to emit this window. Escaped-weight
  // deltas are collected on every rank (collide tallies are rank-local),
  // summed once globally; the one-step lag is inherent (emission happens
  // at start-of-step, absorption during the previous move).
  double slave_ntarget_total = 0.0;
  if (npmode == SLAVE) {
    double dsum_local = 0.0;
    for (size_t k = 0; k < slave_sc.size(); k++) {
      double cur = slave_sc[k]->escaped_weight();
      dsum_local += cur - slave_lastread[k];
      slave_lastread[k] = cur;
    }
    double escaped_atoms = 0.0;
    MPI_Allreduce(&dsum_local,&escaped_atoms,1,MPI_DOUBLE,MPI_SUM,world);
    slave_ntarget_total = (slave_R * escaped_atoms +
                           slave_gamma * update->dt * nevery) / update->fnum;
  }

  int i,m,n,pcell,isurf,ninsert,nactual,isp,ispecies,ntri,id;
  double indot,scosine,rn,ntarget,vr,alpha,beta;
  double beta_un,normalized_distbn_fn,theta,erot,evib;
  double vnmag,vamag,vbmag;
  double *normal,*p1,*p2,*p3,*atan,*btan,*vstream,*vscale;
  double x[3],v[3],e1[3],e2[3];
  Particle::OnePart *p;

  const double dt = update->dt;
  int *species = particle->mixture[imix]->species;

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

    ninsert = -1;

    if (npmode == FLOW) {
      // Thermal inflow from mixture nrho at the surface temp. Summed per
      // species via mol_inflow (Bird 1994, p.425). For strictly zero mixture
      // vstream this reduces to vscale * fraction / (2*sqrt(pi)) per species.
      nrho = particle->mixture[imix]->nrho;
      double ntarget_all = 0.0;
      for (isp = 0; isp < nspecies; isp++) {
        double phi_in = mol_inflow(indot, vscale[isp], fraction[isp]);
        double nsp = phi_in * nrho * tasks[i].area * dt / fnum;
        ntarget_all += nsp;
        if (perspecies) tasks[i].ntargetsp[isp] = nsp;
      }
      ntarget = ntarget_all;
    } else if (npmode == SLAVE) {
      // SLAVE: flux-closure total distributed across tasks by area fraction
      ntarget = slave_ntarget_total * tasks[i].ntarget;
    } else {
      // CONSTANT: n N means N particles per step distributed across tasks by
      // area fraction (tasks[i].ntarget was set in grid_changed()).
      ntarget = np * tasks[i].ntarget;
    }
    if (ntarget <= 0.0) continue;

    if (perspecies) {
      for (isp = 0; isp < nspecies; isp++) {
        ispecies = species[isp];
        // FLOW: per-species count carries the mol_inflow (vscale) weighting;
        // CONSTANT/SLAVE never populate ntargetsp, use the fraction split
        double ntarget_sp = (npmode == FLOW) ? tasks[i].ntargetsp[isp]
                                             : ntarget * fraction[isp];
        ninsert = static_cast<int>(ntarget_sp + random->uniform());
        if (cap_remaining >= 0) {
          bigint slot = cap_remaining - static_cast<bigint>(nsingle);
          if (slot <= 0) { ninsert = 0; }
          else if (static_cast<bigint>(ninsert) > slot) ninsert = static_cast<int>(slot);
        }
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
      if (ninsert < 0) {
        ninsert = static_cast<int>(ntarget + random->uniform());
        if (cap_remaining >= 0) {
          bigint slot = cap_remaining - static_cast<bigint>(nsingle);
          if (slot <= 0) { ninsert = 0; }
          else if (static_cast<bigint>(ninsert) > slot) ninsert = static_cast<int>(slot);
        }
        if (ninsert <= 0) continue;
      }

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

void FixSurfaceEmitPuff::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks,ntaskmax*sizeof(Task),
                                    "surface/emit/puff:tasks");

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

int FixSurfaceEmitPuff::option(int narg, char **arg)
{
  if (strcmp(arg[0],"n") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/puff command");
    np = atoi(arg[1]);
    if (np <= 0) npmode = FLOW;
    else npmode = CONSTANT;
    return 2;
  }

  if (strcmp(arg[0],"normal") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/puff command");
    if (strcmp(arg[1],"yes") == 0) normalflag = 1;
    else if (strcmp(arg[1],"no") == 0) normalflag = 0;
    else error->all(FLERR,"Illegal fix surface/emit/puff command");
    return 2;
  }

  if (strcmp(arg[0],"slave") == 0) {
    // slave R Gamma_ext scID1[,scID2,...] : emit R x (weight absorbed by the
    // listed vanish collides last window) + Gamma_ext [atoms/s] external feed
    if (4 > narg) error->all(FLERR,"Illegal fix surface/emit/puff command");
    slave_R = atof(arg[1]);
    slave_gamma = atof(arg[2]);
    if (slave_R < 0.0 || slave_gamma < 0.0)
      error->all(FLERR,"fix surface/emit/puff slave R and Gamma must be >= 0");
    delete [] slave_scstr;
    int n = strlen(arg[3]) + 1;
    slave_scstr = new char[n];
    strcpy(slave_scstr,arg[3]);
    npmode = SLAVE;
    return 4;
  }

  if (strcmp(arg[0],"stop_at_np") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/puff command");
    stop_at_np = ATOBIGINT(arg[1]);
    if (stop_at_np < 0)
      error->all(FLERR,"fix surface/emit/puff stop_at_np must be >= 0");
    return 2;
  }

  error->all(FLERR,"Illegal fix surface/emit/puff command");
  return 0;
}
