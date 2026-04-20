/* ----------------------------------------------------------------------
    OpenEdge: fix emit/surf/recycle
    EIRENE-faithful wall recycling driven by local plasma Bohm flux.
    See header for algorithm summary.
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_emit_surf_recycle.h"
#include "fix_plasma_data.h"
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

#define DELTATASK 256

namespace {
  constexpr double QE    = 1.602176634e-19;   // C
  constexpr double AMU   = 1.66053906660e-27; // kg
  constexpr double KB    = 1.380649e-23;      // J/K
}

/* ---------------------------------------------------------------------- */

FixEmitSurfRecycle::FixEmitSurfRecycle(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  // Usage: fix ID emit/surf/recycle mix group plasma_fix_ID
  //              [mass <amu>] [R <val>] [twall <K>]
  if (narg < 5) error->all(FLERR,"Illegal fix emit/surf/recycle command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf/recycle mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf/recycle group ID does not exist");
  groupbit = surf->bitmask[igroup];

  ifix_plasma = modify->find_fix(arg[4]);
  if (ifix_plasma < 0)
    error->all(FLERR,"Fix emit/surf/recycle plasma fix ID does not exist");
  plasma = dynamic_cast<FixPlasmaData *>(modify->fix[ifix_plasma]);
  if (!plasma)
    error->all(FLERR,"Fix emit/surf/recycle requires a fix plasma/data");

  // defaults
  mass_amu  = 2.0;    // D+
  R_recycle = 0.99;   // EIRENE PRFCT default
  twall     = 400.0;  // K

  int iarg = 5;
  options(narg-iarg, &arg[iarg]);

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf/recycle requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf/recycle not allowed for implicit surfaces");

  tasks = NULL;
  ntask = ntaskmax = 0;

  dimension = domain->dimension;
  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta, domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixEmitSurfRecycle::~FixEmitSurfRecycle()
{
  if (copymode) return;

  for (int i = 0; i < ntaskmax; i++) {
    delete [] tasks[i].path;
    delete [] tasks[i].fracarea;
  }
  memory->sfree(tasks);

  if (dimension == 3) delete cut3d;
  else delete cut2d;
}

/* ---------------------------------------------------------------------- */

void FixEmitSurfRecycle::init()
{
  FixEmit::init();

  fnum = update->fnum;

  nspecies    = particle->mixture[imix]->nspecies;
  fraction    = particle->mixture[imix]->fraction;
  cummulative = particle->mixture[imix]->cummulative;

  if (ifix_plasma < 0 || ifix_plasma >= modify->nfix)
    error->all(FLERR,"Fix emit/surf/recycle plasma fix ID no longer exists");
  plasma = dynamic_cast<FixPlasmaData *>(modify->fix[ifix_plasma]);
  if (!plasma)
    error->all(FLERR,"Fix emit/surf/recycle requires a fix plasma/data");

  grid_changed();
}

/* ---------------------------------------------------------------------- */

void FixEmitSurfRecycle::grid_changed()
{
  create_tasks();
}

/* ----------------------------------------------------------------------
   create task for one grid cell (called by FixEmit::create_tasks)
------------------------------------------------------------------------- */

void FixEmitSurfRecycle::create_task(int icell)
{
  int i, m, isurf, npoint, isplit, subcell;
  double area, areaone;
  double *normal, *p1, *p2, *p3, *path;
  double cpath[36], delta[3], e1[3], e2[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  if (cells[icell].nsurf == 0) return;
  if (cinfo[icell].volume == 0.0) return;

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

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

      p1 = lines[isurf].p1;
      p2 = lines[isurf].p2;
      npoint = cut2d->clip_external(p1, p2, lo, hi, cpath);
      if (npoint < 2) continue;

      tasks[ntask].npoint = 2;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[6];
      path = tasks[ntask].path;
      path[0] = cpath[0]; path[1] = cpath[1]; path[2] = 0.0;
      path[3] = cpath[2]; path[4] = cpath[3]; path[5] = 0.0;

      if (domain->axisymmetric) {
        double sqrtarg = (path[1]-path[4])*(path[1]-path[4]) +
                         (path[0]-path[3])*(path[0]-path[3]);
        area = MY_PI * (path[1]+path[4]) * sqrt(sqrtarg);
      } else {
        MathExtra::sub3(&path[0], &path[3], delta);
        area = MathExtra::len3(delta);
      }
      tasks[ntask].area = area;

      tasks[ntask].tan1[0] =  normal[1];
      tasks[ntask].tan1[1] = -normal[0];
      tasks[ntask].tan1[2] =  0.0;
      tasks[ntask].tan2[0] =  0.0;
      tasks[ntask].tan2[1] =  0.0;
      tasks[ntask].tan2[2] =  1.0;

      tasks[ntask].rmid = 0.5 * (path[0] + path[3]);
      tasks[ntask].zmid = 0.5 * (path[1] + path[4]);

      tasks[ntask].inward[0] = -normal[0];
      tasks[ntask].inward[1] = -normal[1];
      tasks[ntask].inward[2] =  0.0;

    } else {
      normal = tris[isurf].norm;

      p1 = tris[isurf].p1;
      p2 = tris[isurf].p2;
      p3 = tris[isurf].p3;
      npoint = cut3d->clip_external(p1, p2, p3, lo, hi, cpath);
      if (npoint < 3) continue;

      tasks[ntask].npoint = npoint;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[npoint*3];
      path = tasks[ntask].path;
      memcpy(path, cpath, npoint*3*sizeof(double));
      delete [] tasks[ntask].fracarea;
      tasks[ntask].fracarea = new double[npoint-2];

      area = 0.0;
      p1 = &path[0];
      for (m = 0; m < npoint-2; m++) {
        p2 = &path[3*(m+1)];
        p3 = &path[3*(m+2)];
        MathExtra::sub3(p2, p1, e1);
        MathExtra::sub3(p3, p1, e2);
        MathExtra::cross3(e1, e2, delta);
        areaone = fabs(0.5 * MathExtra::len3(delta));
        area += areaone;
        tasks[ntask].fracarea[m] = area;
      }
      tasks[ntask].area = area;
      for (m = 0; m < npoint-2; m++) tasks[ntask].fracarea[m] /= area;

      delta[0] = random->uniform();
      delta[1] = random->uniform();
      delta[2] = random->uniform();
      MathExtra::cross3(tris[isurf].norm, delta, tasks[ntask].tan1);
      MathExtra::norm3(tasks[ntask].tan1);
      MathExtra::cross3(tris[isurf].norm, tasks[ntask].tan1, tasks[ntask].tan2);
      MathExtra::norm3(tasks[ntask].tan2);

      double cx = 0.0, cy = 0.0, cz = 0.0;
      for (int k = 0; k < npoint; k++) {
        cx += path[3*k + 0];
        cy += path[3*k + 1];
        cz += path[3*k + 2];
      }
      cx /= npoint; cy /= npoint; cz /= npoint;
      tasks[ntask].rmid = sqrt(cx*cx + cy*cy);
      tasks[ntask].zmid = cz;

      tasks[ntask].inward[0] = -normal[0];
      tasks[ntask].inward[1] = -normal[1];
      tasks[ntask].inward[2] = -normal[2];
    }

    tasks[ntask].vscale_molec = 0.0;
    tasks[ntask].ntarget = 0.0;
    ntask++;
  }
}

/* ----------------------------------------------------------------------
   per-surface emission rate (particles / step) from local Bohm flux
------------------------------------------------------------------------- */

double FixEmitSurfRecycle::emission_rate_per_surface(int itask)
{
  const double R = tasks[itask].rmid;
  const double Z = tasks[itask].zmid;

  const double ne = plasma->interp2D(plasma->dens_e, R, Z);
  const double te = plasma->interp2D(plasma->temp_e, R, Z);
  const double ti = plasma->temp_i.empty() ? te
                  : plasma->interp2D(plasma->temp_i, R, Z);

  if (!std::isfinite(ne) || !std::isfinite(te)) return 0.0;
  if (ne <= 0.0 || te <= 0.0) return 0.0;

  const double ti_eff = std::isfinite(ti) && ti > 0.0 ? ti : te;
  const double cs_arg = (te + ti_eff) * QE / (mass_amu * AMU);
  if (cs_arg <= 0.0) return 0.0;
  const double cs = std::sqrt(cs_arg);

  double sin_alpha = 1.0;
  if (plasma->has_bfield) {
    double Br, Bz, Bt;
    plasma->bfield_at(R, Z, Br, Bz, Bt);
    const double Bmag = std::sqrt(Br*Br + Bz*Bz + Bt*Bt);
    if (Bmag > 0.0) {
      const double proj = Br * tasks[itask].inward[0]
                        + Bz * tasks[itask].inward[1];
      sin_alpha = std::fabs(proj) / Bmag;
      if (sin_alpha > 1.0) sin_alpha = 1.0;
    }
  }

  const double gamma = ne * cs * sin_alpha;
  const double dot_N = 0.5 * R_recycle * gamma * tasks[itask].area;
  return dot_N * update->dt / fnum;
}

/* ----------------------------------------------------------------------
   perform one step's insertion for every task
------------------------------------------------------------------------- */

void FixEmitSurfRecycle::perform_task()
{
  int i, m, n, pcell, isurf, ninsert, nactual, isp, ispecies, ntri, id;
  double rn, ntarget, vr, alpha, beta, theta, erot, evib;
  double vnmag, vamag, vbmag;
  double *normal, *p1, *p2, *p3, *atan, *btan;
  double x[3], v[3], e1[3], e2[3];
  Particle::OnePart *p;

  const double dt = update->dt;
  int *species = particle->mixture[imix]->species;

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

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

    ntarget = emission_rate_per_surface(i);
    if (ntarget <= 0.0) continue;

    ninsert = static_cast<int>(ntarget + random->uniform());
    if (ninsert <= 0) continue;

    nactual = 0;
    for (m = 0; m < ninsert; m++) {
      rn = random->uniform();
      isp = 0;
      while (isp < nspecies - 1 && cummulative[isp] < rn) isp++;
      ispecies = species[isp];

      if (dimension == 2) {
        rn = random->uniform();
        p1 = &tasks[i].path[0];
        p2 = &tasks[i].path[3];
        x[0] = p1[0] + rn * (p2[0] - p1[0]);
        x[1] = p1[1] + rn * (p2[1] - p1[1]);
        x[2] = 0.0;
      } else {
        rn = random->uniform();
        ntri = tasks[i].npoint - 2;
        for (n = 0; n < ntri; n++)
          if (rn < tasks[i].fracarea[n]) break;
        p1 = &tasks[i].path[0];
        p2 = &tasks[i].path[3*(n+1)];
        p3 = &tasks[i].path[3*(n+2)];
        MathExtra::sub3(p2, p1, e1);
        MathExtra::sub3(p3, p1, e2);
        alpha = random->uniform();
        beta  = random->uniform();
        if (alpha + beta > 1.0) { alpha = 1.0 - alpha; beta = 1.0 - beta; }
        x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
        x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
        x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
      }

      if (region && !region->match(x)) continue;

      // Maxwellian flux velocity at twall, inward along normal.
      // v_perp ~ Rayleigh ~ sqrt(-ln U) * vscale
      // v_parallel ~ two independent Gaussians (tangential)
      const double mspec = particle->species[ispecies].mass;
      const double vscale_twall = std::sqrt(2.0 * KB * twall / mspec);

      vnmag = vscale_twall * std::sqrt(-std::log(random->uniform()));

      theta = MY_2PI * random->uniform();
      vr = vscale_twall * std::sqrt(-std::log(random->uniform()))
           / std::sqrt(2.0);
      vamag = vr * std::sin(theta);
      vbmag = vr * std::cos(theta);

      // normal[] points out of the fluid, so emission = -vnmag * normal
      v[0] = -vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
      v[1] = -vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
      v[2] = -vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

      erot = particle->erot(ispecies, twall, random);
      evib = particle->evib(ispecies, twall, random);
      id = MAXSMALLINT * random->uniform();

      particle->add_particle(id, ispecies, pcell, x, v, erot, evib);
      nactual++;

      p = &particle->particles[particle->nlocal-1];
      p->flag = PSURF + 1 + isurf;
      p->dtremain = dt * random->uniform();

      if (nsurf_tally)
        for (int k = 0; k < nsurf_tally; k++)
          slist_active[k]->surf_tally(p->dtremain, isurf, pcell, 0, NULL, p, NULL);

      if (nfix_update_custom)
        modify->update_custom(particle->nlocal-1, twall, twall, twall, v);
    }
    nsingle += nactual;
  }
}

/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixEmitSurfRecycle::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks, ntaskmax*sizeof(Task),
                                    "emit/surf/recycle:tasks");
  memset(&tasks[oldmax], 0, (ntaskmax-oldmax)*sizeof(Task));

  for (int i = oldmax; i < ntaskmax; i++) {
    tasks[i].path = NULL;
    tasks[i].fracarea = NULL;
  }
}

/* ----------------------------------------------------------------------
   keyword options
------------------------------------------------------------------------- */

int FixEmitSurfRecycle::option(int narg, char **arg)
{
  if (strcmp(arg[0], "mass") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf/recycle command");
    mass_amu = atof(arg[1]);
    if (mass_amu <= 0.0)
      error->all(FLERR,"fix emit/surf/recycle mass must be > 0");
    return 2;
  }
  if (strcmp(arg[0], "R") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf/recycle command");
    R_recycle = atof(arg[1]);
    if (R_recycle < 0.0 || R_recycle > 1.0)
      error->all(FLERR,"fix emit/surf/recycle R must be in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "twall") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf/recycle command");
    twall = atof(arg[1]);
    if (twall <= 0.0)
      error->all(FLERR,"fix emit/surf/recycle twall must be > 0");
    return 2;
  }

  error->all(FLERR,"Illegal fix emit/surf/recycle command");
  return 0;
}
