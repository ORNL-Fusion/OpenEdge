/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_emit_surf_file.h"
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
#include "variable.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include <unordered_map>

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NOSUBSONIC,PTBOTH,PONLY};
enum{FLOW,CONSTANT,VARIABLE};
enum{INT,DOUBLE};                                        // several files

#define DELTATASK 256
#define TEMPLIMIT 1.0e5

/* ---------------------------------------------------------------------- */

FixEmitSurfFile::FixEmitSurfFile(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix emit/surf command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  if (strcmp(arg[4], "subsonic") != 0)
    error->all(FLERR,"Fix emit/surf/file: expected keyword 'subsonic' after MIX and GROUP");

  erodedFluxPath = strdup(arg[5]);   // or utils::strdup(arg[5]) in your codebase

  initializeErodedFluxData(); 

  // optional args

  np = 0;
  npmode = FLOW;
  npstr = NULL;
  normalflag = 0;
  subsonic = 0;
  subsonic_style = NOSUBSONIC;
  subsonic_warning = 0;

  nrho_custom_flag = vstream_custom_flag = speed_custom_flag =
    temp_custom_flag = fractions_custom_flag = 0;
  nrho_custom_id = vstream_custom_id = speed_custom_id =
    temp_custom_id = fractions_custom_id = NULL;

  max_cummulative = 0;
  cummulative_custom = NULL;


  int iarg = 4;
  options(narg-iarg,&arg[iarg]);

  // error checks

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf not allowed for implicit surfaces");
  if ((npmode == CONSTANT || npmode == VARIABLE) && perspecies)
    error->all(FLERR,"Cannot use fix emit/surf with n a constant or variable "
               "with perspecies yes");

  int custom_any = 0;
  if (nrho_custom_flag || vstream_custom_flag || speed_custom_flag ||
      temp_custom_flag || fractions_custom_flag) custom_any = 1;
  if (custom_any && npmode != FLOW)
    error->all(FLERR,"Cannot use fix emit/surf with n != 0 and custom options");
  if (custom_any && subsonic)
    error->all(FLERR,"Cannot use fix emit/surf with subsonic and custom options");

  // task list and subsonic data structs

  tasks = NULL;
  ntask = ntaskmax = 0;

  maxactive = 0;
  activecell = NULL;

  dimension = domain->dimension;

  // create instance of Cut2d,Cut3d for geometry calculations

  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta,domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixEmitSurfFile::~FixEmitSurfFile()
{
  delete [] npstr;

  delete [] nrho_custom_id;
  delete [] vstream_custom_id;
  delete [] speed_custom_id;
  delete [] temp_custom_id;
  delete [] fractions_custom_id;
  memory->destroy(cummulative_custom);

  for (int i = 0; i < ntaskmax; i++) {
    delete [] tasks[i].ntargetsp;
    delete [] tasks[i].vscale;
    delete [] tasks[i].path;
    delete [] tasks[i].fracarea;
  }
  memory->sfree(tasks);
  memory->destroy(activecell);

  // deallocate Cut2d,Cut3d

  if (dimension == 3) delete cut3d;
  else delete cut2d;
}

/* ---------------------------------------------------------------------- */

void FixEmitSurfFile::init()
{
  // invoke FixEmit::init() to set flags

  FixEmit::init();

  // copies of class data before invoking parent init() and count_task()

  fnum = update->fnum;
  dt = update->dt;

  nspecies = particle->mixture[imix]->nspecies;
  fraction = particle->mixture[imix]->fraction;
  cummulative = particle->mixture[imix]->cummulative;

  // subsonic prefactor

  tprefactor = update->mvv2e / (3.0*update->boltz);

  // mixture soundspeed, used by subsonic PONLY as default cell property

  double avegamma = 0.0;
  double avemass = 0.0;

  for (int m = 0; m < nspecies; m++) {
    int ispecies = particle->mixture[imix]->species[m];
    avemass += fraction[m] * particle->species[ispecies].mass;
    avegamma += fraction[m] * (1.0 + 2.0 /
                               (3.0 + particle->species[ispecies].rotdof));
  }

  soundspeed_mixture = sqrt(avegamma * update->boltz *
                            particle->mixture[imix]->temp_thermal / avemass);

  // magvstream = magnitude of mxiture vstream vector
  // norm_vstream = unit vector in stream direction

  double *vstream = particle->mixture[imix]->vstream;
  magvstream = MathExtra::len3(vstream);

  norm_vstream[0] = vstream[0];
  norm_vstream[1] = vstream[1];
  norm_vstream[2] = vstream[2];
  if (norm_vstream[0] != 0.0 || norm_vstream[1] != 0.0 ||
      norm_vstream[2] != 0.0)
    MathExtra::norm3(norm_vstream);

  // if used, reallocate ntargetsp and vscale for each task
  // b/c nspecies count of mixture may have changed

  if (perspecies) {
    for (int i = 0; i < ntask; i++) {
      delete [] tasks[i].ntargetsp;
      tasks[i].ntargetsp = new double[nspecies];
    }
  }
  if (subsonic_style == PONLY) {
    for (int i = 0; i < ntask; i++) {
      delete [] tasks[i].vscale;
      tasks[i].vscale = new double[nspecies];
    }
  }
  // create tasks for all grid cells

  grid_changed();
}


/* ----------------------------------------------------------------------
   grid changed operation
   invoke create_tasks() to rebuild entire task list
   invoked after per-processor list of grid cells has changed
------------------------------------------------------------------------- */

void FixEmitSurfFile::grid_changed()
{
  // if any custom attributes are used,
  // ensure owned custom values are spread to nlocal+nghost surfs

  if (nrho_custom_flag && surf->estatus[nrho_custom_index] == 0)
    surf->spread_custom(nrho_custom_index);
  if (vstream_custom_flag && surf->estatus[vstream_custom_index] == 0)
    surf->spread_custom(vstream_custom_index);
  if (speed_custom_flag && surf->estatus[speed_custom_index] == 0)
    surf->spread_custom(speed_custom_index);
  if (temp_custom_flag && surf->estatus[temp_custom_index] == 0)
    surf->spread_custom(temp_custom_index);
  if (fractions_custom_flag && surf->estatus[fractions_custom_index] == 0)
    surf->spread_custom(fractions_custom_index);

  // create tasks for grid cell / surf pairs

  create_tasks();

  // if custom fractions requested and perspecies = 0,
  // setup cummulaitve_custom array for nlocal surfs

  if (fractions_custom_flag && !perspecies) {
    int nslocal = surf->nlocal;
    if (nslocal > max_cummulative) {
      memory->destroy(cummulative_custom);
      max_cummulative = nslocal;
      memory->create(cummulative_custom,nslocal,nspecies,"fix/emit/surf:cummulative_custom");
    }

    double **fractions = surf->edarray_local[surf->ewhich[fractions_custom_index]];
    int isp;

    for (int isurf = 0; isurf < nslocal; isurf++) {
      for (isp = 0; isp < nspecies; isp++) {
        if (isp) cummulative_custom[isurf][isp] =
                   cummulative_custom[isurf][isp-1] + fractions[isurf][isp];
        else cummulative_custom[isurf][isp] = fractions[isurf][isp];
      }
    }
  }

  // for MODE = CONSTANT or VARIABLE
  // set per-task ntarget to fraction of its area / total area

  if (npmode != FLOW) {
    double areasum_me = 0.0;
    for (int i = 0; i < ntask; i++)
      areasum_me += tasks[i].area;

    double areasum;
    MPI_Allreduce(&areasum_me,&areasum,1,MPI_DOUBLE,MPI_SUM,world);

    for (int i = 0; i < ntask; i++)
      tasks[i].ntarget = tasks[i].area / areasum;
  }
}

/* ----------------------------------------------------------------------
   create task for one grid cell
   add them to tasks list and increment ntasks
------------------------------------------------------------------------- */

void FixEmitSurfFile::create_task(int icell)
{
  int i,m,isurf,isp,npoint,isplit,subcell;
  double indot,area,areaone,ntargetsp;
  double *normal,*p1,*p2,*p3,*path;
  double cpath[36],delta[3],e1[3],e2[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  // no tasks if no surfs in cell

  if (cells[icell].nsurf == 0) return;

  // no tasks if cell is outside of flow volume

  if (cinfo[icell].volume == 0.0) return;

  // setup for loop over surfs

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  double nrho = particle->mixture[imix]->nrho;
  double *vstream = particle->mixture[imix]->vstream;
  double *vscale = particle->mixture[imix]->vscale;
  double temp_thermal = particle->mixture[imix]->temp_thermal;

  if (nrho_custom_flag) nrho_custom = surf->edvec_local[surf->ewhich[nrho_custom_index]];
  if (vstream_custom_flag) vstream_custom = surf->edarray_local[surf->ewhich[vstream_custom_index]];
  if (speed_custom_flag) speed_custom = surf->edvec_local[surf->ewhich[speed_custom_index]];
  if (temp_custom_flag) temp_custom = surf->edvec_local[surf->ewhich[temp_custom_index]];
  if (fractions_custom_flag) fractions_custom =
                               surf->edarray_local[surf->ewhich[fractions_custom_index]];
  double temp_thermal_custom;

  double *lo = cells[icell].lo;
  double *hi = cells[icell].hi;
  surfint *csurfs = cells[icell].csurfs;
  int nsurf = cells[icell].nsurf;

  // loop over surfs in cell
  // use Cut2d/Cut3d to find overlap area and geoemtry of overlap

  for (i = 0; i < nsurf; i++) {
    isurf = csurfs[i];

    if (dimension == 2) {
      if (!(lines[isurf].mask & groupbit)) continue;
    } else {
      if (!(tris[isurf].mask & groupbit)) continue;
    }

    // if requested, override mixture properties with custom per-surf attributes

    if (nrho_custom_flag) nrho = nrho_custom[isurf];
    if (vstream_custom_flag) vstream = vstream_custom[isurf];
    if (speed_custom_flag) magvstream = speed_custom[isurf];
    if (temp_custom_flag) temp_thermal_custom = temp_custom[isurf];
    if (fractions_custom_flag) fraction = fractions_custom[isurf];

    // set cell parameters of task
    // pcell = sub cell for particles if a split cell

    if (ntask == ntaskmax) grow_task();

    tasks[ntask].icell = icell;
    tasks[ntask].isurf = isurf;
    if (cells[icell].nsplit == 1) tasks[ntask].pcell = icell;
    else {
      isplit = cells[icell].isplit;
      subcell = sinfo[isplit].csplits[i];
      tasks[ntask].pcell = sinfo[isplit].csubs[subcell];
    }

    // set geometry-dependent params of task
    // indot = vstream magnitude for normalflag = 1
    // indot = vstream dotted with surface normal for normalflag = 0
    // area = area for insertion = extent of line/triangle inside grid cell

    double x_center[3];
    if (dimension == 2) {
      normal = lines[isurf].norm;
      if (normalflag) indot = magvstream;
      else indot = vstream[0]*normal[0] + vstream[1]*normal[1];

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

      // axisymmetric "area" of line segment = surf area of truncated cone
      // PI (y1+y2) sqrt( (y1-y2)^2 + (x1-x2)^2) )

      if (domain->axisymmetric) {
        double sqrtarg = (path[1]-path[4])*(path[1]-path[4]) +
          (path[0]-path[3])*(path[0]-path[3]);
        area = MY_PI * (path[1]+path[4]) * sqrt(sqrtarg);
      } else {
        MathExtra::sub3(&path[0],&path[3],delta);
        area = MathExtra::len3(delta);
      }
      tasks[ntask].area = area;

      // set 2 tangent vectors to surf normal
      // tan1 is in xy plane, 90 degrees from normal
      // tan2 is unit +z vector

      tasks[ntask].tan1[0] = normal[1];
      tasks[ntask].tan1[1] = -normal[0];
      tasks[ntask].tan1[2] = 0.0;
      tasks[ntask].tan2[0] = 0.0;
      tasks[ntask].tan2[1] = 0.0;
      tasks[ntask].tan2[2] = 1.0;

       x_center[0] = 0.5 * (p1[0] + p2[0]);
       x_center[1] = 0.5 * (p1[1] + p2[1]);
       x_center[2] = 0;

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
      for (m = 0; m < npoint-2; m++)
        tasks[ntask].fracarea[m] /= area;

      // set 2 random tangent vectors to surf normal
      // tangent vectors are also normal to each other

      delta[0] = random->uniform();
      delta[1] = random->uniform();
      delta[2] = random->uniform();
      MathExtra::cross3(tris[isurf].norm,delta,tasks[ntask].tan1);
      MathExtra::norm3(tasks[ntask].tan1);
      MathExtra::cross3(tris[isurf].norm,tasks[ntask].tan1,tasks[ntask].tan2);
      MathExtra::norm3(tasks[ntask].tan2);

      x_center[0] = (p1[0] + p2[0] + p3[0]) / 3.0;
      x_center[1] = (p1[1] + p2[1] + p3[1]) / 3.0;
      x_center[2] = (p1[2] + p2[2] + p3[2]) / 3.0;

    }


    // set ntarget and ntargetsp via mol_inflow()
    // will be overwritten if mode != FLOW
    // skip task if final ntarget = 0.0, due to large outbound vstream
    // do not skip for subsonic since it resets ntarget every step

    tasks[ntask].ntarget = 0.0;
    for (isp = 0; isp < nspecies; isp++) {

      double flux = interpErodedFluxAt( tasks[i].isurf,x_center[0], x_center[1], eroded_flux_data, /*clamp_outside=*/true);
      ntargetsp *= flux*area*dt / fnum;
      ntargetsp /= cinfo[icell].weight;
      tasks[ntask].ntarget += ntargetsp;

    }

    if (!subsonic) {
      if (tasks[ntask].ntarget == 0.0) continue;
      if (tasks[ntask].ntarget >= MAXSMALLINT)
        error->one(FLERR,
                   "Fix emit/surf insertion count exceeds 32-bit int");
    }

    // initialize other task values with mixture or per-surf custom properties
    // may be overwritten by subsonic methods

    double utemp;

    tasks[ntask].nrho = nrho;
    if (temp_custom_flag) {
      tasks[ntask].temp_thermal = temp_thermal_custom;
      tasks[ntask].temp_rot = temp_thermal_custom;
      tasks[ntask].temp_vib = temp_thermal_custom;
      utemp = temp_thermal_custom;
    } else {
      tasks[ntask].temp_thermal = temp_thermal;
      tasks[ntask].temp_rot = particle->mixture[imix]->temp_rot;
      tasks[ntask].temp_vib = particle->mixture[imix]->temp_vib;
      utemp = temp_thermal;
    }
    tasks[ntask].magvstream = magvstream;
    tasks[ntask].vstream[0] = vstream[0];
    tasks[ntask].vstream[1] = vstream[1];
    tasks[ntask].vstream[2] = vstream[2];

    // increment task counter

    ntask++;
  }
}

/* ----------------------------------------------------------------------
   insert particles in grid cells with emitting surface elements
------------------------------------------------------------------------- */

void FixEmitSurfFile::perform_task()
{
  if (!subsonic) error->all(FLERR,"FixEmitSurfFile::perform_task(): subsonic-only build");

  const double dt = update->dt;
  int *species = particle->mixture[imix]->species;

  // Recompute per-task inflow counts & vstream/temp for subsonic
  subsonic_inflow();

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

  const int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;
  const int nfix_update_custom = modify->n_update_custom;

  for (int i = 0; i < ntask; i++) {
    const int pcell = tasks[i].pcell;
    const int isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"Bad surf index");

    double *normal = (dimension == 2) ? lines[isurf].norm : tris[isurf].norm;
    double *atan   = tasks[i].tan1;
    double *btan   = tasks[i].tan2;

    const double temp_rot    = tasks[i].temp_rot;
    const double temp_vib    = tasks[i].temp_vib;
    const double magvstream  = tasks[i].magvstream;
    double *vstream          = tasks[i].vstream;

    // vscale selection from original subsonic path
    double *vscale = (subsonic_style == PONLY) ?
                      tasks[i].vscale : particle->mixture[imix]->vscale;

    // stream component along surface normal
    const double indot = normalflag
                           ? magvstream
                           : (vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2]);

    // SUBSONIC: always per-species using ntargetsp computed by subsonic_inflow()
    for (int isp = 0; isp < nspecies; isp++) {
      const int ispecies = species[isp];
      // target insertions (floor + jitter)
      const double ntarget = tasks[i].ntargetsp[isp] + random->uniform();
      const int ninsert = static_cast<int>(ntarget);



      int nactual = 0;
      for (int m = 0; m < ninsert; m++) {
        // --- sample position on the emitting face (segment or triangle fan) ---
        double x[3], v[3], e1[3], e2[3];
        double *p1, *p2, *p3;

        if (dimension == 2) {
          double rn = random->uniform();
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3];
          x[0] = p1[0] + rn*(p2[0]-p1[0]);
          x[1] = p1[1] + rn*(p2[1]-p1[1]);
          x[2] = 0.0;
        } else {
          double rn = random->uniform();
          const int ntri = tasks[i].npoint - 2;
          int n = 0;
          for (; n < ntri; n++) if (rn < tasks[i].fracarea[n]) break;
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3*(n+1)];
          p3 = &tasks[i].path[3*(n+2)];
          MathExtra::sub3(p2,p1,e1);
          MathExtra::sub3(p3,p1,e2);
          double alpha = random->uniform();
          double beta  = random->uniform();
          if (alpha + beta > 1.0) { alpha = 1.0 - alpha; beta = 1.0 - beta; }
          x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
          x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
          x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
        }

        // --- Simple inflow: sample normal from thermal scale, no Bird/A-R, no drift ---
        const double vnmag = vscale[isp] * fabs(random->gaussian());  // half-normal into domain
        // keep your tangential sampling (Rayleigh magnitude + random angle)
        const double theta = MY_2PI * random->uniform();
        const double vr    = vscale[isp] * sqrt(-log(random->uniform()));

        double vamag, vbmag;
        vamag = vr * sin(theta);
        vbmag = vr * cos(theta);

        v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
        v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
        v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];


        const double erot = particle->erot(ispecies,temp_rot,random);
        const double evib = particle->evib(ispecies,temp_vib,random);
        const int id = MAXSMALLINT*random->uniform();

        particle->add_particle(id,ispecies,pcell,x,v,erot,evib);

        auto *pp = &particle->particles[particle->nlocal-1];
        pp->flag = PSURF + 1 + isurf;
        pp->dtremain = dt * random->uniform();

        if (nsurf_tally)
          for (int k = 0; k < nsurf_tally; k++)
            // slist_active[k]->surf_tally(pp->dtremain,isurf,pcell,0,NULL,pp,NULL);
            slist_active[k]->surf_tally(pp->dtremain, isurf, pcell, 0, NULL, pp, NULL);


        if (nfix_update_custom)
          modify->update_custom(particle->nlocal-1,tasks[i].temp_thermal,
                                temp_rot,temp_vib,vstream);

        nactual++;
      } // m insertions

      nsingle += nactual;
    } // species
  }   // tasks
}

/* ----------------------------------------------------------------------
   recalculate task properties based on subsonic BC
------------------------------------------------------------------------- */

void FixEmitSurfFile::subsonic_inflow() {
  if (!particle->sorted) subsonic_sort();
  subsonic_grid();

  int isp, icell;
  double mass, indot, area, nrho, temp_thermal, vscale, ntargetsp;
  double *vstream, *normal;

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

  Particle::Species *species = particle->species;
  Grid::ChildInfo   *cinfo   = grid->cinfo;
  int   *mspecies = particle->mixture[imix]->species;
  double fnum = update->fnum;
  double boltz = update->boltz;
  double dt = update->dt;               // make sure you have this

  double x_center[3];

  for (int i = 0; i < ntask; i++) {
    vstream = tasks[i].vstream;

    // normal · vstream
    if (dimension == 2) {
      if (normalflag) indot = magvstream;
      else {
        normal = lines[tasks[i].isurf].norm;
        indot = vstream[0]*normal[0] + vstream[1]*normal[1];
      }
      x_center[0] = 0.5*(lines[tasks[i].isurf].p1[0] + lines[tasks[i].isurf].p2[0]);
      x_center[1] = 0.5*(lines[tasks[i].isurf].p1[1] + lines[tasks[i].isurf].p2[1]);
      x_center[2] = 0.0;
    } else {
      if (normalflag) indot = magvstream;
      else {
        normal = tris[tasks[i].isurf].norm;
        indot = vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];
      }
      const auto &tri = tris[tasks[i].isurf];
      x_center[0] = (tri.p1[0] + tri.p2[0] + tri.p3[0]) / 3.0;
      x_center[1] = (tri.p1[1] + tri.p2[1] + tri.p3[1]) / 3.0;
      x_center[2] = (tri.p1[2] + tri.p2[2] + tri.p3[2]) / 3.0;
    }

    area         = tasks[i].area;
    nrho         = tasks[i].nrho;
    temp_thermal = tasks[i].temp_thermal;
    icell        = tasks[i].icell;

    // reset totals
    tasks[i].ntarget = 0.0;
    for (isp = 0; isp < nspecies; isp++) tasks[i].ntargetsp[isp] = 0.0;   // <<< reset

    // interpolate once per task (or per species if species-dependent)
    const double flux = interpErodedFluxAt(tasks[i].isurf,x_center[0], x_center[1],
                                           eroded_flux_data, /*clamp_outside=*/true);

    for (isp = 0; isp < nspecies; isp++) {
      mass   = species[mspecies[isp]].mass;
      vscale = sqrt(2.0 * boltz * temp_thermal / mass);
      // if you use PONLY later, keep vscale per species:
      if (subsonic_style == PONLY) tasks[i].vscale[isp] = vscale;         // <<< store
      ntargetsp  = flux * area * dt / fnum * fraction[isp];
      ntargetsp /= cinfo[icell].weight;
      tasks[i].ntargetsp[isp] = ntargetsp;
    }

    if (tasks[i].ntarget >= MAXSMALLINT)
      error->one(FLERR,"Fix emit/surf subsonic insertion count exceeds 32-bit int");
  }


}

/* ----------------------------------------------------------------------
   identify particles in grid cells associated with a task
   store count and linked list, same as for particle sorting
------------------------------------------------------------------------- */

void FixEmitSurfFile::subsonic_sort()
{
  int i,icell;

  // initialize particle sort lists for grid cells assigned to tasks
  // use task pcell, not icell

  Grid::ChildInfo *cinfo = grid->cinfo;

  for (i = 0; i < ntask; i++) {
    icell = tasks[i].pcell;
    cinfo[icell].first = -1;
    cinfo[icell].count = 0;
  }

  // reallocate particle next list if necessary

  particle->sort_allocate();

  // update list of active grid cells if necessary
  // active cells = those assigned to tasks
  // active_current flag set by parent class

  if (!active_current) {
    if (grid->nlocal > maxactive) {
      memory->destroy(activecell);
      maxactive = grid->nlocal;
      memory->create(activecell,maxactive,"emit/face:active");
    }
    memset(activecell,0,maxactive*sizeof(int));
    for (i = 0; i < ntask; i++) activecell[tasks[i].pcell] = 1;
    active_current = 1;
  }

  // loop over particles to store linked lists for active cells
  // not using reverse loop like Particle::sort(),
  //   since this should only be created/used occasionally

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nlocal = particle->nlocal;

  for (i = 0; i < nlocal; i++) {
    icell = particles[i].icell;
    if (!activecell[icell]) continue;
    next[i] = cinfo[icell].first;
    cinfo[icell].first = i;
    cinfo[icell].count++;
  }
}

/* ----------------------------------------------------------------------
   compute number density, thermal temperature, stream velocity
   only for grid cells associated with a task
   first compute for grid cells, then adjust due to boundary conditions
------------------------------------------------------------------------- */

void FixEmitSurfFile::subsonic_grid()
{
  int m,ip,np,icell,ispecies;
  double mass,masstot,gamma,ke;
  double nrho_cell,massrho_cell,temp_thermal_cell,press_cell;
  double mass_cell,gamma_cell,soundspeed_cell,vsmag;
  double mv[4];
  double *v,*vstream,*vscale,*normal;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  Particle::Species *species = particle->species;
  double boltz = update->boltz;

  int temp_exceed_flag = 0;
  double tempmax = 0.0;

  for (int i = 0; i < ntask; i++) {
    icell = tasks[i].pcell;
    np = cinfo[icell].count;

    // accumulate needed per-particle quantities
    // mv = mass*velocity terms, masstot = total mass
    // gamma = rotational/tranlational DOFs

    mv[0] = mv[1] = mv[2] = mv[3] = 0.0;
    masstot = gamma = 0.0;

    ip = cinfo[icell].first;

    while (ip >= 0) {
      ispecies = particles[ip].ispecies;
      mass = species[ispecies].mass;
      v = particles[ip].v;
      mv[0] += mass*v[0];
      mv[1] += mass*v[1];
      mv[2] += mass*v[2];
      mv[3] += mass * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      masstot += mass;
      gamma += 1.0 + 2.0 / (3.0 + species[ispecies].rotdof);
      ip = next[ip];
    }

    // compute/store nrho, 3 temps, vstream for task
    // also vscale for PONLY
    // if sound speed = 0.0 due to <= 1 particle in cell or
    //   all particles having COM velocity, set via mixture properties

    vstream = tasks[i].vstream;
    vstream[0] = vstream[1] = vstream[2] = 0.0;

    tasks[i].nrho = nsubsonic;
    tasks[i].nrho = nsubsonic;
    // temp_thermal_cell = tsubsonic;
    tasks[i].temp_thermal = tsubsonic;
    tasks[i].temp_rot = tasks[i].temp_vib = tsubsonic;

  }

  // // test if any task has invalid thermal temperature for first time

  // if (!subsonic_warning)
  //   subsonic_warning = subsonic_temperature_check(temp_exceed_flag,tempmax);
}

/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixEmitSurfFile::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks,ntaskmax*sizeof(Task),
                                    "emit/face:tasks");

  // set all new task bytes to 0 so valgrind won't complain
  // if bytes between fields are uninitialized

  memset(&tasks[oldmax],0,(ntaskmax-oldmax)*sizeof(Task));

  // allocate vectors in each new task or set to NULL
  // path and fracarea are allocated later to specific sizes

  if (perspecies) {
    for (int i = oldmax; i < ntaskmax; i++)
      tasks[i].ntargetsp = new double[nspecies];
  } else {
    for (int i = oldmax; i < ntaskmax; i++)
      tasks[i].ntargetsp = NULL;
  }

  if (subsonic_style == PONLY) {
    for (int i = oldmax; i < ntaskmax; i++)
      tasks[i].vscale = new double[nspecies];
  } else {
    for (int i = oldmax; i < ntaskmax; i++)
      tasks[i].vscale = NULL;
  }

  for (int i = oldmax; i < ntaskmax; i++) {
    tasks[i].path = NULL;
    tasks[i].fracarea = NULL;
  }
}

/* ----------------------------------------------------------------------
   process keywords specific to this class
------------------------------------------------------------------------- */

int FixEmitSurfFile::option(int narg, char **arg)
{

  if (strcmp(arg[0],"subsonic") == 0) {
    // if (3 > narg) error->all(FLERR,"Illegal fix emit/surf command");
    subsonic = 1;
    subsonic_style = PTBOTH;
    psubsonic = input->numeric(FLERR,arg[2]);
    tsubsonic = input->numeric(FLERR,arg[2]) * update->ev2kelvin;
    nsubsonic = psubsonic / (update->boltz * tsubsonic);
    subsonic_style = PONLY;
    nsubsonic = psubsonic / (update->boltz * tsubsonic);

    return 3;
  }

  error->all(FLERR,"Illegal fix emit/surf command");
  return 0;
}


/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
ErodedFluxData FixEmitSurfFile::readErodedFlux(const std::string& filePath) {
    printf("Reading eroded flux data from file: %s\n", filePath.c_str());
    ErodedFluxData data;

    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);

        // Utility to read 1D dataset
        auto read1D = [&](const std::string& name) -> std::vector<double> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dim;
            space.getSimpleExtentDims(&dim);
            std::vector<double> vec(dim);
            ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
            return vec;
        };

        // First read coordinates
        data.r = read1D("r");
        data.z = read1D("z");
        size_t nr = data.r.size();
        size_t nz = data.z.size();

        // Load eroded flux at wall: 1D
        data.eroded_flux = read1D("eroded_flux");

    } catch (const H5::Exception& e) {
        fprintf(stderr, "HDF5 error: %s\n", e.getCDetailMsg());
        throw;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        throw;
    }

    printf("Finished reading plasma data from file: %s\n", filePath.c_str());
    return data;
}


/*----------------------------------------------------------------------
   broadcast magnetic field data
------------------------------------------------------------------------- */
void FixEmitSurfFile::broadcastErodedFluxData(ErodedFluxData& data) {
  int me = comm->me;

  // Broadcast sizes of 1D vectors (e.g., r and z for the magnetic field)
  int r_size = data.r.size();
  int z_size = data.z.size();
  int ef_size = data.eroded_flux.size();
  MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&z_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&ef_size, 1, MPI_INT, 0, world);

  // Resize vectors on non-root processes
  if (me != 0) {
      data.r.resize(r_size);
      data.z.resize(z_size);
      data.eroded_flux.resize(ef_size);
  }

  // Broadcast 1D vector data (r and z and eroded_flux)
  MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.eroded_flux.data(), ef_size, MPI_DOUBLE, 0, world);

}

/*---------------------------------
  initialize magnetic field data
-----------------------------------*/
void FixEmitSurfFile::initializeErodedFluxData() {
  int me = comm->me;
// 

  // Load magnetic field data only on the root process
  if (me == 0) {
      eroded_flux_data = readErodedFlux(erodedFluxPath);
  }

  // Broadcast the eroded flux data to all processes
  broadcastErodedFluxData(eroded_flux_data);
}

double FixEmitSurfFile::interpErodedFluxAt(int isurf,
                                           double r_val,
                                           double z_val,
                                           const ErodedFluxData& data,
                                           bool clamp_outside) const
{
  // ---- cache check only by surface id ----
  auto it = flux_cache.find(isurf);
  if (it != flux_cache.end()) {
    return it->second;
  }

  const auto& R  = data.r;
  const auto& Z  = data.z;
  const auto& EF = data.eroded_flux;

  if (R.empty() || Z.empty() || EF.empty()) {
    flux_cache[isurf] = 0.0;
    return 0.0;
  }

  const int nr = static_cast<int>(R.size());
  const int nz = static_cast<int>(Z.size());

  double result = 0.0;

  // ---------- scattered-point IDW branch ----------
  if (static_cast<int>(EF.size()) != nr * nz) {
    if (clamp_outside) {
      auto rmm = std::minmax_element(R.begin(), R.end());
      auto zmm = std::minmax_element(Z.begin(), Z.end());
      r_val = std::min(std::max(r_val, *rmm.first),  *rmm.second);
      z_val = std::min(std::max(z_val, *zmm.first),  *zmm.second);
    }

    const double eps = 1e-12;
    const double p   = 2.0;
    const int    N   = static_cast<int>(EF.size());
    const int    K   = std::min(4, N);

    // exact hit short-circuit
    for (int i = 0; i < N; ++i) {
      const double dr = R[i] - r_val, dz = Z[i] - z_val;
      if (dr*dr + dz*dz < eps*eps) {
        result = EF[i];
        flux_cache[isurf] = result;
        return result;
      }
    }

    // build distance index array
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;

    auto dist2 = [&](int i) {
      const double dr = R[i] - r_val, dz = Z[i] - z_val;
      return dr*dr + dz*dz;
    };

    // partition so first K are nearest
    std::nth_element(idx.begin(), idx.begin() + (K-1), idx.end(),
                     [&](int a, int b){ return dist2(a) < dist2(b); });

    // IDW over the K nearest
    double wsum = 0.0, vsum = 0.0;
    for (int t = 0; t < K; ++t) {
      int i = idx[t];
      double d = std::sqrt(dist2(i));
      double w = 1.0 / std::pow(d + eps, p);
      wsum += w;
      vsum += w * EF[i];
    }
    result = (wsum > 0.0) ? (vsum / wsum) : 0.0;
    flux_cache[isurf] = result;
    return result;
  }

  // ---------- tensor-grid bilinear path ----------
  const bool r_out = (r_val < R.front() || r_val > R.back());
  const bool z_out = (z_val < Z.front() || z_val > Z.back());
  if ((r_out || z_out) && !clamp_outside) {
    flux_cache[isurf] = 0.0;
    return 0.0;
  }

  if (clamp_outside) {
    if (r_val < R.front()) r_val = R.front();
    if (r_val > R.back())  r_val = R.back();
    if (z_val < Z.front()) z_val = Z.front();
    if (z_val > Z.back())  z_val = Z.back();
  }

  auto r_it = std::lower_bound(R.begin(), R.end(), r_val);
  auto z_it = std::lower_bound(Z.begin(), Z.end(), z_val);

  int r1 = std::max(0, int(r_it - R.begin()) - 1);
  int r2 = std::min(nr - 1, r1 + 1);
  int z1 = std::max(0, int(z_it - Z.begin()) - 1);
  int z2 = std::min(nz - 1, z1 + 1);

  const double R1 = R[r1], R2 = R[r2];
  const double Z1 = Z[z1], Z2 = Z[z2];
  const double denom = (R2 - R1) * (Z2 - Z1);

  auto idx = [&](int zi, int ri) -> int { return zi * nr + ri; };

  const double Q11 = EF[idx(z1, r1)];
  const double Q21 = EF[idx(z1, r2)];
  const double Q12 = EF[idx(z2, r1)];
  const double Q22 = EF[idx(z2, r2)];

  if (denom == 0.0) {
    result = 0.25 * (Q11 + Q12 + Q21 + Q22);
    flux_cache[isurf] = result;
    return result;
  }

  const double tR1 = (R2 - r_val), tR2 = (r_val - R1);
  const double tZ1 = (Z2 - z_val), tZ2 = (z_val - Z1);

  result = (Q11 * tR1 * tZ1 +
            Q21 * tR2 * tZ1 +
            Q12 * tR1 * tZ2 +
            Q22 * tR2 * tZ2) / denom;

  flux_cache[isurf] = result;
  return result;
}
