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
#include "variable.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NOSUBSONIC,PTBOTH,PONLY};
enum{FLOW,CONSTANT};
enum{INT,DOUBLE};                                        // several files

#define DELTATASK 256
#define TEMPLIMIT 1.0e5

/* ---------------------------------------------------------------------- */

FixEmitSurfPmi::FixEmitSurfPmi(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix emit/surf command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  // optional args

  np = 0;
  npmode = FLOW;
  npstr = NULL;
  normalflag = 0;
  subsonic = 0;
  subsonic_style = NOSUBSONIC;
  subsonic_warning = 0;

  max_cummulative = 0;
  cummulative_custom = NULL;

  int iarg = 4;
  options(narg-iarg,&arg[iarg]);

  // error checks

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf requires surface elements");
  // if (surf->implicit)
  //   error->all(FLERR,"Fix emit/surf not allowed for implicit surfaces");
  // if ((npmode == CONSTANT ))
  //   error->all(FLERR,"Cannot use fix emit/surf with n a constant or variable "
  //              "with perspecies yes");

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

FixEmitSurfPmi::~FixEmitSurfPmi()
{
  delete [] npstr;


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

void FixEmitSurfPmi::init()
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

  if (subsonic_style == PONLY) {
    for (int i = 0; i < ntask; i++) {
      delete [] tasks[i].vscale;
      tasks[i].vscale = new double[nspecies];
    }
  }


  // check custom per-surf vectors or arrays

  // if custom fractions is set, reset any fractions which are less than zero
  // do this for owned custom surfs, grid_changed() will propagate to nlocal+nghost surfs

  // create tasks for all grid cells

  grid_changed();
}


/* ----------------------------------------------------------------------
   grid changed operation
   invoke create_tasks() to rebuild entire task list
   invoked after per-processor list of grid cells has changed
------------------------------------------------------------------------- */

void FixEmitSurfPmi::grid_changed()
{
  // if any custom attributes are used,
  // create tasks for grid cell / surf pairs

  create_tasks();

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

void FixEmitSurfPmi::create_task(int icell)
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
    }

    // set ntarget and ntargetsp via mol_inflow()
    // will be overwritten if mode != FLOW
    // skip task if final ntarget = 0.0, due to large outbound vstream
    // do not skip for subsonic since it resets ntarget every step

    tasks[ntask].ntarget = 0.0;
    for (isp = 0; isp < nspecies; isp++) {
      
    double gamma_Li = get_flux(icell);  // define this however you store the flux

    ntargetsp = gamma_Li * area * dt / fnum;
    // printf("icell %d gamma_Li %g\n",icell,gamma_Li);
    // ntargetsp /= cinfo[icell].weight;
    // printf(" weight icell %d cinfo[icell].weight %g\n",icell,cinfo[icell].weight);
    tasks[ntask].ntarget += ntargetsp;
    // printf("icell %d ntargetsp %g\n",icell,ntargetsp);
    // exit(0);
    }


    // initialize other task values with mixture or per-surf custom properties
    // may be overwritten by subsonic methods

    double utemp;

    tasks[ntask].nrho = nrho;
    tasks[ntask].temp_thermal = temp_thermal;
    tasks[ntask].temp_rot = particle->mixture[imix]->temp_rot;
    tasks[ntask].temp_vib = particle->mixture[imix]->temp_vib;
    utemp = temp_thermal;

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

void FixEmitSurfPmi::perform_task()
{
  int i,m,n,pcell,isurf,ninsert,nactual,isp,ispecies,ntri,id;
  double indot,scosine,rn,ntarget,vr,alpha,beta;
  double beta_un,normalized_distbn_fn,theta,erot,evib;
  double vnmag,vamag,vbmag;
  double *normal,*p1,*p2,*p3,*atan,*btan,*vstream,*vscale;
  double x[3],v[3],e1[3],e2[3];
  Particle::OnePart *p;

  double dt = update->dt;
  int *species = particle->mixture[imix]->species;

  // if subsonic, re-compute particle inflow counts for each task
  // also computes current per-task temp_thermal and vstream

  // insert particles for each task = cell/surf pair
  // ntarget/ninsert is either perspecies or for all species
  // for one particle:
  //   x = random position with overlap of surf with cell
  //   v = randomized thermal velocity + vstream
  //       if normalflag, mag of vstream is applied to surf normal dir
  //       first stage: normal dimension (normal)
  //       second stage: parallel dimensions (tan1,tan2)

  // double while loop until randomized particle velocity meets 2 criteria
  // inner do-while loop:
  //   v = vstream-component + vthermal is into simulation box
  //   see Bird 1994, p 425
  // outer do-while loop:
  //   shift Maxwellian distribution by stream velocity component
  //   see Bird 1994, p 259, eq 12.5

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;

  int nfix_update_custom = modify->n_update_custom;

  for (i = 0; i < ntask; i++) {
    pcell = tasks[i].pcell;
    isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"BAD surf index\n");
    if (dimension == 2) normal = lines[isurf].norm;
    else normal = tris[isurf].norm;
    atan = tasks[i].tan1;
    btan = tasks[i].tan2;

    temp_thermal = tasks[i].temp_thermal;
    temp_rot = tasks[i].temp_rot;
    temp_vib = tasks[i].temp_vib;
    magvstream = tasks[i].magvstream;
    vstream = tasks[i].vstream;

    if (subsonic_style == PONLY) vscale = tasks[i].vscale;
    else vscale = particle->mixture[imix]->vscale;
    if (normalflag) indot = magvstream;
    else indot = vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];

    //  double nrho_cell = np * fnum / cinfo[icell].volume;

    double nrho_cell = update->plasma_data_map[pcell].dens_i;
    tasks[i].nrho = nrho_cell;
    // perspecies yes get_flux(int icell)
      // set ntarget for insertion mode FLOW, CONSTANT, or VARIABLE
      // for FLOW: ntarget is already set within task
      // ninsert = rounded-down (ntarget + random number)

      if (npmode == CONSTANT) {
        ntarget = np * tasks[i].ntarget;
      }
      ninsert = static_cast<int> (ntarget + random->uniform());

      // loop over ninsert for all species
      // use cummulative fractions to assign species for each insertion
      // if requested, override cummulative from mixture with cummulative for isurf

      nactual = 0;
      for (int m = 0; m < ninsert; m++) {
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
          do {
            beta_un = (6.0*random->uniform() - 3.0);
          } while (beta_un + scosine < 0.0);
          normalized_distbn_fn = 2.0 * (beta_un + scosine) /
            (scosine + sqrt(scosine*scosine + 2.0)) *
            exp(0.5 + (0.5*scosine)*(scosine-sqrt(scosine*scosine + 2.0)) -
                beta_un*beta_un);
        } while (normalized_distbn_fn < random->uniform());

             double mass = update->target_material_mass * update->proton_mass;
        // printf("mass %g\n",mass);
        double target_Es = update->target_material_binding_energy/2.0;
        double twall = target_Es;
        double vrm = sqrt(2.0 * update->boltz * update->ev2kelvin * twall / mass);
        // printf("vrm %g\n",vrm);

        if (normalflag) {
        // vnmag = beta_un*vscale[isp] + magvstream;
        vnmag = vrm;
        }

        else vnmag = beta_un*vscale[isp] + indot;

        theta = MY_2PI * random->uniform();
        // vr = vscale[isp] * sqrt(-log(random->uniform()));
        vr = vrm * sqrt(-log(random->uniform()));

        vamag = vr * sin(theta);
        vbmag = vr * cos(theta);

        v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
        v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
        v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];
        // printf("v %g %g %g\n",v[0],v[1],v[2]);

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
            slist_active[k]->surf_tally(isurf,pcell,0,NULL,p,NULL);

        if (nfix_update_custom)
          modify->update_custom(particle->nlocal-1,temp_thermal,
                               temp_rot,temp_vib,vstream);
      }

      nsingle += nactual;
    
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
                                    "emit/face:tasks");

  // set all new task bytes to 0 so valgrind won't complain
  // if bytes between fields are uninitialized

  memset(&tasks[oldmax],0,(ntaskmax-oldmax)*sizeof(Task));

  // allocate vectors in each new task or set to NULL
  // path and fracarea are allocated later to specific sizes

  for (int i = oldmax; i < ntaskmax; i++)
    tasks[i].ntargetsp = NULL;

    for (int i = oldmax; i < ntaskmax; i++)
      tasks[i].vscale = NULL;
  

  for (int i = oldmax; i < ntaskmax; i++) {
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
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf command");
      np = atoi(arg[1]);
      if (np == 0) npmode = FLOW;
      else npmode = CONSTANT;
    
    return 2;
  }

  if (strcmp(arg[0],"normal") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf command");
    if (strcmp(arg[1],"yes") == 0) normalflag = 1;
    else if (strcmp(arg[1],"no") == 0) normalflag = 0;
    else error->all(FLERR,"Illegal fix emit/surf command");
    return 2;
  }

  error->all(FLERR,"Illegal fix emit/surf command");
  return 0;
}

/* ----------------------------------------------------------------------
  comppute total flux at cell face
  for all species
  ----------------------------------------------------------------------- */

double FixEmitSurfPmi::get_flux(int icell)
{
  
  double dens = update->plasma_data_map[icell].dens_i;
  double v_parr = update->plasma_data_map[icell].parr_flow;
  double flux_density = dens * abs(v_parr);
  printf(" icell %d dens %g v_parr %g flux_density %g\n",icell,dens,v_parr,flux_density);
  return flux_density;


}
