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
#include "fix_emit_droplet.h"
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
#include <cmath>  // make sure this is present at top

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NOSUBSONIC,PTBOTH,PONLY};
enum{FLOW,CONSTANT,VARIABLE};
enum{INT,DOUBLE};                                        // several files

#define DELTATASK 256
#define TEMPLIMIT 1.0e5

/* ---------------------------------------------------------------------- */

FixEmitDroplet::FixEmitDroplet(SPARTA *sparta, int narg, char **arg) :
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

  max_cummulative = 0;

// set safe defaults FIRST
xpos = ypos = zpos = 0.0;
energy_eV = 0.0;
phi_deg = 0.0;
theta_deg = 0.0;

int iarg = 4;
options(narg-iarg,&arg[iarg]);  // now options can override

printf("[emit/droplet] parsed options: energy=%g eV, phi=%g deg, theta=%g deg, pos=(%g,%g,%g)\n",
       energy_eV, phi_deg, theta_deg, xpos, ypos, zpos);



  // error checks

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf not allowed for implicit surfaces");
  if ((npmode == CONSTANT || npmode == VARIABLE) && perspecies)
    error->all(FLERR,"Cannot use fix emit/surf with n a constant or variable "
               "with perspecies yes");
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

FixEmitDroplet::~FixEmitDroplet()
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

void FixEmitDroplet::init()
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

  // create tasks for all grid cells

  grid_changed();
}


/* ----------------------------------------------------------------------
   grid changed operation
   invoke create_tasks() to rebuild entire task list
   invoked after per-processor list of grid cells has changed
------------------------------------------------------------------------- */

void FixEmitDroplet::grid_changed()
{
  // create tasks for grid cell / surf pairs

  create_tasks();


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

void FixEmitDroplet::create_task(int icell)
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
      ntargetsp = mol_inflow(indot,vscale[isp],fraction[isp]);
      ntargetsp *= nrho*area*dt / fnum;
      ntargetsp /= cinfo[icell].weight;
      tasks[ntask].ntarget += ntargetsp;
      if (perspecies) tasks[ntask].ntargetsp[isp] = ntargetsp;
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

void FixEmitDroplet::perform_task()
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

  // if npmode = VARIABLE, set npcurrent to variable evaluation

  double npcurrent;


  // insert particles for each task = cell/surf pair

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;

  for (i = 0; i < ntask; i++) {
    pcell = tasks[i].pcell;
    isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"BAD surf index\n");
    if (dimension == 2) normal = lines[isurf].norm;
    else normal = tris[isurf].norm;
    atan = tasks[i].tan1;
    btan = tasks[i].tan2;

    tasks[i].temp_thermal  = energy_eV * 11604.525; // eV to K
    temp_thermal = tasks[i].temp_thermal;
    temp_rot = tasks[i].temp_rot;
    temp_vib = tasks[i].temp_vib;
    magvstream = tasks[i].magvstream;
    vstream = tasks[i].vstream;

    if (subsonic_style == PONLY) vscale = tasks[i].vscale;
    else vscale = particle->mixture[imix]->vscale;
    if (normalflag) indot = magvstream;
    else indot = vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];


      // set ntarget for insertion mode FLOW, CONSTANT, or VARIABLE
      // for FLOW: ntarget is already set within task
      // for CONSTANT or VARIABLE: task narget is fraction of its surf's area
      //   scale fraction by np or npcurrent (variable evaluation)
      // ninsert = rounded-down (ntarget + random number)

      if (npmode == FLOW) ntarget = tasks[i].ntarget;
      else if (npmode == CONSTANT) ntarget = np * tasks[i].ntarget;
      else if (npmode == VARIABLE) ntarget = npcurrent * tasks[i].ntarget;
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
        printf("Inserting particle of species %d\n", ispecies);
        scosine = indot / vscale[isp];

            // set fixed position first
        if (dimension == 3) {
          x[0] = xpos; x[1] = ypos; x[2] = zpos;
        } else {
          x[0] = xpos; x[1] = ypos; x[2] = 0.0;
        }

        // now it’s safe to check the region
        // if (region && !region->match(x)) continue;

          double mass_kg = particle->species[ispecies].mass;
          double energy_J = energy_eV *  update->echarge;
          double vtotal   = sqrt(2.0 * energy_J / mass_kg);

          double rad_phi   = phi_deg   * M_PI / 180.0;
          double rad_theta = theta_deg * M_PI / 180.0;

          v[0] = vtotal * sin(rad_phi) * cos(rad_theta);
          v[1] = vtotal * sin(rad_phi) * sin(rad_theta);
          v[2] = vtotal * cos(rad_phi);

          auto bad = [](double a){ return (a != a) || !std::isfinite(a); }; // NaN or Inf
          static int debug_once = 0;

          if (bad(v[0]) || bad(v[1]) || bad(v[2])) {
            printf("[emit/droplet][BAD-V] mass=%g, E_J=%g, vtotal=%g, "
                  "phi=%g, theta=%g, v=(%g,%g,%g)\n",
                  particle->species[ispecies].mass,
                  energy_eV * update->echarge,
                  sqrt(2.0 * energy_eV * update->echarge / particle->species[ispecies].mass),
                  phi_deg, theta_deg, v[0], v[1], v[2]);
            error->one(FLERR,"NaN/Inf in injected velocity");
          }
          if (bad(x[0]) || bad(x[1]) || bad(x[2])) {
        printf("[emit/droplet][BAD-X] pos=(%g,%g,%g)\n", x[0],x[1],x[2]);
        error->one(FLERR,"NaN/Inf in injected position");
      }

      if (!debug_once) {
        printf("[emit/droplet] will insert id=%d type=%d pos=(%g,%g,%g) v=(%g,%g,%g)\n",
              id, ispecies, x[0],x[1],x[2], v[0],v[1],v[2]);
        debug_once = 1;
      }


        erot = particle->erot(ispecies,temp_rot,random);
        evib = particle->evib(ispecies,temp_vib,random);
        id = MAXSMALLINT*random->uniform();

        if (std::isnan(x[0]) || std::isnan(x[1]) || std::isnan(x[2])) {
          error->one(FLERR,"NaN in injected position (x,y,z)");
        }

        particle->add_particle(id,ispecies,pcell,x,v,erot,evib);
        nactual++;

        p = &particle->particles[particle->nlocal-1];
        p->flag = PSURF + 1 + isurf;
        p->dtremain = dt * random->uniform();

        if (nsurf_tally)
          for (int k = 0; k < nsurf_tally; k++)
            slist_active[k]->surf_tally(p->dtremain,isurf,pcell,0,NULL,p,NULL);


      }
      nsingle += nactual;
    }
  
}


/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixEmitDroplet::grow_task()
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

int FixEmitDroplet::option(int narg, char **arg)
{
  if (strcmp(arg[0],"n") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf command");

    if (strstr(arg[1],"v_") == arg[1]) {
      npmode = VARIABLE;
      int n = strlen(&arg[1][2]) + 1;
      npstr = new char[n];
      strcpy(npstr,&arg[1][2]);
    } else {
      np = atoi(arg[1]);
      if (np == 0) npmode = FLOW;
      else npmode = CONSTANT;
    }
    return 2;
  }


  if (strcmp(arg[0],"normal") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf command");
    if (strcmp(arg[1],"yes") == 0) normalflag = 1;
    else if (strcmp(arg[1],"no") == 0) normalflag = 0;
    else error->all(FLERR,"Illegal fix emit/surf command");
    return 2;
  }
/// add an option angle {value}
  if (strcmp(arg[0],"angle") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix emit/surf command");
    requested_angle = atof(arg[1]);
    if (requested_angle < 0.0 || requested_angle > 90.0)
      error->all(FLERR,"Illegal fix emit/surf command");
    return 2;
  }

  // option to add particle positions xpos, ypos, zpos
  if (strcmp(arg[0],"position") == 0) {
    if (4 > narg) error->all(FLERR,"Illegal fix emit/surf command");
      xpos = atof(arg[1]);
      ypos = atof(arg[2]);
      zpos = atof(arg[3]);
      return 4;
  }

  if (strcmp(arg[0],"energy") == 0) {
    energy_eV = atof(arg[1]);
    return 2;
  }
  if (strcmp(arg[0],"phi") == 0) {
    phi_deg = atof(arg[1]);
    return 2;
  }
  if (strcmp(arg[0],"theta") == 0) {
    theta_deg = atof(arg[1]);
    return 2;
  }

  error->all(FLERR,"Illegal fix emit/surf command");
  return 0;
}