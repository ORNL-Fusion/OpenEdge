/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "spatype.h"
#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"
#include "math_const.h"
#include "particle.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "domain.h"
#include "comm.h"
#include "collide.h"
#include "grid.h"
#include "surf.h"
#include "surf_collide.h"
#include "surf_react.h"
#include "input.h"
#include "output.h"
#include "geometry.h"
#include "random_mars.h"
#include "timer.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace SPARTA_NS;

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // same as Domain
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain
enum{OUTSIDE,INSIDE,ONSURF2OUT,ONSURF2IN};      // several files
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Grid
enum{TALLYAUTO,TALLYREDUCE,TALLYRVOUS};         // same as Surf
enum{PERAUTO,PERCELL,PERSURF};                  // several files
enum{NOFIELD,CFIELD,PFIELD,GFIELD};             // several files


#define MAXSTUCK 20
#define EPSPARAM 1.0e-7
#define MAXLINE 16384

#define CORE_GROUP_NAME "CORE"
#define BIG 1.0e20

// either set ID or PROC/INDEX, set other to -1

//#define MOVE_DEBUG 1              // un-comment to debug one particle
#define MOVE_DEBUG_ID 308143534  // particle ID
#define MOVE_DEBUG_PROC -1        // owning proc
#define MOVE_DEBUG_INDEX -1   // particle index on owning proc
#define MOVE_DEBUG_STEP 4107    // timestep

/* ---------------------------------------------------------------------- */

Update::Update(SPARTA *sparta) : Pointers(sparta)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  ntimestep = 0;
  runflag = 0;
  firststep = laststep = 0;
  beginstep = endstep = 0;
  first_update = 0;

  time = 0.0;
  time_last_update = 0;

  unit_style = NULL;
  set_units("si");

  fnum = 1.0;
  nrho = 1.0;
  vstream[0] = vstream[1] = vstream[2] = 0.0;
  temp_thermal = 273.15;
  optmove_flag = 0;
  fstyle = NOFIELD;
  fieldID = NULL;
  efstyle = NOFIELD;
  efieldID = NULL;

  bfstyle = NOFIELD;
  bfieldID = NULL;

  ethermalstyle = NOFIELD;
  ethermalID = NULL;

  ithermalstyle = NOFIELD;
  ithermalID = NULL;
  ithermalflag = 0;
  ithermal_active = NULL;


  maxmigrate = 0;
  mlist = NULL;

  nslist_compute = nblist_compute = 0;
  slist_compute = blist_compute = NULL;
  slist_active = blist_active = NULL;

  nulist_surfcollide  = 0;
  ulist_surfcollide = NULL;

  ranmaster = new RanMars(sparta);

  reorder_period = 0;
  global_mem_limit = 0;
  mem_limit_grid_flag = 0;

  copymode = 0;

  plasma_state = NULL;
  plasmaStyle = 0;
  // plasma_data = NULL;
  efield[0] = efield[1] = efield[2] = 0.0;
  bfield[0] = bfield[1] = bfield[2] = 0.0;
  flow_v[0] = flow_v[1] = flow_v[2] = 0.0;
  grad_ti_r = grad_ti_t = grad_ti_z = 0.0;
  grad_te_r = grad_te_t = grad_te_z = 0.0;
  grad_temp_i[0] = grad_temp_i[1] = grad_temp_i[2] = 0.0;
  grad_temp_e[0] = grad_temp_e[1] = grad_temp_e[2] = 0.0;


  target_material = NULL;
  target_material_charge = 74.0;  //tungsten default
  target_material_mass   = 184.0;
  target_material_binding_energy = 8.79;

  std::string magneticFieldsPath = "";
  std::string plasmaStatePath = "";
  ionization_flag = 0;
  recombination_flag = 0;
  materials = std::vector<int>();
  magneticFieldsStyle = 0;
  cross_field_diffusion_flag = 0;
  background_collision_flag = 0;
  d_perp = 0;
  sheath_field_flag = 0;
  thermal_gradient_forces_flag = 0;
  cross_diffusion_flag = 0;

    // Plasma background material
  plasma_background_charge = 1.0; // Default to Deuterium
  plasma_background_mass = 2.0;
  plasma_background_material = NULL;

  // wall target material
  target_material = NULL;
  target_material_charge = 74.0;  // tungsten default
  target_material_mass   = 184.0;
  target_material_binding_energy = 8.79;

}

/* ---------------------------------------------------------------------- */

Update::~Update()
{
  if (copymode) return;

  delete [] unit_style;
  delete [] fieldID;
  memory->destroy(mlist);
  delete [] slist_compute;
  delete [] blist_compute;
  delete [] slist_active;
  delete [] blist_active;
  delete [] ulist_surfcollide;
  delete ranmaster;
}

/* ---------------------------------------------------------------------- */

void Update::set_units(const char *style)
{
  // physical constants from:
  // http://physics.nist.gov/cuu/Constants/Table/allascii.txt

  if (strcmp(style,"cgs") == 0) {
    boltz = 1.380649e-16;
    mvv2e = 1.0;
    dt = 1.0;

  } else if (strcmp(style,"si") == 0) {
    boltz = 1.380649e-23;
    mvv2e = 1.0;
    dt = 1.0;
    echarge = 1.60217646e-19;
    ev2kelvin = 11604.505;
    proton_mass = 1.6726219e-27;
    epsilon_0 = 8.854187817e-12;
    electron_mass = 9.10938215e-31;
    ANGSTROM = 1e-10;
    hbar = 1.0545718e-34;
    joule2ev = 6.242e18;
    c = 299792458.0;



  } else error->all(FLERR,"Illegal units command");

  delete [] unit_style;
  int n = strlen(style) + 1;
  unit_style = new char[n];
  strcpy(unit_style,style);
}

/* ---------------------------------------------------------------------- */

void Update::init()
{
  // init the Update class if borisorming a run, else just return
  // only set first_update if a run is being borisormed

  if (runflag == 0) return;
  first_update = 1;

  if (optmove_flag) {
    if (!grid->uniform)
      error->all(FLERR,"Cannot use optimized move with non-uniform grid");
    else if (surf->exist)
      error->all(FLERR,"Cannot use optimized move when surfaces are defined");
    else {
      for (int ifix = 0; ifix < modify->nfix; ifix++) {
        if (strstr(modify->fix[ifix]->style,"adapt") != NULL)
          error->all(FLERR,"Cannot use optimized move with fix adapt");
      }
    }
  }

  // choose the appropriate move method

  if (domain->dimension == 3) {
    if (surf->exist)
      moveptr = &Update::move<3,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<3,0,1>;
      else moveptr = &Update::move<3,0,0>;
    }
  } else if (domain->axisymmetric) {
    if (surf->exist)
      moveptr = &Update::move<1,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<1,0,1>;
      else moveptr = &Update::move<1,0,0>;
    }
  } else if (domain->dimension == 2) {
    if (surf->exist)
      moveptr = &Update::move<2,1,0>;
    else {
      if (optmove_flag) moveptr = &Update::move<2,0,1>;
      else moveptr = &Update::move<2,0,0>;
    }
  }

  // checks on external field options

  if (fstyle == CFIELD) {
    if (domain->dimension == 2 && field[2] != 0.0)
      error->all(FLERR,"External field in z not allowed for 2d");
    if (domain->axisymmetric && field[1] != 0.0)
      error->all(FLERR,
                 "External field in y not allowed for axisymmetric model");
  } else if (fstyle == PFIELD) {
    ifieldfix = modify->find_fix(fieldID);
    if (ifieldfix < 0) error->all(FLERR,"External field fix ID not found");
    if (!modify->fix[ifieldfix]->per_particle_field)
      error->all(FLERR,"External field fix does not compute necessary field");
  } else if (fstyle == GFIELD) {
    ifieldfix = modify->find_fix(fieldID);
    if (ifieldfix < 0) error->all(FLERR,"External field fix ID not found");
    if (!modify->fix[ifieldfix]->per_grid_field)
      error->all(FLERR,"External field fix does not compute necessary field");
  }
  // checks options for external electric field only particle perturbation
  // similar to above fstyle checks
  eperturbflag = 0;
  if (efstyle == PFIELD) {
    efieldfix = modify->find_fix(efieldID);        // <-- NO 'int' here
    if (efieldfix < 0) error->all(FLERR,"External electric field fix ID not found");
    if (!modify->fix[efieldfix]->per_particle_field)
      error->all(FLERR,"External electric field fix does not compute necessary field");
    efield_active = modify->fix[efieldfix]->field_active;  // packed columns
    eperturbflag = 1;
  }
    // add GFIELD now
    if (efstyle == GFIELD) {
    efieldfix = modify->find_fix(efieldID);        // <-- NO 'int' here
    if (efieldfix < 0) error->all(FLERR,"External electric field fix ID not found");
    if (!modify->fix[efieldfix]->per_grid_field)
      error->all(FLERR,"External electric field fix does not compute necessary field");
    efield_active = modify->fix[efieldfix]->field_active;  // packed columns
    eperturbflag = 1;
  }
  // checks options for external magnetic field only particle perturbation
  bperturbflag = 0;
  if (bfstyle == PFIELD) {
    bfieldfix = modify->find_fix(bfieldID);        
    if (bfieldfix < 0) error->all(FLERR,"External magnetic field fix ID not found");
    if (!modify->fix[bfieldfix]->per_particle_field)
      error->all(FLERR,"External magnetic field fix does not compute necessary field");
    bfield_active = modify->fix[bfieldfix]->field_active;  // packed columns
    bperturbflag = 1;
  }
  if (bfstyle == GFIELD) {
    bfieldfix = modify->find_fix(bfieldID);     
    if (bfieldfix < 0) error->all(FLERR,"External magnetic field fix ID not found");
    if (!modify->fix[bfieldfix]->per_grid_field)
      error->all(FLERR,"External magnetic field fix does not compute necessary field");
    bfield_active = modify->fix[bfieldfix]->field_active;  // packed columns
    bperturbflag = 1;
  }
  ethermalflag = 0;
  if (ethermalstyle==GFIELD) {
    ethermalfix = modify->find_fix(ethermalID);        
    if (ethermalfix < 0) error->all(FLERR,"External electron thermal gradient field fix ID not found");
    if (!modify->fix[ethermalfix]->per_grid_field)
      error->all(FLERR,"External electron thermal gradient field fix does not compute necessary field");
    ethermal_active = modify->fix[ethermalfix]->field_active;  // packed columns
    ethermalflag = 1;
  }
  ithermalflag = 0;
  if (ithermalstyle==GFIELD) {
    ithermalfix = modify->find_fix(ithermalID);        
    if (ithermalfix < 0) error->all(FLERR,"External ion thermal gradient field fix ID not found");
    if (!modify->fix[ithermalfix]->per_grid_field)
      error->all(FLERR,"External ion thermal gradient field fix does not compute necessary field");
    ithermal_active = modify->fix[ithermalfix]->field_active;  // packed columns
    ithermalflag = 1;
  }
  // moveperturb method is set if external field perturbs particle motion
  moveperturb = NULL;

  if (fstyle == CFIELD) {
    if (domain->dimension == 2) moveperturb = &Update::field2d;
    if (domain->dimension == 3) moveperturb = &Update::field3d;
  } else if (fstyle == PFIELD) {
    moveperturb = &Update::field_per_particle;
    field_active = modify->fix[ifieldfix]->field_active;
  } else if (fstyle == GFIELD) {
    moveperturb = &Update::field_per_grid;
    field_active = modify->fix[ifieldfix]->field_active;
  }

  if (moveperturb) perturbflag = 1;
  else perturbflag = 0;

  // if (plasmaStyle == 1) 
  initializePlasmaData();
  if (magneticFieldsStyle == 1) initializeMagneticData();
}

/* ---------------------------------------------------------------------- */

void Update::setup()
{
  // initialize counters in case stats outputs them
  // initialize running stats before each run

  ntouch_one = ncomm_one = 0;
  nboundary_one = nexit_one = 0;
  nscheck_one = nscollide_one = 0;
  surf->nreact_one = 0;

  first_running_step = update->ntimestep;
  niterate_running = 0;
  nmove_running = ntouch_running = ncomm_running = 0;
  nboundary_running = nexit_running = 0;
  nscheck_running = nscollide_running = 0;
  surf->nreact_running = 0;
  nstuck = naxibad = 0;

  collide_react = collide_react_setup();
  bounce_tally = bounce_setup();

  dynamic = 0;
  dynamic_setup();

  modify->setup();
  if (dynamic) dynamic_update();
  output->setup(1);

  
}

/* ---------------------------------------------------------------------- */

void Update::run(int nsteps)
{
  int n_start_of_step = modify->n_start_of_step;
  int n_end_of_step = modify->n_end_of_step;

  // external per grid cell field
  // only evaluate once at beginning of run b/c time-independent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq == 0)
    modify->fix[ifieldfix]->compute_field();

    // external per grid cell electric field
  if (efstyle == GFIELD && fieldfreq == 0)
    modify->fix[efieldfix]->compute_field();

    // external per grid cell magnetic field
  if (bfstyle == GFIELD && fieldfreq == 0)
    modify->fix[bfieldfix]->compute_field();

    // external per grid cell electron thermal gradient field
  // cellweightflag = 1 if grid-based particle weighting is ON

  int cellweightflag = 0;
  if (grid->cellweightflag) cellweightflag = 1;

  // loop over timesteps

  for (int i = 0; i < nsteps; i++) {

    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep++;

    if (collide_react) collide_react_reset();
    if (bounce_tally) bounce_set(ntimestep);


    timer->stamp();

    // dynamic parameter updates

    if (dynamic) dynamic_update();

    // start of step fixes

    if (n_start_of_step) {
      modify->start_of_step();
      timer->stamp(TIME_MODIFY);
    }

    // move particles

    if (cellweightflag) particle->pre_weight();
    (this->*moveptr)();
    timer->stamp(TIME_MOVE);

    // communicate particles

    comm->migrate_particles(nmigrate,mlist);
    if (cellweightflag) particle->post_weight();
    timer->stamp(TIME_COMM);

    if (collide) {
      particle->sort();
      timer->stamp(TIME_SORT);

      collide->collisions();
      timer->stamp(TIME_COLLIDE);
    }


    if (collide_react) collide_react_update();

    // diagnostic fixes

    if (n_end_of_step) {
      modify->end_of_step();
      timer->stamp(TIME_MODIFY);
    }

    // all output

    if (ntimestep == output->next) {
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
  }

  modify->post_run();
}

/* ----------------------------------------------------------------------
   advect particles thru grid
   DIM = 2/3 for 2d/3d, 1 for 2d axisymmetric
   SURF = 0/1 for no surfs or surfs
   use multiple iterations of move/comm if necessary
------------------------------------------------------------------------- */

template < int DIM, int SURF, int OPT > void Update::move()
{
  bool hitflag;
  int m,icell,icell_original,nmask,outface,bflag,nflag,pflag,itmp;
  int side,minside,minsurf,nsurf,cflag,isurf,exclude,stuck_iterate;
  int pstart,pstop,entryexit,any_entryexit,reaction;
  surfint *csurfs;
  cellint *neigh;
  double dtremain,frac,newfrac,param,minparam,rnew,dtsurf,tc,tmp;
  double xnew[3],xhold[3],xc[3],vc[3],minxc[3],minvc[3];
  double *x,*v,*lo,*hi;
  double Lx,Ly,Lz,dx,dy,dz;
  double *boxlo, *boxhi;
  Grid::ParentCell *pcell;
  Surf::Tri *tri;
  Surf::Line *line;
  Particle::OnePart iorig;
  Particle::OnePart *particles;
  Particle::OnePart *ipart,*jpart;
  Particle::Species* species = particle->species;


  if (OPT) {
    boxlo = domain->boxlo;
    boxhi = domain->boxhi;
    Lx = boxhi[0] - boxlo[0];
    Ly = boxhi[1] - boxlo[1];
    Lz = boxhi[2] - boxlo[2];
    dx = Lx/grid->unx;
    dy = Ly/grid->uny;
    dz = Lz/grid->unz;
  }

  // for 2d and axisymmetry only
  // xnew,xc passed to geometry routines which use or set z component

  if (DIM < 3) xnew[2] = xc[2] = 0.0;

  // extend migration list if necessary

  int nlocal = particle->nlocal;
  int maxlocal = particle->maxlocal;

  if (nlocal > maxmigrate) {
    maxmigrate = maxlocal;
    memory->destroy(mlist);
    memory->create(mlist,maxmigrate,"particle:mlist");
  }

  // counters

  niterate = 0;
  ntouch_one = ncomm_one = 0;
  nboundary_one = nexit_one = 0;
  nscheck_one = nscollide_one = 0;
  surf->nreact_one = 0;

  // move/migrate iterations

  Grid::ChildCell *cells = grid->cells;
  Grid::ParentCell *pcells = grid->pcells;
  Surf::Tri *tris = surf->tris;
  Surf::Line *lines = surf->lines;
  Grid::ChildInfo *cinfo = grid->cinfo;

  double dt = update->dt;

  // external per particle field
  // fix calculates field acting on all owned particles

  if (fstyle == PFIELD) modify->fix[ifieldfix]->compute_field();

  // external per grid cell field
  // evaluate once every fieldfreq steps b/c time-dependent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ifieldfix]->compute_field();

  // per-particle E
  if (efstyle == PFIELD) modify->fix[efieldfix]->compute_field();

  // per-particle B 
  if (bfstyle == PFIELD) modify->fix[bfieldfix]->compute_field();

  // per-grid E
  if (efstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[efieldfix]->compute_field();

  // // per-grid B
  if (bfstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[bfieldfix]->compute_field();

    // per-grid ethermal
  if (ethermalstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ethermalfix]->compute_field();

  // per grid ithermal
  if (ithermalstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ithermalfix]->compute_field();
  // one or more loops over particles
  // first iteration = all my particles
  // subsequent iterations = received particles

  while (1) {

    niterate++;
    particles = particle->particles;
    nmigrate = 0;
    entryexit = 0;

    if (niterate == 1) {
      pstart = 0;
      pstop = nlocal;
    }

    for (int i = pstart; i < pstop; i++) {
      pflag = particles[i].flag;

      // received from another proc and move is done
      // if first iteration, PDONE is from a previous step,
      //   set pflag to PKEEP so move the particle on this step
      // else do nothing

      if (pflag == PDONE) {
        pflag = particles[i].flag = PKEEP;
        if (niterate > 1) continue;
      }
    
      x = particles[i].x;
      v = particles[i].v;
      exclude = -1;

      double mass = species[particles[i].ispecies].mass;
      double charge = species[particles[i].ispecies].charge;
      
      // apply moveperturb() to PKEEP and PINSERT since are computing xnew
      // not to PENTRY,PEXIT since are just re-computing xnew of sender
      // set xnew[2] to linear move for axisymmetry, will be remapped later
      // let pflag = PEXIT persist to check during axisymmetric cell crossing
      // 

      if (pflag == PKEEP) {
        dtremain = dt;
        if (DIM == 2)
        {
          pusherBoris2D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3)
        {
          pusher_boris3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PINSERT) {
        if (DIM == 2) {
          pusherBoris2D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3) {
          pusher_boris3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PENTRY) {
        icell = particles[i].icell;
        if (cells[icell].nsplit > 1) {
          if (DIM == 3 && SURF) icell = split3d(icell,x);
          if (DIM < 3 && SURF) icell = split2d(icell,x);
          particles[i].icell = icell;
        }
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      } else if (pflag == PEXIT) {
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      } else if (pflag >= PSURF) {
        dtremain = particles[i].dtremain;
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
        if (pflag > PSURF) exclude = pflag - PSURF - 1;
      }

      // optimized move

      if (OPT) {
        int optmove = 1;

        if (xnew[0] < boxlo[0] || xnew[0] > boxhi[0])
          optmove = 0;

        if (xnew[1] < boxlo[1] || xnew[1] > boxhi[1])
          optmove = 0;

        if (DIM == 3) {
          if (xnew[2] < boxlo[2] || xnew[2] > boxhi[2])
            optmove = 0;
        }

        if (optmove) {
          const int ip = static_cast<int>((xnew[0] - boxlo[0])/dx);
          const int jp = static_cast<int>((xnew[1] - boxlo[1])/dy);
          int kp = 0;
          if (DIM == 3) kp = static_cast<int>((xnew[2] - boxlo[2])/dz);

          int cellIdx = (kp*grid->uny + jp)*grid->unx + ip + 1;

          // particle outside ghost grid halo must use standard move

          if (grid->hash->find(cellIdx) != grid->hash->end()) {

            int icell = (*(grid->hash))[cellIdx];

            // reset particle cell and coordinates

            particles[i].icell = icell;
            particles[i].flag = PKEEP;
            x[0] = xnew[0];
            x[1] = xnew[1];
            x[2] = xnew[2];

            if (cells[icell].proc != me) {
              mlist[nmigrate++] = i;
              particles[i].flag = PDONE;
              ncomm_one++;
            }

            continue;
          }
        }
      }

      particles[i].flag = PKEEP;
      icell = particles[i].icell;
      lo = cells[icell].lo;
      hi = cells[icell].hi;
      neigh = cells[icell].neigh;
      nmask = cells[icell].nmask;
      stuck_iterate = 0;
      ntouch_one++;

      // advect one particle from cell to cell and thru surf collides til done

      //int iterate = 0;

      while (1) {

#ifdef MOVE_DEBUG
        if (DIM == 3) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g %g: xnew %g %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g %g: hi %g %g %g: DTR %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                   icell,cells[icell].id,
                   lo[0],lo[1],lo[2],hi[0],hi[1],hi[2],dtremain);
        }
        if (DIM == 2) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],xnew[0],xnew[1],
                   icell,cells[icell].id,
                   lo[0],lo[1],hi[0],hi[1],dtremain);
        }
        if (DIM == 1) {
          if (ntimestep == MOVE_DEBUG_STEP &&
              (MOVE_DEBUG_ID == particles[i].id ||
               (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
            printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
                   CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
                   me,update->ntimestep,i,particles[i].id,
                   cells[icell].nsurf,
                   x[0],x[1],xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                   icell,cells[icell].id,
                   lo[0],lo[1],hi[0],hi[1],dtremain);
        }
#endif

        // check if particle crosses any cell face
        // frac = fraction of move completed before hitting cell face
        // this section should be as efficient as possible,
        //   since most particles won't do anything else
        // axisymmetric y cell face crossings:
        //   these faces are curved cylindrical shells
        //   axi_horizontal_line() checks for intersection of
        //     straight-line y,z move with circle in y,z
        //   always check move against lower y face
        //     except when particle starts on face and
        //     PEXIT is set (just received) or particle is moving downward in y
        //   only check move against upper y face
        //     if remapped final y position (rnew) is within cell,
        //     or except when particle starts on face and
        //     PEXIT is set (just received) or particle is moving upward in y
        //   unset pflag so not checked again for this particle

        outface = INTERIOR;
        frac = 1.0;

        if (xnew[0] < lo[0]) {
          frac = (lo[0]-x[0]) / (xnew[0]-x[0]);
          outface = XLO;
        } else if (xnew[0] >= hi[0]) {
          frac = (hi[0]-x[0]) / (xnew[0]-x[0]);
          outface = XHI;
        }

        if (DIM != 1) {
          if (xnew[1] < lo[1]) {
            newfrac = (lo[1]-x[1]) / (xnew[1]-x[1]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = YLO;
            }
          } else if (xnew[1] >= hi[1]) {
            newfrac = (hi[1]-x[1]) / (xnew[1]-x[1]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = YHI;
            }
          }
        }

        if (DIM == 1) {
          if (x[1] == lo[1] && (pflag == PEXIT || v[1] < 0.0)) {
            frac = 0.0;
            outface = YLO;
          } else if (Geometry::
                     axi_horizontal_line(dtremain,x,v,lo[1],itmp,tc,tmp)) {
            newfrac = tc/dtremain;
            if (newfrac < frac) {
              frac = newfrac;
              outface = YLO;
            }
          }

          if (x[1] == hi[1] && (pflag == PEXIT || v[1] > 0.0)) {
            frac = 0.0;
            outface = YHI;
          } else {
            rnew = sqrt(xnew[1]*xnew[1] + xnew[2]*xnew[2]);
            if (rnew >= hi[1]) {
              if (Geometry::
                  axi_horizontal_line(dtremain,x,v,hi[1],itmp,tc,tmp)) {
                newfrac = tc/dtremain;
                if (newfrac < frac) {
                  frac = newfrac;
                  outface = YHI;
                }
              }
            }
          }

          pflag = 0;
        }

        if (DIM == 3) {
          if (xnew[2] < lo[2]) {
            newfrac = (lo[2]-x[2]) / (xnew[2]-x[2]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = ZLO;
            }
          } else if (xnew[2] >= hi[2]) {
            newfrac = (hi[2]-x[2]) / (xnew[2]-x[2]);
            if (newfrac < frac) {
              frac = newfrac;
              outface = ZHI;
            }
          }
        }

        //if (iterate == 10) exit(1);
        //iterate++;

#ifdef MOVE_DEBUG
        if (ntimestep == MOVE_DEBUG_STEP &&
            (MOVE_DEBUG_ID == particles[i].id ||
             (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX))) {
          if (outface != INTERIOR)
            printf("  OUTFACE %d out: %d %d, frac %g\n",
                   outface,grid->neigh_decode(nmask,outface),
                   neigh[outface],frac);
          else
            printf("  INTERIOR %d %d\n",outface,INTERIOR);
        }
#endif

        // START of code specific to surfaces

        if (SURF) {

          // skip surf checks if particle flagged as EXITing this cell
          // then unset pflag so not checked again for this particle

          nsurf = cells[icell].nsurf;
          if (pflag == PEXIT) {
            nsurf = 0;
            pflag = 0;
          }
          nscheck_one += nsurf;

          if (nsurf) {

            // particle crosses cell face, reset xnew exactly on face of cell
            // so surface check occurs only for particle path within grid cell
            // xhold = saved xnew so can restore below if no surf collision

            if (outface != INTERIOR) {
              xhold[0] = xnew[0];
              xhold[1] = xnew[1];
              if (DIM != 2) xhold[2] = xnew[2];

              xnew[0] = x[0] + frac*(xnew[0]-x[0]);
              xnew[1] = x[1] + frac*(xnew[1]-x[1]);
              if (DIM != 2) xnew[2] = x[2] + frac*(xnew[2]-x[2]);

              if (outface == XLO) xnew[0] = lo[0];
              else if (outface == XHI) xnew[0] = hi[0];
              else if (outface == YLO) xnew[1] = lo[1];
              else if (outface == YHI) xnew[1] = hi[1];
              else if (outface == ZLO) xnew[2] = lo[2];
              else if (outface == ZHI) xnew[2] = hi[2];
            }

            // for axisymmetric, dtsurf = time that particle stays in cell
            // used as arg to axi_line_intersect()

            if (DIM == 1) {
              if (outface == INTERIOR) dtsurf = dtremain;
              else dtsurf = dtremain * frac;
            }

            // check for collisions with triangles or lines in cell
            // find 1st surface hit via minparam
            // skip collisions with previous surf, but not for axisymmetric
            // not considered collision if 2 params are tied and one INSIDE surf
            // if collision occurs, borisorm collision with surface model
            // reset x,v,xnew,dtremain and continue single particle trajectory

            cflag = 0;
            minparam = 2.0;
            csurfs = cells[icell].csurfs;

            for (m = 0; m < nsurf; m++) {
              isurf = csurfs[m];

              if (DIM > 1) {
                if (isurf == exclude) continue;
              }
              if (DIM == 3) {
                tri = &tris[isurf];
                hitflag = Geometry::
                  line_tri_intersect(x,xnew,tri->p1,tri->p2,tri->p3,
                                     tri->norm,xc,param,side);
              }
              if (DIM == 2) {
                line = &lines[isurf];
                hitflag = Geometry::
                  line_line_intersect(x,xnew,line->p1,line->p2,
                                      line->norm,xc,param,side);
              }
              if (DIM == 1) {
                line = &lines[isurf];
                hitflag = Geometry::
                  axi_line_intersect(dtsurf,x,v,outface,lo,hi,line->p1,line->p2,
                                     line->norm,exclude == isurf,
                                     xc,vc,param,side);
              }

#ifdef MOVE_DEBUG
              if (DIM == 3) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE: %d %d %d %d: "
                         "P1 %g %g %g: P2 %g %g %g: "
                         "T1 %g %g %g: T2 %g %g %g: T3 %g %g %g: "
                         "TN %g %g %g: XC %g %g %g: "
                         "Param %g: Side %d\n",
                         MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                         tri->p1[0],tri->p1[1],tri->p1[2],
                         tri->p2[0],tri->p2[1],tri->p2[2],
                         tri->p3[0],tri->p3[1],tri->p3[2],
                         tri->norm[0],tri->norm[1],tri->norm[2],
                         xc[0],xc[1],xc[2],param,side);
              }
              if (DIM == 2) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE: %d %d %d %d: P1 %g %g: P2 %g %g: "
                         "L1 %g %g: L2 %g %g: LN %g %g: XC %g %g: "
                         "Param %g: Side %d\n",
                         MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],xnew[0],xnew[1],
                         line->p1[0],line->p1[1],line->p2[0],line->p2[1],
                         line->norm[0],line->norm[1],
                         xc[0],xc[1],param,side);
              }
              if (DIM == 1) {
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("SURF COLLIDE %d %ld: %d %d %d %d: P1 %g %g: P2 %g %g: "
                         "L1 %g %g: L2 %g %g: LN %g %g: XC %g %g: "
                         "VC %g %g %g: Param %g: Side %d\n",
                         hitflag,ntimestep,MOVE_DEBUG_INDEX,icell,nsurf,isurf,
                         x[0],x[1],
                         xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                         line->p1[0],line->p1[1],line->p2[0],line->p2[1],
                         line->norm[0],line->norm[1],
                         xc[0],xc[1],vc[0],vc[1],vc[2],param,side);
                double edge1[3],edge2[3],xfinal[3],cross[3];
                MathExtra::sub3(line->p2,line->p1,edge1);
                MathExtra::sub3(x,line->p1,edge2);
                MathExtra::cross3(edge2,edge1,cross);
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    MOVE_DEBUG_ID == particles[i].id)
                  printf("CROSSSTART %g %g %g\n",cross[0],cross[1],cross[2]);
                xfinal[0] = xnew[0];
                xfinal[1] = sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]);
                xfinal[2] = 0.0;
                MathExtra::sub3(xfinal,line->p1,edge2);
                MathExtra::cross3(edge2,edge1,cross);
                if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                    MOVE_DEBUG_ID == particles[i].id)
                  printf("CROSSFINAL %g %g %g\n",cross[0],cross[1],cross[2]);
              }
#endif

              if (hitflag && param < minparam && side == OUTSIDE) {
                cflag = 1;
                minparam = param;
                minside = side;
                minsurf = isurf;
                minxc[0] = xc[0];
                minxc[1] = xc[1];
                if (DIM == 3) minxc[2] = xc[2];
                if (DIM == 1) {
                  minvc[1] = vc[1];
                  minvc[2] = vc[2];
                }
              }

            } // END of for loop over surfs

            // tri/line = surf that particle hit first

            if (cflag) {
              if (DIM == 3) tri = &tris[minsurf];
              if (DIM != 3) line = &lines[minsurf];

              // set x to collision point
              // if axisymmetric, set v to remapped velocity at collision pt

              x[0] = minxc[0];
              x[1] = minxc[1];
              if (DIM == 3) x[2] = minxc[2];
              if (DIM == 1) {
                v[1] = minvc[1];
                v[2] = minvc[2];
              }

              // borisorm surface collision using surface collision model
              // surface chemistry may destroy particle or create new one
              // must update particle's icell to current icell so that
              //   if jpart is created, it will be added to correct cell
              // if jpart, add new particle to this iteration via pstop++
              // tally surface statistics if requested using iorig

              ipart = &particles[i];
              ipart->icell = icell;
              dtremain *= 1.0 - minparam*frac;

              if (nsurf_tally)
                memcpy(&iorig,&particles[i],sizeof(Particle::OnePart));

              if (DIM == 3)
                jpart = surf->sc[tri->isc]->
                  collide(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction);
              if (DIM != 3)
                jpart = surf->sc[line->isc]->
                  collide(ipart,dtremain,minsurf,line->norm,line->isr,reaction);

              if (jpart) {
                particles = particle->particles;
                x = particles[i].x;
                v = particles[i].v;
                jpart->flag = PSURF + 1 + minsurf;
                jpart->dtremain = dtremain;
                jpart->weight = particles[i].weight;
                pstop++;
              }

              if (nsurf_tally)
                for (m = 0; m < nsurf_tally; m++)
                      slist_active[m]->surf_tally(dtremain,minsurf,icell,reaction,
                                                                    &iorig,ipart,jpart);

              // stuck_iterate = consecutive iterations particle is immobile

              if (minparam == 0.0) stuck_iterate++;
              else stuck_iterate = 0;

              // reset post-bounce xnew

              xnew[0] = x[0] + dtremain*v[0];
              xnew[1] = x[1] + dtremain*v[1];
              if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];

              exclude = minsurf;
              nscollide_one++;

#ifdef MOVE_DEBUG
              if (DIM == 3) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g %g: %g %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                         minparam,frac,dtremain);
              }
              if (DIM == 2) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g: %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],xnew[0],xnew[1],
                         minparam,frac,dtremain);
              }
              if (DIM == 1) {
                if (ntimestep == MOVE_DEBUG_STEP &&
                    (MOVE_DEBUG_ID == particles[i].id ||
                     (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
                  printf("POST COLLISION %d: %g %g: %g %g: vel %g %g %g: %g %g %g\n",
                         MOVE_DEBUG_INDEX,
                         x[0],x[1],
                         xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
                         v[0],v[1],v[2],
                         minparam,frac,dtremain);
              }
#endif

              // if ipart = NULL, particle discarded due to surface chem
              // else if particle not stuck, continue advection while loop
              // if stuck, mark for DISCARD, and drop out of SURF code

              if (ipart == NULL) particles[i].flag = PDISCARD;
              else if (stuck_iterate < MAXSTUCK) continue;
              else {
                particles[i].flag = PDISCARD;
                nstuck++;
              }

            } // END of cflag if section that borisormed collision

            // no collision, so restore saved xnew if changed it above

            if (outface != INTERIOR) {
              xnew[0] = xhold[0];
              xnew[1] = xhold[1];
              if (DIM != 2) xnew[2] = xhold[2];
            }

          } // END of if test for any surfs in this cell
        } // END of code specific to surfaces

        // break from advection loop if discarding particle

        if (particles[i].flag == PDISCARD) break;

        // no cell crossing and no surface collision
        // set final particle position to xnew, then break from advection loop
        // for axisymmetry, must first remap linear xnew and v
        // for axisymmetry, check if final particle position is within cell
        //   can be rare epsilon round-off cases where particle ends up outside
        //     of final cell curved surf when move logic thinks it is inside
        //   example is when Geom::axi_horizontal_line() says no crossing of cell edge
        //     but axi_remap() puts particle outside the cell
        //   in this case, just DISCARD particle and tally it to naxibad
        // if migrating to another proc,
        //   flag as PDONE so new proc won't move it more on this step

        if (outface == INTERIOR) {
          if (DIM == 1) axi_remap(xnew,v);
          x[0] = xnew[0];
          x[1] = xnew[1];
          if (DIM == 3) x[2] = xnew[2];
          if (DIM == 1) {
            if (x[1] < lo[1] || x[1] > hi[1]) {
              particles[i].flag = PDISCARD;
              naxibad++;
              break;
            }
          }
          if (cells[icell].proc != me) particles[i].flag = PDONE;
          break;
        }

        // particle crosses cell face
        // decrement dtremain in case particle is passed to another proc
        // for axisymmetry, must then remap linear x and v
        // reset particle x to be exactly on cell face
        // for axisymmetry, must reset xnew for next iteration since v changed

        dtremain *= 1.0-frac;
        exclude = -1;

        x[0] += frac * (xnew[0]-x[0]);
        x[1] += frac * (xnew[1]-x[1]);
        if (DIM != 2) x[2] += frac * (xnew[2]-x[2]);
        if (DIM == 1) axi_remap(x,v);

        if (outface == XLO) x[0] = lo[0];
        else if (outface == XHI) x[0] = hi[0];
        else if (outface == YLO) x[1] = lo[1];
        else if (outface == YHI) x[1] = hi[1];
        else if (outface == ZLO) x[2] = lo[2];
        else if (outface == ZHI) x[2] = hi[2];

        if (DIM == 1) {
          xnew[0] = x[0] + dtremain*v[0];
          xnew[1] = x[1] + dtremain*v[1];
          xnew[2] = x[2] + dtremain*v[2];
        }

        // nflag = type of neighbor cell: child, parent, unknown, boundary
        // if parent, use id_find_child to identify child cell
        //   result can be -1 for unknown cell, occurs when:
        //   (a) particle hits face of ghost child cell
        //   (b) the ghost cell extends beyond ghost halo
        //   (c) cell on other side of face is a parent
        //   (d) its child, which the particle is in, is entirely beyond my halo
        // if new cell is child and surfs exist, check if a split cell

        nflag = grid->neigh_decode(nmask,outface);
        icell_original = icell;

        if (nflag == NCHILD) {
          icell = neigh[outface];
          if (DIM == 3 && SURF) {
            if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
              icell = split3d(icell,x);
          }
          if (DIM < 3 && SURF) {
            if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
              icell = split2d(icell,x);
          }
        } else if (nflag == NPARENT) {
          pcell = &pcells[neigh[outface]];
          icell = grid->id_find_child(pcell->id,cells[icell].level,
                                      pcell->lo,pcell->hi,x);
          if (icell >= 0) {
            if (DIM == 3 && SURF) {
              if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                icell = split3d(icell,x);
            }
            if (DIM < 3 && SURF) {
              if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                icell = split2d(icell,x);
            }
          }
        } else if (nflag == NUNKNOWN) icell = -1;

        // neighbor cell is global boundary
        // tally boundary stats if requested using iorig
        // collide() updates x,v,xnew as needed due to boundary interaction
        //   may also update dtremain (piston BC)
        // for axisymmetric, must recalculate xnew since v may have changed
        // surface chemistry may destroy particle or create new one
        // if jpart, add new particle to this iteration via pstop++
        // OUTFLOW: exit with particle flag = PDISCARD
        // PERIODIC: new cell via same logic as above for child/parent/unknown
        // OTHER: reflected particle stays in same grid cell

        else {
          ipart = &particles[i];

          if (nboundary_tally)
            memcpy(&iorig,&particles[i],sizeof(Particle::OnePart));

          bflag = domain->collide(ipart,outface,icell,xnew,dtremain,
                                  jpart,reaction);

          if (jpart) {
            particles = particle->particles;
            x = particles[i].x;
            v = particles[i].v;
          }

          if (nboundary_tally)
            for (m = 0; m < nboundary_tally; m++)
              blist_active[m]->
                // boundary_tally(outface,bflag,reaction,&iorig,ipart,jpart);
                boundary_tally(dtremain,outface,bflag,reaction,&iorig,ipart,jpart);


          if (DIM == 1) {
            xnew[0] = x[0] + dtremain*v[0];
            xnew[1] = x[1] + dtremain*v[1];
            xnew[2] = x[2] + dtremain*v[2];
          }

          if (bflag == OUTFLOW) {
            particles[i].flag = PDISCARD;
            nexit_one++;
            break;

          } else if (bflag == PERIODIC) {
            if (nflag == NPBCHILD) {
              icell = neigh[outface];
              if (DIM == 3 && SURF) {
                if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                  icell = split3d(icell,x);
              }
              if (DIM < 3 && SURF) {
                if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                  icell = split2d(icell,x);
              }
            } else if (nflag == NPBPARENT) {
              pcell = &pcells[neigh[outface]];
              icell = grid->id_find_child(pcell->id,cells[icell].level,
                                          pcell->lo,pcell->hi,x);
              if (icell >= 0) {
                if (DIM == 3 && SURF) {
                  if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                    icell = split3d(icell,x);
                }
                if (DIM < 3 && SURF) {
                  if (cells[icell].nsplit > 1 && cells[icell].nsurf >= 0)
                    icell = split2d(icell,x);
                }
              } else domain->uncollide(outface,x);
            } else if (nflag == NPBUNKNOWN) {
              icell = -1;
              domain->uncollide(outface,x);
            }

          } else if (bflag == SURFACE) {
            if (ipart == NULL) {
              particles[i].flag = PDISCARD;
              break;
            } else if (jpart) {
              jpart->flag = PSURF;
              jpart->dtremain = dtremain;
              jpart->weight = particles[i].weight;
              pstop++;
            }
            nboundary_one++;
            ntouch_one--;    // decrement here since will increment below

          } else {
            nboundary_one++;
            ntouch_one--;    // decrement here since will increment below
          }
        }

        // neighbor cell is unknown
        // reset icell to original icell which must be a ghost cell
        // exit with particle flag = PEXIT, so receiver can identify neighbor

        if (icell < 0) {
          icell = icell_original;
          particles[i].flag = PEXIT;
          particles[i].dtremain = dtremain;
          entryexit = 1;
          break;
        }

        // if nsurf < 0, new cell is EMPTY ghost
        // exit with particle flag = PENTRY, so receiver can continue move

        if (cells[icell].nsurf < 0) {
          particles[i].flag = PENTRY;
          particles[i].dtremain = dtremain;
          entryexit = 1;
          break;
        }

        // move particle into new grid cell for next stage of move

        lo = cells[icell].lo;
        hi = cells[icell].hi;
        neigh = cells[icell].neigh;
        nmask = cells[icell].nmask;
        ntouch_one++;
      }

      // END of while loop over advection of single particle

#ifdef MOVE_DEBUG
      if (ntimestep == MOVE_DEBUG_STEP &&
          (MOVE_DEBUG_ID == particles[i].id ||
           (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
        printf("MOVE DONE %d %d %d: %g %g %g: DTR %g\n",
               MOVE_DEBUG_INDEX,particles[i].flag,icell,
               x[0],x[1],x[2],dtremain);
#endif

      // move is complete, or as much as can be done on this proc
      // update particle's grid cell
      // if particle flag set, add particle to migrate list
      // if discarding, migration will delete particle

      particles[i].icell = icell;

      if (particles[i].flag != PKEEP) {
        mlist[nmigrate++] = i;
        if (particles[i].flag != PDISCARD) {
          if (cells[icell].proc == me) {
            char str[128];
            sprintf(str,
                    "Particle %d on proc %d being sent to self "
                    "on step " BIGINT_FORMAT,
                    i,me,update->ntimestep);
            error->one(FLERR,str);
          }
          ncomm_one++;
        }
      }
    }

    // END of pstart/pstop loop advecting all particles

    // if gridcut >= 0.0, check if another iteration of move is required
    // only the case if some particle flag = PENTRY/PEXIT
    //   in which case pusher_Bororm particle migration
    // if not, move is done and final particle comm will occur in run()
    // if iterating, reset pstart/pstop and extend migration list if necessary

    if (grid->cutoff < 0.0) break;

    timer->stamp(TIME_MOVE);
    MPI_Allreduce(&entryexit,&any_entryexit,1,MPI_INT,MPI_MAX,world);
    timer->stamp();

    if (any_entryexit) {
      timer->stamp(TIME_MOVE);
      pstart = comm->migrate_particles(nmigrate,mlist);
      timer->stamp(TIME_COMM);
      pstop = particle->nlocal;
      if (pstop-pstart > maxmigrate) {
        maxmigrate = pstop-pstart;
        memory->destroy(mlist);
        memory->create(mlist,maxmigrate,"particle:mlist");
      }
    } else break;

    // END of single move/migrate iteration

  }

  // END of all move/migrate iterations

  particle->sorted = 0;

  // accumulate running totals

  niterate_running += niterate;
  nmove_running += particle->nlocal;
  ntouch_running += ntouch_one;
  ncomm_running += ncomm_one;
  nboundary_running += nboundary_one;
  nexit_running += nexit_one;
  nscheck_running += nscheck_one;
  nscollide_running += nscollide_one;
  surf->nreact_running += surf->nreact_one;
}

/* ----------------------------------------------------------------------
   calculate motion perturbation for a single particle I
     due to external per particle field
   array in fix[ifieldfix] stores per particle perturbations for x and v
------------------------------------------------------------------------- */

void Update::field_per_particle(int i, int icell, double dt, double* x, double* v)
{
    // Grab per-particle field row, accounting for packed active components
    double** arr = modify->fix[ifieldfix]->array_particle;

    // Map columns according to field_active[]
    int col = 0;
    double E[3] = {0,0,0};
    double B[3] = {0,0,0};

    // If your layout really is Bx,By,Bz in 0..2 and E is zero, keep:
    B[0] = arr[i][0]; B[1] = arr[i][1]; B[2] = arr[i][2];
    // E remains zeros unless you actually store it.

    // Species & charge/mass
    Particle::OnePart* parts  = particle->particles;
    Particle::Species* specs  = particle->species;
    const int is = parts[i].ispecies;
    const double m  = specs[is].mass;
    const double q  = specs[is].charge * update->echarge;   // Z*e in Coulombs
    const double qom = q / m;

    const double half_dt = 0.5 * dt;

    // 1) Half E-kick
    double v_minus[3] = {
        v[0] + qom * E[0] * half_dt,
        v[1] + qom * E[1] * half_dt,
        v[2] + qom * E[2] * half_dt
    };

    // 2) B-rotation (standard Boris)
    double t[3] = { qom * B[0] * half_dt, qom * B[1] * half_dt, qom * B[2] * half_dt };
    double t2 = MathExtra::dot3(t, t);
    double s[3] = { 2.0 * t[0] / (1.0 + t2), 2.0 * t[1] / (1.0 + t2), 2.0 * t[2] / (1.0 + t2) };

    double v_prime[3];
    MathExtra::cross3(v_minus, t, v_prime);
    v_prime[0] += v_minus[0];
    v_prime[1] += v_minus[1];
    v_prime[2] += v_minus[2];

    double v_plus[3];
    MathExtra::cross3(v_prime, s, v_plus);
    v_plus[0] += v_minus[0];
    v_plus[1] += v_minus[1];
    v_plus[2] += v_minus[2];

    // 3) Second half E-kick → v_new
    double v_new[3] = {
        v_plus[0] + qom * E[0] * half_dt,
        v_plus[1] + qom * E[1] * half_dt,
        v_plus[2] + qom * E[2] * half_dt
    };

    // 4) Position update with v_new
    x[0] += v_new[0] * dt;
    x[1] += v_new[1] * dt;
    x[2] += v_new[2] * dt;

    // Write back velocity
    v[0] = v_new[0];
    v[1] = v_new[1];
    v[2] = v_new[2];
}




// /* ----------------------------------------------------------------------
//    calculate motion perturbation for a single particle I
//      due to external per particle field
//    array in fix[ifieldfix] stores per particle perturbations for x and v
// ------------------------------------------------------------------------- */
// void Update::field_per_particle(int i, int icell, double dt, double *x, double *v)
// {
//   Particle::Species *species = particle->species;
//   Particle::OnePart *particles = particle->particles;
//   Grid::ChildInfo *cinfo = grid->cinfo;
//   double **array = modify->fix[ifieldfix]->array_particle;

//   const PlasmaDataParams &plasma_data = plasma_data_map[icell];

//   // Physical parameters from plasma data
//   double temp_i = std::max(0.0, plasma_data.temp_i);
//   double temp_e = plasma_data.temp_e;
//   double dens_i = std::max(0.0, plasma_data.dens_i);
//   double dens_e = plasma_data.dens_e;
//   double v_parr = plasma_data.parr_flow;

//   double flow_velocity[3] = {
//     plasma_data.parr_flow_r,
//     plasma_data.parr_flow_z,
//     plasma_data.parr_flow_t
//   };

//   // Physical constants
//   // constexpr double Rho = 534.0;                // lithium density [kg/m^3]
//   constexpr double background_mass = 2.0 * 1.67e-27 * 1e3;  // kg
//   constexpr double background_charge = 1.0; // un
//   constexpr double viscosity_scale = 0.8E-2;
//   constexpr double pi = M_PI;

//   // Compute viscosity (if valid inputs)
//   double viscosity = 0.0;
//   if (temp_i > 0.0 && dens_i > 0.0) {
//     double dens_i_cm = dens_i * 1e-6;
//     viscosity = eta(dens_i_cm, background_mass, background_charge, temp_i) * viscosity_scale;
//   }

//   // Compute relative velocity between particle and plasma flow
//   double relative_velocity[3] = {
//     v[0] - flow_velocity[0],
//     v[1] - flow_velocity[1],
//     v[2] - flow_velocity[2]
//   };

//   // Particle properties
//   double droplet_mass = species[i].mass;
//   double radius = particles[i].radius;

//   // Avoid division by zero
//   if (droplet_mass <= 0.0) return;

//   // Stokes drag (in-plane only)
//   double drag_acc_x = 6.0 * pi * viscosity * radius * relative_velocity[0] / droplet_mass;
//   double drag_acc_y = 6.0 * pi * viscosity * radius * relative_velocity[1] / droplet_mass;

//   // Update position and velocity due to field and drag
//   double dtsq = 0.5 * dt * dt;
//   int icol = 0;

//   if (field_active[0]) {
//     x[0] += dtsq * array[i][icol] + dtsq * drag_acc_x;
//     v[0] += dt * array[i][icol] + dt * drag_acc_x;
//     icol++;
//   }
//   if (field_active[1]) {
//     x[1] += dtsq * array[i][icol] + dtsq * drag_acc_y;
//     v[1] += dt * array[i][icol] + dt * drag_acc_y;
//     icol++;
//   }
//   if (field_active[2]) {
//     x[2] += dtsq * array[i][icol];
//     v[2] += dt * array[i][icol];
//     icol++;
//   }
// }


/* ----------------------------------------------------------------------
   calculate motion perturbation for a single particle I in grid cell Icell
     due to external per grid cell field
   array in fix[ifieldfix] stores per grid cell perturbations for x and v
------------------------------------------------------------------------- */

void Update::field_per_grid(int i, int icell, double dt, double *x, double *v)
{
  double dtsq = 0.5*dt*dt;
  double **array = modify->fix[ifieldfix]->array_grid;

  int icol = 0;
  if (field_active[0]) {
    x[0] += dtsq*array[icell][icol];
    v[0] += dt*array[icell][icol];
    icol++;
  }
  if (field_active[1]) {
    x[1] += dtsq*array[icell][icol];
    v[1] += dt*array[icell][icol];
    icol++;
  }
  if (field_active[2]) {
    x[2] += dtsq*array[icell][icol];
    v[2] += dt*array[icell][icol];
    icol++;
  }
};

/* ----------------------------------------------------------------------
   particle is entering split parent icell at x
   determine which split child cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

int Update::split3d(int icell, double *x)
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Tri *tri;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Surf::Tri *tris = surf->tris;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider tris that are mapped via csplits to a split cell
  //   unmapped tris only touch cell surf at xnew
  //   another mapped tri should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particles starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = cells[icell].nsurf;
  surfint *csurfs = cells[icell].csurfs;
  int isplit = cells[icell].isplit;
  int *csplits = sinfo[isplit].csplits;
  double *xnew = sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;

  for (m = 0; m < nsurf; m++) {
    if (csplits[m] < 0) continue;
    isurf = csurfs[m];
    tri = &tris[isurf];
    hitflag = Geometry::
      line_tri_intersect(x,xnew,tri->p1,tri->p2,tri->p3,
                         tri->norm,xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  if (!cflag) return sinfo[isplit].csubs[sinfo[isplit].xsub];
  int index = csplits[minsurfindex];
  return sinfo[isplit].csubs[index];
}

/* ----------------------------------------------------------------------
   particle is entering split ICELL at X
   determine which split sub-cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

int Update::split2d(int icell, double *x)
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Line *line;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Surf::Line *lines = surf->lines;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider lines that are mapped via csplits to a split cell
  //   unmapped lines only touch cell surf at xnew
  //   another mapped line should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particle starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = cells[icell].nsurf;
  surfint *csurfs = cells[icell].csurfs;
  int isplit = cells[icell].isplit;
  int *csplits = sinfo[isplit].csplits;
  double *xnew = sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;
  for (m = 0; m < nsurf; m++) {
    if (csplits[m] < 0) continue;
    isurf = csurfs[m];
    line = &lines[isurf];
    hitflag = Geometry::
      line_line_intersect(x,xnew,line->p1,line->p2,line->norm,xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  if (!cflag) return sinfo[isplit].csubs[sinfo[isplit].xsub];
  int index = csplits[minsurfindex];
  return sinfo[isplit].csubs[index];
}

/* ----------------------------------------------------------------------
   check if any surface collision or reaction models are defined
   return 1 if there are any, 0 if not
------------------------------------------------------------------------- */

int Update::collide_react_setup()
{
  nsc = surf->nsc;
  sc = surf->sc;
  nsr = surf->nsr;
  sr = surf->sr;

  if (nsc || nsr) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   zero counters for tallying surface collisions/reactions
   done at start of each timestep
   done within individual SurfCollide and SurfReact instances
------------------------------------------------------------------------- */

void Update::collide_react_reset()
{
  for (int i = 0; i < nsc; i++) sc[i]->tally_reset();
  for (int i = 0; i < nsr; i++) sr[i]->tally_reset();
}

/* ----------------------------------------------------------------------
   update cummulative counters for tallying surface collisions/reactions
   done at end of each timestep
   this is done within individual SurfCollide and SurfReact instances
------------------------------------------------------------------------- */

void Update::collide_react_update()
{
  for (int i = 0; i < nsc; i++) sc[i]->tally_update();
  for (int i = 0; i < nsr; i++) sr[i]->tally_update();
}

/* ----------------------------------------------------------------------
   setup lists of all computes that tally surface and boundary bounce info
   return 1 if there are any, 0 if not
------------------------------------------------------------------------- */

int Update::bounce_setup()
{
  delete [] slist_compute;
  delete [] blist_compute;
  delete [] slist_active;
  delete [] blist_active;
  slist_compute = blist_compute = NULL;

  nslist_compute = nblist_compute = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->surf_tally_flag) nslist_compute++;
    if (modify->compute[i]->boundary_tally_flag) nblist_compute++;
  }

  if (nslist_compute) slist_compute = new Compute*[nslist_compute];
  if (nblist_compute) blist_compute = new Compute*[nblist_compute];
  if (nslist_compute) slist_active = new Compute*[nslist_compute];
  if (nblist_compute) blist_active = new Compute*[nblist_compute];

  nslist_compute = nblist_compute = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->surf_tally_flag)
      slist_compute[nslist_compute++] = modify->compute[i];
    if (modify->compute[i]->boundary_tally_flag)
      blist_compute[nblist_compute++] = modify->compute[i];
  }

  if (nslist_compute || nblist_compute) return 1;
  nsurf_tally = nboundary_tally = 0;
  return 0;
}

/* ----------------------------------------------------------------------
   set bounce tally flags for current timestep
   nsurf_tally = # of surface computes needing bounce info on this step
   nboundary_tally = # of boundary computes needing bounce info on this step
   clear accumulators in computes that will be invoked this step
------------------------------------------------------------------------- */

void Update::bounce_set(bigint ntimestep)
{
  int i;

  nsurf_tally = 0;
  if (nslist_compute) {
    for (i = 0; i < nslist_compute; i++)
      if (slist_compute[i]->matchstep(ntimestep)) {
        slist_active[nsurf_tally++] = slist_compute[i];
        slist_compute[i]->clear();
      }
  }

  nboundary_tally = 0;
  if (nblist_compute) {
    for (i = 0; i < nblist_compute; i++)
      if (blist_compute[i]->matchstep(ntimestep)) {
        blist_active[nboundary_tally++] = blist_compute[i];
        blist_compute[i]->clear();
      }
  }
}

/* ----------------------------------------------------------------------
   make list of classes that reset dynamic parameters
   currently only surf collision models
------------------------------------------------------------------------- */

void Update::dynamic_setup()
{
  delete [] ulist_surfcollide;
  ulist_surfcollide = NULL;

  nulist_surfcollide = 0;
  for (int i = 0; i < surf->nsc; i++)
    if (surf->sc[i]->dynamicflag) nulist_surfcollide++;

  if (nulist_surfcollide)
    ulist_surfcollide = new SurfCollide*[nulist_surfcollide];

  nulist_surfcollide = 0;
  for (int i = 0; i < surf->nsc; i++)
    if (surf->sc[i]->dynamicflag)
      ulist_surfcollide[nulist_surfcollide++] = surf->sc[i];

  if (nulist_surfcollide) dynamic = 1;
}

/* ----------------------------------------------------------------------
   invoke class methods that reset dynamic parameters
------------------------------------------------------------------------- */

void Update::dynamic_update()
{
  if (nulist_surfcollide) {
    for (int i = 0; i < nulist_surfcollide; i++)
      ulist_surfcollide[i]->dynamic();
  }
}

/* ----------------------------------------------------------------------
   set global properites via global command in input script
------------------------------------------------------------------------- */

void Update::global(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal global command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"fnum") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      fnum = input->numeric(FLERR,arg[iarg+1]);
      if (fnum <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"optmove") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) optmove_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) optmove_flag = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nrho") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      nrho = input->numeric(FLERR,arg[iarg+1]);
      if (nrho <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"vstream") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal global command");
      vstream[0] = input->numeric(FLERR,arg[iarg+1]);
      vstream[1] = input->numeric(FLERR,arg[iarg+2]);
      vstream[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      temp_thermal = input->numeric(FLERR,arg[iarg+1]);
      if (temp_thermal <= 0.0) error->all(FLERR,"Illegal global command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"field") == 0) {
      if (iarg+1 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"none") == 0) {
        fstyle = NOFIELD;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"constant") == 0) {
        if (iarg+6 > narg) error->all(FLERR,"Illegal global field command");
        fstyle = CFIELD;
        double fmag = input->numeric(FLERR,arg[iarg+2]);
        field[0] = input->numeric(FLERR,arg[iarg+3]);
        field[1] = input->numeric(FLERR,arg[iarg+4]);
        field[2] = input->numeric(FLERR,arg[iarg+5]);
        if (fmag <= 0.0) error->all(FLERR,"Illegal global field command");
        if (field[0] == 0.0 && field[1] == 0.0 && field[2] == 0.0)
          error->all(FLERR,"Illegal global field command");
        MathExtra::snorm3(fmag,field);
        iarg += 6;
      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal global field command");
        delete [] fieldID;
        fstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        fieldID = new char[n];
        strcpy(fieldID,arg[iarg+2]);
        iarg += 3;
      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global field command");
        delete [] fieldID;
        fstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        fieldID = new char[n];
        strcpy(fieldID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (fieldfreq < 0) error->all(FLERR,"Illegal global field command");
        iarg += 4;
      } else error->all(FLERR,"Illegal global field command");

    } 
      // --------------- E field ----------------
    else if (strcmp(arg[iarg],"efield") == 0) {
      if (iarg+1 > narg) error->all(FLERR,"Illegal global command");

      if (strcmp(arg[iarg+1],"none") == 0) {
        efstyle = NOFIELD;
        iarg += 2;

      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        iarg += 3;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);   // <— own freq
        if (fieldfreq < 0) error->all(FLERR,"Illegal global e field command");
        iarg += 4;

      } else error->all(FLERR,"Illegal global e field command");
    }

    // --------------- B field ----------------
    else if (strcmp(arg[iarg],"bfield") == 0) {
      if (iarg+1 > narg) error->all(FLERR,"Illegal global command");

      if (strcmp(arg[iarg+1],"none") == 0) {
        bfstyle = NOFIELD;
        iarg += 2;

      } else if (strcmp(arg[iarg+1],"particle") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);   // <— own freq
        if (fieldfreq < 0) error->all(FLERR,"Illegal global b field command");
        iarg += 4;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);   // <— own freq
        if (fieldfreq < 0) error->all(FLERR,"Illegal global b field command");
        iarg += 4;

      } else error->all(FLERR,"Illegal global b field command");


    }
    // Ethermal field
    else if (strcmp(arg[iarg],"ethermal") == 0) {
      if (iarg+1 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"none") == 0) {
        ethermalstyle = NOFIELD;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global ethermal field command");
        delete [] ethermalID;
        ethermalstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        ethermalID = new char[n];
        strcpy(ethermalID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (fieldfreq < 0) error->all(FLERR,"Illegal global field command");
        iarg += 4;
      } else error->all(FLERR,"Illegal global ethermal field command");

    }
    // Ithermal field
    else if (strcmp(arg[iarg],"ithermal") == 0) {
      if (iarg+1 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"none") == 0) {
        ithermalstyle = NOFIELD;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global ithermal field command");
        delete [] ithermalID;
        ithermalstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        ithermalID = new char[n];
        strcpy(ithermalID,arg[iarg+2]);
        fieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (fieldfreq < 0) error->all(FLERR,"Illegal global field command");
        iarg += 4;
      } else error->all(FLERR,"Illegal global ithermal field command");

    }

    else if (strcmp(arg[iarg],"surfs") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      surf->global(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"surfgrid") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global surfgrid when surfaces already exist");
      if (strcmp(arg[iarg+1],"auto") == 0) grid->surfgrid_algorithm = PERAUTO;
      else if (strcmp(arg[iarg+1],"percell") == 0)
        grid->surfgrid_algorithm = PERCELL;
      else if (strcmp(arg[iarg+1],"persurf") == 0)
        grid->surfgrid_algorithm = PERSURF;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"surfmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global surfmax when surfaces already exist");
      grid->maxsurfpercell = atoi(arg[iarg+1]);
      if (grid->maxsurfpercell <= 0) error->all(FLERR,"Illegal global command");
      // reallocate paged data structs for variable-length surf info
      grid->allocate_surf_arrays();
      iarg += 2;
    } else if (strcmp(arg[iarg],"splitmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (surf->exist)
        error->all(FLERR,
                   "Cannot set global splitmax when surfaces already exist");
      grid->maxsplitpercell = atoi(arg[iarg+1]);
      if (grid->maxsplitpercell <= 0) error->all(FLERR,"Illegal global command");
      // reallocate paged data structs for variable-length cell info
      grid->allocate_surf_arrays();
      iarg += 2;
    } else if (strcmp(arg[iarg],"gridcut") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      grid->cutoff = input->numeric(FLERR,arg[iarg+1]);
      if (grid->cutoff < 0.0 && grid->cutoff != -1.0)
        error->all(FLERR,"Illegal global command");
      // force ghost info to be regenerated with new cutoff
      grid->remove_ghosts();
      iarg += 2;
    } else if (strcmp(arg[iarg],"weight") == 0) {
      // for now assume just one arg after "cell"
      // may need to generalize later
      if (iarg+3 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"cell") == 0) grid->weight(1,&arg[iarg+2]);
      else error->all(FLERR,"Illegal weight command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"comm/sort") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) comm->commsortflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) comm->commsortflag = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/style") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"neigh") == 0) comm->commpartstyle = 1;
      else if (strcmp(arg[iarg+1],"all") == 0) comm->commpartstyle = 0;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"surftally") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"auto") == 0) surf->tally_comm = TALLYAUTO;
      else if (strcmp(arg[iarg+1],"reduce") == 0) surf->tally_comm = TALLYREDUCE;
      else if (strcmp(arg[iarg+1],"rvous") == 0) surf->tally_comm = TALLYRVOUS;
      else error->all(FLERR,"Illegal global command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"particle/reorder") == 0) {
      reorder_period = input->inumeric(FLERR,arg[iarg+1]);
      if (reorder_period < 0) error->all(FLERR,"Illegal global command");
      iarg += 2;
    } 
    // PMI

     // target material take mass and charge
    else if (strcmp(arg[iarg], "target_material") == 0) {
        if (iarg + 5 > narg) error->all(FLERR, "Illegal global command");
        int n = strlen(arg[iarg+1]) + 1;
        target_material = new char[n];
        strcpy(target_material, arg[iarg+1]);
        iarg += 2;

        // Parse mass
        if (strcmp(arg[iarg], "mass") != 0) 
            error->all(FLERR, "Expected 'mass' in global command");
        double target_material_mass = atof(arg[iarg+1]);
        iarg += 2;

        // Parse charge
        if (strcmp(arg[iarg], "charge") != 0) 
            error->all(FLERR, "Expected 'charge' in global command");
        double target_material_charge = atof(arg[iarg+1]);
        iarg += 2;

        // parse target_material_binding_energy
        if (strcmp(arg[iarg], "binding_energy") != 0) 
            error->all(FLERR, "Expected 'binding_energy' in global command");
        double target_material_binding_energy = atof(arg[iarg+1]);
        iarg += 2;
    }

    else if (strcmp(arg[iarg], "plasma_background_material") == 0) {
            if (iarg + 6 > narg) error->all(FLERR, "Illegal global command");
            // Read the material name
            const char* material_name = arg[iarg + 1];
            iarg += 2;
            if (strcmp(arg[iarg], "mass") != 0)
                error->all(FLERR, "Expected 'mass' in global command");
            plasma_background_mass = atof(arg[iarg + 1]);
            iarg += 2;
            if (strcmp(arg[iarg], "charge") != 0)
                error->all(FLERR, "Expected 'charge' in global command");
            plasma_background_charge = atof(arg[iarg + 1]);
            iarg += 2;
        }

    else if (strcmp(arg[iarg], "target_material") == 0) {
        if (iarg + 5 > narg) error->all(FLERR, "Illegal global command");
        int n = strlen(arg[iarg+1]) + 1;
        target_material = new char[n];
        strcpy(target_material, arg[iarg+1]);
        iarg += 2;

        // Parse mass
        if (strcmp(arg[iarg], "mass") != 0) 
            error->all(FLERR, "Expected 'mass' in global command");
         target_material_mass = atof(arg[iarg+1]);
        iarg += 2;

        // Parse charge
        if (strcmp(arg[iarg], "charge") != 0) 
            error->all(FLERR, "Expected 'charge' in global command");
         target_material_charge = atof(arg[iarg+1]);
        iarg += 2;

        // parse target_material_binding_energy
        if (strcmp(arg[iarg], "binding_energy") != 0) 
            error->all(FLERR, "Expected 'binding_energy' in global command");
         target_material_binding_energy = atof(arg[iarg+1]);
        iarg += 2;
    }

        else if (strcmp(arg[iarg], "cross_field_diffusion") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global cross_diffusion_flag command");
      
      if (strcmp(arg[iarg + 1], "yes") == 0) {
          cross_diffusion_flag = 1;
      } else if (strcmp(arg[iarg + 1], "no") == 0) {
          cross_diffusion_flag = 0;
      } else {
          error->all(FLERR, "Illegal global cross_diffusion_flag command, expected 'yes' or 'no'");
      }
      
      iarg += 2;  
    }

    else if (strcmp(arg[iarg], "thermal_gradient_forces") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global thermal_gradient_forces_flag command");
      
      if (strcmp(arg[iarg + 1], "yes") == 0) {
          thermal_gradient_forces_flag = 1;
      } else if (strcmp(arg[iarg + 1], "no") == 0) {
          thermal_gradient_forces_flag = 0;
      } else {
          error->all(FLERR, "Illegal global thermal_gradient_forces_flag command, expected 'yes' or 'no'");
      }
      
      iarg += 2;  
  }


    else if (strcmp(arg[iarg], "sheath") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
      
      if (strcmp(arg[iarg + 1], "yes") == 0) {
          sheath_field_flag = 1;
      } else if (strcmp(arg[iarg + 1], "no") == 0) {
          sheath_field_flag = 0;
      } else {
          error->all(FLERR, "Illegal global sheath command, expected 'yes' or 'no'");
      }
      
      iarg += 2;  // Move past "sheath" and "yes"/"no"
  }
    else if (strcmp(arg[iarg], "ionization") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal global command");
      if (strcmp(arg[iarg + 1], "yes") == 0) ionization_flag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) ionization_flag = 0;
      else error->all(FLERR, "Illegal global command");
      iarg += 2;
  } else if (strcmp(arg[iarg], "recombination") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal global command");
      if (strcmp(arg[iarg + 1], "yes") == 0) recombination_flag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) recombination_flag = 0;
      else error->all(FLERR, "Illegal global command");
      iarg += 2;
  } else if (strcmp(arg[iarg], "adas_rates_path") == 0) {
    if (iarg + 2 > narg) error->all(FLERR, "Illegal global command");
    adas_rates_path = arg[iarg + 1]; // Store the path
    iarg += 2;
  } else if (strcmp(arg[iarg], "cross_field_diffusion") == 0) {
    if (iarg + 2 > narg) error->all(FLERR, "Illegal global command");
    if (strcmp(arg[iarg + 1], "yes") == 0) {
        cross_field_diffusion_flag = 1;
        iarg += 2;
        if (iarg + 2 > narg) error->all(FLERR, "Illegal global command");
        if (strcmp(arg[iarg], "d_perp") != 0)
            error->all(FLERR, "Expected 'd_perp' in global command");
        d_perp = atof(arg[iarg + 1]);
        iarg += 2;
    } else if (strcmp(arg[iarg + 1], "no") == 0) {
        cross_field_diffusion_flag = 0;
        iarg += 2;
    } else {
        error->all(FLERR, "Illegal global command");
    }
}


else if (strcmp(arg[iarg], "background_collisions") == 0) {
    if (iarg + 1 >= narg) error->all(FLERR, "Illegal global background_collisions command");
    if (strcmp(arg[iarg + 1], "yes") == 0) {
        background_collision_flag = 1;
    } else if (strcmp(arg[iarg + 1], "no") == 0) {
        background_collision_flag = 0;
    } else {
        error->all(FLERR, "Illegal global background_collisions command, expected 'yes' or 'no'");
    }
    iarg += 2;
}

  else if (strcmp(arg[iarg], "materials") == 0) {
    iarg++; // Move to the first material atomic number
    // Clear existing materials list if any
    materials.clear();
    // Continue reading and storing material atomic numbers until the end of arguments
    while (iarg < narg && strcmp(arg[iarg], "materials") != 0) {
        int atomic_number = atoi(arg[iarg]); // Convert string to integer
        if (atomic_number <= 0) {
            error->all(FLERR, "Invalid atomic number in materials command");
        }
        materials.push_back(atomic_number);
        iarg++; // Move to the next argument
      }
    }else if (strcmp(arg[iarg],"mem/limit") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"grid") == 0) mem_limit_grid_flag = 1;
      else {
        double factor = input->numeric(FLERR,arg[iarg+1]);
        bigint global_mem_limit_big = static_cast<bigint> (factor * 1024*1024);
        if (global_mem_limit_big < 0) error->all(FLERR,"Illegal global command");
        if (global_mem_limit_big > MAXSMALLINT)
          error->all(FLERR,"Global mem/limit setting cannot exceed 2GB");
        global_mem_limit = global_mem_limit_big;
      }
      iarg += 2;
    } else error->all(FLERR,"Illegal global command");
  }
}

/* ----------------------------------------------------------------------
   reset timestep as called from input script
------------------------------------------------------------------------- */

void Update::reset_timestep(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal reset_timestep command");
  bigint newstep = ATOBIGINT(arg[0]);
  reset_timestep(newstep);
}

/* ----------------------------------------------------------------------
   reset timestep
   set atimestep to new timestep, so future update_time() calls will be correct
   trigger reset of timestep for output and for fixes that require it
   do not allow any timestep-dependent fixes to be defined
   reset eflag/vflag global so nothing will think eng/virial are current
   reset invoked flags of computes,
     so nothing will think they are current between runs
   clear timestep list of computes that store future invocation times
   called from rerun command and input script (indirectly)
------------------------------------------------------------------------- */

void Update::reset_timestep(bigint newstep)
{
  ntimestep = newstep;
  if (ntimestep < 0) error->all(FLERR,"Timestep must be >= 0");
  if (ntimestep > MAXBIGINT) error->all(FLERR,"Too big a timestep");

  output->reset_timestep(ntimestep);

  for (int i = 0; i < modify->nfix; i++) {
    if (modify->fix[i]->time_depend)
      error->all(FLERR,
                 "Cannot reset timestep with a time-dependent fix defined");
    //modify->fix[i]->reset_timestep(ntimestep);
  }

  for (int i = 0; i < modify->ncompute; i++) {
    modify->compute[i]->invoked_scalar = -1;
    modify->compute[i]->invoked_vector = -1;
    modify->compute[i]->invoked_array = -1;
    modify->compute[i]->invoked_per_particle = -1;
    modify->compute[i]->invoked_per_grid = -1;
    modify->compute[i]->invoked_per_surf = -1;
  }

  for (int i = 0; i < modify->ncompute; i++)
    if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();
}

/* ----------------------------------------------------------------------
   get mem/limit based on grid memory
------------------------------------------------------------------------- */

void Update::set_mem_limit_grid(int gnlocal)
{
  if (gnlocal == 0) gnlocal = grid->nlocal;

  bigint global_mem_limit_big = static_cast<bigint> (gnlocal*sizeof(Grid::ChildCell));

  if (global_mem_limit_big > MAXSMALLINT)
    error->one(FLERR,"Global mem/limit setting cannot exceed 2GB");

  global_mem_limit = global_mem_limit_big;
}

/* ----------------------------------------------------------------------
   get mem/limit based on grid memory
------------------------------------------------------------------------- */

int Update::have_mem_limit()
{
  if (mem_limit_grid_flag)
    set_mem_limit_grid();

  int mem_limit_flag = 0;

  if (global_mem_limit > 0 || (mem_limit_grid_flag && !grid->nlocal))
    mem_limit_flag = 1;

  return mem_limit_flag;
}

// END of SPARTA

// Begin PMI

/* ----------------------------------------------------------------------
   set global properites via read_plasma_state
------------------------------------------------------------------------- */
void Update::read_plasma_state(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal read_plasma_state command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "plasma_state") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal plasma_background command");
      
      if (strcmp(arg[iarg+1], "constant") == 0) {
        plasmaStyle = 0;
        printf("Setting constant plasma background\n");
        iarg += 2; // Move past 'plasma_background constant'
        while (iarg < narg && strcmp(arg[iarg], "file") != 0) {
          if (strcmp(arg[iarg], "temp_i") == 0) {
            if (iarg+1 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            temp_i = input->numeric(FLERR, arg[iarg+1]);
            temp_thermal = temp_i * ev2kelvin;
            iarg += 2;
          } else if (strcmp(arg[iarg], "temp_e") == 0) {
            if (iarg+1 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            temp_e = input->numeric(FLERR, arg[iarg+1]);
            iarg += 2;
          } else if (strcmp(arg[iarg], "dens_i") == 0) {
            if (iarg+1 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            dens_i = input->numeric(FLERR, arg[iarg+1]);
            nrho = dens_i;
            iarg += 2;
          } else if (strcmp(arg[iarg], "dens_e") == 0) {
            if (iarg+1 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            dens_e = input->numeric(FLERR, arg[iarg+1]);
            iarg += 2;
          } else if (strcmp(arg[iarg], "flow") == 0) {
            if (iarg+3 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            flow_v[0] = input->numeric(FLERR, arg[iarg+1]);
            flow_v[1] = input->numeric(FLERR, arg[iarg+2]);
            flow_v[2] = input->numeric(FLERR, arg[iarg+3]);
            vstream[0] = flow_v[0];
            vstream[1] = flow_v[1];
            vstream[2] = flow_v[2];
            iarg += 4;
          } else if (strcmp(arg[iarg], "grad_te") == 0) {
            if (iarg+3 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            grad_te_r = input->numeric(FLERR, arg[iarg+1]);
            grad_te_t = input->numeric(FLERR, arg[iarg+2]);
            grad_te_z = input->numeric(FLERR, arg[iarg+3]);
            iarg += 4;
          } else if (strcmp(arg[iarg], "grad_ti") == 0) {
            if (iarg+3 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            grad_ti_r = input->numeric(FLERR, arg[iarg+1]);
            grad_ti_t = input->numeric(FLERR, arg[iarg+2]);
            grad_ti_z = input->numeric(FLERR, arg[iarg+3]);
            iarg += 4;
          }  else if (strcmp(arg[iarg], "efield") == 0) {
            if (iarg+3 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
            efield[0] = input->numeric(FLERR, arg[iarg+1]);
            efield[1] = input->numeric(FLERR, arg[iarg+2]);
            efield[2] = input->numeric(FLERR, arg[iarg+3]);
            iarg += 4;
          } else {
            error->all(FLERR, "Unrecognized parameter in plasma_background constant");
          }
        }
      } else if (strcmp(arg[iarg+1], "file") == 0) {
        plasmaStyle = 1;
        if (iarg+2 >= narg) error->all(FLERR, "Illegal read_plasma_state command");
        plasmaStatePath = arg[iarg+2];
        iarg += 3;  
      } else {
        error->all(FLERR, "Illegal read_plasma_state command");
      }
    } else {
      error->all(FLERR, "Illegal read_plasma_state command");
    }
  }
}


/*---------------------------------
  Bilinear interpolation plasma
-----------------------------------*/
MagneticFieldDataParams Update::bilinearInterpolationMagneticField(int icell, const MagneticFieldData& data) {
   // Check if the result is already cached
   auto cache_it = magneticFieldDataCache.find(icell);
   if (cache_it != magneticFieldDataCache.end()) {
       return cache_it->second;
   }

   if (data.r.empty() || data.z.empty()) {
       printf("Data arrays are empty.\n");
       throw std::runtime_error("Data arrays are empty.");
   }

   const std::vector<double>& r_values = data.r;
   const std::vector<double>& z_values = data.z;

   // Access cell and calculate midpoints
   Grid::ChildCell* cell = &grid->cells[icell];
   double r_val = 0.5 * (cell->lo[0] + cell->hi[0]);
   double z_val = 0.5 * (cell->lo[1] + cell->hi[1]);

   // Ensure r and z are within the data bounds
   if (r_val < r_values.front() || r_val > r_values.back() ||
       z_val < z_values.front() || z_val > z_values.back()) {
       // printf("Interpolation point (r_val: %f, z_val: %f) is outside the bounds of the data grid.\n", r_val, z_val);
       MagneticFieldDataParams params = {};
       return params;
   }


    // Locate indices for surrounding grid points
    auto r_it = std::lower_bound(r_values.begin(), r_values.end(), r_val);
    auto z_it = std::lower_bound(z_values.begin(), z_values.end(), z_val);

    int r1_idx = std::max(0, int(r_it - r_values.begin()) - 1);
    int r2_idx = std::min(int(r_values.size()) - 1, r1_idx + 1);
    int z1_idx = std::max(0, int(z_it - z_values.begin()) - 1);
    int z2_idx = std::min(int(z_values.size()) - 1, z1_idx + 1);

    // Get surrounding grid values
    double r1 = r_values[r1_idx];
    double r2 = r_values[r2_idx];
    double z1 = z_values[z1_idx];
    double z2 = z_values[z2_idx];

    // Lambda for bilinear interpolation
    auto bilinearInterpolation = [&](const std::vector<std::vector<double>>& field) -> double {
        double Q11 = field[z1_idx][r1_idx];
        double Q12 = field[z2_idx][r1_idx];
        double Q21 = field[z1_idx][r2_idx];
        double Q22 = field[z2_idx][r2_idx];
        double denom = (r2 - r1) * (z2 - z1);

        // Check for zero denominators
        if (denom == 0.0) {
            return (Q11 + Q12 + Q21 + Q22) / 4.0;
        }

        return (Q11 * (r2 - r_val) * (z2 - z_val) +
                Q21 * (r_val - r1) * (z2 - z_val) +
                Q12 * (r2 - r_val) * (z_val - z1) +
                Q22 * (r_val - r1) * (z_val - z1)) / denom;
    };


   // pusher_Bororm bilinear interpolation for each field component
    MagneticFieldDataParams params;
    params.br = bilinearInterpolation(data.br);
    params.bt = bilinearInterpolation(data.bt);
    params.bz = bilinearInterpolation(data.bz);

  
 // Cache the result
    magneticFieldDataCache[icell] = params;

   return params;
}


/*---------------------------------
 Bilinear interpolation
-----------------------------------*/
double Update::bilinearInterpolation(double r, double z,
                            const std::vector<double>& r_values,
                            const std::vector<double>& z_values,
                            const std::vector<std::vector<double>>& data) {
   // Ensure r and z are within the data bounds
   if (r < r_values.front() || r > r_values.back() ||
       z < z_values.front() || z > z_values.back()) {
       // printf("Interpolation point is outside the bounds of the data grid.\n");
       return 0.0;
   }

   // Locate indices for surrounding grid points
   auto r_it = std::lower_bound(r_values.begin(), r_values.end(), r);
   auto z_it = std::lower_bound(z_values.begin(), z_values.end(), z);

   int r1_idx = std::max(0, int(r_it - r_values.begin()) - 1);
   int r2_idx = std::min(int(r_values.size()) - 1, r1_idx + 1);
   int z1_idx = std::max(0, int(z_it - z_values.begin()) - 1);
   int z2_idx = std::min(int(z_values.size()) - 1, z1_idx + 1);

   // Get surrounding grid values
   double r1 = r_values[r1_idx];
   double r2 = r_values[r2_idx];
   double z1 = z_values[z1_idx];
   double z2 = z_values[z2_idx];
   double Q11 = data[z1_idx][r1_idx];
   double Q12 = data[z2_idx][r1_idx];
   double Q21 = data[z1_idx][r2_idx];
   double Q22 = data[z2_idx][r2_idx];

   // Check for zero denominators
   double denom = (r2 - r1) * (z2 - z1);
   if (denom == 0.0) {
       // printf("Degenerate grid detected, returning average of surrounding points.\n");
       return (Q11 + Q12 + Q21 + Q22) / 4.0;
   }

   // Perform bilinear interpolation
   double interp_value = Q11 * (r2 - r) * (z2 - z) +
                         Q21 * (r - r1) * (z2 - z) +
                         Q12 * (r2 - r) * (z - z1) +
                         Q22 * (r - r1) * (z - z1);
   return interp_value / denom;
}


/* ----------------------------------------------------------------------
  read magnetic field data from file
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
PlasmaData Update::readPlasmaData(const std::string& filePath) {
    printf("Reading plasma data from file: %s\n", filePath.c_str());
    PlasmaData data;

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

        // Utility to read 2D dataset with shape validation
        auto read2D = [&](const std::string& name) -> std::vector<std::vector<double>> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dims[2];
            space.getSimpleExtentDims(dims);

            if (dims[0] != nz || dims[1] != nr) {
                throw std::runtime_error("Dataset '" + name + "' shape mismatch: expected " +
                                         std::to_string(nz) + " x " + std::to_string(nr) +
                                         ", got " + std::to_string(dims[0]) + " x " + std::to_string(dims[1]));
            }

            std::vector<double> raw(dims[0] * dims[1]);
            ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

            std::vector<std::vector<double>> grid(nz, std::vector<double>(nr));
            for (size_t i = 0; i < nz; ++i) {
                for (size_t j = 0; j < nr; ++j) {
                    grid[i][j] = raw[i * nr + j];
                }
            }
            return grid;
        };

        // Load 2D fields with strict shape check
        data.dens_e        = read2D("dens_e");
        data.temp_e        = read2D("temp_e");
        data.dens_i        = read2D("dens_i");
        data.temp_i        = read2D("temp_i");

        data.parr_flow     = read2D("parr_flow");
        data.parr_flow_r   = read2D("parr_flow_r");
        data.parr_flow_t   = read2D("parr_flow_t");
        data.parr_flow_z   = read2D("parr_flow_z");

        data.grad_temp_e_r = read2D("grad_te_r");
        data.grad_temp_e_t = read2D("grad_te_t");
        data.grad_temp_e_z = read2D("grad_te_z");

        data.grad_temp_i_r = read2D("grad_ti_r");
        data.grad_temp_i_t = read2D("grad_ti_t");
        data.grad_temp_i_z = read2D("grad_ti_z");

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



/*---------------------------------
  initialize plasma data
-----------------------------------*/
void Update::initializePlasmaData() {
  const int me     = comm->me;
  const int ncells = grid->nlocal;

  if (plasmaStyle == 0) {
    // // -------- CONSTANT BACKGROUND --------

    for (int icell = 0; icell < ncells; ++icell) {
      PlasmaDataParams params;     // value-init in case the struct grows later
      params.dens_e = dens_e;
      params.temp_e = temp_e;
      params.dens_i = dens_i;
      params.temp_i = temp_i;

      const double v0 = flow_v[0], v1 = flow_v[1], v2 = flow_v[2];
      params.parr_flow   = std::sqrt(v0*v0 + v1*v1 + v2*v2);
      params.parr_flow_r = v0;
      params.parr_flow_t = v1;
      params.parr_flow_z = v2;

      params.grad_temp_e_r = grad_te_r;
      params.grad_temp_e_t = grad_te_t;
      params.grad_temp_e_z = grad_te_z;

      params.grad_temp_i_r = grad_ti_r;
      params.grad_temp_i_t = grad_ti_t;
      params.grad_temp_i_z = grad_ti_z;

      plasma_data_map[icell] = params;
    }
  } else if (plasmaStyle == 1) {
    if (me == 0) plasma_data = readPlasmaData(plasmaStatePath);
     broadcastPlasmaData(plasma_data);

  for (int icell = 0; icell < ncells; ++icell) {
    plasma_data_map[icell] = bilinearInterpolationPlasma(icell, plasma_data);
  }
  }
  else {
    error->all(FLERR, "Unknown plasmaStyle:");
  }
}



/*----------------------------------------------------------------------
   broadcast plasma data
------------------------------------------------------------------------- */
void Update::broadcastPlasmaData(PlasmaData& data) {
    int me = comm->me;

    // Broadcast sizes of 1D vectors (r and z)
    int r_size = data.r.size();
    int z_size = data.z.size();
    MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
    MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

    // Resize vectors on non-root processes
    if (me != 0) {
        data.r.resize(r_size);
        data.z.resize(z_size);
    }

    // Broadcast 1D vector data
    MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
    MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

    // Broadcast 2D vectors (dens_e, temp_e, dens_i, temp_i, parr_flow, grad_temp_e, grad_temp_i)
    auto broadcast2DVector = [&](std::vector<std::vector<double>>& vec) {
        int dim1 = vec.size();
        int dim2 = dim1 ? vec[0].size() : 0;

        MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
        MPI_Bcast(&dim2, 1, MPI_INT, 0, world);

        // Resize the outer vector and each inner vector on non-root processes
        if (me != 0) {
            vec.resize(dim1, std::vector<double>(dim2));
        }

        for (int i = 0; i < dim1; ++i) {
            MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
        }
    };

    broadcast2DVector(data.dens_e);
    broadcast2DVector(data.temp_e);
    broadcast2DVector(data.dens_i);
    broadcast2DVector(data.temp_i);

    broadcast2DVector(data.parr_flow);
    broadcast2DVector(data.parr_flow_r);
    broadcast2DVector(data.parr_flow_t);
    broadcast2DVector(data.parr_flow_z);

    broadcast2DVector(data.grad_temp_e_r);
    broadcast2DVector(data.grad_temp_e_t);
    broadcast2DVector(data.grad_temp_e_z);

    broadcast2DVector(data.grad_temp_i_r);
    broadcast2DVector(data.grad_temp_i_t);
    broadcast2DVector(data.grad_temp_i_z);
}

/* ----------------------------------------------------------------------
   set global properites via read_magnetic_fields
------------------------------------------------------------------------- */
void Update::read_magnetic_fields(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal read_plasma_state command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "magnetic_fields") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal magnetic_fields command");
      if (strcmp(arg[iarg+1], "constant") == 0) {
        magneticFieldsStyle = 0;
        iarg += 2; // Move past 'constant'
        while (iarg < narg && strcmp(arg[iarg], "file") != 0) {
         if (strcmp(arg[iarg], "bfield") == 0) {
            if (iarg+3 >= narg) error->all(FLERR, "Illegal magnetic_fields command");
            bfield[0] = input->numeric(FLERR, arg[iarg+1]);
            bfield[1] = input->numeric(FLERR, arg[iarg+2]);
            bfield[2] = input->numeric(FLERR, arg[iarg+3]);
            iarg += 4;
          } else {
            error->all(FLERR, "Unrecognized parameter in magnetic_fields constant");
          }
        }
      } else if (strcmp(arg[iarg+1], "file") == 0) {
        magneticFieldsStyle = 1;
        if (iarg+2 >= narg) error->all(FLERR, "Illegal magnetic_fields command");
        magneticFieldsPath = arg[iarg+2];
        iarg += 3;  
      } else {
        error->all(FLERR, "Illegal magnetic_fields command");
      }
    } else {
      error->all(FLERR, "Illegal magnetic_fields command");
    }
  }
}
/*----------------------------------------------------------------------
   broadcast magnetic field data
------------------------------------------------------------------------- */
void Update::broadcastMagneticData(MagneticFieldData& data) {
  int me = comm->me;

  // Broadcast sizes of 1D vectors (e.g., r and z for the magnetic field)
  int r_size = data.r.size();
  int z_size = data.z.size();
  MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

  // Resize vectors on non-root processes
  if (me != 0) {
      data.r.resize(r_size);
      data.z.resize(z_size);
  }

  // Broadcast 1D vector data (r and z)
  MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

  // Broadcast 2D vectors (e.g., br, bt, bz)
  auto broadcast2DVector = [&](std::vector<std::vector<double>>& vec) {
      int dim1 = vec.size();
      int dim2 = dim1 ? vec[0].size() : 0;

      MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
      MPI_Bcast(&dim2, 1, MPI_INT, 0, world);

      // Resize the outer vector and each inner vector on non-root processes
      if (me != 0) {
          vec.resize(dim1, std::vector<double>(dim2));
      }

      for (int i = 0; i < dim1; ++i) {
          MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
      }
  };

  // Broadcast the magnetic field components
  broadcast2DVector(data.br);
  broadcast2DVector(data.bt);
  broadcast2DVector(data.bz);
}

/* ----------------------------------------------------------------------
   read magnetic field data from file
------------------------------------------------------------------------- */
MagneticFieldData Update::readMagneticFieldData(const std::string& filePath) {
  printf("Reading magnetic field data from file: %s\n", filePath.c_str());
    MagneticFieldData data; // Initialize an empty MagneticFieldData struct
    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        
        auto read1DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(1);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<double> data(dims[0]);
            ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
            return data;
        };
        
        auto read2DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(2);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
            std::vector<double> rawData(dims[0] * dims[1]);
            ds.read(rawData.data(), H5::PredType::NATIVE_DOUBLE);
            
            for (hsize_t i = 0; i < dims[0]; ++i) {
                for (hsize_t j = 0; j < dims[1]; ++j) {
                    data[i][j] = rawData[i * dims[1] + j];
                }
            }
            return data;
        };
        
        // Read the required datasets
        data.r = read1DDataSet("r");
        data.z = read1DDataSet("z");
        data.br = read2DDataSet("br");
        data.bz = read2DDataSet("bz");
        data.bt = read2DDataSet("bt");
        file.close();
    } catch (const H5::Exception& e) {
        printf("Error reading magnetic field file file: %s\n", e.getCDetailMsg());
        throw;  // Re-throw the exception to handle it outside
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        throw;
    }
    printf("Finished reading magnetic field data from file: %s\n", filePath.c_str());
    return data;
}
/*---------------------------------
  initialize magnetic field data
-----------------------------------*/
void Update::initializeMagneticData() {
  int me = comm->me;
// 

  // Load magnetic field data only on the root process
  if (me == 0) {
      magnetic_data = readMagneticFieldData(magneticFieldsPath);
  }

  // Broadcast the magnetic field data to all processes
  broadcastMagneticData(magnetic_data);

  // Perform interpolation for each cell on all processes
  // int ncells = grid->ncell;
  int ncells = grid->nlocal;
  for (int icell = 0; icell < ncells; ++icell) {
      magnetic_data_map[icell] = bilinearInterpolationMagneticField(icell, magnetic_data);      
  }

}


/*---------------------------------
  Bilinear interpolation plasma
-----------------------------------*/

PlasmaDataParams Update::bilinearInterpolationPlasma(int icell, const PlasmaData& data) {
    // Check cache
    auto it = plasmaDataCache.find(icell);
    if (it != plasmaDataCache.end()) return it->second;

    // Validate coordinate arrays
    if (data.r.empty() || data.z.empty()) {
        throw std::runtime_error("Plasma data coordinate arrays are empty.");
    }

    const auto& r_vals = data.r;
    const auto& z_vals = data.z;

    // Get cell midpoint
    Grid::ChildCell* cell = &grid->cells[icell];
    double r = 0.5 * (cell->lo[0] + cell->hi[0]);
    double z = 0.5 * (cell->lo[1] + cell->hi[1]);

    // Out-of-bounds check
    if (r < r_vals.front() || r > r_vals.back() || z < z_vals.front() || z > z_vals.back()) {
        return PlasmaDataParams{};
    }

    // Locate surrounding grid indices
    auto r_it = std::lower_bound(r_vals.begin(), r_vals.end(), r);
    auto z_it = std::lower_bound(z_vals.begin(), z_vals.end(), z);
    int r1 = std::max(0, static_cast<int>(r_it - r_vals.begin()) - 1);
    int r2 = std::min(static_cast<int>(r_vals.size()) - 1, r1 + 1);
    int z1 = std::max(0, static_cast<int>(z_it - z_vals.begin()) - 1);
    int z2 = std::min(static_cast<int>(z_vals.size()) - 1, z1 + 1);

    double R1 = r_vals[r1], R2 = r_vals[r2];
    double Z1 = z_vals[z1], Z2 = z_vals[z2];
    double denom = (R2 - R1) * (Z2 - Z1);

    // Bilinear interpolation lambda
    auto interp = [&](const std::vector<std::vector<double>>& field) -> double {
        if (field.size() <= z2 || field[0].size() <= r2) return 0.0;

        double Q11 = field[z1][r1];
        double Q21 = field[z1][r2];
        double Q12 = field[z2][r1];
        double Q22 = field[z2][r2];

        if (denom == 0.0) return (Q11 + Q21 + Q12 + Q22) / 4.0;

        return (
            Q11 * (R2 - r) * (Z2 - z) +
            Q21 * (r - R1) * (Z2 - z) +
            Q12 * (R2 - r) * (z - Z1) +
            Q22 * (r - R1) * (z - Z1)
        ) / denom;
    };

    // Fill and cache interpolated values
    PlasmaDataParams result;
    result.dens_e        = interp(data.dens_e);
    result.temp_e        = interp(data.temp_e);
    result.dens_i        = interp(data.dens_i);
    result.temp_i        = interp(data.temp_i);
    result.parr_flow     = interp(data.parr_flow);
    result.parr_flow_r   = interp(data.parr_flow_r);
    result.parr_flow_t   = interp(data.parr_flow_t);
    result.parr_flow_z   = interp(data.parr_flow_z);
    result.grad_temp_e_r = interp(data.grad_temp_e_r);
    result.grad_temp_e_t = interp(data.grad_temp_e_t);
    result.grad_temp_e_z = interp(data.grad_temp_e_z);
    result.grad_temp_i_r = interp(data.grad_temp_i_r);
    result.grad_temp_i_t = interp(data.grad_temp_i_t);
    result.grad_temp_i_z = interp(data.grad_temp_i_z);

    plasmaDataCache[icell] = result;
    return result;
}


/* ----------------------------------------------------------------------
   Boris pusher for 3D cartesian coordinates  (minimal, API-compatible fix)
------------------------------------------------------------------------- */
void Update::pusher_boris3D(int i, int icell, double dt,
                            double *x, double *v, double *xnew,
                            double charge, double mass)
{
  // Use q/m consistently
  const double qm = (charge * echarge) / mass;

  // --- Load fields (use independent indices for E and B)
  double E[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};

  if (eperturbflag) {
    double **arr;
    int colE = 0;
    if (efstyle == GFIELD) {
      arr = modify->fix[efieldfix]->array_grid;
      if (efield_active[0]) E[0] = arr[icell][colE++];
      if (efield_active[1]) E[1] = arr[icell][colE++];
      if (efield_active[2]) E[2] = arr[icell][colE++];
    } else if (efstyle == PFIELD) {
      arr = modify->fix[efieldfix]->array_particle;
      if (efield_active[0]) E[0] = arr[i][colE++];
      if (efield_active[1]) E[1] = arr[i][colE++];
      if (efield_active[2]) E[2] = arr[i][colE++];
    }
  }

  if (bperturbflag) {
    double **arr;
    int colB = 0;
    if (bfstyle == GFIELD) {
      arr = modify->fix[bfieldfix]->array_grid;
      if (bfield_active[0]) B[0] = arr[icell][colB++];
      if (bfield_active[1]) B[1] = arr[icell][colB++];
      if (bfield_active[2]) B[2] = arr[icell][colB++];
    } else if (bfstyle == PFIELD) {
      arr = modify->fix[bfieldfix]->array_particle;
      if (bfield_active[0]) B[0] = arr[i][colB++];
      if (bfield_active[1]) B[1] = arr[i][colB++];
      if (bfield_active[2]) B[2] = arr[i][colB++];
    }
  }

  // --- Boris: half E
  double v_minus[3] = {
    v[0] + qm * E[0] * 0.5 * dt,
    v[1] + qm * E[1] * 0.5 * dt,
    v[2] + qm * E[2] * 0.5 * dt
  };

  // --- Boris: rotation
  const double t[3] = { qm * B[0] * 0.5 * dt,
                        qm * B[1] * 0.5 * dt,
                        qm * B[2] * 0.5 * dt };
  const double t2   = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
  const double s[3] = { 2.0*t[0]/(1.0 + t2),
                        2.0*t[1]/(1.0 + t2),
                        2.0*t[2]/(1.0 + t2) };

  double v_prime[3], v_plus[3];
  MathExtra::cross3(v_minus, t, v_prime);
  v_prime[0] += v_minus[0]; v_prime[1] += v_minus[1]; v_prime[2] += v_minus[2];

  MathExtra::cross3(v_prime, s, v_plus);
  v_plus[0] += v_minus[0];  v_plus[1] += v_minus[1];  v_plus[2] += v_minus[2];

  // --- Boris: final half E
  v_plus[0] += qm * E[0] * 0.5 * dt;
  v_plus[1] += qm * E[1] * 0.5 * dt;
  v_plus[2] += qm * E[2] * 0.5 * dt;

  // --- Trial position ONLY; do NOT mutate x[] here
  xnew[0] = x[0] + v_plus[0] * dt;
  xnew[1] = x[1] + v_plus[1] * dt;
  xnew[2] = x[2] + v_plus[2] * dt;

  // It's OK to output new velocity via v[] (no separate vnew in API)
  v[0] = v_plus[0];
  v[1] = v_plus[1];
  v[2] = v_plus[2];
}


/* ----------------------------------------------------------------------
   Cross-field diffusion step
   Random walk in plane perpendicular to B
------------------------------------------------------------------------- */  
void Update::apply_cross_field_diffusion(int icell, double dt, double* B, double* x) {
  // Compute |B| and normalize it
  double Bnorm = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  double Bx[3];

  if (Bnorm < 1e-10) {
      // Default to toroidal field if B is nearly zero
      Bx[0] = 0.0; Bx[1] = 1.0; Bx[2] = 0.0;
  } else {
      for (int i = 0; i < 3; ++i) Bx[i] = B[i] / Bnorm;
  }

  // Find a vector ex perpendicular to B
  double ex[3];
  double zdir[3] = {0.0, 0.0, 1.0};
  MathExtra::cross3(Bx, zdir, ex);
  double ex_norm = std::sqrt(ex[0]*ex[0] + ex[1]*ex[1] + ex[2]*ex[2]);

  // If B is aligned with z, pick a different reference vector
  if (ex_norm < 1e-10) {
      double ydir[3] = {0.0, 1.0, 0.0};
      MathExtra::cross3(Bx, ydir, ex);
      ex_norm = std::sqrt(ex[0]*ex[0] + ex[1]*ex[1] + ex[2]*ex[2]);
  }

  // Normalize ex
  for (int i = 0; i < 3; ++i) ex[i] /= ex_norm;

  // Compute ey = B × ex (also perpendicular to B)
  double ey[3];
  MathExtra::cross3(Bx, ex, ey);

  // Compute random walk step in the perpendicular plane
  double step = std::sqrt(4.0 * d_perp * dt);
  double phi = 2.0 * M_PI * update->ranmaster->uniform();
  double cos_phi = std::cos(phi);
  double sin_phi = std::sin(phi);

  // Final displacement vector in perpendicular plane
  for (int i = 0; i < 3; ++i) {
      double delta = step * (ex[i] * cos_phi + ey[i] * sin_phi);
      x[i] += delta;
  }
}

// All-SI sheath E-field (Brooks/Stangeby style)
// Inputs:
//   B[3]          : tesla
//   Te_eV, Ti_eV  : eV
//   ne_m3         : m^-3
//   t_sheath      : dimensionless model constant
//   alpha_deg     : degrees between B and surface (incidence angle)
//   d_wall_m      : minimum distance to the wall [m]
//   n_hat[3]      : outward surface normal (need not be unit)
// Outputs:
//   Efield[3]     : V/m  (points from plasma toward wall, i.e. opposite outward normal)
//   ne_sheath     : m^-3 (placeholder here; keep ne for now)
//
// Stangeby P.C. 2000 The Plasma Boundary of Magnetic Fusion
// Sheath Model from Stangeby
void Update::sheathEfieldBrooks(double *B, double Te_eV, double ne_m3, double Ti_eV, double t_sheath, double alpha_deg,double d_wall_m, const double* n_hat, double& Emag, double& ne_sheath) 
{
  // --- constants (SI) ---
  const double e    = update->echarge;          // 1.602e-19 C
  const double me   = update->electron_mass;    // 9.11e-31 kg
  const double mi   = update->proton_mass;      // 1.67e-27 kg  (TODO: species mass)
  const double eps0 = update->epsilon_0;         // 8.854e-12 F/m

  // --- angles & magnitudes ---
  const double alpha = alpha_deg * M_PI / 180.0;
  const double s_alpha = std::max(std::abs(std::sin(alpha)), 1e-6); // clamp to avoid log(0)
  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

  // --- Debye length [m] ---
  // λ_D = sqrt( ε0 * kTe / (n e^2) ) with kTe = e * Te_eV
  const double lambda_D = std::sqrt( eps0 * (Te_eV*e) / ( std::max(ne_m3,1e5) * e*e ) );

  // --- ion sound speed and gyro ---
  // c_s ≈ sqrt( (Te+Ti) * e / mi )  (γ factors omitted for simplicity)
  const double cs       = std::sqrt( std::max(Te_eV+Ti_eV, 1e-6) * e / mi );
  const double omega_ci = (Bmag > 0.0 ? e * Bmag / mi : 1e-9);

  // --- Larmor radius and MPS scale length [m] ---
  const double rho_i = cs / omega_ci; // [m]
  const double L_MPS = std::abs(rho_i * std::cos(alpha)) * std::sqrt(6.0) * t_sheath;

  // --- potentials [V] ---
  // kTe/e = Te_eV (1 eV corresponds to 1 V); so "normalizationFactor" is just Te_eV.
  const double TeV = Te_eV;
  const double phi_wall  = 0.5 * std::log(2.0 * M_PI * me / mi) * (1.0 + Ti_eV / std::max(Te_eV,1e-9)) * TeV;
  const double phi_VMPS  = std::log(s_alpha) * TeV;  // <= 0
  const double phi_UDS   = phi_wall - phi_VMPS;

  // --- field magnitudes at the wall [V/m] ---
  const double lambda_safe = std::max(lambda_D, 1e-9);   // avoid divide by ~0
  const double LMPS_safe   = std::max(L_MPS,   1e-9);

  const double E_DS0  =  phi_UDS / lambda_safe; // V/m
  const double E_MPS0 =  phi_VMPS / LMPS_safe;  // V/m

  // --- radial profiles vs distance to wall [m] ---
  const double E_DS   = E_DS0  * std::exp( -0.5 * d_wall_m / lambda_safe );
  const double E_MPS  = E_MPS0 * std::exp(       d_wall_m / (-LMPS_safe) );

  Emag = E_DS + E_MPS;

  // If we are far from the wall in terms of both scales, just zero out
  const double L_char = std::max(lambda_safe, LMPS_safe);
  if ( !std::isfinite(Emag) || d_wall_m > 5.0 * L_char ) {
    Emag = 0.0;
  }

  // --- unit normal & field direction (into the wall) ---
  double nx = n_hat[0], ny = n_hat[1], nz = n_hat[2];
  const double nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
  if (nlen > 0.0) { nx /= nlen; ny /= nlen; nz /= nlen; }

  // sheath density model (placeholder: keep upstream ne for now)
  ne_sheath = ne_m3;

}



inline void cross_rphiz(const double a[3], const double b[3], double out[3]) {
    // (r, φ, z) behaves like (x, y, z)
    out[0] = a[1]*b[2] - a[2]*b[1]; // r
    out[1] = a[2]*b[0] - a[0]*b[2]; // φ
    out[2] = a[0]*b[1] - a[1]*b[0]; // z
}


// Find bracketing index i0 such that x[i0] <= xq <= x[i0+1], plus weight wx in [0,1]
inline void bracket_clamped(const std::vector<double>& x, double xq, int& i0, double& wx) {
    const int N = (int)x.size();
    if (N < 2) { i0 = 0; wx = 0.0; return; }
    if (xq <= x.front()) { i0 = 0; wx = 0.0; return; }
    if (xq >= x.back())  { i0 = N-2; wx = 1.0; return; }
    auto it = std::upper_bound(x.begin(), x.end(), xq);
    int i1 = int(it - x.begin());
    i0 = std::max(0, std::min(i1 - 1, N - 2));
    double x0 = x[i0], x1 = x[i0+1];
    wx = (x1 > x0) ? (xq - x0) / (x1 - x0) : 0.0;
}

// Bilinear sample of a scalar field stored as data[z][r] (Z-major, R-minor)
inline double bilinear_sample_scalar(
    double R, double Z,
    const std::vector<double>& rgrid,
    const std::vector<double>& zgrid,
    const std::vector<std::vector<double>>& data)
{
    int ir; double wr;
    int iz; double wz;
    bracket_clamped(rgrid, R, ir, wr);
    bracket_clamped(zgrid, Z, iz, wz);

    // corners
    double Q00 = data[iz][ir];
    double Q10 = data[iz][ir+1];
    double Q01 = data[iz+1][ir];
    double Q11 = data[iz+1][ir+1];

    double Q0 = Q00*(1.0 - wr) + Q10*wr;
    double Q1 = Q01*(1.0 - wr) + Q11*wr;
    return Q0*(1.0 - wz) + Q1*wz;
}



// Axisymmetric Boris pusher using cylindrical fields (Br, Bz, Bphi) on (R,Z).
// State: x = [R, Z, phi], v = [v_r, v_z, v_phi].  We rotate to Cartesian,
// do Boris, then rotate back.  The sheath E is applied ALONG poloidal B
// and INTO the wall (so E·n <= 0, E×B has no poloidal contribution).

// Helper: cylindrical -> Cartesian rotation for B at angle phi
inline void cylB_to_cart(double phi, double Br, double Bphi, double Bz_c,
                         double &Bx, double &By, double &Bz)
{
  const double c = std::cos(phi), s = std::sin(phi);
  // e_R   = ( c,  s, 0)
  // e_phi = (-s,  c, 0)
  Bx =  Br*c - Bphi*s;
  By =  Br*s + Bphi*c;
  Bz =  Bz_c;
}

/* ----------------------------------------------------------------------
   Boris pusher for 2D axisymmetric (R,Z,phi) with optional thermal forces.
   - x  = {R, Z, phi}
   - v  = {v_r, v_z, v_phi}
   - E,B stored in cylindrical comps (Er, Ephi, Ez) at icell/i
   - Thermal "E_th" built from parallel ∇T_e, ∇T_i (in eV/m!)
------------------------------------------------------------------------- */
void Update::pusherBoris2D(int i, int icell, double dt,
                           double *x, double *v, double *xnew,
                           double charge, double mass)
{

    if (perturbflag) {
      // pull cylindrical accelerations (a_r, a_z, a_phi) from the active columns
      double **array = modify->fix[ifieldfix]->array_grid;
      double ar = 0.0, az = 0.0, aphi = 0.0;
      int icol = 0;
      if (field_active[0]) ar   = array[icell][icol++];  // a_r [m/s^2]
      if (field_active[1]) az   = array[icell][icol++];  // a_z [m/s^2]
      if (field_active[2]) aphi = array[icell][icol++];  // a_phi [m/s^2]

      const double dt2  = dt*dt;
      const double R0   = x[0];
      const double Z0   = x[1];
      const double phi0 = x[2];
      const double vr0  = v[0];
      const double vz0  = v[1];
      const double vphi0= v[2];

      // velocities (explicit Euler on a)
      const double vr1   = vr0   + ar   * dt;
      const double vz1   = vz0   + az   * dt;
      const double vphi1 = vphi0 + aphi * dt;

      // positions: include drift + 0.5 a dt^2
      const double R1    = R0  + vr0   * dt + 0.5 * ar   * dt2;
      const double Z1    = Z0  + vz0   * dt + 0.5 * az   * dt2;

      // angular advance: use averaged v_phi and averaged R to be stable
      const double Ravg  = std::max(0.5*(R0 + R1), 1e-12);
      const double vphi_avg = 0.5*(vphi0 + vphi1);
      const double phi1  = phi0 + (vphi_avg / Ravg) * dt;

      // write back
      v[0]    = vr1;
      v[1]    = vz1;
      v[2]    = vphi1;
      xnew[0] = R1;
      xnew[1] = Z1;
      xnew[2] = phi1;

      // helpful debug: print the actual a and the NEW x
      // printf("Particle %d: a=(%g,%g,%g) m/s^2, xnew=(%g,%g,%g)\n",
            // i, ar, az, aphi, xnew[0], xnew[1], xnew[2]);
      return;
    }


  // ---------- load fields ----------
  double E_cyl[3] = {0.0, 0.0, 0.0};
  double B_cyl[3] = {0.0, 0.0, 0.0};
  double gradTe_e[3] = {0.0, 0.0, 0.0}; // MUST be eV/m
  double gradTe_i[3] = {0.0, 0.0, 0.0}; // MUST be eV/m

  if (eperturbflag) {
    double **arr; int col=0;
    if (efstyle == GFIELD) {
      arr = modify->fix[efieldfix]->array_grid;
      if (efield_active[0]) E_cyl[0] = arr[icell][col++];
      if (efield_active[1]) E_cyl[1] = arr[icell][col++];
      if (efield_active[2]) E_cyl[2] = arr[icell][col++];
    } else if (efstyle == PFIELD) {
      arr = modify->fix[efieldfix]->array_particle;
      if (efield_active[0]) E_cyl[0] = arr[i][col++];
      if (efield_active[1]) E_cyl[1] = arr[i][col++];
      if (efield_active[2]) E_cyl[2] = arr[i][col++];
    }
  }

  if (bperturbflag) {
    double **arr; int col=0;
    if (bfstyle == GFIELD) {
      arr = modify->fix[bfieldfix]->array_grid;
      if (bfield_active[0]) B_cyl[0] = arr[icell][col++];
      if (bfield_active[1]) B_cyl[1] = arr[icell][col++];
      if (bfield_active[2]) B_cyl[2] = arr[icell][col++];
    } else if (bfstyle == PFIELD) {
      arr = modify->fix[bfieldfix]->array_particle;
      if (bfield_active[0]) B_cyl[0] = arr[i][col++];
      if (bfield_active[1]) B_cyl[1] = arr[i][col++];
      if (bfield_active[2]) B_cyl[2] = arr[i][col++];
    }
  }

  if (ethermalflag) {
    double **arr; int col=0;
    if (ethermalstyle == GFIELD) {
      arr = modify->fix[ethermalfix]->array_grid;
      if (ethermal_active[0]) gradTe_e[0] += arr[icell][col++];
      if (ethermal_active[1]) gradTe_e[1] += arr[icell][col++];
      if (ethermal_active[2]) gradTe_e[2] += arr[icell][col++];
    }
  }
  if (ithermalflag) {
    double **arr; int col=0;
    if (ithermalstyle == GFIELD) {
      arr = modify->fix[ithermalfix]->array_grid;
      if (ithermal_active[0]) gradTe_i[0] += arr[icell][col++];
      if (ithermal_active[1]) gradTe_i[1] += arr[icell][col++];
      if (ithermal_active[2]) gradTe_i[2] += arr[icell][col++];
    }
  }


  // if no perturbations, do simple ballistic step and return
  if (!eperturbflag && !bperturbflag && !ethermalflag && !ithermalflag) {
        double dtremain = dt;
        xnew[0] = x[0] + v[0]*dtremain;
        xnew[1] = x[1] + v[1]*dtremain;
        // xnew[2] = x[2] + (v[2]/std::
        return;
    }

  // ---------- geometry ----------
  const double R   = std::max(x[0], 1e-12);
  const double Zc  = x[1];
  const double phi = x[2];
  const double cphi = std::cos(phi), sphi = std::sin(phi);

  auto cyl_to_cart = [&](double vr, double vphi, double vz,
                         double &vx, double &vy, double &vz_out) {
    vx =  vr*cphi - vphi*sphi;
    vy =  vr*sphi + vphi*cphi;
    vz_out = vz;
  };
  auto cart_to_cyl = [&](double vx, double vy, double vz,
                         double phi_now, double &vr, double &vphi, double &vz_out) {
    const double c = std::cos(phi_now), s = std::sin(phi_now);
    vr     =  vx*c + vy*s;
    vphi   = -vx*s + vy*c;
    vz_out =  vz;
  };

  // ---------- q/m & early exit for neutrals ----------
  const double Zabs = std::abs(charge);            // charge in units of e
  if (Zabs == 0.0) { // neutral: ballistic
    xnew[0] = R + v[0]*dt;
    xnew[1] = Zc + v[1]*dt;
    xnew[2] = phi + (v[2]/std::max(R,1e-12))*dt;   // crude φ advance if needed
    return;
  }
  const double qm = (charge * echarge) / mass;

  // ---------- B (cyl->cart) once; used for b-hat and rotation ----------
  double Bx=0, By=0, Bz=0;
  cylB_to_cart(phi, B_cyl[0], B_cyl[1], B_cyl[2], Bx, By, Bz);
  const double Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);
  double bx=0, by=0, bz=0;
  if (Bmag > 0.0) { bx = Bx/Bmag; by = By/Bmag; bz = Bz/Bmag; }

  // ---------- build thermal E_parallel (∇T in eV/m) ----------
  // coefficients from your doc (force-level): F_T = - α_e ∇T_e - β_i ∇T_i
  // => E_th = F_T / (Ze) = -(α_e/Z) ∇T_e - (β_i/Z) ∇T_i (parallel component only)
  if ((ethermalflag || ithermalflag) && Zabs > 0 && Bmag > 0) {
    // Rotate gradients to Cartesian to dot with b-hat
    const double dTe_x =  gradTe_e[0]*cphi - gradTe_e[1]*sphi;
    const double dTe_y =  gradTe_e[0]*sphi + gradTe_e[1]*cphi;
    const double dTe_z =  gradTe_e[2];
    const double dTi_x =  gradTe_i[0]*cphi - gradTe_i[1]*sphi;
    const double dTi_y =  gradTe_i[0]*sphi + gradTe_i[1]*cphi;
    const double dTi_z =  gradTe_i[2];

    const double gTe_par = dTe_x*bx + dTe_y*by + dTe_z*bz; // eV/m
    const double gTi_par = dTi_x*bx + dTi_y*by + dTi_z*bz; // eV/m

    // impurity mass ratio μ = m_s / (m_s + m_i_main)
    const double m_i_main = proton_mass * plasma_background_mass; // e.g., 2.0 for D
    const double mu = mass / (mass + m_i_main);

    const double alpha_e = 0.71 * Zabs * Zabs; // your Eq. (6), force-level
    const double beta_i  = (3.0*( mu
                           + 5.0*std::sqrt(2.0)*Zabs*Zabs
                             *(1.1*std::pow(mu,2.5) - 0.35*std::pow(mu,1.5))
                           - 1.0))
                           /(2.6 - 2.0*mu + 5.4*mu*mu); // your Eq. (7/8)

    const double Eth_par = -(alpha_e/Zabs)*gTe_par - (beta_i/Zabs)*gTi_par; // V/m

    // Back to components; add into cylindrical E
    const double Ex_th = Eth_par * bx;
    const double Ey_th = Eth_par * by;
    const double Ez_th = Eth_par * bz;
    const double Er_th   =  Ex_th*cphi + Ey_th*sphi;
    const double Ephi_th = -Ex_th*sphi + Ey_th*cphi;

    E_cyl[0] += Er_th;
    E_cyl[1] += Ephi_th;
    E_cyl[2] += Ez_th;
  }

  // ---------- transform final E to Cartesian ----------
  const double Ex =  E_cyl[0]*cphi - E_cyl[1]*sphi;
  const double Ey =  E_cyl[0]*sphi + E_cyl[1]*cphi;
  const double Ez =  E_cyl[2];

  // ---------- velocities to Cartesian ----------
  double vx, vy, vz;
  // v[] ordering: {v_r, v_z, v_phi}
  cyl_to_cart(/*vr=*/v[0], /*vphi=*/v[2], /*vz=*/v[1], vx, vy, vz);

  // ---------- Boris: half E-kick ----------
  double v_minus[3] = {
    vx + qm * Ex * 0.5 * dt,
    vy + qm * Ey * 0.5 * dt,
    vz + qm * Ez * 0.5 * dt
  };

  // ---------- Boris: rotation (t= q B dt / 2m) ----------
  const double tvec[3] = { qm * Bx * 0.5 * dt,
                           qm * By * 0.5 * dt,
                           qm * Bz * 0.5 * dt };
  const double t2   = tvec[0]*tvec[0] + tvec[1]*tvec[1] + tvec[2]*tvec[2];
  const double svec[3] = { 2.0*tvec[0]/(1.0 + t2),
                           2.0*tvec[1]/(1.0 + t2),
                           2.0*tvec[2]/(1.0 + t2) };

  double v_prime[3], v_plus[3];
  MathExtra::cross3(v_minus, tvec, v_prime);
  v_prime[0] += v_minus[0]; v_prime[1] += v_minus[1]; v_prime[2] += v_minus[2];

  MathExtra::cross3(v_prime, svec, v_plus);
  v_plus[0] += v_minus[0];  v_plus[1] += v_minus[1];  v_plus[2] += v_minus[2];

  // ---------- Boris: final half E-kick ----------
  v_plus[0] += qm * Ex * 0.5 * dt;
  v_plus[1] += qm * Ey * 0.5 * dt;
  v_plus[2] += qm * Ez * 0.5 * dt;

  // ---------- advance positions ----------
  const double X0 = R * cphi, Y0 = R * sphi;
  const double Xn = X0 + v_plus[0] * dt;
  const double Yn = Y0 + v_plus[1] * dt;
  const double Zn = Zc + v_plus[2] * dt;

  const double Rn   = std::hypot(Xn, Yn);
  const double phin = std::atan2(Yn, Xn);

  // ---------- back to cylindrical velocities at new angle ----------
  double vr_out, vphi_out, vz_out;
  cart_to_cyl(v_plus[0], v_plus[1], v_plus[2], phin, vr_out, vphi_out, vz_out);

  v[0] = vr_out;   // v_r
  v[1] = vz_out;   // v_z
  v[2] = vphi_out; // v_phi

  xnew[0] = Rn;
  xnew[1] = Zn;
  xnew[2] = phin;
}



PlasmaDataParams Update::interpolatePlasma_RZ_clamped(
    double R, double Z, const PlasmaData& data)
{
  PlasmaDataParams out{};  // zeros by default

  // Fast reject if coords missing
  if (data.r.size() < 2 || data.z.size() < 2) return out;

  // Sample each field if present (same Z-major, R-minor layout)
  auto S = [&](const std::vector<std::vector<double>>& f)->double {
    if (f.empty() || f[0].empty()) return 0.0;
    return bilinear_sample_scalar(R, Z, data.r, data.z, f);
  };

  out.dens_e        = S(data.dens_e);
  out.temp_e        = S(data.temp_e);
  out.dens_i        = S(data.dens_i);
  out.temp_i        = S(data.temp_i);

  out.parr_flow     = S(data.parr_flow);
  out.parr_flow_r   = S(data.parr_flow_r);
  out.parr_flow_t   = S(data.parr_flow_t);
  out.parr_flow_z   = S(data.parr_flow_z);

  out.grad_temp_e_r = S(data.grad_temp_e_r);
  out.grad_temp_e_t = S(data.grad_temp_e_t);
  out.grad_temp_e_z = S(data.grad_temp_e_z);
  out.grad_temp_i_r = S(data.grad_temp_i_r);
  out.grad_temp_i_t = S(data.grad_temp_i_t);
  out.grad_temp_i_z = S(data.grad_temp_i_z);

  return out;
}



PlasmaDataParams Update::interpolatePlasma_RZ_constant()
{
  PlasmaDataParams out{};  // zeros by default


  out.dens_e        =  dens_e;
  out.temp_e        =  temp_e;

  out.parr_flow     = 0; //parr_flow;
  out.parr_flow_r   = 0; //parr_flow_r;
  out.parr_flow_t   =0; // parr_flow_t;
  out.parr_flow_z   = 0; //parr_flow_z;

  out.grad_temp_e_r = 0; //grad_temp_e_r;
  out.grad_temp_e_t = 0; //grad_temp_e_t;
  out.grad_temp_e_z = 0; //grad_temp_e_z;
  out.grad_temp_i_r = 0; //grad_temp_i_r;
  out.grad_temp_i_t = 0; //grad_temp_i_t;
  out.grad_temp_i_z = 0; //grad_temp_i_z;

  return out;
}


// F_Te = α ∇T_e ,  F_Ti = β ∇T_i     (T in eV, ∇T in eV/m)
// Effective field:  q E_th = F  ⇒  E_th = (e/ q) (α ∇T_e + β ∇T_i)
// For an ion with charge q = Z e  →  E_th = (α/Z) ∇T_e + (β/Z) ∇T_i.
void Update::thermal_gradient_Efield(const double mass, const double charge,
                                    const double gradTe_e[3],  // ∇T_e in eV/m (R,φ,Z)
                                    const double gradTe_i[3],  // ∇T_i in eV/m (R,φ,Z)
                                    double E_th[3])            // out: (Er,Eφ,Ez) in V/m
{
    if (std::abs(charge) < 1e-16) { E_th[0]=E_th[1]=E_th[2]=0.0; return; }

    // main-ion mass used in β(μ); set from your background species
    const double m_i_main = proton_mass * plasma_background_mass; // e.g. D/T -> set factor
    const double mu = mass / (m_i_main + mass);

    const double Z = (charge != 0.0) ? (charge ) : 0.0;

    const double alpha = 0.71 * Z * Z;
    const double beta  = 3.0 * (mu + 5.0 * std::sqrt(2.0) * Z * Z *
                               (1.1 * std::pow(mu, 2.5) - 0.35 * std::pow(mu, 1.5)) - 1.0)
                         / (2.6 - 2.0 * mu + 5.4 * std::pow(mu, 2));

    const double factor = ev2kelvin * boltz / (charge * echarge);

    E_th[0] = factor * (alpha * gradTe_e[0] + beta * gradTe_i[0]); // Er
    E_th[1] = factor * (alpha * gradTe_e[1] + beta * gradTe_i[1]); // Eφ
    E_th[2] = factor * (alpha * gradTe_e[2] + beta * gradTe_i[2]); // Ez
}


// Brownian step in plane ⟂ B with <|ΔX⊥|^2> = 4 Dperp dt
void Update::apply_cross_field_diffusion_cart(
    const double dt,
    const double Bx, const double By, const double Bz,
    const double u01,                // uniform(0,1)
    double &Xn, double &Yn, double &Zn)
{
  // b-hat
  const double Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);
  double bx=0, by=0, bz=1; // fallback if |B|≈0
  if (Bmag > 1e-14) { bx = Bx/Bmag; by = By/Bmag; bz = Bz/Bmag; }

  // Orthonormal basis e1, e2 spanning plane ⟂ b̂
  // choose a ref not ~parallel to b̂
  double rx = (std::fabs(bx) < 0.9) ? 1.0 : 0.0;
  double ry = (rx==0.0) ? 1.0 : 0.0;
  double rz = 0.0;

  // e1 = normalize(b̂ × ref), e2 = b̂ × e1
  double e1x =  by*rz - bz*ry;
  double e1y =  bz*rx - bx*rz;
  double e1z =  bx*ry - by*rx;
  double n1  = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
  if (n1 < 1e-14) { // try ẑ if ref was nearly parallel
    rx = 0; ry = 0; rz = 1;
    e1x =  by*rz - bz*ry;
    e1y =  bz*rx - bx*rz;
    e1z =  bx*ry - by*rx;
    n1  = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
    if (n1 < 1e-14) return; // pathological: give up quietly
  }
  e1x/=n1; e1y/=n1; e1z/=n1;
  const double e2x = bz*e1y - by*e1z;
  const double e2y = bx*e1z - bz*e1x;
  const double e2z = by*e1x - bx*e1y;

  // random direction in ⟂ plane
  const double ang = 2.0*M_PI * std::fmod(std::max(0.0, u01), 1.0);
  const double ux  = e1x*std::cos(ang) + e2x*std::sin(ang);
  const double uy  = e1y*std::cos(ang) + e2y*std::sin(ang);
  const double uz  = e1z*std::cos(ang) + e2z*std::sin(ang);

  // step length for 2D diffusion in the ⟂ plane
  const double step = std::sqrt(std::max(0.0, 4.0 * dt));

  // apply to Cartesian position
  Xn += step * ux;
  Yn += step * uy;
  Zn += step * uz;
}


double Update::normal01_from_uniforms_() {
  const double u1 = std::max(1e-12, ranmaster->uniform());
  const double u2 = std::max(1e-12, ranmaster->uniform());
  return std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2);
}

// ------- BGK collision step (member) -------
void Update::apply_bgk_collision_step(double dt,
                                      double R, double Z, double phi,
                                      double charge, double mass,
                                      double &vx, double &vy, double &vz)
{
  // 1) Background plasma at (R,Z)
  const PlasmaDataParams Pf = interpolatePlasma_RZ_clamped(R, Z, this->plasma_data);

  const double uflow_r = Pf.parr_flow_r;
  const double uflow_t = Pf.parr_flow_t;
  const double uflow_z = Pf.parr_flow_z;
  const double Te_eV   = Pf.temp_e;
  const double Ti_eV   = Pf.temp_i;
  const double ne_m3   = Pf.dens_e;
  if (Te_eV <= 0.0 || Ti_eV <= 0.0 || ne_m3 <= 0.0) return;

  // 2) Rotate flow to Cartesian
  const double c = std::cos(phi), s = std::sin(phi);
  const double ux =  uflow_r * c - uflow_t * s;
  const double uy =  uflow_r * s + uflow_t * c;
  const double uz =  uflow_z;

  // 3) Relative velocity
  double vrx = vx - ux;
  double vry = vy - uy;
  double vrz = vz - uz;
  double vrel = std::sqrt(std::max(1e-30, vrx*vrx + vry*vry + vrz*vrz));

  // 4) Constants from Update members
  const double eQ   = this->echarge;
  const double eps0 = this->epsilon_0;
  const double kB   = this->boltz;
  const double eV2K = this->ev2kelvin;
  const double me   = this->electron_mass;
  const double mp   = this->proton_mass;

  const double Z_bg = (double)this->plasma_background_charge; // dimensionless
  const double A_bg = (double)this->plasma_background_mass;   // amu
  const double M_bg = A_bg * mp;

  // 5) Debye length & Coulomb log
  const double lam_D = std::sqrt(std::max(0.0,
      eps0 * kB * eV2K * Te_eV / ( ne_m3 * (Z_bg*eQ)*(Z_bg*eQ) )));

  const double Z_imp  = charge;
  const double Lambda = std::max(1.0, 12.0 * M_PI * ne_m3 * lam_D*lam_D*lam_D / std::max(1e-30, Z_imp));
  const double lnL    = std::log(Lambda);

  // 6) GITR-like prefactors
  const double A_imp  = mass / mp; // amu
  const double pref    = 0.238762895 * (Z_imp*Z_imp) * lnL / std::max(1e-30, A_imp*A_imp);
  const double gam_eBG = std::max(0.0, pref);
  const double gam_iBG = std::max(0.0, pref * (Z_bg*Z_bg));

  // 7) Dimensionless speeds and special functions
  const double xx_i = vrel*vrel * M_bg / (2.0 * eQ * Ti_eV);
  auto safe_sqrt = [](double x){ return std::sqrt(std::max(0.0, x)); };

  const double psi_prime_i    = 2.0 * safe_sqrt(xx_i/M_PI) * std::exp(-xx_i);
  const double psi_psiprime_i = std::erf(safe_sqrt(xx_i));
  const double psi_i          = psi_psiprime_i - psi_prime_i;

  // 8) Base collision frequencies
  const double vrel3 = vrel*vrel*vrel;
  const double nu0_i = gam_eBG * ne_m3 / std::max(1e-30, vrel3);
  // const double nu0_e = gam_iBG * ne_m3 / std::max(1e-30, vrel3); // reserved

  const double nu_friction   = (1.0 + A_imp/std::max(1e-30, A_bg)) * psi_i * nu0_i;
  const double nu_deflection = 2.0 * (psi_psiprime_i - psi_i/std::max(1e-30, 2.0*xx_i)) * nu0_i;
  const double nu_parallel   = (psi_i / std::max(1e-30, xx_i)) * nu0_i;
  const double nu_energy     = 2.0 * ((A_imp/std::max(1e-30, A_bg)) * psi_i - psi_prime_i) * nu0_i;

  // 9) Substep for stability: keep max(nu)*dt_sub <= 0.2
  const double nu_max = std::max({ std::fabs(nu_friction),
                                   std::fabs(nu_deflection),
                                   std::fabs(nu_parallel),
                                   std::fabs(nu_energy) });
  const int nsub = std::max(1, (int)std::ceil(nu_max * dt / 0.2));
  const double dt_sub = dt / nsub;

  auto build_triad = [](double px, double py, double pz,
                        double &ex1x, double &ex1y, double &ex1z,
                        double &ex2x, double &ex2y, double &ex2z)
  {
    // perp1 = normalize(p × ẑ)
    ex1x =  py*0.0 - pz*1.0;
    ex1y =  pz*0.0 - px*0.0;
    ex1z =  px*1.0 - py*0.0;
    double n1 = std::sqrt(ex1x*ex1x + ex1y*ex1y + ex1z*ex1z);
    if (n1 < 1e-14) { // p ~ ẑ; use ŷ
      ex1x =  py*1.0 - pz*0.0;
      ex1y =  pz*0.0 - px*1.0;
      ex1z =  px*0.0 - py*0.0;
      n1 = std::sqrt(ex1x*ex1x + ex1y*ex1y + ex1z*ex1z);
      if (n1 < 1e-14) { ex1x=1.0; ex1y=0.0; ex1z=0.0; n1=1.0; }
    }
    ex1x/=n1; ex1y/=n1; ex1z/=n1;
    // perp2 = p × perp1
    ex2x = py*ex1z - pz*ex1y;
    ex2y = pz*ex1x - px*ex1z;
    ex2z = px*ex1y - py*ex1x;
  };

  for (int is=0; is<nsub; ++is) {
    // direction from current v_rel
    double px = vrx / vrel, py = vry / vrel, pz = vrz / vrel;
    double ex1x, ex1y, ex1z, ex2x, ex2y, ex2z;
    build_triad(px, py, pz, ex1x, ex1y, ex1z, ex2x, ex2y, ex2z);

    // randoms from member RNG
    const double n1  = normal01_from_uniforms_();
    const double n2  = normal01_from_uniforms_();
    const double xsi = ranmaster->uniform();
    const double cosX = std::min(1.0, std::cos(2.0*M_PI*xsi) - 0.0028);
    const double sinX = std::sin(2.0*M_PI*xsi);

    const double c_par   = n1 * std::sqrt(std::max(0.0, 2.0*nu_parallel  * dt_sub));
    const double c_perp1 = cosX * std::sqrt(std::max(0.0, 0.5*nu_deflection* dt_sub));
    const double c_perp2 = sinX * std::sqrt(std::max(0.0, 0.5*nu_deflection* dt_sub));

    double nuEdt = nu_energy * dt_sub;
    if (nuEdt < -1.0) nuEdt = -1.0;

    const double vNormEt = vrel * (1.0 - 0.5*nuEdt);
    const double vPar_x  = (1.0 + c_par) * px;
    const double vPar_y  = (1.0 + c_par) * py;
    const double vPar_z  = (1.0 + c_par) * pz;

    const double vPerp0_x = c_perp1*ex1x + c_perp2*ex2x;
    const double vPerp0_y = c_perp1*ex1y + c_perp2*ex2y;
    const double vPerp0_z = c_perp1*ex1z + c_perp2*ex2z;
    const double vPerp_x  = std::fabs(n2) * vPerp0_x;
    const double vPerp_y  = std::fabs(n2) * vPerp0_y;
    const double vPerp_z  = std::fabs(n2) * vPerp0_z;

    const double vPar2_x  = vrel * dt_sub * nu_friction * px;
    const double vPar2_y  = vrel * dt_sub * nu_friction * py;
    const double vPar2_z  = vrel * dt_sub * nu_friction * pz;

    // new relative velocity
    vrx = vNormEt*(vPar_x + vPerp_x) - vPar2_x;
    vry = vNormEt*(vPar_y + vPerp_y) - vPar2_y;
    vrz = vNormEt*(vPar_z + vPerp_z) - vPar2_z;
    vrel = std::sqrt(std::max(1e-30, vrx*vrx + vry*vry + vrz*vrz));
  }

  // add background flow back
  vx = vrx + ux;
  vy = vry + uy;
  vz = vrz + uz;

  if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
    vx = ux; vy = uy; vz = uz; // safety fallback
  }
}
