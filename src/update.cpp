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
#include "boris_grid.h"
#include "gca_pusher.h"
#include "random_mars.h"
#include "sheath_models.h"
#include "compute_sheath_geometry_grid.h"
#include "compute_plasma_fields.h"
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
  move_flag = 1;
  fstyle = NOFIELD;
  fieldID = NULL;
  efstyle = NOFIELD;
  efieldID = NULL;
  efieldfreq = 0;

  bfstyle = NOFIELD;
  bfieldID = NULL;
  bfieldfreq = 0;

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


  thermal_gradient_forces_flag = 0;
  boris_dump_flag = 0;
  boris_dump_every = 1;
  boris_subcycles = 1;
  boris_bad_dt_check = 1;
  boris_bad_dt_warned = 0;
  boris_bad_dt_limit = 0.1;

  gca_flag = 0;
  gca_switch_factor = 2.5;
  gca_plasma_cid = NULL;
  gca_plasma_cidx = -1;
  gca_x_custom = -1;
  gca_y_custom = -1;
  gca_z_custom = -1;
  gca_vpar_custom = -1;
  gca_mu_custom = -1;
  gca_on_custom = -1;

  cd_flag = 0;
  cd_nmax = 0;
  dx_cd = NULL;

  psi_reflect_flag = 0;
  psi_reflect_threshold = 1.0;
  psi_nw = psi_nh = 0;
  psi_axis = psi_bry = 0.0;
  psi_r_grid = NULL;
  psi_z_grid = NULL;
  psi_rz = NULL;

  sheath_flag = 0;
  sheath_geom_cid = NULL;
  sheath_plasma_cid = NULL;
  sheath_geom_cidx = -1;
  sheath_plasma_cidx = -1;
  sheath_model = 0;             // 0=borodkina, 1=coulette_manfredi
  sheath_dmax = 0.02;
  sheath_pot_mult = 2.5;
  sheath_mD_amu = 2.01410177811;
  sheath_kick = 0;

  plasma_cache_flag = 0;
  pc_te_custom = pc_ti_custom = pc_ne_custom = pc_ni_custom = -1;
  pc_vpar_custom = -1;
  pc_bx_custom = pc_by_custom = pc_bz_custom = -1;

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
  delete [] sheath_geom_cid;
  delete [] sheath_plasma_cid;
  delete [] gca_plasma_cid;
  memory->destroy(dx_cd);
  // psi_r_grid, psi_z_grid, psi_rz are owned by fix_reflect_psi, not freed here
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

  // Resolve per-particle sheath compute IDs
  if (sheath_flag) {
    sheath_geom_cidx = modify->find_compute(sheath_geom_cid);
    if (sheath_geom_cidx < 0)
      error->all(FLERR,"global sheath: geometry compute ID not found");
    if (!modify->compute[sheath_geom_cidx]->per_grid_flag)
      error->all(FLERR,"global sheath: geometry compute must be per-grid");

    sheath_plasma_cidx = modify->find_compute(sheath_plasma_cid);
    if (sheath_plasma_cidx < 0)
      error->all(FLERR,"global sheath: plasma compute ID not found");
    if (!modify->compute[sheath_plasma_cidx]->per_grid_flag)
      error->all(FLERR,"global sheath: plasma compute must be per-grid");
  }

  // Resolve GCA plasma/fields compute for grad(B)
  if (gca_flag) {
    if (!gca_plasma_cid)
      error->all(FLERR,"global gca requires plasma_compute ID");
    gca_plasma_cidx = modify->find_compute(gca_plasma_cid);
    if (gca_plasma_cidx < 0)
      error->all(FLERR,"global gca: plasma compute ID not found");
    if (!modify->compute[gca_plasma_cidx]->per_grid_flag)
      error->all(FLERR,"global gca: plasma compute must be per-grid");

    // ERO2.0-style persistent guiding-center state per particle.
    // Keep this state across timesteps to avoid re-initializing from
    // instantaneous gyromotion every step.
    const int custom_double = 1;
    if (gca_x_custom < 0) {
      gca_x_custom = particle->find_custom((char *) "gca_x");
      if (gca_x_custom < 0)
        gca_x_custom = particle->add_custom((char *) "gca_x", custom_double, 0);
    }
    if (gca_y_custom < 0) {
      gca_y_custom = particle->find_custom((char *) "gca_y");
      if (gca_y_custom < 0)
        gca_y_custom = particle->add_custom((char *) "gca_y", custom_double, 0);
    }
    if (gca_z_custom < 0) {
      gca_z_custom = particle->find_custom((char *) "gca_z");
      if (gca_z_custom < 0)
        gca_z_custom = particle->add_custom((char *) "gca_z", custom_double, 0);
    }
    if (gca_vpar_custom < 0) {
      gca_vpar_custom = particle->find_custom((char *) "gca_vpar");
      if (gca_vpar_custom < 0)
        gca_vpar_custom = particle->add_custom((char *) "gca_vpar", custom_double, 0);
    }
    if (gca_mu_custom < 0) {
      gca_mu_custom = particle->find_custom((char *) "gca_mu");
      if (gca_mu_custom < 0)
        gca_mu_custom = particle->add_custom((char *) "gca_mu", custom_double, 0);
    }
    if (gca_on_custom < 0) {
      gca_on_custom = particle->find_custom((char *) "gca_on");
      if (gca_on_custom < 0)
        gca_on_custom = particle->add_custom((char *) "gca_on", custom_double, 0);
    }
  }

  // Register per-particle plasma cache vectors.
  // Active when any plasma compute is available (sheath or GCA).
  {
    int plasma_cidx = -1;
    if (sheath_flag && sheath_plasma_cidx >= 0) plasma_cidx = sheath_plasma_cidx;
    else if (gca_flag && gca_plasma_cidx >= 0) plasma_cidx = gca_plasma_cidx;

    if (plasma_cidx >= 0) {
      const int custom_double = 1;
      auto reg = [&](int &idx, const char *name) {
        if (idx < 0) {
          idx = particle->find_custom((char *) name);
          if (idx < 0) idx = particle->add_custom((char *) name, custom_double, 0);
        }
      };
      reg(pc_te_custom,   "pc_te");
      reg(pc_ti_custom,   "pc_ti");
      reg(pc_ne_custom,   "pc_ne");
      reg(pc_ni_custom,   "pc_ni");
      reg(pc_vpar_custom, "pc_vpar");
      reg(pc_bx_custom,   "pc_bx");
      reg(pc_by_custom,   "pc_by");
      reg(pc_bz_custom,   "pc_bz");
      plasma_cache_flag = 1;
    }
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
  if (efstyle == GFIELD && efieldfreq == 0)
    modify->fix[efieldfix]->compute_field();

    // external per grid cell magnetic field
  if (bfstyle == GFIELD && bfieldfreq == 0)
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

    // cache plasma fields at particle positions (one query per particle)
    if (plasma_cache_flag) cache_plasma_particles();

    // move particles (skip when global move no)

    if (move_flag) {
      if (cellweightflag) particle->pre_weight();
      (this->*moveptr)();
      timer->stamp(TIME_MOVE);

      // communicate particles

      comm->migrate_particles(nmigrate,mlist);
      if (cellweightflag) particle->post_weight();
      timer->stamp(TIME_COMM);
    }

    // halt early if all particles have been absorbed/lost
    {
      bigint nlocal = particle->nlocal;
      bigint nglobal;
      MPI_Allreduce(&nlocal, &nglobal, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
      if (nglobal == 0) {
        // force final output before breaking
        output->next = ntimestep;
        output->write(ntimestep);
        update->nsteps = i;
        break;
      }
    }

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
   Cache plasma fields at every particle position using custom vectors.
   Called once per timestep before the move.  All consumers (Boris sheath,
   ADAS chemistry, Nanbu collisions) read from these cached values.
------------------------------------------------------------------------- */

void Update::cache_plasma_particles()
{
  // resolve the plasma compute (sheath or GCA — whichever is available)
  int plasma_cidx = -1;
  if (sheath_flag && sheath_plasma_cidx >= 0) plasma_cidx = sheath_plasma_cidx;
  else if (gca_flag && gca_plasma_cidx >= 0) plasma_cidx = gca_plasma_cidx;
  if (plasma_cidx < 0) return;

  Compute *c_base = modify->compute[plasma_cidx];
  auto *cp = dynamic_cast<ComputePlasmaFields *>(c_base);
  if (!cp) return;

  // ensure per-grid data is computed this timestep
  if (!(c_base->invoked_flag & 16)) {  // INVOKED_PER_GRID = 16
    c_base->compute_per_grid();
    c_base->invoked_flag |= 16;
  }

  // resolve per-particle custom vectors
  // Guard: bail out if custom indices are invalid, vectors are NULL,
  // or no local particles on this rank
  if (particle->nlocal == 0) return;
  if (pc_te_custom < 0 || pc_ti_custom < 0 || pc_ne_custom < 0 ||
      pc_ni_custom < 0 || pc_vpar_custom < 0 ||
      pc_bx_custom < 0 || pc_by_custom < 0 || pc_bz_custom < 0) return;
  if (particle->ewhich[pc_te_custom] < 0) return;

  double *te_vec   = particle->edvec[particle->ewhich[pc_te_custom]];
  double *ti_vec   = particle->edvec[particle->ewhich[pc_ti_custom]];
  double *ne_vec   = particle->edvec[particle->ewhich[pc_ne_custom]];
  double *ni_vec   = particle->edvec[particle->ewhich[pc_ni_custom]];
  double *vpar_vec = particle->edvec[particle->ewhich[pc_vpar_custom]];
  double *bx_vec   = particle->edvec[particle->ewhich[pc_bx_custom]];
  double *by_vec   = particle->edvec[particle->ewhich[pc_by_custom]];
  double *bz_vec   = particle->edvec[particle->ewhich[pc_bz_custom]];
  if (!te_vec || !ti_vec || !ne_vec || !ni_vec ||
      !vpar_vec || !bx_vec || !by_vec || !bz_vec) return;

  // Sheath geometry compute (for Boltzmann ne correction)
  ComputeSheathGeometryGrid *csg = nullptr;
  if (sheath_flag && sheath_geom_cidx >= 0) {
    Compute *cg = modify->compute[sheath_geom_cidx];
    csg = dynamic_cast<ComputeSheathGeometryGrid *>(cg);
  }

  Particle::OnePart *particles = particle->particles;
  Grid::ChildCell *cells = grid->cells;
  const int nlocal = particle->nlocal;
  const int dim = domain->dimension;

  for (int i = 0; i < nlocal; i++) {
    const double *x = particles[i].x;
    PlasmaFileParams pf = cp->query_plasma_at_point(x);

    te_vec[i]   = pf.temp_e;
    ti_vec[i]   = pf.temp_i;
    ne_vec[i]   = pf.dens_e;
    ni_vec[i]   = pf.dens_i;
    vpar_vec[i] = pf.parr_flow;

    MagneticFieldFileDataParams bf = cp->query_bfield_at_point(x);
    // store Cartesian B-field at particle position
    const double rx = x[0], ry = x[1];
    const double rmag = std::sqrt(rx*rx + ry*ry);
    double bx, by, bz;
    if (rmag > 1.0e-20 && dim == 3) {
      const double cphi = rx / rmag, sphi = ry / rmag;
      bx = bf.br * cphi - bf.bt * sphi;
      by = bf.br * sphi + bf.bt * cphi;
    } else {
      bx = bf.br;
      by = (dim == 3) ? 0.0 : bf.bt;
    }
    bz = bf.bz;
    bx_vec[i] = bx;
    by_vec[i] = by;
    bz_vec[i] = bz;

    // Boltzmann ne correction: ne_local = ne_upstream * exp(-phi/Te)
    // where phi = sheath potential drop at particle distance from wall
    if (csg && pf.temp_e > 0.0 && pf.dens_e > 0.0) {
      int icell = particles[i].icell;
      int gcell = icell;
      if (cells[icell].nsplit <= 0 && cells[icell].isplit >= 0)
        gcell = grid->sinfo[cells[icell].isplit].icell;

      int midx = csg->midx_grid[gcell];

      // refine to nearest surface at particle position (same as pusher)
      Grid::ChildCell *pc = &cells[gcell];
      if (pc->nsurf > 0) {
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          double d;
          if (dim == 2) {
            if (!(surf->lines[m].mask & sbit)) continue;
            Surf::Line *ln = &surf->lines[m];
            d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                          (x[1]-ln->p1[1])*ln->norm[1]);
          } else {
            if (!(surf->tris[m].mask & sbit)) continue;
            Surf::Tri *tr = &surf->tris[m];
            d = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                          (x[1]-tr->p1[1])*tr->norm[1] +
                          (x[2]-tr->p1[2])*tr->norm[2]);
          }
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0) {
        // get surface normal and reference point
        double sh_nx, sh_ny, sh_nz;
        double sref[3];
        if (dim == 2) {
          Surf::Line *ln = &surf->lines[midx];
          sh_nx = ln->norm[0]; sh_ny = ln->norm[1]; sh_nz = 0.0;
          sref[0] = 0.5*(ln->p1[0]+ln->p2[0]);
          sref[1] = 0.5*(ln->p1[1]+ln->p2[1]);
          sref[2] = 0.0;
        } else {
          Surf::Tri *tr = &surf->tris[midx];
          sh_nx = tr->norm[0]; sh_ny = tr->norm[1]; sh_nz = tr->norm[2];
          sref[0] = (tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0;
          sref[1] = (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0;
          sref[2] = (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0;
        }

        const double d_particle = std::fabs(
          (x[0]-sref[0])*sh_nx + (x[1]-sref[1])*sh_ny + (x[2]-sref[2])*sh_nz);

        if (d_particle > 0.0 && d_particle < sheath_dmax) {
          const double te = pf.temp_e;
          const double ti = pf.temp_i;
          const double ne = pf.dens_e;
          const double bmag = std::sqrt(bx*bx + by*by + bz*bz);

          double alpha_deg = 90.0;
          if (bmag > 0.0) {
            double bvec[3] = {bx, by, bz};
            double nvec[3] = {sh_nx, sh_ny, sh_nz};
            SheathModels::ChoduraMetrics cm =
              SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
            alpha_deg = cm.alpha_deg;
          }

          SheathModels::BorodkinaSheathResult sr;
          if (sheath_model == 1) {
            sr = SheathModels::coulette_manfredi_sheath_at_distance(
              d_particle, te, ti, ne, bmag,
              alpha_deg, sheath_mD_amu, sheath_pot_mult);
          } else {
            sr = SheathModels::borodkina_sheath_at_distance(
              d_particle, te, ti, ne, bmag,
              alpha_deg, sheath_mD_amu, sheath_pot_mult);
          }

          // Boltzmann: ne_local = ne * exp(-phi/Te), phi = esheath_eV (positive)
          if (sr.esheath_eV > 0.0 && te > 0.0) {
            ne_vec[i] = ne * std::exp(-sr.esheath_eV / te);
          }
        }
      }
    }
  }
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

  // Optional built-in per-particle first-hit tracking.
  // Attributes are created by Particle ctor and dumped via p_hit_flag/p_hit_surf_id.
  const int hit_flag_index = particle->find_custom((char *) "hit_flag");
  const int hit_surf_index = particle->find_custom((char *) "hit_surf_id");

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
  if (efstyle == GFIELD && efieldfreq && ((ntimestep-1) % efieldfreq == 0))
    modify->fix[efieldfix]->compute_field();

  // per-grid B
  if (bfstyle == GFIELD && bfieldfreq && ((ntimestep-1) % bfieldfreq == 0))
    modify->fix[bfieldfix]->compute_field();

  // per-grid ethermal
  if (ethermalstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ethermalfix]->compute_field();

  // per-grid ithermal
  if (ithermalstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0))
    modify->fix[ithermalfix]->compute_field();
  // Pre-compute sheath geometry and plasma BEFORE the particle loop.
  // Geometry is static (surfaces don't move) — compute once, reuse forever.
  // Plasma may update if coupled to a solver; for analytic profiles it's also static.
  if (sheath_flag && sheath_geom_cidx >= 0 && sheath_plasma_cidx >= 0) {
    Compute *cg = modify->compute[sheath_geom_cidx];
    if (cg->invoked_per_grid < 0) cg->compute_per_grid();  // only first time
    Compute *cp = modify->compute[sheath_plasma_cidx];
    if (cp->invoked_per_grid < 0) cp->compute_per_grid();   // only first time
  }

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

      // Use per-particle mass when available (e.g. evaporation updates),
      // fall back to species mass for legacy particles.
      double mass = (particles[i].mass > 0.0)
                    ? particles[i].mass
                    : species[particles[i].ispecies].mass;
      double charge = species[particles[i].ispecies].charge;
      
      // apply moveperturb() to PKEEP and PINSERT since are computing xnew
      // not to PENTRY,PEXIT since are just re-computing xnew of sender
      // set xnew[2] to linear move for axisymmetry, will be remapped later
      // let pflag = PEXIT persist to check during axisymmetric cell crossing
      // 

      if (pflag == PKEEP) {
        dtremain = dt;
        if (DIM == 1 || DIM == 2)
        {
          pusherBoris2D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3)
        {
          if (gca_flag)
            pusher_hybrid3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher_boris3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PINSERT) {
        dtremain = dt;
        if (DIM == 1 || DIM == 2) {
          pusherBoris2D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
        else if (DIM == 3) {
          if (gca_flag)
            pusher_hybrid3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
          else
            pusher_boris3D(i,particles[i].icell,dtremain,x,v,xnew,charge,mass);
        }
      } else if (pflag == PENTRY) {
        // printf("We are in PENTRY move\n");
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

      // Apply cross-field diffusion displacement (if active).
      // Only on the initial push (PKEEP/PINSERT), not on re-entries
      // after migration or surface collision (PENTRY/PEXIT/PSURF).
      // The displacement was pre-computed by fix cross_diffusion in
      // START_OF_STEP and stored in dx_cd[i].

      if (cd_flag && (pflag == PKEEP || pflag == PINSERT)) {
        xnew[0] += dx_cd[i][0];
        xnew[1] += dx_cd[i][1];
        if (DIM == 3) xnew[2] += dx_cd[i][2];
      }

      // Psi-based core boundary: check if xnew is inside the core
      // (psi_norm < threshold). If so, reflect the radial component of
      // both displacement and velocity so the particle bounces outward.
      // This check is done here (after Boris push, before surface tracing)
      // so that the reflected trajectory goes through proper surface checks.
      if (psi_reflect_flag && (pflag == PKEEP || pflag == PINSERT)) {
        double Rnew, Znew;
        if (DIM == 3) {
          Rnew = sqrt(xnew[0]*xnew[0] + xnew[1]*xnew[1]);
          Znew = xnew[2];
        } else {
          Rnew = xnew[0];
          Znew = xnew[1];
        }

        // Bilinear interpolation of psi_norm at xnew
        double psi_n = 1.0;
        if (psi_nw > 1 && psi_nh > 1 && psi_rz) {
          double Rc = Rnew;
          double Zc = Znew;
          if (Rc < psi_r_grid[0]) Rc = psi_r_grid[0];
          if (Rc > psi_r_grid[psi_nw-1]) Rc = psi_r_grid[psi_nw-1];
          if (Zc < psi_z_grid[0]) Zc = psi_z_grid[0];
          if (Zc > psi_z_grid[psi_nh-1]) Zc = psi_z_grid[psi_nh-1];

          double dr = psi_r_grid[1] - psi_r_grid[0];
          double dz = psi_z_grid[1] - psi_z_grid[0];
          int ii = (int)((Rc - psi_r_grid[0]) / dr);
          int jj = (int)((Zc - psi_z_grid[0]) / dz);
          if (ii < 0) ii = 0; if (ii > psi_nw-2) ii = psi_nw-2;
          if (jj < 0) jj = 0; if (jj > psi_nh-2) jj = psi_nh-2;
          double t = (Rc - psi_r_grid[ii]) / dr;
          double u = (Zc - psi_z_grid[jj]) / dz;
          if (t < 0.0) t = 0.0; if (t > 1.0) t = 1.0;
          if (u < 0.0) u = 0.0; if (u > 1.0) u = 1.0;

          double psi_val = (1-t)*(1-u)*psi_rz[jj*psi_nw+ii]
                         + t*(1-u)*psi_rz[jj*psi_nw+ii+1]
                         + (1-t)*u*psi_rz[(jj+1)*psi_nw+ii]
                         + t*u*psi_rz[(jj+1)*psi_nw+ii+1];
          double dpsi = psi_bry - psi_axis;
          if (fabs(dpsi) > 1e-30) psi_n = (psi_val - psi_axis) / dpsi;
        }

        if (psi_n < psi_reflect_threshold) {
          // Reject the move: particle stays at current position
          xnew[0] = x[0];
          xnew[1] = x[1];
          if (DIM == 3) xnew[2] = x[2];

          // Reverse radial velocity so particle moves outward next step
          if (DIM == 3) {
            double R0 = sqrt(x[0]*x[0] + x[1]*x[1]);
            if (R0 > 1e-10) {
              double cphi = x[0] / R0;
              double sphi = x[1] / R0;
              double vr = particles[i].v[0]*cphi + particles[i].v[1]*sphi;
              double vp = -particles[i].v[0]*sphi + particles[i].v[1]*cphi;
              vr = -vr;
              particles[i].v[0] = vr*cphi - vp*sphi;
              particles[i].v[1] = vr*sphi + vp*cphi;
            }
          } else {
            particles[i].v[0] = -particles[i].v[0];
          }
        }
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

              // Record the first explicit surface hit for this particle ID.
              if (hit_flag_index >= 0 && hit_surf_index >= 0) {
                int *hit_flag_vec = particle->eivec[particle->ewhich[hit_flag_index]];
                int *hit_surf_vec = particle->eivec[particle->ewhich[hit_surf_index]];
                if (hit_flag_vec && hit_surf_vec && hit_flag_vec[i] == 0) {
                  hit_flag_vec[i] = 1;
                  surfint sid = (DIM == 3) ? tris[minsurf].id : lines[minsurf].id;
                  hit_surf_vec[i] = static_cast<int>(sid);
                }
              }

              // --- Sheath kick: apply sheath energy as velocity boost at wall ---
              if (sheath_kick && sheath_flag &&
                  sheath_geom_cidx >= 0 && sheath_plasma_cidx >= 0) {
                // Get surface normal (outward, toward plasma)
                const double *snorm = (DIM == 3) ? tri->norm : line->norm;

                // Plasma conditions at particle position (point query)
                Compute *cp_base = modify->compute[sheath_plasma_cidx];
                auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
                if (cp) {
                  PlasmaFileParams sk_pf = cp->query_plasma_at_point(x);
                  const double sk_te = sk_pf.temp_e;
                  const double sk_ti = sk_pf.temp_i;
                  if (sk_te > 0.0) {
                    constexpr double QE_kick = 1.602176634e-19;
                    constexpr double ME_kick = 9.1093837015e-31;
                    constexpr double AMU_kick = 1.66053906660e-27;
                    constexpr double PI_kick = 3.14159265358979323846;

                    const double mD_kg = sheath_mD_amu * AMU_kick;

                    // Floating potential: phi = 0.5*ln(mD/(2*pi*me)/(1+Ti/Te)) * Te (eV)
                    const double ti_ratio = (sk_ti > 0.0) ? (sk_ti / sk_te) : 0.0;
                    const double phi_float_mult =
                      0.5 * std::log(mD_kg / (2.0 * PI_kick * ME_kick) / (1.0 + ti_ratio));
                    const double phi_eV = (sheath_pot_mult > 0.0)
                      ? (sheath_pot_mult * sk_te)
                      : (std::max(phi_float_mult, 0.0) * sk_te);

                    // Particle charge and mass (from species table, not particle struct)
                    const int isp = particles[i].ispecies;
                    const double Z = std::abs(particle->species[isp].charge);
                    const double pmass = particle->species[isp].mass;

                    if (Z > 0.0 && pmass > 0.0 && phi_eV > 0.0) {
                      // Normal component of velocity (toward wall = negative v·n)
                      const double vdotn = v[0]*snorm[0] + v[1]*snorm[1] + v[2]*snorm[2];
                      // v·n < 0 means particle moving toward wall (against outward normal)
                      const double vn_toward = -vdotn;  // positive = toward wall

                      // New normal speed: sqrt(vn^2 + 2*Z*e*phi/m)
                      const double dE_J = Z * QE_kick * phi_eV;
                      const double vn_new = std::sqrt(vn_toward * vn_toward + 2.0 * dE_J / pmass);
                      const double dv = vn_new - vn_toward;  // always >= 0

                      // Add kick toward wall (against outward normal)
                      v[0] -= dv * snorm[0];
                      v[1] -= dv * snorm[1];
                      v[2] -= dv * snorm[2];
                    }
                  }
                }
              }

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

              // check if surf_collide teleported particle outside current cell
              // (occurs with toroidal periodic boundary collision model)

              if (DIM == 3 &&
                  (x[0] < lo[0] || x[0] > hi[0] ||
                   x[1] < lo[1] || x[1] > hi[1] ||
                   x[2] < lo[2] || x[2] > hi[2])) {
                int newcell = grid->id_find_child(0,0,
                                  domain->boxlo,domain->boxhi,x);
                if (newcell >= 0) {
                  if (cells[newcell].proc == me) {
                    if (DIM == 3 && SURF) {
                      if (cells[newcell].nsplit > 1 &&
                          cells[newcell].nsurf >= 0)
                        newcell = split3d(newcell,x);
                    }
                    icell = newcell;
                    lo = cells[icell].lo;
                    hi = cells[icell].hi;
                    neigh = cells[icell].neigh;
                    nmask = cells[icell].nmask;
                    exclude = -1;
                    nscollide_one++;
                    continue;
                  } else {
                    icell = newcell;
                    particles[i].icell = icell;
                    particles[i].flag = PEXIT;
                    particles[i].dtremain = dtremain;
                    entryexit = 1;
                    nscollide_one++;
                    break;
                  }
                } else {
                  particles[i].flag = PDISCARD;
                  nscollide_one++;
                  break;
                }
              }

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

void Update::field_per_particle(int i, int icell, double dt, double *x, double *v)
{
  double dtsq = 0.5*dt*dt;
  double **array = modify->fix[ifieldfix]->array_particle;

  int icol = 0;
  if (field_active[0]) {
    x[0] += dtsq*array[i][icol];
    v[0] += dt*array[i][icol];
    icol++;
  }
  if (field_active[1]) {
    x[1] += dtsq*array[i][icol];
    v[1] += dt*array[i][icol];
    icol++;
  }
  if (field_active[2]) {
    x[2] += dtsq*array[i][icol];
    v[2] += dt*array[i][icol];
    icol++;
  }
};

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
   Boris pusher for 2D (x,y) positions with full 3-component velocity
------------------------------------------------------------------------- */

void Update::pusherBoris2D(int i, int icell, double dt,
                           double *x, double *v, double *xnew,
                           double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Boris pusher requires positive particle mass");
  // Fast path for neutrals: pure advection, no E/B field reads or Boris algebra.
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * echarge) / mass;
  const int nsub = (boris_subcycles > 0) ? boris_subcycles : 1;
  const double dt_sub = dt / static_cast<double>(nsub);

  double xcur[2] = {x[0], x[1]};
  double zcur = x[2];
  double vcur[3] = {v[0], v[1], v[2]};
  double E[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};

  // Field arrays are precomputed before particle motion. Cache them once per
  // Boris call to avoid redundant per-subcycle reads.
  if (eperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[efieldfix], (efstyle == GFIELD),
                                   efield_active, i, icell, E);
  if (bperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[bfieldfix], (bfstyle == GFIELD),
                                   bfield_active, i, icell, B);

  for (int isub = 0; isub < nsub; isub++) {
    if (boris_bad_dt_check && !boris_bad_dt_warned) {
      const double bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
      const double bad = std::fabs(qm) * bmag * dt_sub;
      if (bad > boris_bad_dt_limit) {
        if (comm->me == 0)
          error->warning(FLERR, "OpenEdge Boris warning: |q/m|*|B|*dt_sub is large");
        boris_bad_dt_warned = 1;
      }
    }

    double xold[2] = {xcur[0], xcur[1]};

    BorisGrid::push_velocity(qm, dt_sub, E, B, vcur);
    xcur[0] += vcur[0] * dt_sub;
    xcur[1] += vcur[1] * dt_sub;
    zcur += vcur[2] * dt_sub;

    if (boris_dump_flag && (ntimestep % boris_dump_every == 0)) {
      const double emag = std::sqrt(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]);
      if (comm->me == 0 && i == 0 && emag > 0.0) {
        printf("boris2D step=%lld icell=%d sub=%d/%d qm=%g E=(%g,%g,%g) B=(%g,%g,%g)\n",
               (long long) ntimestep, icell, isub+1, nsub, qm,
               E[0], E[1], E[2], B[0], B[1], B[2]);
      }
    }

    // Per-subcycle surface crossing guard (2D version).
    // Same logic as 3D guard: check segment xold→xcur against every line
    // surface in the particle's grid cell.  If crossing detected, stop
    // subcycling and return to move loop for collision handling.

    if (nsub > 1) {
      int gcell = icell;
      Grid::ChildCell *cells_local = grid->cells;
      if (cells_local[icell].nsplit <= 0 && cells_local[icell].isplit >= 0)
        gcell = grid->sinfo[cells_local[icell].isplit].icell;

      int nsurf_cell = cells_local[gcell].nsurf;
      if (nsurf_cell > 0) {
        surfint *csurfs_local = cells_local[gcell].csurfs;
        Surf::Line *lines_local = surf->lines;
        double xc[2];
        double param;
        int side;
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = static_cast<int>(csurfs_local[m]);
          Surf::Line *line = &lines_local[isurf];
          if (Geometry::line_line_intersect(xold, xcur,
                                             line->p1, line->p2,
                                             line->norm, xc, param, side)) {
            v[0] = vcur[0];
            v[1] = vcur[1];
            v[2] = vcur[2];
            xnew[0] = xcur[0];
            xnew[1] = xcur[1];
            xnew[2] = zcur;
            return;
          }
        }
      }
    }
  }

  v[0] = vcur[0];
  v[1] = vcur[1];
  v[2] = vcur[2];
  xnew[0] = xcur[0];
  xnew[1] = xcur[1];
  xnew[2] = zcur;
}

/* ----------------------------------------------------------------------
   Boris pusher for 3D cartesian coordinates
------------------------------------------------------------------------- */

void Update::pusher_boris3D(int i, int icell, double dt,
                            double *x, double *v, double *xnew,
                            double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Boris pusher requires positive particle mass");

  // Fast path for neutrals: pure advection, no E/B field reads or Boris algebra.
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * echarge) / mass;
  const int nsub = (boris_subcycles > 0) ? boris_subcycles : 1;
  const double dt_sub = dt / static_cast<double>(nsub);

  double xcur[3] = {x[0], x[1], x[2]};
  double vcur[3] = {v[0], v[1], v[2]};

  // --- Pre-fetch per-particle sheath data from grid-cached computes ---
  // Grid cell's cached nearest-surface geometry and plasma parameters.
  // These are invariant during subcycling (grid data doesn't change mid-step).

  double sh_nx = 0.0, sh_ny = 0.0, sh_nz = 0.0;  // raw surface normal (unit)
  double sh_sref[3] = {0.0, 0.0, 0.0};  // reference point on nearest surface element
  double sh_te = 0.0, sh_ti = 0.0, sh_ne = 0.0;
  double sh_bmag = 0.0, sh_alpha_deg = 90.0;
  int sh_active = 0;

  if (sheath_flag && sheath_geom_cidx >= 0 && sheath_plasma_cidx >= 0) {
    Compute *cg = modify->compute[sheath_geom_cidx];

    // If particle is in a sub-cell (split cell), resolve to parent cell
    // for geometry/plasma lookup — the compute skips sub-cells.
    int gcell = icell;
    Grid::ChildCell *cells_tmp = grid->cells;
    if (cells_tmp[icell].nsplit <= 0 && cells_tmp[icell].isplit >= 0)
      gcell = grid->sinfo[cells_tmp[icell].isplit].icell;

    // Get nearest surface element index from geometry compute
    auto *csg = dynamic_cast<ComputeSheathGeometryGrid *>(cg);
    if (csg) {
      int midx = csg->midx_grid[gcell];

      // When the parent cell contains surface elements, refine midx by
      // finding the surface nearest to the PARTICLE position (not the
      // cell center used by the compute).  This fixes wrong-face
      // selection when a thin slab (top+bottom+side faces) intersects a
      // single cell and the cell center sits between the faces.
      Grid::ChildCell *pc = &grid->cells[gcell];
      if (pc->nsurf > 0) {
        const int dim = domain->dimension;
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          double d;
          if (dim == 2) {
            if (!(surf->lines[m].mask & sbit)) continue;
            Surf::Line *ln = &surf->lines[m];
            d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                          (x[1]-ln->p1[1])*ln->norm[1]);
          } else {
            if (!(surf->tris[m].mask & sbit)) continue;
            Surf::Tri *tr = &surf->tris[m];
            d = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                          (x[1]-tr->p1[1])*tr->norm[1] +
                          (x[2]-tr->p1[2])*tr->norm[2]);
          }
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0) {
        // Use RAW triangle/line normal directly from the surface element,
        // not the per-cell flipped version.  This avoids normal sign flips
        // in split cells where the cell center is on the opposite side of
        // the surface from the particle.
        if (domain->dimension == 2) {
          Surf::Line *ln = &surf->lines[midx];
          sh_nx = ln->norm[0];
          sh_ny = ln->norm[1];
          sh_nz = 0.0;
          sh_sref[0] = 0.5*(ln->p1[0]+ln->p2[0]);
          sh_sref[1] = 0.5*(ln->p1[1]+ln->p2[1]);
          sh_sref[2] = 0.0;
        } else {
          Surf::Tri *tr = &surf->tris[midx];
          sh_nx = tr->norm[0];
          sh_ny = tr->norm[1];
          sh_nz = tr->norm[2];
          sh_sref[0] = (tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0;
          sh_sref[1] = (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0;
          sh_sref[2] = (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0;
        }
        const double nmag = std::sqrt(sh_nx*sh_nx + sh_ny*sh_ny + sh_nz*sh_nz);
        if (nmag > 0.0) {
          sh_nx /= nmag;  sh_ny /= nmag;  sh_nz /= nmag;
        }

        // Read pre-computed plasma arrays
        Compute *cp_base = modify->compute[sheath_plasma_cidx];
        auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
        if (cp) {
          sh_te = cp->plasma_arr[gcell].temp_e;
          sh_ti = cp->plasma_arr[gcell].temp_i;
          sh_ne = cp->plasma_arr[gcell].dens_e;
          const double br = cp->mag_arr[gcell].br;
          const double bt = cp->mag_arr[gcell].bt;
          const double bz = cp->mag_arr[gcell].bz;

          // Convert cylindrical B to Cartesian at particle position
          const double rx = x[0], ry = x[1];
          const double rmag = std::sqrt(rx*rx + ry*ry);
          double bvec[3];
          if (rmag > 1.0e-20) {
            const double cphi = rx / rmag, sphi = ry / rmag;
            bvec[0] = br * cphi - bt * sphi;
            bvec[1] = br * sphi + bt * cphi;
            bvec[2] = bz;
          } else {
            bvec[0] = br;  bvec[1] = 0.0;  bvec[2] = bz;
          }
          sh_bmag = std::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1] + bvec[2]*bvec[2]);

          // Chodura angle: angle between B and surface normal
          // (chodura_metrics uses abs(B·n) so result is independent of normal sign)
          if (sh_bmag > 0.0) {
            double nvec[3] = {sh_nx, sh_ny, sh_nz};
            SheathModels::ChoduraMetrics cm =
              SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
            sh_alpha_deg = cm.alpha_deg;
          }

          sh_active = (sh_te > 0.0 && sh_ne > 0.0);
        }
      }
    }
  }

  // Record which side of the wall the particle starts on (sign of d_raw).
  // During subcycling, only apply sheath E-field while particle remains on
  // this side.  If it overshoots past the wall, skip E-field to prevent
  // reverse-field deceleration that causes energy loss.
  double sh_d0_sign = 0.0;
  if (sh_active) {
    const double d0 =
      (xcur[0] - sh_sref[0]) * sh_nx
    + (xcur[1] - sh_sref[1]) * sh_ny
    + (xcur[2] - sh_sref[2]) * sh_nz;
    sh_d0_sign = (d0 >= 0.0) ? 1.0 : -1.0;
  }

  for (int isub = 0; isub < nsub; isub++) {
    double E[3] = {0.0, 0.0, 0.0};
    double B[3] = {0.0, 0.0, 0.0};

    if (eperturbflag)
      BorisGrid::read_field_from_fix(modify->fix[efieldfix], (efstyle == GFIELD),
                                     efield_active, i, icell, E);
    if (bperturbflag)
      BorisGrid::read_field_from_fix(modify->fix[bfieldfix], (bfstyle == GFIELD),
                                     bfield_active, i, icell, B);

    // Per-particle sheath E-field: evaluate at particle position using
    // grid-cached geometry (which surface, normal) and plasma (Te, ne, B).
    // Distance is computed directly from particle to surface plane via dot
    // product.  The signed distance determines both |d| for the model and
    // the E-field direction (always toward the wall from whichever side).
    if (sh_active && !sheath_kick) {
      const double d_raw =
        (xcur[0] - sh_sref[0]) * sh_nx
      + (xcur[1] - sh_sref[1]) * sh_ny
      + (xcur[2] - sh_sref[2]) * sh_nz;
      const double d_particle = std::fabs(d_raw);

      // Skip if particle overshot past wall (sign flipped from initial side)
      const double d_sign = (d_raw >= 0.0) ? 1.0 : -1.0;
      if (d_sign == sh_d0_sign && d_particle > 0.0 && d_particle < sheath_dmax) {
        double emag = 0.0;
        if (sheath_model == 1) {
          SheathModels::BorodkinaSheathResult sr =
            SheathModels::coulette_manfredi_sheath_at_distance(
              d_particle, sh_te, sh_ti, sh_ne, sh_bmag,
              sh_alpha_deg, sheath_mD_amu, sheath_pot_mult);
          emag = sr.emag_vpm;
        } else {
          SheathModels::BorodkinaSheathResult sr =
            SheathModels::borodkina_sheath_at_distance(
              d_particle, sh_te, sh_ti, sh_ne, sh_bmag,
              sh_alpha_deg, sheath_mD_amu, sheath_pot_mult);
          emag = sr.emag_vpm;
        }
        // E-field points toward wall (decreasing |d|):
        //   d_raw > 0 → particle on +nhat side → E along -nhat
        //   d_raw < 0 → particle on -nhat side → E along +nhat
        const double dir = (d_raw > 0.0) ? -1.0 : 1.0;
        E[0] += dir * emag * sh_nx;
        E[1] += dir * emag * sh_ny;
        E[2] += dir * emag * sh_nz;
      }
    }
// printf("E field due to sheath is %g %g %g\n", E[0], E[1], E[2]);

    if (boris_bad_dt_check && !boris_bad_dt_warned) {
      const double bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
      const double bad = std::fabs(qm) * bmag * dt_sub;
      if (bad > boris_bad_dt_limit) {
        if (comm->me == 0)
          error->warning(FLERR, "OpenEdge Boris warning: |q/m|*|B|*dt_sub is large");
        boris_bad_dt_warned = 1;
      }
    }

    double xold[3] = {xcur[0], xcur[1], xcur[2]};

    BorisGrid::push_velocity(qm, dt_sub, E, B, vcur);
    xcur[0] += vcur[0] * dt_sub;
    xcur[1] += vcur[1] * dt_sub;
    xcur[2] += vcur[2] * dt_sub;

    if (boris_dump_flag && (ntimestep % boris_dump_every == 0)) {
      const double emag = std::sqrt(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]);
      if (comm->me == 0 && i == 0 && emag > 0.0) {
        printf("boris3D step=%lld icell=%d sub=%d/%d qm=%g E=(%g,%g,%g) B=(%g,%g,%g)\n",
               (long long) ntimestep, icell, isub+1, nsub, qm,
               E[0], E[1], E[2], B[0], B[1], B[2]);
      }
    }

    // Per-subcycle surface crossing guard.
    // When subcycling, the move loop only sees the straight line from x
    // (start of timestep) to xnew (end of all subcycles).  If the curved
    // gyro-orbit crosses a surface during an intermediate subcycle but the
    // endpoints are on the same side, the crossing is invisible to the
    // move loop and the particle leaks through.
    //
    // Fix: after each subcycle, test the straight-line segment xold→xcur
    // against every triangle in the particle's grid cell.  If a crossing
    // is found, stop subcycling immediately and return xcur to the move
    // loop.  The move loop's straight-line check from x to xnew will then
    // see the particle on the far side of the surface and handle the
    // collision normally.
    //
    // This check is skipped when nsub==1 (no subcycling) since the move
    // loop already handles that single segment.

    if (nsub > 1) {
      int gcell = icell;
      Grid::ChildCell *cells_local = grid->cells;
      if (cells_local[icell].nsplit <= 0 && cells_local[icell].isplit >= 0)
        gcell = grid->sinfo[cells_local[icell].isplit].icell;

      int nsurf_cell = cells_local[gcell].nsurf;
      if (nsurf_cell > 0) {
        surfint *csurfs_local = cells_local[gcell].csurfs;
        Surf::Tri *tris_local = surf->tris;
        double xc[3];
        double param;
        int side;
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = static_cast<int>(csurfs_local[m]);
          Surf::Tri *tri = &tris_local[isurf];
          if (Geometry::line_tri_intersect(xold, xcur,
                                           tri->p1, tri->p2, tri->p3,
                                           tri->norm, xc, param, side)) {
            v[0] = vcur[0];
            v[1] = vcur[1];
            v[2] = vcur[2];
            xnew[0] = xcur[0];
            xnew[1] = xcur[1];
            xnew[2] = xcur[2];
            return;
          }
        }
      }
    }
  }

  v[0] = vcur[0];
  v[1] = vcur[1];
  v[2] = vcur[2];
  xnew[0] = xcur[0];
  xnew[1] = xcur[1];
  xnew[2] = xcur[2];
}

/* ----------------------------------------------------------------------
   Hybrid Boris/GCA 3D pusher (ERO2.0-style)
   Uses full Boris when Larmor radius is well-resolved by the B gradient
   scale, and switches to GCA when the gyration is fast (small rho_L).
   Criterion: use GCA when L_B < switch_factor * rho_L
   where L_B = B / |grad B| and rho_L = v_perp / (|q/m| * B)
------------------------------------------------------------------------- */

void Update::pusher_hybrid3D(int i, int icell, double dt,
                              double *x, double *v, double *xnew,
                              double charge, double mass)
{
  if (mass <= 0.0) error->all(FLERR, "Hybrid pusher requires positive particle mass");

  // Neutrals: pure advection
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt;
    xnew[1] = x[1] + v[1] * dt;
    xnew[2] = x[2] + v[2] * dt;
    return;
  }

  const double qm = (charge * echarge) / mass;
  const double qm_abs = std::fabs(qm);

  double *gca_x_vec = NULL;
  double *gca_y_vec = NULL;
  double *gca_z_vec = NULL;
  double *gca_vpar_vec = NULL;
  double *gca_mu_vec = NULL;
  double *gca_on_vec = NULL;
  if (gca_x_custom >= 0 && gca_y_custom >= 0 && gca_z_custom >= 0 &&
      gca_vpar_custom >= 0 && gca_mu_custom >= 0 && gca_on_custom >= 0) {
    gca_x_vec = particle->edvec[particle->ewhich[gca_x_custom]];
    gca_y_vec = particle->edvec[particle->ewhich[gca_y_custom]];
    gca_z_vec = particle->edvec[particle->ewhich[gca_z_custom]];
    gca_vpar_vec = particle->edvec[particle->ewhich[gca_vpar_custom]];
    gca_mu_vec = particle->edvec[particle->ewhich[gca_mu_custom]];
    gca_on_vec = particle->edvec[particle->ewhich[gca_on_custom]];
  }
  const bool have_gca_state =
    (gca_x_vec && gca_y_vec && gca_z_vec && gca_vpar_vec && gca_mu_vec && gca_on_vec);

  // --- Read E and B fields ---
  double E[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};

  if (eperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[efieldfix], (efstyle == GFIELD),
                                   efield_active, i, icell, E);

  ComputePlasmaFields *cp_bfield = NULL;
  if (gca_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[gca_plasma_cidx];
    cp_bfield = dynamic_cast<ComputePlasmaFields *>(cp_base);
  }

  MagneticFieldFileDataParams Bcyl{};
  bool have_point_b = false;
  if (cp_bfield) {
    Bcyl = cp_bfield->query_bfield_at_point(x);
    if (Bcyl.Bmag > 0.0) {
      have_point_b = true;
      if (domain->dimension == 2) {
        B[0] = Bcyl.br;
        B[1] = Bcyl.bz;
        B[2] = Bcyl.bt;
      } else {
        const double rx = x[0], ry = x[1];
        const double rxy = std::sqrt(rx*rx + ry*ry);
        double cphi = 1.0, sphi = 0.0;
        if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
        B[0] = Bcyl.br * cphi - Bcyl.bt * sphi;
        B[1] = Bcyl.br * sphi + Bcyl.bt * cphi;
        B[2] = Bcyl.bz;
      }
    }
  }

  if (!have_point_b && bperturbflag)
    BorisGrid::read_field_from_fix(modify->fix[bfieldfix], (bfstyle == GFIELD),
                                   bfield_active, i, icell, B);

  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

  // --- Read grad(|B|) from point-query if available, else cell cache ---
  double gradBmag_cart[3] = {0.0, 0.0, 0.0};
  double kappa_cart[3] = {0.0, 0.0, 0.0};
  double curlb_cart[3] = {0.0, 0.0, 0.0};
  double gradBmag_magnitude = 0.0;

  if (Bmag > 0.0) {
    if (have_point_b) {
      if (domain->dimension == 2) {
        // 2D slots: [0]=R, [1]=Z, [2]=toroidal (=0 for axisymmetric)
        gradBmag_cart[0] = Bcyl.dBmag_dr;
        gradBmag_cart[1] = Bcyl.dBmag_dz;
        gradBmag_cart[2] = 0.0;
      } else {
        const double rx = x[0], ry = x[1];
        const double rxy = std::sqrt(rx*rx + ry*ry);
        if (rxy > 1.0e-20) {
          const double cphi = rx / rxy, sphi = ry / rxy;
          gradBmag_cart[0] = Bcyl.dBmag_dr * cphi;
          gradBmag_cart[1] = Bcyl.dBmag_dr * sphi;
        } else {
          gradBmag_cart[0] = Bcyl.dBmag_dr;
          gradBmag_cart[1] = 0.0;
        }
        gradBmag_cart[2] = Bcyl.dBmag_dz;
      }
    }
    // No cell-cached fallback: if point query failed, gradBmag stays zero.

    gradBmag_magnitude = std::sqrt(gradBmag_cart[0]*gradBmag_cart[0] +
                                   gradBmag_cart[1]*gradBmag_cart[1] +
                                   gradBmag_cart[2]*gradBmag_cart[2]);

    // Compute kappa and curl(b̂) at particle position from point-query
    // B-component derivatives, avoiding cell-center geom_arr cache.
    if (have_point_b && Bcyl.Bmag > 0.0) {
      const double invBm = 1.0 / Bcyl.Bmag;
      const double bR = Bcyl.br * invBm;
      const double bphi = Bcyl.bt * invBm;
      const double bZ = Bcyl.bz * invBm;

      // ∂b̂_i/∂x = (1/|B|)(∂B_i/∂x - b̂_i ∂|B|/∂x)
      const double dbR_dR = invBm * (Bcyl.dBr_dr - bR * Bcyl.dBmag_dr);
      const double dbR_dZ = invBm * (Bcyl.dBr_dz - bR * Bcyl.dBmag_dz);
      const double dbphi_dR = invBm * (Bcyl.dBt_dr - bphi * Bcyl.dBmag_dr);
      const double dbphi_dZ = invBm * (Bcyl.dBt_dz - bphi * Bcyl.dBmag_dz);
      const double dbZ_dR = invBm * (Bcyl.dBz_dr - bZ * Bcyl.dBmag_dr);
      const double dbZ_dZ = invBm * (Bcyl.dBz_dz - bZ * Bcyl.dBmag_dz);

      double R_pt;
      if (domain->dimension == 2) R_pt = x[0];
      else R_pt = std::sqrt(x[0]*x[0] + x[1]*x[1]);
      if (R_pt < 1.0e-10) R_pt = 1.0e-10;
      const double invR_pt = 1.0 / R_pt;

      // κ = (b̂·∇)b̂ in cylindrical (axisymmetric, ∂/∂φ = 0)
      const double kR = bR * dbR_dR + bZ * dbR_dZ - bphi * bphi * invR_pt;
      const double kphi = bR * dbphi_dR + bZ * dbphi_dZ + bR * bphi * invR_pt;
      const double kZ = bR * dbZ_dR + bZ * dbZ_dZ;

      // curl(b̂) in cylindrical (axisymmetric, ∂/∂φ = 0)
      const double cR = -dbphi_dZ;
      const double cphi_c = dbR_dZ - dbZ_dR;
      const double cZ = bphi * invR_pt + dbphi_dR;

      if (domain->dimension == 2) {
        // 2D slots: [0]=R, [1]=Z, [2]=toroidal
        kappa_cart[0] = kR;   kappa_cart[1] = kZ;   kappa_cart[2] = kphi;
        curlb_cart[0] = cR;   curlb_cart[1] = cZ;   curlb_cart[2] = cphi_c;
      } else {
        const double rx = x[0], ry = x[1];
        const double rxy = std::sqrt(rx*rx + ry*ry);
        double cphi_a = 1.0, sphi_a = 0.0;
        if (rxy > 1.0e-20) { cphi_a = rx / rxy; sphi_a = ry / rxy; }

        kappa_cart[0] = kR * cphi_a - kphi * sphi_a;
        kappa_cart[1] = kR * sphi_a + kphi * cphi_a;
        kappa_cart[2] = kZ;

        curlb_cart[0] = cR * cphi_a - cphi_c * sphi_a;
        curlb_cart[1] = cR * sphi_a + cphi_c * cphi_a;
        curlb_cart[2] = cZ;
      }
    }
    // No cell-cached fallback: if point query failed, kappa/curl_b stay zero.
  }

  // --- Per-particle sheath E-field (same approach as boris3D) ---
  if (sheath_flag && !sheath_kick && sheath_geom_cidx >= 0 && sheath_plasma_cidx >= 0) {
    Compute *cg = modify->compute[sheath_geom_cidx];

    // Resolve sub-cell to parent for geometry/plasma lookup
    int gcell = icell;
    Grid::ChildCell *cells_h = grid->cells;
    if (cells_h[icell].nsplit <= 0 && cells_h[icell].isplit >= 0)
      gcell = grid->sinfo[cells_h[icell].isplit].icell;

    auto *csg = dynamic_cast<ComputeSheathGeometryGrid *>(cg);
    if (csg) {
      int midx = csg->midx_grid[gcell];

      // Refine midx using particle position when cell contains surfaces
      Grid::ChildCell *pc = &grid->cells[gcell];
      if (pc->nsurf > 0) {
        const int dim = domain->dimension;
        const int sbit = csg->sgroupbit;
        surfint *cs = pc->csurfs;
        double best_d = 1.0e20;
        int best_m = -1;
        for (int j = 0; j < pc->nsurf; j++) {
          int m = static_cast<int>(cs[j]);
          double d;
          if (dim == 2) {
            if (!(surf->lines[m].mask & sbit)) continue;
            Surf::Line *ln = &surf->lines[m];
            d = std::fabs((x[0]-ln->p1[0])*ln->norm[0] +
                          (x[1]-ln->p1[1])*ln->norm[1]);
          } else {
            if (!(surf->tris[m].mask & sbit)) continue;
            Surf::Tri *tr = &surf->tris[m];
            d = std::fabs((x[0]-tr->p1[0])*tr->norm[0] +
                          (x[1]-tr->p1[1])*tr->norm[1] +
                          (x[2]-tr->p1[2])*tr->norm[2]);
          }
          if (d < best_d) { best_d = d; best_m = m; }
        }
        if (best_m >= 0) midx = best_m;
      }

      if (midx >= 0) {
        // Use RAW triangle/line normal from the surface element
        double sh_nx, sh_ny, sh_nz;
        double sref[3];
        if (domain->dimension == 2) {
          Surf::Line *ln = &surf->lines[midx];
          sh_nx = ln->norm[0]; sh_ny = ln->norm[1]; sh_nz = 0.0;
          sref[0] = 0.5*(ln->p1[0]+ln->p2[0]);
          sref[1] = 0.5*(ln->p1[1]+ln->p2[1]);
          sref[2] = 0.0;
        } else {
          Surf::Tri *tr = &surf->tris[midx];
          sh_nx = tr->norm[0]; sh_ny = tr->norm[1]; sh_nz = tr->norm[2];
          sref[0] = (tr->p1[0]+tr->p2[0]+tr->p3[0]) / 3.0;
          sref[1] = (tr->p1[1]+tr->p2[1]+tr->p3[1]) / 3.0;
          sref[2] = (tr->p1[2]+tr->p2[2]+tr->p3[2]) / 3.0;
        }
        const double nmag = std::sqrt(sh_nx*sh_nx + sh_ny*sh_ny + sh_nz*sh_nz);
        if (nmag > 0.0) { sh_nx /= nmag; sh_ny /= nmag; sh_nz /= nmag; }

        // Signed distance: positive = particle on +nhat side
        const double d_raw =
          (x[0]-sref[0])*sh_nx + (x[1]-sref[1])*sh_ny + (x[2]-sref[2])*sh_nz;
        const double d_particle = std::fabs(d_raw);

        if (d_particle > 0.0 && d_particle < sheath_dmax) {
          Compute *cp_base = modify->compute[sheath_plasma_cidx];
          auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
          if (cp) {
            // Point-query plasma data at particle position
            PlasmaFileParams sh_pf = cp->query_plasma_at_point(x);
            const double sh_te = sh_pf.temp_e;
            const double sh_ne = sh_pf.dens_e;
            const double sh_ti = sh_pf.temp_i;
            if (sh_te > 0.0 && sh_ne > 0.0) {
              // Reuse B[] and Bmag already computed at particle position
              double bvec[3] = {B[0], B[1], B[2]};
              const double sh_bmag = Bmag;
              double sh_alpha_deg = 90.0;
              if (sh_bmag > 0.0) {
                double nvec[3] = {sh_nx, sh_ny, sh_nz};
                SheathModels::ChoduraMetrics cm =
                  SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
                sh_alpha_deg = cm.alpha_deg;
              }

              double emag = 0.0;
              if (sheath_model == 1) {
                SheathModels::BorodkinaSheathResult sr =
                  SheathModels::coulette_manfredi_sheath_at_distance(
                    d_particle, sh_te, sh_ti, sh_ne, sh_bmag,
                    sh_alpha_deg, sheath_mD_amu, sheath_pot_mult);
                emag = sr.emag_vpm;
              } else {
                SheathModels::BorodkinaSheathResult sr =
                  SheathModels::borodkina_sheath_at_distance(
                    d_particle, sh_te, sh_ti, sh_ne, sh_bmag,
                    sh_alpha_deg, sheath_mD_amu, sheath_pot_mult);
                emag = sr.emag_vpm;
              }
                    // E-field toward wall: direction from signed distance
              const double dir = (d_raw > 0.0) ? -1.0 : 1.0;
              E[0] += dir * emag * sh_nx;
              E[1] += dir * emag * sh_ny;
              E[2] += dir * emag * sh_nz;
            }
          }
        }
      }
    }
  }

  // --- Switching criterion ---
  // Compute v_perp and check whether GCA is appropriate
  bool use_gca = false;

  if (Bmag > 0.0 && qm_abs > 0.0) {
    const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
    double v_perp = 0.0;
    if (have_gca_state && gca_on_vec[i] > 0.5) {
      // In persistent GCA mode, use stored mu to evaluate rho_L.
      const double mu_eff = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
      const double vperp2 = (2.0 * mu_eff * Bmag) / mass;
      v_perp = (vperp2 > 0.0) ? std::sqrt(vperp2) : 0.0;
    } else {
      const double v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
      const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
      double vperp2 = v2 - v_par * v_par;
      if (vperp2 < 0.0) vperp2 = 0.0;
      v_perp = std::sqrt(vperp2);
    }

    const double rho_L = GCAPusher::larmor_radius(v_perp, qm_abs, Bmag);
    const double L_B = GCAPusher::grad_b_length(Bmag, gradBmag_magnitude);

    // ERO2.0 criterion: use GCA when L_B < switch_factor * rho_L
    // i.e., the gradient scale is smaller than the Larmor orbit
    // Equivalently: rho_L is large relative to gradient scale → fast gyration
    // Actually: use GCA when rho_L is SMALL (fast gyration), i.e.,
    // the particle gyrates many times within one gradient scale length.
    // ERO2.0: GCA when d_char < 2.5 * rho_L  →  rho_L > L_B / 2.5
    // Rearranged: use GCA when rho_L < L_B / switch_factor
    // This means: gradient is gentle relative to orbit → GCA is valid
    if (rho_L > 0.0 && L_B < 1.0e19) {
      use_gca = (rho_L < L_B / gca_switch_factor);
    }
  }

  if (use_gca) {
    // --- GCA path ---
    GCAPusher::GCAState gca;
    if (have_gca_state && gca_on_vec[i] > 0.5) {
      gca.X[0] = gca_x_vec[i];
      gca.X[1] = gca_y_vec[i];
      gca.X[2] = gca_z_vec[i];
      gca.v_par = gca_vpar_vec[i];
      gca.mu = (gca_mu_vec[i] > 0.0) ? gca_mu_vec[i] : 0.0;
    } else {
      gca = GCAPusher::init_from_particle(x, v, mass, B);
    }

    // Full GCA integration (Littlejohn B* form) with RK4.
    GCAPusher::push_gca_rk4(qm, dt, mass, E, B, Bmag, gradBmag_cart,
                            kappa_cart, curlb_cart, gca);

    if (have_gca_state) {
      gca_x_vec[i] = gca.X[0];
      gca_y_vec[i] = gca.X[1];
      gca_z_vec[i] = gca.X[2];
      gca_vpar_vec[i] = gca.v_par;
      gca_mu_vec[i] = gca.mu;
      gca_on_vec[i] = 1.0;
    }

    // Keep persistent GC state, but reconstruct full v for diagnostics and
    // clean Boris fallback if regime switching happens.
    const double phi_golden = 0.6180339887498949;
    const double two_pi = 2.0 * M_PI;
    const double omega_c = std::fabs(qm) * Bmag;
    const double phase_turns =
      particle->particles[i].id * phi_golden +
      (omega_c * dt * static_cast<double>(ntimestep)) / two_pi;
    double rand_u = phase_turns - std::floor(phase_turns);
    GCAPusher::gca_to_particle(gca, B, mass, rand_u, xnew, v);
  } else {
    // --- Boris path (with subcycling) ---
    const int nsub = (boris_subcycles > 0) ? boris_subcycles : 1;
    const double dt_sub = dt / static_cast<double>(nsub);

    double xcur[3] = {x[0], x[1], x[2]};
    double vcur[3] = {v[0], v[1], v[2]};

    for (int isub = 0; isub < nsub; isub++) {
      // For point-interpolated B, re-evaluate at the current particle location.
      if (cp_bfield) {
        MagneticFieldFileDataParams Bc = cp_bfield->query_bfield_at_point(xcur);
        if (Bc.Bmag > 0.0) {
          if (domain->dimension == 2) {
            B[0] = Bc.br;
            B[1] = Bc.bz;
            B[2] = Bc.bt;
          } else {
            const double rx = xcur[0], ry = xcur[1];
            const double rxy = std::sqrt(rx*rx + ry*ry);
            double cphi = 1.0, sphi = 0.0;
            if (rxy > 1.0e-20) { cphi = rx / rxy; sphi = ry / rxy; }
            B[0] = Bc.br * cphi - Bc.bt * sphi;
            B[1] = Bc.br * sphi + Bc.bt * cphi;
            B[2] = Bc.bz;
          }
        }
      }

      BorisGrid::push_velocity(qm, dt_sub, E, B, vcur);
      xcur[0] += vcur[0] * dt_sub;
      xcur[1] += vcur[1] * dt_sub;
      xcur[2] += vcur[2] * dt_sub;
    }

    v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
    xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];

    if (have_gca_state) gca_on_vec[i] = 0.0;
  }
}

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
    } else if (strcmp(arg[iarg],"move") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal global command");
      if (strcmp(arg[iarg+1],"yes") == 0) move_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) move_flag = 0;
      else error->all(FLERR,"Illegal global command");
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
        if (iarg+4 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        efieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (efieldfreq < 0) error->all(FLERR,"Illegal global e field command");
        iarg += 4;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global e field command");
        delete [] efieldID;
        efstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        efieldID = new char[n];
        strcpy(efieldID,arg[iarg+2]);
        efieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (efieldfreq < 0) error->all(FLERR,"Illegal global e field command");
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
        if (iarg+4 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = PFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        bfieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (bfieldfreq < 0) error->all(FLERR,"Illegal global b field command");
        iarg += 4;

      } else if (strcmp(arg[iarg+1],"grid") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal global b field command");
        delete [] bfieldID;
        bfstyle = GFIELD;
        int n = strlen(arg[iarg+2]) + 1;
        bfieldID = new char[n];
        strcpy(bfieldID,arg[iarg+2]);
        bfieldfreq = input->inumeric(FLERR,arg[iarg+3]);
        if (bfieldfreq < 0) error->all(FLERR,"Illegal global b field command");
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
        if (fieldfreq < 0) error->all(FLERR,"Illegal global ethermal field command");
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
        if (fieldfreq < 0) error->all(FLERR,"Illegal global ithermal field command");
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
    // OpenEdge additions
    else if (strcmp(arg[iarg], "thermal_gradient_forces") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global thermal_gradient_forces command");
      if (strcmp(arg[iarg + 1], "yes") == 0) thermal_gradient_forces_flag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) thermal_gradient_forces_flag = 0;
      else error->all(FLERR, "Illegal global thermal_gradient_forces command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_dump") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global boris_dump command");
      if (strcmp(arg[iarg + 1], "yes") == 0) boris_dump_flag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) boris_dump_flag = 0;
      else error->all(FLERR, "Illegal global boris_dump command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_dump_every") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global boris_dump_every command");
      boris_dump_every = input->inumeric(FLERR, arg[iarg + 1]);
      if (boris_dump_every <= 0) error->all(FLERR, "Illegal global boris_dump_every command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_subcycles") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global boris_subcycles command");
      boris_subcycles = input->inumeric(FLERR, arg[iarg + 1]);
      if (boris_subcycles <= 0) error->all(FLERR, "Illegal global boris_subcycles command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_bad_dt_check") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global boris_bad_dt_check command");
      if (strcmp(arg[iarg + 1], "yes") == 0) boris_bad_dt_check = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) boris_bad_dt_check = 0;
      else error->all(FLERR, "Illegal global boris_bad_dt_check command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boris_bad_dt_limit") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Illegal global boris_bad_dt_limit command");
      boris_bad_dt_limit = input->numeric(FLERR, arg[iarg + 1]);
      if (boris_bad_dt_limit <= 0.0) error->all(FLERR, "Illegal global boris_bad_dt_limit command");
      iarg += 2;

    // Hybrid Boris/GCA pusher (ERO2.0-style).
    // Usage: global gca plasma_compute <ID> [switch_factor 2.5]
    } else if (strcmp(arg[iarg], "gca") == 0) {
      iarg++;
      gca_flag = 1;
      while (iarg < narg) {
        if (strcmp(arg[iarg], "plasma_compute") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global gca command");
          delete [] gca_plasma_cid;
          int n = strlen(arg[iarg+1]) + 1;
          gca_plasma_cid = new char[n];
          strcpy(gca_plasma_cid, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "switch_factor") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global gca command");
          gca_switch_factor = input->numeric(FLERR, arg[iarg+1]);
          if (gca_switch_factor <= 0.0) error->all(FLERR, "global gca switch_factor must be > 0");
          iarg += 2;
        } else break;
      }
      if (!gca_plasma_cid)
        error->all(FLERR, "global gca requires plasma_compute <ID>");

    // Per-particle sheath E-field from grid-cached geometry + plasma computes.
    // Usage: global sheath geom_compute <ID> plasma_compute <ID>
    //          [model borodkina/coulette_manfredi] [dmax 0.02] [pot_mult 2.5]
    //          [mD_amu 2.014]
    } else if (strcmp(arg[iarg], "sheath") == 0) {
      iarg++;
      sheath_flag = 1;
      while (iarg < narg) {
        if (strcmp(arg[iarg], "geom_compute") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          delete [] sheath_geom_cid;
          int n = strlen(arg[iarg+1]) + 1;
          sheath_geom_cid = new char[n];
          strcpy(sheath_geom_cid, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "plasma_compute") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          delete [] sheath_plasma_cid;
          int n = strlen(arg[iarg+1]) + 1;
          sheath_plasma_cid = new char[n];
          strcpy(sheath_plasma_cid, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "model") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          if (strcmp(arg[iarg+1], "borodkina") == 0) sheath_model = 0;
          else if (strcmp(arg[iarg+1], "coulette_manfredi") == 0) sheath_model = 1;
          else error->all(FLERR, "global sheath model must be borodkina or coulette_manfredi");
          iarg += 2;
        } else if (strcmp(arg[iarg], "dmax") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          sheath_dmax = input->numeric(FLERR, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "pot_mult") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          sheath_pot_mult = input->numeric(FLERR, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "mD_amu") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          sheath_mD_amu = input->numeric(FLERR, arg[iarg+1]);
          iarg += 2;
        } else if (strcmp(arg[iarg], "kick") == 0) {
          if (iarg + 1 >= narg) error->all(FLERR, "Illegal global sheath command");
          if (strcmp(arg[iarg+1], "yes") == 0) sheath_kick = 1;
          else if (strcmp(arg[iarg+1], "no") == 0) sheath_kick = 0;
          else error->all(FLERR, "global sheath kick must be yes or no");
          iarg += 2;
        } else break;  // next keyword belongs to a different global option
      }
      if (!sheath_geom_cid || !sheath_plasma_cid)
        error->all(FLERR, "global sheath requires geom_compute and plasma_compute");

    } else if (strcmp(arg[iarg],"mem/limit") == 0) {
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
