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

#define INVOKED_PER_GRID 16

#include "spatype.h"
#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "update_kokkos.h"
#include "pusher.h"
#include "compute_plasma_fields_kokkos.h"
#include "compute_nearest_surf_grid.h"
#include "sheath_models.h"
#include "surf.h"
#include "math_const.h"
#include "particle_kokkos.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "domain.h"
#include "comm_kokkos.h"
#include "collide.h"
#include "collide_vss_kokkos.h"
#include "grid_kokkos.h"
#include "surf_kokkos.h"
#include "surf_collide.h"
#include "surf_react.h"
#include "output.h"
#include "geometry_kokkos.h"
#include "random_mars.h"
#include "timer.h"
#include "math_extra.h"
#include "memory_kokkos.h"
#include "error.h"
#include <unistd.h>
#include "kokkos.h"
#include "sparta_masks.h"
#include "surf_collide_specular_kokkos.h"
#include "kokkos_base.h"

using namespace SPARTA_NS;

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // same as Domain
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain
//enum{OUTSIDE,INSIDE,ONSURF2OUT,ONSURF2IN};      // several files
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Grid
enum{TALLYAUTO,TALLYREDUCE,TALLYLOCAL};         // same as Surf
enum{PERAUTO,PERCELL,PERSURF};                  // several files
enum{NOFIELD,CFIELD,PFIELD,GFIELD};             // several files

#define MAXSTUCK 20
#define EPSPARAM 1.0e-7

// either set ID or PROC/INDEX, set other to -1

//#define MOVE_DEBUG 1              // un-comment to debug one particle
#define MOVE_DEBUG_ID 308143534  // particle ID
#define MOVE_DEBUG_PROC -1        // owning proc
#define MOVE_DEBUG_INDEX -1   // particle index on owning proc
#define MOVE_DEBUG_STEP 4107    // timestep

#define VAL_1(X) X
#define VAL_2(X) VAL_1(X), VAL_1(X)

/* ---------------------------------------------------------------------- */

UpdateKokkos::UpdateKokkos(SPARTA *sparta) : Update(sparta),
  grid_kk_copy(sparta),
  domain_kk_copy(sparta),
  // Virtual functions are not yet supported on the GPU, which leads to pain:
  sc_kk_specular_copy{VAL_2(KKCopy<SurfCollideSpecularKokkos>(sparta))},
  sc_kk_diffuse_copy{VAL_2(KKCopy<SurfCollideDiffuseKokkos>(sparta))},
  sc_kk_vanish_copy{VAL_2(KKCopy<SurfCollideVanishKokkos>(sparta))},
  sc_kk_piston_copy{VAL_2(KKCopy<SurfCollidePistonKokkos>(sparta))},
  sc_kk_transparent_copy{VAL_2(KKCopy<SurfCollideTransparentKokkos>(sparta))},
  blist_active_copy{VAL_2(KKCopy<ComputeBoundaryKokkos>(sparta))},
  slist_active_copy{VAL_2(KKCopy<ComputeSurfKokkos>(sparta))}
{

  // use 1D view for scalars to reduce GPU memory operations

  d_scalars = t_int_14("collide:scalars");
  h_scalars = t_host_int_14("collide:scalars_mirror");

  d_ncomm_one     = Kokkos::subview(d_scalars,0);
  d_nexit_one     = Kokkos::subview(d_scalars,1);
  d_nboundary_one = Kokkos::subview(d_scalars,2);
  d_nmigrate      = Kokkos::subview(d_scalars,3);
  d_entryexit     = Kokkos::subview(d_scalars,4);
  d_ntouch_one    = Kokkos::subview(d_scalars,5);
  d_nscheck_one   = Kokkos::subview(d_scalars,6);
  d_nscollide_one = Kokkos::subview(d_scalars,7);
  d_nreact_one    = Kokkos::subview(d_scalars,8);
  d_nstuck        = Kokkos::subview(d_scalars,9);
  d_naxibad       = Kokkos::subview(d_scalars,10);
  d_error_flag    = Kokkos::subview(d_scalars,11);
  d_retry         = Kokkos::subview(d_scalars,12);
  d_nlocal        = Kokkos::subview(d_scalars,13);

  h_ncomm_one     = Kokkos::subview(h_scalars,0);
  h_nexit_one     = Kokkos::subview(h_scalars,1);
  h_nboundary_one = Kokkos::subview(h_scalars,2);
  h_nmigrate      = Kokkos::subview(h_scalars,3);
  h_entryexit     = Kokkos::subview(h_scalars,4);
  h_ntouch_one    = Kokkos::subview(h_scalars,5);
  h_nscheck_one   = Kokkos::subview(h_scalars,6);
  h_nscollide_one = Kokkos::subview(h_scalars,7);
  h_nreact_one    = Kokkos::subview(h_scalars,8);
  h_nstuck        = Kokkos::subview(h_scalars,9);
  h_naxibad       = Kokkos::subview(h_scalars,10);
  h_error_flag    = Kokkos::subview(h_scalars,11);
  h_retry         = Kokkos::subview(h_scalars,12);
  h_nlocal        = Kokkos::subview(h_scalars,13);

  nboundary_tally = 0;
}

/* ---------------------------------------------------------------------- */

UpdateKokkos::~UpdateKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_mlist,mlist);
  mlist = NULL;

  grid_kk_copy.uncopy();
  domain_kk_copy.uncopy();

  for (int i=0; i<KOKKOS_MAX_SURF_COLL_PER_TYPE; i++) {
    sc_kk_specular_copy[i].uncopy();
    sc_kk_diffuse_copy[i].uncopy();
    sc_kk_vanish_copy[i].uncopy();
    sc_kk_piston_copy[i].uncopy();
    sc_kk_transparent_copy[i].uncopy();
  }

  for (int i=0; i<KOKKOS_MAX_BLIST; i++) {
    blist_active_copy[i].uncopy();
  }

  for (int i=0; i<KOKKOS_MAX_SLIST; i++) {
    slist_active_copy[i].uncopy();
  }
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::init()
{
  // Call base Update::init() first for OpenEdge-specific initialization:
  // plasma cache custom vectors, sheath setup, Boris config, field fixes.
  // UpdateKokkos then overrides moveptr and field fix resolution below.
  Update::init();

  if (runflag == 0) return;

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
    if (surf->exist) {
      if (surf->nsr) moveptr = &UpdateKokkos::move<3,1,1,0>;
      else moveptr = &UpdateKokkos::move<3,1,0,0>;
    } else {
      if (optmove_flag) moveptr = &UpdateKokkos::move<3,0,0,1>;
      else moveptr = &UpdateKokkos::move<3,0,0,0>;
    }
  } else if (domain->axisymmetric) {
    if (surf->exist) {
      if (surf->nsr) moveptr = &UpdateKokkos::move<1,1,1,0>;
      else moveptr = &UpdateKokkos::move<1,1,0,0>;
    } else {
      if (optmove_flag) moveptr = &UpdateKokkos::move<1,0,0,1>;
      else moveptr = &UpdateKokkos::move<1,0,0,0>;
    }
  } else if (domain->dimension == 2) {
    if (surf->exist) {
      if (surf->nsr) moveptr = &UpdateKokkos::move<2,1,1,0>;
      else moveptr = &UpdateKokkos::move<2,1,0,0>;
    } else {
      if (optmove_flag) moveptr = &UpdateKokkos::move<2,0,0,1>;
      else moveptr = &UpdateKokkos::move<2,0,0,0>;
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

  if (optmove_flag) {
    xlo = domain->boxlo[0];
    ylo = domain->boxlo[1];
    zlo = domain->boxlo[2];
    xhi = domain->boxhi[0];
    yhi = domain->boxhi[1];
    zhi = domain->boxhi[2];
    Lx = xhi-xlo;
    Ly = yhi-ylo;
    Lz = zhi-zlo;
    ncx = grid->unx;
    ncy = grid->uny;
    ncz = grid->unz;
    dx = Lx/ncx;
    dy = Ly/ncy;
    dz = Lz/ncz;
  }

  if (fstyle == PFIELD) {
    field_active[0] = modify->fix[ifieldfix]->field_active[0];
    field_active[1] = modify->fix[ifieldfix]->field_active[1];
    field_active[2] = modify->fix[ifieldfix]->field_active[2];
    KKBaseFieldFix = dynamic_cast<KokkosBase*>(modify->fix[ifieldfix]);
    if (!KKBaseFieldFix)
      error->all(FLERR,"External field fix is not Kokkos-enabled");
  } else if (fstyle == GFIELD) {
    field_active[0] = modify->fix[ifieldfix]->field_active[0];
    field_active[1] = modify->fix[ifieldfix]->field_active[1];
    field_active[2] = modify->fix[ifieldfix]->field_active[2];
    KKBaseFieldFix = dynamic_cast<KokkosBase*>(modify->fix[ifieldfix]);
    if (!KKBaseFieldFix)
      error->all(FLERR,"External field fix is not Kokkos-enabled");
  }

  // OpenEdge: Boris config — read B/E directly from plasma compute view
  oe_pusher_subcycles = pusher->pusher_subcycles;
  oe_pusher_mode = pusher->pusher_mode;
  oe_echarge = echarge;
  oe_bx_col = oe_by_col = oe_bz_col = -1;
  oe_ex_col = oe_ey_col = oe_ez_col = -1;

  // OpenEdge Phase A: equilibrium-based point-query B (defaults off).
  // Actual binding to ComputePlasmaFieldsKokkos's d_equ_* views happens
  // at the same site where d_oe_plasma_compute is bound (see below).
  oe_has_equilibrium = 0;
  oe_equ_jm = oe_equ_km = 0;
  oe_equ_btf = oe_equ_rtf = 0.0;
  oe_dim = domain->dimension;
  oe_axisymmetric = domain->axisymmetric;

  // OpenEdge Phase B: mesh-triangulation B (defaults off).
  oe_has_mesh_b = 0;
  oe_mesh_ntri = 0;
  oe_mesh_hash_nr = oe_mesh_hash_nz = 0;
  oe_mesh_hash_rmin = oe_mesh_hash_zmin = 0.0;
  oe_mesh_hash_dr = oe_mesh_hash_dz = 1.0;

  // OpenEdge Phase D: sheath spatial-mode cache (defaults off).
  oe_has_sheath_spatial = 0;
  oe_sheath_mD_amu = sheath_mD_amu;

  // OpenEdge Phase C: GCA persistent state binding (defaults off).
  // Activated at the same site that binds the plasma compute, when the
  // pusher is in hybrid mode and the custom attrs were registered by
  // Pusher::init() on the host side.
  oe_has_gca_state = 0;
  oe_pusher_gca_switch = pusher->pusher_gca_switch;
  if (pusher->pusher_mode == Pusher::PUSHER_HYBRID &&
      pusher->gca_x_custom >= 0 && pusher->gca_y_custom >= 0 &&
      pusher->gca_z_custom >= 0 && pusher->gca_vpar_custom >= 0 &&
      pusher->gca_mu_custom >= 0 && pusher->gca_on_custom >= 0) {
    auto *pkk = (ParticleKokkos*) particle;
    d_oe_gca_x    = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_x_custom]].k_view.d_view;
    d_oe_gca_y    = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_y_custom]].k_view.d_view;
    d_oe_gca_z    = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_z_custom]].k_view.d_view;
    d_oe_gca_vpar = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_vpar_custom]].k_view.d_view;
    d_oe_gca_mu   = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_mu_custom]].k_view.d_view;
    d_oe_gca_on   = pkk->k_edvec.h_view[particle->ewhich[pusher->gca_on_custom]].k_view.d_view;
    oe_has_gca_state = 1;
  }
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::setup()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  GridKokkos* grid_kk = (GridKokkos*) grid;
  SurfKokkos* surf_kk = (SurfKokkos*) surf;

  // Sync particle data EXCLUDING custom vectors (which may be zero-sized
  // at this point if no particles exist yet). Custom vectors get synced
  // after setup() populates them.
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  particle_kk->sorted_kk = 0;

  if (sparta->kokkos->prewrap) {

    // particle

    particle_kk->wrap_kokkos();

    // grid

    grid_kk->wrap_kokkos();
    grid_kk->update_hash();

    // surf

    if (surf->exist)
      surf_kk->wrap_kokkos();

    sparta->kokkos->prewrap = 0;
  } else {
    grid_kk->modify(Host,ALL_MASK);
    grid_kk->update_hash();

    if (surf->exist) {
      surf_kk->modify(Host,ALL_MASK);
      grid_kk->wrap_kokkos_graphs();
    }
  }
  hash_kk = grid_kk->hash_kk;

  Update::setup(); // must come after prewrap since computes are called by setup()

  // For MPI debugging
  //
  //  volatile int i = 0;
  //  char hostname[256];
  //  gethostname(hostname, sizeof(hostname));
  //  printf("PID %d on %s ready for attach, i = %i\n", getpid(), hostname, i);
  //  fflush(stdout);
  //  sleep(30);
  //  printf("Continuing...\n");
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::run(int nsteps)
{
  sparta->kokkos->auto_sync = 0;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;

  int n_start_of_step = modify->n_start_of_step;
  int n_end_of_step = modify->n_end_of_step;

  // external per grid cell field
  // only evaluate once at beginning of run b/c time-independent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq == 0) {
    modify->fix[ifieldfix]->compute_field();
    d_fieldfix_array_grid = KKBaseFieldFix->d_array_grid;
  }

  // OpenEdge: fetch plasma compute view for Boris B-field (bypass fixes)
  // The plasma/fields compute stores bx,by,bz as the first 3 columns
  if (oe_pusher_subcycles > 0 && pusher->pusher_plasma_cidx >= 0) {
    Compute *cp = modify->compute[pusher->pusher_plasma_cidx];
    if (!(cp->invoked_flag & INVOKED_PER_GRID)) {
      cp->compute_per_grid();
      cp->invoked_flag |= INVOKED_PER_GRID;
    }
    KokkosBase *kk_cp = dynamic_cast<KokkosBase*>(cp);
    if (kk_cp) {
      d_oe_plasma_compute = kk_cp->d_array_grid;
      // Column mapping from compute: bx=0, by=1, bz=2 (first 3 values)
      oe_bx_col = 0; oe_by_col = 1; oe_bz_col = 2;
    }

    // Phase A: bind to the device-resident equilibrium psi map (if any)
    // for smooth point-query B inside oe_boris3d. Falls back to cell-center
    // columns when no equilibrium is loaded.
    auto *cp_pf = dynamic_cast<ComputePlasmaFieldsKokkos*>(cp);
    if (cp_pf && cp_pf->d_has_equilibrium) {
      d_oe_equ_r        = cp_pf->d_equ_r;
      d_oe_equ_z        = cp_pf->d_equ_z;
      d_oe_equ_psi      = cp_pf->d_equ_psi;
      oe_equ_btf        = cp_pf->d_equ_btf;
      oe_equ_rtf        = cp_pf->d_equ_rtf;
      oe_equ_jm         = cp_pf->d_equ_jm;
      oe_equ_km         = cp_pf->d_equ_km;
      oe_has_equilibrium = 1;
      oe_dim            = domain->dimension;
      oe_axisymmetric   = domain->axisymmetric;
    }

    // Phase B: bind to the mesh-triangulation B Views (SOLPS / SOLEDGE3X
    // plasmas). Mesh takes precedence over equilibrium in the per-particle
    // dispatch — matches CPU semantics in compute_plasma_fields.cpp.
    if (cp_pf && cp_pf->d_has_mesh_b) {
      d_oe_mesh_vtx_r    = cp_pf->d_mesh_vtx_r;
      d_oe_mesh_vtx_z    = cp_pf->d_mesh_vtx_z;
      d_oe_mesh_tri      = cp_pf->d_mesh_tri;
      d_oe_mesh_tri_br   = cp_pf->d_mesh_tri_br;
      d_oe_mesh_tri_bz   = cp_pf->d_mesh_tri_bz;
      d_oe_mesh_tri_bt   = cp_pf->d_mesh_tri_bt;
      d_oe_mesh_tri_rmin = cp_pf->d_mesh_tri_rmin;
      d_oe_mesh_tri_rmax = cp_pf->d_mesh_tri_rmax;
      d_oe_mesh_tri_zmin = cp_pf->d_mesh_tri_zmin;
      d_oe_mesh_tri_zmax = cp_pf->d_mesh_tri_zmax;
      d_oe_hash_offset   = cp_pf->d_hash_offset;
      d_oe_hash_entries  = cp_pf->d_hash_entries;
      oe_mesh_hash_rmin  = cp_pf->d_mesh_hash_rmin;
      oe_mesh_hash_zmin  = cp_pf->d_mesh_hash_zmin;
      oe_mesh_hash_dr    = cp_pf->d_mesh_hash_dr;
      oe_mesh_hash_dz    = cp_pf->d_mesh_hash_dz;
      oe_mesh_hash_nr    = cp_pf->d_mesh_hash_nr;
      oe_mesh_hash_nz    = cp_pf->d_mesh_hash_nz;
      oe_mesh_ntri       = cp_pf->d_mesh_ntri;
      oe_has_mesh_b      = 1;
    }

    // Phase D: build per-cell sheath spatial-mode cache once at run() setup.
    // Static-plasma assumption: cache is not refreshed per step. Triggered
    // only when `global pusher ... sheath spatial geom <ID>` is configured
    // (sheath_flag=1 && sheath_kick=0 && sheath_geom_cidx resolved).
    if (sheath_flag && !sheath_kick && sheath_geom_cidx >= 0)
      build_oe_sheath_cache();
  }

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
    if (tallyflag) tally_set(ntimestep);
    if (dynamic) dynamic_update();

    timer->stamp();

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

    if (nmigrate) {
      k_mlist_small = Kokkos::subview(k_mlist,std::make_pair(0,nmigrate));
      k_mlist_small.sync_host();
    }
    auto mlist_small = k_mlist_small.view_host().data();

    ((CommKokkos*)comm)->migrate_particles(nmigrate,mlist_small,k_mlist_small.view_device());
    if (cellweightflag) particle->post_weight();
    timer->stamp(TIME_COMM);

    const int reorder_flag = (update->reorder_period &&
        (update->ntimestep % update->reorder_period == 0));

    if (collide || reorder_flag) {
      particle_kk->sort_kokkos();
      timer->stamp(TIME_SORT);
    }

    if (collide) {
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
      particle_kk->sync(Host,ALL_MASK);
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
  }
  sparta->kokkos->auto_sync = 1;

  particle_kk->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   advect particles thru grid
   DIM = 2/3 for 2d/3d, 1 for 2d axisymmetric
   SURF = 0/1 for no surfs or surfs
   use multiple iterations of move/comm if necessary
------------------------------------------------------------------------- */

template < int DIM, int SURF, int REACT, int OPT > void UpdateKokkos::move()
{
  int pstart,pstop,entryexit,any_entryexit;
  int continue_loop_flag = 0;

  // extend migration list if necessary

  int maxlocal = particle->maxlocal;

  if (particle->nlocal > maxmigrate) {
    maxmigrate = maxlocal;
    memoryKK->destroy_kokkos(k_mlist,mlist);
    memoryKK->create_kokkos(k_mlist,mlist,maxmigrate,"particle:mlist");
  }

  // counters

  niterate = 0;
  ntouch_one = ncomm_one = 0;
  nboundary_one = nexit_one = 0;
  nscheck_one = nscollide_one = 0;
  surf->nreact_one = 0;

  if (!sparta->kokkos->need_atomics || sparta->kokkos->atomic_reduction) {
    h_ntouch_one() = 0;
    h_nexit_one() = 0;
    h_nboundary_one() = 0;
    h_ncomm_one() = 0;
    h_nscheck_one() = 0;
    h_nscollide_one() = 0;
    h_nreact_one() = 0;
  }

  h_error_flag() = 0;

  // move/migrate iterations

  dt = update->dt;

  ParticleKokkos* particle_kk = ((ParticleKokkos*)particle);

  // external per particle field
  // fix calculates field acting on all owned particles

  if (fstyle == PFIELD) {
    modify->fix[ifieldfix]->compute_field();
    d_fieldfix_array_particle = KKBaseFieldFix->d_array_particle;
  }

  // external per grid cell field
  // evaluate once every fieldfreq steps b/c time-dependent
  // fix calculates field acting at center point of all grid cells

  if (fstyle == GFIELD && fieldfreq && ((ntimestep-1) % fieldfreq == 0)) {
    modify->fix[ifieldfix]->compute_field();
    d_fieldfix_array_grid = KKBaseFieldFix->d_array_grid;
  }

  // OpenEdge: plasma compute view refresh (already done per-step by computes)

  // one or more loops over particles
  // first iteration = all my particles
  // subsequent iterations = received particles

  while (1) {

    if (!continue_loop_flag)
      niterate++;

    d_particles = particle_kk->k_particles.view_device();

    GridKokkos* grid_kk = ((GridKokkos*)grid);
    d_cells = grid_kk->k_cells.view_device();
    d_sinfo = grid_kk->k_sinfo.view_device();
    d_pcells = grid_kk->k_pcells.view_device();

    d_csurfs = grid_kk->d_csurfs;
    d_csplits = grid_kk->d_csplits;
    d_csubs = grid_kk->d_csubs;

    if (surf->exist) {
      SurfKokkos* surf_kk = ((SurfKokkos*)surf);
      surf_kk->sync(Device,ALL_MASK);
      d_lines = surf_kk->k_lines.view_device();
      d_tris = surf_kk->k_tris.view_device();
    }

    if (surf->nsr) {
      double extra_factor = 1.0;
      if (!sparta->kokkos->react_retry_flag)
        extra_factor = sparta->kokkos->react_extra;

      int nlocal_extra = particle->nlocal*extra_factor;
      if (d_particles.extent(0) < nlocal_extra) {
        particle->grow(nlocal_extra - particle->nlocal); // this!
        d_particles = particle_kk->k_particles.view_device();
      }
    }

    particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
    grid_kk->sync(Device,CELL_MASK|PCELL_MASK|SINFO_MASK|PLEVEL_MASK);
    d_species = particle_kk->k_species.d_view;

    // may be able to move this outside of the while loop
    grid_kk_copy.copy(grid_kk);
    domain_kk_copy.copy((DomainKokkos*)domain);

    if (surf->nsc > KOKKOS_MAX_TOT_SURF_COLL)
      error->all(FLERR,"Kokkos currently supports two instances of each surface collide method");

    if (surf->nsc > 0) {
      int nspec,ndiff,nvan,npist,ntrans;
      nspec = ndiff = nvan = npist = ntrans = 0;
      for (int n = 0; n < surf->nsc; n++) {
        if (!surf->sc[n]->kokkosable)
          error->all(FLERR,"Must use Kokkos-enabled surface collide method with Kokkos");
        if (strcmp(surf->sc[n]->style,"specular") == 0) {
          sc_kk_specular_copy[nspec].copy((SurfCollideSpecularKokkos*)(surf->sc[n]));
          sc_kk_specular_copy[nspec].obj.pre_collide();
          sc_type_list[n] = 0;
          sc_map[n] = nspec;
          nspec++;
        } else if (strcmp(surf->sc[n]->style,"diffuse") == 0) {
          sc_kk_diffuse_copy[ndiff].copy((SurfCollideDiffuseKokkos*)(surf->sc[n]));
          sc_kk_diffuse_copy[ndiff].obj.pre_collide();
          sc_type_list[n] = 1;
          sc_map[n] = ndiff;
          ndiff++;
        } else if (strcmp(surf->sc[n]->style,"vanish") == 0) {
          sc_kk_vanish_copy[nvan].copy((SurfCollideVanishKokkos*)(surf->sc[n]));
          sc_kk_vanish_copy[nvan].obj.pre_collide();
          sc_type_list[n] = 2;
          sc_map[n] = nvan;
          nvan++;
        } else if (strcmp(surf->sc[n]->style,"piston") == 0) {
          sc_kk_piston_copy[npist].copy((SurfCollidePistonKokkos*)(surf->sc[n]));
          sc_kk_piston_copy[npist].obj.pre_collide();
          sc_type_list[n] = 3;
          sc_map[n] = npist;
          npist++;
        } else if (strcmp(surf->sc[n]->style,"transparent") == 0) {
          sc_kk_transparent_copy[ntrans].copy((SurfCollideTransparentKokkos*)(surf->sc[n]));
          sc_kk_transparent_copy[ntrans].obj.pre_collide();
          sc_type_list[n] = 4;
          sc_map[n] = ntrans;
          ntrans++;
        } else {
          error->all(FLERR,"Unknown Kokkos surface collide method");
        }
      }
      if (nspec > KOKKOS_MAX_SURF_COLL_PER_TYPE || ndiff > KOKKOS_MAX_SURF_COLL_PER_TYPE ||
          nvan > KOKKOS_MAX_SURF_COLL_PER_TYPE || npist > KOKKOS_MAX_SURF_COLL_PER_TYPE ||
          ntrans > KOKKOS_MAX_SURF_COLL_PER_TYPE)
        error->all(FLERR,"Kokkos currently supports two instances of each surface collide method");
    }

    Kokkos::deep_copy(h_scalars,0);

    if (!continue_loop_flag) {
      nmigrate = 0;
      entryexit = 0;
    }

    if (niterate == 1 && !continue_loop_flag) {
      pstart = 0;
      pstop = particle->nlocal;
    }

    UPDATE_REDUCE reduce;

    // Reactions may create or delete more particles than existing views can hold.
    //  Cannot grow a Kokkos view in a parallel loop, so
    //  if the capacity of the view is exceeded, break out of parallel loop,
    //  reallocate on the host, and then repeat the parallel loop again.
    //  Unfortunately this leads to really messy code.

    h_retry() = 1;

    while (h_retry()) {

      if (surf->nsr && sparta->kokkos->react_retry_flag)
        backup();

      h_retry() = 0;
      h_nlocal() = particle->nlocal;
      if (continue_loop_flag) h_nmigrate() = nmigrate;

      Kokkos::deep_copy(d_scalars,h_scalars);

      copymode = 1;

    /* ATOMIC_REDUCTION: 1 = use atomics
                         0 = don't need atomics
                        -1 = use parallel_reduce
    */

#if defined SPARTA_KOKKOS_GPU
  #if defined(KOKKOS_ARCH_AMD_GFX940) || defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,-1> >(pstart,pstop),*this,reduce);
  #else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,1> >(pstart,pstop),*this);
  #endif
#elif defined KOKKOS_ENABLE_SERIAL
      if constexpr(std::is_same<DeviceType,Kokkos::Serial>::value)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,0> >(pstart,pstop),*this);
      else {
        if (!sparta->kokkos->need_atomics)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,0> >(pstart,pstop),*this);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,-1> >(pstart,pstop),*this,reduce);
      }
#else
      if (!sparta->kokkos->need_atomics)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,0> >(pstart,pstop),*this);
      else
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagUpdateMove<DIM,SURF,REACT,OPT,-1> >(pstart,pstop),*this,reduce);
#endif

      copymode = 0;

      Kokkos::deep_copy(h_scalars,d_scalars);

      if (h_retry()) {
        int nlocal_new = h_nlocal();

        if (!sparta->kokkos->react_retry_flag) {
          error->one(FLERR,"Ran out of space for Kokkos reactions, increase react/extra"
                           " or use react/retry");
        } else
          restore();

        //  reset counters

        Kokkos::deep_copy(h_scalars,0);
        reduce = UPDATE_REDUCE();
        h_retry() = 1;

        if (d_particles.extent(0) < nlocal_new) {
          particle->grow(nlocal_new - particle->nlocal);
          d_particles = particle_kk->k_particles.view_device();
        }
      }
    }

    particle_kk->modify(Device,PARTICLE_MASK);
    d_particles = t_particle_1d(); // destroy reference to reduce memory use

    k_mlist.modify_device();

    // END of pstart/pstop loop advecting all particles

    nmigrate = h_nmigrate();

    particle->nlocal = h_nlocal();

    int error_flag;

    if (!sparta->kokkos->need_atomics || sparta->kokkos->atomic_reduction) {
      ntouch_one += h_ntouch_one();
      nexit_one += h_nexit_one();
      nboundary_one += h_nboundary_one();
      ncomm_one += h_ncomm_one();
      nscheck_one += h_nscheck_one();
      nscollide_one += h_nscollide_one();
      surf->nreact_one += h_nreact_one();
      nstuck += h_nstuck();
      naxibad += h_naxibad();
    } else {
      ntouch_one       += reduce.ntouch_one   ;
      nexit_one        += reduce.nexit_one    ;
      nboundary_one    += reduce.nboundary_one;
      ncomm_one        += reduce.ncomm_one    ;
      nscheck_one      += reduce.nscheck_one  ;
      nscollide_one    += reduce.nscollide_one;
      surf->nreact_one += reduce.nreact_one   ;
      nstuck           += reduce.nstuck       ;
      naxibad          += reduce.naxibad      ;
    }

    entryexit += h_entryexit();

    error_flag = h_error_flag();

    if (error_flag) {
      char str[128];
      sprintf(str,
              "Particle being sent to self proc "
              "on step " BIGINT_FORMAT,
              update->ntimestep);
      error->one(FLERR,str);
    }

    if (surf->nsc > 0) {
      int nspec,ndiff,nvan,npist,ntrans;
      nspec = ndiff = nvan = npist = ntrans = 0;
      for (int n = 0; n < surf->nsc; n++) {
        if (strcmp(surf->sc[n]->style,"specular") == 0) {
          sc_kk_specular_copy[nspec].obj.post_collide();
          nspec++;
        } else if (strcmp(surf->sc[n]->style,"diffuse") == 0) {
          sc_kk_diffuse_copy[ndiff].obj.post_collide();
          ndiff++;
        } else if (strcmp(surf->sc[n]->style,"vanish") == 0) {
          sc_kk_vanish_copy[nvan].obj.post_collide();
          nvan++;
        } else if (strcmp(surf->sc[n]->style,"piston") == 0) {
          sc_kk_piston_copy[npist].obj.post_collide();
          npist++;
        } else if (strcmp(surf->sc[n]->style,"transparent") == 0) {
          sc_kk_transparent_copy[ntrans].obj.post_collide();
          ntrans++;
        }
      }
    }

    // move newly created particles from surface reactions

    continue_loop_flag = 0;

    if (surf->nsr && pstop < particle->nlocal) {
      pstart = pstop;
      pstop = particle->nlocal;
      continue_loop_flag = 1;
      continue;
    }

    // if gridcut >= 0.0, check if another iteration of move is required
    // only the case if some particle flag = PENTRY/PEXIT
    //   in which case perform particle migration
    // if not, move is done and final particle comm will occur in run()
    // if iterating, reset pstart/pstop and extend migration list if necessary

    if (grid->cutoff < 0.0) break;

    timer->stamp(TIME_MOVE);
    MPI_Allreduce(&entryexit,&any_entryexit,1,MPI_INT,MPI_MAX,world);
    timer->stamp();

    if (any_entryexit) {
      if (nmigrate) {
        k_mlist_small = Kokkos::subview(k_mlist,std::make_pair(0,nmigrate));
        k_mlist_small.sync_host();
      }
      auto mlist_small = k_mlist_small.view_host().data();
      timer->stamp(TIME_MOVE);
      pstart = ((CommKokkos*)comm)->migrate_particles(nmigrate,mlist_small,k_mlist_small.view_device());
      timer->stamp(TIME_COMM);
      pstop = particle->nlocal;
      if (pstop-pstart > maxmigrate) {
        maxmigrate = pstop-pstart;
        memoryKK->destroy_kokkos(k_mlist,mlist);
        memoryKK->create_kokkos(k_mlist,mlist,maxmigrate,"particle:mlist");
      }
    } else break;

    // END of single move/migrate iteration
  }

  // END of all move/migrate iterations

  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

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

  if (nsurf_tally) {
    for (int m = 0; m < nsurf_tally; m++) {
      ComputeSurfKokkos* compute_surf_kk = (ComputeSurfKokkos*)(slist_active[m]);
      compute_surf_kk->post_surf_tally();
    }
  }

  if (nboundary_tally) {
    for (int m = 0; m < nboundary_tally; m++) {
      ComputeBoundaryKokkos* compute_boundary_kk = (ComputeBoundaryKokkos*)(blist_active[m]);
      compute_boundary_kk->post_boundary_tally();
    }
  }
}

/* ---------------------------------------------------------------------- */

template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
KOKKOS_INLINE_FUNCTION
void UpdateKokkos::operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int &i) const {
  UPDATE_REDUCE reduce;
  this->template operator()<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>(), i, reduce);
}

/*-----------------------------------------------------------------------------*/

template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
KOKKOS_INLINE_FUNCTION
void UpdateKokkos::operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int &i, UPDATE_REDUCE &reduce) const {
  if (d_error_flag() || d_retry()) return;

  // int m;
  bool hitflag;
  int icell,icell_original,outface,bflag,nflag,pflag,itmp;
  int side,minsurf,nsurf,cflag,isurf,exclude,stuck_iterate;
  double dtremain,frac,newfrac,param,minparam,rnew,dtsurf,tc,tmp;
  double xnew[3],xhold[3],xc[3],vc[3],minxc[3],minvc[3];
  double *x,*v;
  Surf::Tri *tri;
  Surf::Line *line;
  int reaction;

  Particle::OnePart &particle_i = d_particles[i];
  pflag = particle_i.flag;

  Particle::OnePart iorig;
  Particle::OnePart *ipart,*jpart;
  jpart = NULL;

  // received from another proc and move is done
  // if first iteration, PDONE is from a previous step,
  //   set pflag to PKEEP so move the particle on this step
  // else do nothing

  if (pflag == PDONE) {
    pflag = particle_i.flag = PKEEP;
    if (niterate > 1) return;
  }

  x = particle_i.x;
  v = particle_i.v;
  exclude = -1;

  // for 2d and axisymmetry only
  // xnew,xc passed to geometry routines which use or set z component

  if (DIM < 3) xnew[2] = xc[2] = 0.0;

  // apply moveperturb() to PKEEP and PINSERT since are computing xnew
  // not to PENTRY,PEXIT since are just re-computing xnew of sender
  // set xnew[2] to linear move for axisymmetry, will be remapped later
  // let pflag = PEXIT persist to check during axisymmetric cell crossing

  if (DIM < 3) xnew[2] = 0.0;
  if (pflag == PKEEP) {
    dtremain = dt;
    // OpenEdge: dispatch to hybrid Boris/GCA when mode==hybrid, else Boris
    if (DIM == 3 && oe_pusher_subcycles > 0 && d_oe_plasma_compute.data()) {
      const int ispecies = particle_i.ispecies;
      const double charge = d_species[ispecies].charge;
      const double mass = d_species[ispecies].mass;
      if (oe_pusher_mode == 1)
        oe_hybrid3d(i, particle_i.icell, dtremain, x, v, xnew, charge, mass);
      else
        oe_boris3d(i, particle_i.icell, dtremain, x, v, xnew, charge, mass);
    } else {
      xnew[0] = x[0] + dtremain*v[0];
      xnew[1] = x[1] + dtremain*v[1];
      if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      if (fstyle == CFIELD) {
        if (DIM == 3) field3d(dtremain,xnew,v);
        else if (DIM == 2) field2d(dtremain,xnew,v);
      } else if (fstyle == PFIELD) field_per_particle(i,particle_i.icell,dtremain,xnew,v);
      else if (fstyle == GFIELD) field_per_grid(i,particle_i.icell,dtremain,xnew,v);
    }
  } else if (pflag == PINSERT) {
    dtremain = particle_i.dtremain;
    // OpenEdge: same hybrid/Boris dispatch for newly inserted particles
    if (DIM == 3 && oe_pusher_subcycles > 0 && d_oe_plasma_compute.data()) {
      const int ispecies = particle_i.ispecies;
      const double charge = d_species[ispecies].charge;
      const double mass = d_species[ispecies].mass;
      if (oe_pusher_mode == 1)
        oe_hybrid3d(i, particle_i.icell, dtremain, x, v, xnew, charge, mass);
      else
        oe_boris3d(i, particle_i.icell, dtremain, x, v, xnew, charge, mass);
    } else {
      xnew[0] = x[0] + dtremain*v[0];
      xnew[1] = x[1] + dtremain*v[1];
      if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
      if (fstyle == CFIELD) {
        if (DIM == 3) field3d(dtremain,xnew,v);
        else if (DIM == 2) field2d(dtremain,xnew,v);
      } else if (fstyle == PFIELD) field_per_particle(i,particle_i.icell,dtremain,xnew,v);
      else if (fstyle == GFIELD) field_per_grid(i,particle_i.icell,dtremain,xnew,v);
    }
  } else if (pflag == PENTRY) {
    icell = particle_i.icell;
    if (d_cells[icell].nsplit > 1) {
      if (DIM == 3 && SURF) icell = split3d(icell,x);
      if (DIM < 3 && SURF) icell = split2d(icell,x);
      particle_i.icell = icell;
    }
    dtremain = particle_i.dtremain;
    xnew[0] = x[0] + dtremain*v[0];
    xnew[1] = x[1] + dtremain*v[1];
    if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
  } else if (pflag == PEXIT) {
    dtremain = particle_i.dtremain;
    xnew[0] = x[0] + dtremain*v[0];
    xnew[1] = x[1] + dtremain*v[1];
    if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
  } else if (pflag >= PSURF) {
    dtremain = particle_i.dtremain;
    xnew[0] = x[0] + dtremain*v[0];
    xnew[1] = x[1] + dtremain*v[1];
    if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
    if (pflag > PSURF) exclude = pflag - PSURF - 1;
  }

  // optimized move

  if (OPT) {
    int optmove = 1;

    if (xnew[0] < xlo || xnew[0] > xhi)
      optmove = 0;

    if (xnew[1] < ylo || xnew[1] > yhi)
      optmove = 0;

    if (DIM == 3) {
      if (xnew[2] < zlo || xnew[2] > zhi)
        optmove = 0;
    }

    if (optmove) {

      const int ip = static_cast<int>((xnew[0] - xlo)/dx);
      const int jp = static_cast<int>((xnew[1] - ylo)/dy);
      int kp = 0;
      if (DIM == 3) kp = static_cast<int>((xnew[2] - zlo)/dz);

      int cellIdx = (kp*ncy + jp)*ncx + ip + 1;
      auto index = hash_kk.find(static_cast<GridKokkos::key_type>(cellIdx));

      // particle moving outside ghost halo will be flagged for standard move

      if (hash_kk.valid_at(index)) {

        int icell = static_cast<int>(hash_kk.value_at(index));

        // reset particle cell and coordinates

        particle_i.icell = icell;
        particle_i.flag = PKEEP;
        x[0] = xnew[0];
        x[1] = xnew[1];
        x[2] = xnew[2];

        if (d_cells[icell].proc != me) {
          int indx;
          if (ATOMIC_REDUCTION == 0) {
            indx = d_nmigrate();
            d_nmigrate()++;
          } else {
            indx = Kokkos::atomic_fetch_add(&d_nmigrate(),1);
          }
          k_mlist.view_device()[indx] = i;

          particle_i.flag = PDONE;

          if (ATOMIC_REDUCTION == 1)
            Kokkos::atomic_inc(&d_ncomm_one());
          else if (ATOMIC_REDUCTION == 0)
            d_ncomm_one()++;
          else
            reduce.ncomm_one++;
        }

        return;
      }
    }
  }

  particle_i.flag = PKEEP;
  icell = particle_i.icell;
  double* lo = d_cells[icell].lo;
  double* hi = d_cells[icell].hi;
  cellint* neigh = d_cells[icell].neigh;
  int nmask = d_cells[icell].nmask;
  stuck_iterate = 0;
  if (ATOMIC_REDUCTION == 1)
    Kokkos::atomic_inc(&d_ntouch_one());
  else if (ATOMIC_REDUCTION == 0)
    d_ntouch_one()++;
  else
    reduce.ntouch_one++;

  // advect one particle from cell to cell and thru surf collides til done

  while (1) {

#ifdef MOVE_DEBUG
    if (DIM == 3) {
      if (ntimestep == MOVE_DEBUG_STEP &&
          (MOVE_DEBUG_ID == d_particles[i].id ||
           (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
        printf("PARTICLE %d %ld: %d %d: %d: x %g %g %g: xnew %g %g %g: %d "
               CELLINT_FORMAT ": lo %g %g %g: hi %g %g %g: DTR %g\n",
               me,ntimestep,i,d_particles[i].id,
               d_cells[icell].nsurf,
               x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
               icell,d_cells[icell].id,
               lo[0],lo[1],lo[2],hi[0],hi[1],hi[2],dtremain);
    }
    if (DIM == 2) {
      if (ntimestep == MOVE_DEBUG_STEP &&
          (MOVE_DEBUG_ID == d_particles[i].id ||
           (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
        printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
               CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
               me,ntimestep,i,d_particles[i].id,
               d_cells[icell].nsurf,
               x[0],x[1],xnew[0],xnew[1],
               icell,d_cells[icell].id,
               lo[0],lo[1],hi[0],hi[1],dtremain);
    }
    if (DIM == 1) {
      if (ntimestep == MOVE_DEBUG_STEP &&
          (MOVE_DEBUG_ID == d_particles[i].id ||
           (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
        printf("PARTICLE %d %ld: %d %d: %d: x %g %g: xnew %g %g: %d "
               CELLINT_FORMAT ": lo %g %g: hi %g %g: DTR: %g\n",
               me,ntimestep,i,d_particles[i].id,
               d_cells[icell].nsurf,
               x[0],x[1],xnew[0],sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]),
               icell,d_cells[icell].id,
               lo[0],lo[1],hi[0],hi[1],dtremain);
    }
#endif

    // check if particle crosses any cell face
    // frac = fraction of move completed before hitting cell face
    // this section should be as efficient as possible,
    //   since most particles won't do anything else
    // axisymmetric cell face crossings:
    //   use linear xnew to check vertical faces
    //   must always check move against curved lower y face of cell
    //   use remapped rnew to check horizontal lines
    //   for y faces, if pflag = PEXIT, particle was just received
    //     from another proc and is exiting this cell from face:
    //       axi_horizontal_line() will not detect correct crossing,
    //       so set frac and outface directly to move into adjacent cell,
    //       then unset pflag so not checked again for this particle

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
      } else if (GeometryKokkos::
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
          if (GeometryKokkos::
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

#ifdef MOVE_DEBUG
    if (ntimestep == MOVE_DEBUG_STEP &&
        (MOVE_DEBUG_ID == d_particles[i].id ||
         (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX))) {
      if (outface != INTERIOR)
        printf("  OUTFACE %d out: %d %d, frac %g\n",
               outface,grid_kk_copy.obj.neigh_decode(nmask,outface),
               neigh[outface],frac);
      else
        printf("  INTERIOR %d %d\n",outface,INTERIOR);
    }
#endif

    // START of code specific to surfaces

    if (SURF) {

      // skip surf checks if particle flagged as EXITing this cell
      // then unset pflag so not checked again for this particle

      nsurf = d_cells[icell].nsurf;
      if (pflag == PEXIT) {
        nsurf = 0;
        pflag = 0;
      }

      if (ATOMIC_REDUCTION == 1)
        Kokkos::atomic_add(&d_nscheck_one(),nsurf);
      else if (ATOMIC_REDUCTION == 0)
        d_nscheck_one() += nsurf;
      else
        reduce.nscheck_one += nsurf;

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
        // if collision occurs, perform collision with surface model
        // reset x,v,xnew,dtremain and continue single particle trajectory

        cflag = 0;
        minparam = 2.0;
        auto csurfs_begin = d_csurfs.row_map(icell);

        for (int m = 0; m < nsurf; m++) {
          isurf = d_csurfs.entries(csurfs_begin + m);

          if (DIM > 1) {
            if (isurf == exclude) continue;
          }
          if (DIM == 3) {
            tri = &d_tris[isurf];
            hitflag = GeometryKokkos::
              line_tri_intersect(x,xnew,
                                 tri->p1,tri->p2,
                                 tri->p3,tri->norm,xc,param,side);
          }
          if (DIM == 2) {
            line = &d_lines[isurf];
            hitflag = GeometryKokkos::
              line_line_intersect(x,xnew,
                                  line->p1,line->p2,
                                  line->norm,xc,param,side);
          }
          if (DIM == 1) {
            line = &d_lines[isurf];
            hitflag = GeometryKokkos::
              axi_line_intersect(dtsurf,x,v,outface,lo,hi,
                                 line->p1,line->p2,
                                 line->norm,exclude == isurf,
                                 xc,vc,param,side);
          }

#ifdef MOVE_DEBUG
          if (DIM == 3) {
            if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                (MOVE_DEBUG_ID == d_particles[i].id ||
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
                (MOVE_DEBUG_ID == d_particles[i].id ||
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
                (MOVE_DEBUG_ID == d_particles[i].id ||
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
            MathExtraKokkos::sub3(line->p2,line->p1,edge1);
            MathExtraKokkos::sub3(x,line->p1,edge2);
            MathExtraKokkos::cross3(edge2,edge1,cross);
            if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                MOVE_DEBUG_ID == d_particles[i].id)
              printf("CROSSSTART %g %g %g\n",cross[0],cross[1],cross[2]);
            xfinal[0] = xnew[0];
            xfinal[1] = sqrt(xnew[1]*xnew[1]+xnew[2]*xnew[2]);
            xfinal[2] = 0.0;
            MathExtraKokkos::sub3(xfinal,line->p1,edge2);
            MathExtraKokkos::cross3(edge2,edge1,cross);
            if (hitflag && ntimestep == MOVE_DEBUG_STEP &&
                MOVE_DEBUG_ID == d_particles[i].id)
              printf("CROSSFINAL %g %g %g\n",cross[0],cross[1],cross[2]);
          }
#endif

          if (hitflag && param < minparam && side == OUTSIDE) {

            // NOTE: these were the old checks
            //       think it is now sufficient to test for particle
            //       in an INSIDE cell in fix grid/check

          //if (hitflag && side != ONSURF2OUT && param <= minparam)

            // this if test is to avoid case where particle
            // previously hit 1 of 2 (or more) touching angled surfs at
            // common edge/corner, on this iteration first surf
            // is excluded, but others may be hit on inside:
            // param will be epsilon and exclude must be set
            // skip the hits of other touching surfs

            //if (side == INSIDE && param < EPSPARAM && exclude >= 0)
            // continue;

            // this if test is to avoid case where particle
            // hits 2 touching angled surfs at common edge/corner
            // from far away:
            // param is same, but hits one on outside, one on inside
            // only keep surf hit on outside

            //if (param == minparam && side == INSIDE) continue;

            cflag = 1;
            minparam = param;
            // minside = side;
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
          if (DIM == 3) tri = &d_tris[minsurf];
          if (DIM != 3) line = &d_lines[minsurf];

          // set x to collision point
          // if axisymmetric, set v to remapped velocity at collision pt

          x[0] = minxc[0];
          x[1] = minxc[1];
          if (DIM == 3) x[2] = minxc[2];
          if (DIM == 1) {
            v[1] = minvc[1];
            v[2] = minvc[2];
          }

          // perform surface collision using surface collision model
          // surface chemistry may destroy particle or create new one
          // must update particle's icell to current icell so that
          //   if jpart is created, it will be added to correct cell
          // if jpart, add new particle to this iteration via pstop++
          // tally surface collision stats if requested using iorig

          ipart = &particle_i;
          ipart->icell = icell;
          dtremain *= 1.0 - minparam*frac;

          if (nsurf_tally)
            iorig = particle_i;
          int n = DIM == 3 ? tri->isc : line->isc;
          int sc_type = sc_type_list[n];
          int m = sc_map[n];

          if (DIM == 3) {
            if (sc_type == 0) {
              jpart = sc_kk_specular_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 1) {
              jpart = sc_kk_diffuse_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 2) {
              jpart = sc_kk_vanish_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 3) {
              jpart = sc_kk_piston_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 4) {
              jpart = sc_kk_transparent_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,tri->norm,tri->isr,reaction,d_retry,d_nlocal);
            }
          }

          if (DIM != 3) {
            if (sc_type == 0) {
              jpart = sc_kk_specular_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,line->norm,line->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 1) {
              jpart = sc_kk_diffuse_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,line->norm,line->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 2) {
              jpart = sc_kk_vanish_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,line->norm,line->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 3) {
              jpart = sc_kk_piston_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,line->norm,line->isr,reaction,d_retry,d_nlocal);
            } else if (sc_type == 4) {
              jpart = sc_kk_transparent_copy[m].obj.
                collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,minsurf,line->norm,line->isr,reaction,d_retry,d_nlocal);
            }
          }

          if (jpart) {
            x = particle_i.x;
            v = particle_i.v;
            jpart->flag = PSURF + 1 + minsurf;
            jpart->dtremain = dtremain;
            jpart->weight = particle_i.weight;
          }

          if (nsurf_tally)
            for (m = 0; m < nsurf_tally; m++)
              slist_active_copy[m].obj.
                    surf_tally_kk<ATOMIC_REDUCTION>(dtremain,minsurf,icell,reaction,&iorig,ipart,jpart);

          // stuck_iterate = consecutive iterations particle is immobile

          if (minparam == 0.0) stuck_iterate++;
          else stuck_iterate = 0;

          // reset post-bounce xnew

          xnew[0] = x[0] + dtremain*v[0];
          xnew[1] = x[1] + dtremain*v[1];
          if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];

          exclude = minsurf;
          if (ATOMIC_REDUCTION == 1)
            Kokkos::atomic_inc(&d_nscollide_one());
          else if (ATOMIC_REDUCTION == 0)
            d_nscollide_one()++;
          else
            reduce.nscollide_one++;

#ifdef MOVE_DEBUG
          if (DIM == 3) {
            if (ntimestep == MOVE_DEBUG_STEP &&
                (MOVE_DEBUG_ID == d_particles[i].id ||
                 (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
              printf("POST COLLISION %d: %g %g %g: %g %g %g: %g %g %g\n",
                     MOVE_DEBUG_INDEX,
                     x[0],x[1],x[2],xnew[0],xnew[1],xnew[2],
                     minparam,frac,dtremain);
          }
          if (DIM == 2) {
            if (ntimestep == MOVE_DEBUG_STEP &&
                (MOVE_DEBUG_ID == d_particles[i].id ||
                 (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
              printf("POST COLLISION %d: %g %g: %g %g: %g %g %g\n",
                     MOVE_DEBUG_INDEX,
                     x[0],x[1],xnew[0],xnew[1],
                     minparam,frac,dtremain);
          }
          if (DIM == 1) {
            if (ntimestep == MOVE_DEBUG_STEP &&
                (MOVE_DEBUG_ID == d_particles[i].id ||
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

          if (ipart == NULL) particle_i.flag = PDISCARD;
          else if (stuck_iterate < MAXSTUCK) continue;
          else {
            particle_i.flag = PDISCARD;
            if (ATOMIC_REDUCTION == 1)
              Kokkos::atomic_inc(&d_nstuck());
            else if (ATOMIC_REDUCTION == 0)
              d_nstuck()++;
            else
              reduce.nstuck++;
          }

        } // END of cflag if section that performed collision

        // no collision, so restore saved xnew if changed it above

        if (outface != INTERIOR) {
          xnew[0] = xhold[0];
          xnew[1] = xhold[1];
          if (DIM != 2) xnew[2] = xhold[2];
        }

      } // END of if test for any surfs in this cell
    } // END of code specific to surfaces

    // break from advection loop if discarding particle

    if (particle_i.flag == PDISCARD) break;

    // no cell crossing
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
          particle_i.flag = PDISCARD;
          if (ATOMIC_REDUCTION == 1)
            Kokkos::atomic_inc(&d_naxibad());
          else if (ATOMIC_REDUCTION == 0)
            d_naxibad()++;
          else
            reduce.naxibad++;
          break;
        }
      }
      if (d_cells[icell].proc != me) particle_i.flag = PDONE;
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

    nflag = grid_kk_copy.obj.neigh_decode(nmask,outface);
    icell_original = icell;

    if (nflag == NCHILD) {
      icell = neigh[outface];
      if (DIM == 3 && SURF) {
        if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
          icell = split3d(icell,x);
      }
      if (DIM < 3 && SURF) {
        if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
          icell = split2d(icell,x);
      }
    } else if (nflag == NPARENT) {
      auto pcell = &d_pcells[neigh[outface]];
      icell = grid_kk_copy.obj.id_find_child(pcell->id,d_cells[icell].level,
                                             pcell->lo,pcell->hi,x);
      if (icell >= 0) {
        if (DIM == 3 && SURF) {
          if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
            icell = split3d(icell,x);
        }
        if (DIM < 3 && SURF) {
          if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
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
    // OTHER = reflected particle stays in same grid cell

    else {
      ipart = &particle_i;

      Particle::OnePart iorig;
      if (nboundary_tally)
        memcpy(&iorig,&particle_i,sizeof(Particle::OnePart));

      // from Domain:

      Particle::OnePart* ipart = &particle_i;
      lo = d_cells[icell].lo;
      hi = d_cells[icell].hi;
      if (domain_kk_copy.obj.bflag[outface] == SURFACE) {
        // treat global boundary as a surface
        // particle velocity is changed by surface collision model
        // dtremain may be changed by collision model
        // reset all components of xnew, in case dtremain changed
        // if axisymmetric, caller will reset again, including xnew[2]

        int n = domain_kk_copy.obj.surf_collide[outface];
        int sc_type = sc_type_list[n];
        int m = sc_map[n];

        if (sc_type == 0)
          jpart = sc_kk_specular_copy[m].obj.
            collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,-(outface+1),domain_kk_copy.obj.norm[outface],domain_kk_copy.obj.surf_react[outface],reaction,d_retry,d_nlocal);
        else if (sc_type == 1)
          jpart = sc_kk_diffuse_copy[m].obj.
            collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,-(outface+1),domain_kk_copy.obj.norm[outface],domain_kk_copy.obj.surf_react[outface],reaction,d_retry,d_nlocal);
        else if (sc_type == 2)
          jpart = sc_kk_vanish_copy[m].obj.
            collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,-(outface+1),domain_kk_copy.obj.norm[outface],domain_kk_copy.obj.surf_react[outface],reaction,d_retry,d_nlocal);
        else if (sc_type == 3)
          jpart = sc_kk_piston_copy[m].obj.
            collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,-(outface+1),domain_kk_copy.obj.norm[outface],domain_kk_copy.obj.surf_react[outface],reaction,d_retry,d_nlocal);
        else if (sc_type == 4)
          jpart = sc_kk_transparent_copy[m].obj.
            collide_kokkos<REACT,ATOMIC_REDUCTION>(ipart,dtremain,-(outface+1),domain_kk_copy.obj.norm[outface],domain_kk_copy.obj.surf_react[outface],reaction,d_retry,d_nlocal);

        if (ipart) {
          double *x = ipart->x;
          double *v = ipart->v;
          xnew[0] = x[0] + dtremain*v[0];
          xnew[1] = x[1] + dtremain*v[1];
          if (domain_kk_copy.obj.dimension == 3) xnew[2] = x[2] + dtremain*v[2];
        }
        bflag = SURFACE;
      } else {
        bflag = domain_kk_copy.obj.collide_kokkos(ipart,outface,lo,hi,xnew/*,dtremain*/,reaction);
      }

      if (jpart) {
        x = particle_i.x;
        v = particle_i.v;
      }

      if (nboundary_tally)
        for (int m = 0; m < nboundary_tally; m++)
          blist_active_copy[m].obj.
            boundary_tally_kk<ATOMIC_REDUCTION>(dtremain,outface,bflag,reaction,&iorig,ipart,jpart,domain_kk_copy.obj.norm[outface]);

      if (DIM == 1) {
        xnew[0] = x[0] + dtremain*v[0];
        xnew[1] = x[1] + dtremain*v[1];
        xnew[2] = x[2] + dtremain*v[2];
      }

      if (bflag == OUTFLOW) {
        particle_i.flag = PDISCARD;
        if (ATOMIC_REDUCTION == 1)
          Kokkos::atomic_inc(&d_nexit_one());
        else if (ATOMIC_REDUCTION == 0)
          d_nexit_one()++;
        else
          reduce.nexit_one++;
        break;
      } else if (bflag == PERIODIC) {
        if (nflag == NPBCHILD) {
          icell = neigh[outface];
          if (DIM == 3 && SURF) {
            if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
              icell = split3d(icell,x);
          }
          if (DIM < 3 && SURF) {
            if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
              icell = split2d(icell,x);
          }
        } else if (nflag == NPBPARENT) {
          auto pcell = &d_pcells[neigh[outface]];
          icell = grid_kk_copy.obj.id_find_child(pcell->id,d_cells[icell].level,
                                                 pcell->lo,pcell->hi,x);
          if (icell >= 0) {
            if (DIM == 3 && SURF) {
              if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
                icell = split3d(icell,x);
            }
            if (DIM < 3 && SURF) {
              if (d_cells[icell].nsplit > 1 && d_cells[icell].nsurf >= 0)
                icell = split2d(icell,x);
            }
          } else domain_kk_copy.obj.uncollide_kokkos(outface,x);
        } else if (nflag == NPBUNKNOWN) {
          icell = -1;
          domain_kk_copy.obj.uncollide_kokkos(outface,x);
        }

      } else if (bflag == SURFACE) {
        if (ipart == NULL) {
          particle_i.flag = PDISCARD;
          break;
        } else if (jpart) {
          jpart->flag = PSURF;
          jpart->dtremain = dtremain;
          jpart->weight = particle_i.weight;
        }

        if (ATOMIC_REDUCTION == 1) {
          Kokkos::atomic_inc(&d_nboundary_one());
          Kokkos::atomic_dec(&d_ntouch_one());    // decrement here since will increment below
        } else if (ATOMIC_REDUCTION == 0) {
          d_nboundary_one()++;
          d_ntouch_one()--;    // decrement here since will increment below
        } else {
          reduce.nboundary_one++;
          reduce.ntouch_one--;    // decrement here since will increment below
        }

      } else {
        if (ATOMIC_REDUCTION == 1) {
          Kokkos::atomic_inc(&d_nboundary_one());
          Kokkos::atomic_dec(&d_ntouch_one());    // decrement here since will increment below
        } else if (ATOMIC_REDUCTION == 0) {
          d_nboundary_one()++;
          d_ntouch_one()--;    // decrement here since will increment below
        } else {
          reduce.nboundary_one++;
          reduce.ntouch_one--;    // decrement here since will increment below
        }
      }
    }

    // neighbor cell is unknown
    // reset icell to original icell which must be a ghost cell
    // exit with particle flag = PEXIT, so receiver can identify neighbor

    if (icell < 0) {
      icell = icell_original;
      particle_i.flag = PEXIT;
      particle_i.dtremain = dtremain;
      d_entryexit() = 1;
      break;
    }

    // if nsurf < 0, new cell is EMPTY ghost
    // exit with particle flag = PENTRY, so receiver can continue move

    if (d_cells[icell].nsurf < 0) {
      particle_i.flag = PENTRY;
      particle_i.dtremain = dtremain;
      d_entryexit() = 1;
      break;
    }

    // move particle into new grid cell for next stage of move

    lo = d_cells[icell].lo;
    hi = d_cells[icell].hi;
    neigh = d_cells[icell].neigh;
    nmask = d_cells[icell].nmask;
    if (ATOMIC_REDUCTION == 1)
      Kokkos::atomic_inc(&d_ntouch_one());
    else if (ATOMIC_REDUCTION == 0)
      d_ntouch_one()++;
    else
      reduce.ntouch_one++;
  }

  // END of while loop over advection of single particle

#ifdef MOVE_DEBUG
  if (ntimestep == MOVE_DEBUG_STEP &&
      (MOVE_DEBUG_ID == d_particles[i].id ||
       (me == MOVE_DEBUG_PROC && i == MOVE_DEBUG_INDEX)))
    printf("MOVE DONE %d %d %d: %g %g %g: DTR %g\n",
           MOVE_DEBUG_INDEX,d_particles[i].flag,icell,
           x[0],x[1],x[2],dtremain);
#endif

  // move is complete, or as much as can be done on this proc
  // update particle's grid cell
  // if particle flag set, add particle to migrate list
  // if discarding, migration will delete particle

  particle_i.icell = icell;

  if (particle_i.flag != PKEEP) {
    int index;
    if (ATOMIC_REDUCTION == 0) {
      index = d_nmigrate();
      d_nmigrate()++;
    } else {
      index = Kokkos::atomic_fetch_add(&d_nmigrate(),1);
    }
    k_mlist.view_device()[index] = i;
    if (particle_i.flag != PDISCARD) {
      if (d_cells[icell].proc == me && !d_error_flag()) {
        d_error_flag() = 1;
        return;
      }
      if (ATOMIC_REDUCTION == 1)
        Kokkos::atomic_inc(&d_ncomm_one());
      else if (ATOMIC_REDUCTION == 0)
        d_ncomm_one()++;
      else
        reduce.ncomm_one++;
    }
  }
} // end of Kokkos parallel_reduce

/* ----------------------------------------------------------------------
   particle is entering split parent icell at x
   determine which split child cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int UpdateKokkos::split3d(int icell, double *x) const
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Tri *tri;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider tris that are mapped via csplits to a split cell
  //   unmapped tris only touch cell surf at xnew
  //   another mapped tri should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particles starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = d_cells[icell].nsurf;
  int isplit = d_cells[icell].isplit;
  double *xnew = d_sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;

  auto csplits_begin = d_csplits.row_map(isplit);
  auto csurfs_begin = d_csurfs.row_map(icell);
  for (m = 0; m < nsurf; m++) {
    if (d_csplits.entries(csplits_begin + m) < 0) continue;
    isurf = d_csurfs.entries(csurfs_begin + m);
    tri = &d_tris[isurf];
    hitflag = GeometryKokkos::
      line_tri_intersect(x,xnew,
                         tri->p1,tri->p2,tri->p3,
                         tri->norm,xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  auto csubs_begin = d_csubs.row_map(isplit);
  if (!cflag) return d_csubs.entries(csubs_begin + d_sinfo[isplit].xsub);
  int index = d_csplits.entries(csplits_begin + minsurfindex);
  return d_csubs.entries(csubs_begin + index);
}

/* ----------------------------------------------------------------------
   particle is entering split ICELL at X
   determine which split sub-cell it is in
   return index of sub-cell in ChildCell
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int UpdateKokkos::split2d(int icell, double *x) const
{
  int m,cflag,isurf,hitflag,side,minsurfindex;
  double param,minparam;
  double xc[3];
  Surf::Line *line;

  // check for collisions with lines in cell
  // find 1st surface hit via minparam
  // only consider lines that are mapped via csplits to a split cell
  //   unmapped lines only touch cell surf at xnew
  //   another mapped line should include same xnew
  // NOTE: these next 2 lines do not seem correct compared to code
  // not considered a collision if particle starts on surf, moving out
  // not considered a collision if 2 params are tied and one is INSIDE surf

  int nsurf = d_cells[icell].nsurf;
  int isplit = d_cells[icell].isplit;
  double *xnew = d_sinfo[isplit].xsplit;

  cflag = 0;
  minparam = 2.0;
  auto csplits_begin = d_csplits.row_map(isplit);
  auto csurfs_begin = d_csurfs.row_map(icell);
  for (m = 0; m < nsurf; m++) {
    if (d_csplits.entries(csplits_begin + m) < 0) continue;
    isurf = d_csurfs.entries(csurfs_begin + m);
    line = &d_lines[isurf];
    hitflag = GeometryKokkos::
      line_line_intersect(x,xnew,
                          line->p1,line->p2,line->norm,
                          xc,param,side);

    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  auto csubs_begin = d_csubs.row_map(isplit);
  if (!cflag) return d_csubs.entries(csubs_begin + d_sinfo[isplit].xsub);
  int index = d_csplits.entries(csplits_begin + minsurfindex);
  return d_csubs.entries(csubs_begin + index);
}

/* ----------------------------------------------------------------------
   set bounce tally flags for current timestep
   nsurf_tally = # of computes needing bounce info on this step
   clear accumulators in computes that will be invoked this step
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   OpenEdge Phase D: build the per-cell sheath spatial-mode cache on host.
   Mirrors what CPU pusher.cpp:566-705 computes per Boris call, but does it
   once per run() (static plasma) and stores per-cell rather than recomputing
   per particle. Layout: 13 cols documented in update_kokkos.h.

   Looks up the nearest surface element from compute_nearest_surf_grid,
   reads cell-center plasma + B from compute_plasma_fields, computes the
   Chodura angle and the physics-derived d_max engagement gate. Pure host
   code — feeds a DualView mirrored to device.

   No sheath_kick gating here: kick mode fires at wall-collision time
   (separate code path), so a kick-mode deck never enters this builder.
------------------------------------------------------------------------- */
void UpdateKokkos::build_oe_sheath_cache()
{
  oe_has_sheath_spatial = 0;
  if (!sheath_flag || sheath_kick) return;
  if (sheath_geom_cidx < 0) return;

  Compute *cg = modify->compute[sheath_geom_cidx];
  auto *csg = dynamic_cast<ComputeNearestSurfGrid*>(cg);
  if (!csg) return;
  if (!(cg->invoked_flag & INVOKED_PER_GRID)) {
    cg->compute_per_grid();
    cg->invoked_flag |= INVOKED_PER_GRID;
  }

  // Plasma provider: prefer ComputePlasmaFields (matches CPU pusher).
  // FixBackground point-query fallback would require host-side bfield_at()
  // calls per cell — not wired in this first cut. Decks that use the
  // bench (test_west_axi) drive sheath via compute plasma/fields.
  ComputePlasmaFields *cp = nullptr;
  if (pusher->pusher_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[pusher->pusher_plasma_cidx];
    cp = dynamic_cast<ComputePlasmaFields*>(cp_base);
    if (cp && !(cp_base->invoked_flag & INVOKED_PER_GRID)) {
      cp_base->compute_per_grid();
      cp_base->invoked_flag |= INVOKED_PER_GRID;
    }
  }
  if (!cp || !cp->plasma_arr || !cp->mag_arr) return;

  const int ng = grid->nlocal;
  const int dim = domain->dimension;
  const int ncols = 13;
  // csg->sgroupbit not needed in this first cut: no per-particle
  // nearest-face refinement, only cell-center midx_grid lookup.

  k_oe_sheath_cell = DAT::tdual_float_2d_lr("oe_sheath_cell", ng, ncols);
  d_oe_sheath_cell = k_oe_sheath_cell.d_view;
  auto h_cache = k_oe_sheath_cell.h_view;

  oe_sheath_mD_amu = sheath_mD_amu;

  Grid::ChildCell *cells = grid->cells;
  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

  int n_active = 0;
  for (int icell = 0; icell < ng; icell++) {
    for (int c = 0; c < ncols; c++) h_cache(icell, c) = 0.0;

    // Resolve to parent cell when in a sub-cell (compute skips sub-cells).
    int gcell = icell;
    if (cells[icell].nsplit <= 0 && cells[icell].isplit >= 0)
      gcell = grid->sinfo[cells[icell].isplit].icell;

    int midx = csg->midx_grid[gcell];
    if (midx < 0) continue;

    // Per-cell sheath cache uses the cell-center-nearest surface from the
    // compute. Per-particle nearest-face refinement (the inner loop in
    // CPU pusher.cpp:598-622 that picks among csurfs) is NOT replicated
    // here — that refinement matters only for thin-slab cells where the
    // cell center is between two faces. The bench deck does not have
    // such cells, so we skip the cost.

    double nx, ny, nz;
    double srefx, srefy, srefz;
    if (dim == 2) {
      Surf::Line *ln = &lines[midx];
      nx = ln->norm[0]; ny = ln->norm[1]; nz = 0.0;
      srefx = 0.5*(ln->p1[0] + ln->p2[0]);
      srefy = 0.5*(ln->p1[1] + ln->p2[1]);
      srefz = 0.0;
    } else {
      Surf::Tri *tr = &tris[midx];
      nx = tr->norm[0]; ny = tr->norm[1]; nz = tr->norm[2];
      srefx = (tr->p1[0] + tr->p2[0] + tr->p3[0]) / 3.0;
      srefy = (tr->p1[1] + tr->p2[1] + tr->p3[1]) / 3.0;
      srefz = (tr->p1[2] + tr->p2[2] + tr->p3[2]) / 3.0;
    }
    const double nmag = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nmag <= 0.0) continue;
    nx /= nmag; ny /= nmag; nz /= nmag;

    const double te = cp->plasma_arr[gcell].temp_e;
    const double ti = cp->plasma_arr[gcell].temp_i;
    const double ne = cp->plasma_arr[gcell].dens_e;
    if (!(te > 0.0) || !(ne > 0.0)) continue;

    // Cylindrical (br, bt, bz) → Cartesian at the surface reference point,
    // matching the per-particle CPU treatment in pusher.cpp:677-689.
    const double br = cp->mag_arr[gcell].br;
    const double bt = cp->mag_arr[gcell].bt;
    const double bz = cp->mag_arr[gcell].bz;
    const double rmag = std::sqrt(srefx*srefx + srefy*srefy);
    double bvec[3];
    if (rmag > 1.0e-20) {
      const double cphi = srefx / rmag, sphi = srefy / rmag;
      bvec[0] = br * cphi - bt * sphi;
      bvec[1] = br * sphi + bt * cphi;
      bvec[2] = bz;
    } else {
      bvec[0] = br; bvec[1] = 0.0; bvec[2] = bz;
    }
    const double bmag = std::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1] + bvec[2]*bvec[2]);
    if (!(bmag > 0.0)) continue;

    double nvec[3] = {nx, ny, nz};
    SheathModels::ChoduraMetrics cm =
      SheathModels::chodura_metrics(0.0, 1.0, bvec, nvec);
    const double alpha_deg = cm.alpha_deg;

    // sheath_auto_dmax replicated from pusher.cpp anonymous-namespace
    // helper; identical formula. Couldn't include pusher.cpp here without
    // pulling the whole class.
    constexpr double QE_LOC   = 1.602176634e-19;
    constexpr double AMU_LOC  = 1.66053906660e-27;
    constexpr double EPS0_LOC = 8.8541878128e-12;
    const double mD_kg = std::max(sheath_mD_amu * AMU_LOC, 1.0e-99);
    const double lambdaD = std::sqrt(EPS0_LOC * std::max(te, 1.0e-12)
                                     / (std::max(ne, 1.0e-60) * QE_LOC));
    const double cs = std::sqrt(std::max(te + ti, 0.0) * QE_LOC / (2.0 * mD_kg));
    const double omega_ci = QE_LOC * std::max(std::fabs(bmag), 1.0e-20) / mD_kg;
    const double rho_i = cs / std::max(omega_ci, 1.0e-99);
    const double alpha_n_rad = std::max(0.0, std::min(90.0, alpha_deg)) * M_PI / 180.0;
    const double tan_an = std::min(std::max(std::fabs(std::tan(alpha_n_rad)),
                                            1.0e-3), 30.0);
    const double L_MPS = rho_i * tan_an;
    const double d_max = std::max(5.0 * L_MPS, 10.0 * lambdaD);

    h_cache(icell, 0)  = nx;       h_cache(icell, 1)  = ny;     h_cache(icell, 2)  = nz;
    h_cache(icell, 3)  = srefx;    h_cache(icell, 4)  = srefy;  h_cache(icell, 5)  = srefz;
    h_cache(icell, 6)  = te;       h_cache(icell, 7)  = ti;     h_cache(icell, 8)  = ne;
    h_cache(icell, 9)  = bmag;
    h_cache(icell, 10) = alpha_deg;
    h_cache(icell, 11) = d_max;
    h_cache(icell, 12) = 1.0;
    n_active++;
  }

  k_oe_sheath_cell.modify_host();
  k_oe_sheath_cell.sync_device();
  d_oe_sheath_cell = k_oe_sheath_cell.d_view;
  oe_has_sheath_spatial = 1;

  // CLAUDE.md MPI trap: Allreduce must be called on every rank — gate
  // only the printf, not the collective.
  int n_active_global = 0, ng_global = 0;
  MPI_Allreduce(&n_active, &n_active_global, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&ng,       &ng_global,       1, MPI_INT, MPI_SUM, world);
  if (comm->me == 0 && screen)
    fprintf(screen, "OpenEdge Phase D: sheath spatial cache built — "
            "%d / %d cells active globally\n",
            n_active_global, ng_global);
}

/* ----------------------------------------------------------------------
   OpenEdge: device-callable Boris 3D pusher.
   Reads B directly from plasma compute device view (bypass field fixes).
   No sheath E-field in this version — added incrementally.
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void UpdateKokkos::oe_boris3d(int i, int icell, double dt_full,
                               double *x, double *v, double *xnew,
                               double charge, double mass) const
{
  // Neutrals: pure advection
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt_full;
    xnew[1] = x[1] + v[1] * dt_full;
    xnew[2] = x[2] + v[2] * dt_full;
    return;
  }

  const double qm = (charge * oe_echarge) / mass;
  const int nsub = (oe_pusher_subcycles > 0) ? oe_pusher_subcycles : 1;
  const double dt_sub = dt_full / static_cast<double>(nsub);

  double xcur[3] = {x[0], x[1], x[2]};
  double vcur[3] = {v[0], v[1], v[2]};

  // Phase D: read this cell's sheath cache once before the subcycle loop.
  // Replicates CPU pusher.cpp:566-736 prefetch — Te, Ti, ne, B, alpha,
  // surface normal & ref point are invariant during subcycling.
  bool sh_active = false;
  double sh_nx = 0.0, sh_ny = 0.0, sh_nz = 0.0;
  double sh_srefx = 0.0, sh_srefy = 0.0, sh_srefz = 0.0;
  double sh_te = 0.0, sh_ti = 0.0, sh_ne = 0.0;
  double sh_bmag = 0.0, sh_alpha = 0.0, sh_dmax = 0.0;
  double sh_d0_sign = 0.0;
  if (oe_has_sheath_spatial && d_oe_sheath_cell.data()
      && icell >= 0 && icell < (int)d_oe_sheath_cell.extent(0)
      && d_oe_sheath_cell(icell, 12) > 0.5) {
    sh_nx    = d_oe_sheath_cell(icell, 0);
    sh_ny    = d_oe_sheath_cell(icell, 1);
    sh_nz    = d_oe_sheath_cell(icell, 2);
    sh_srefx = d_oe_sheath_cell(icell, 3);
    sh_srefy = d_oe_sheath_cell(icell, 4);
    sh_srefz = d_oe_sheath_cell(icell, 5);
    sh_te    = d_oe_sheath_cell(icell, 6);
    sh_ti    = d_oe_sheath_cell(icell, 7);
    sh_ne    = d_oe_sheath_cell(icell, 8);
    sh_bmag  = d_oe_sheath_cell(icell, 9);
    sh_alpha = d_oe_sheath_cell(icell, 10);
    sh_dmax  = d_oe_sheath_cell(icell, 11);
    sh_active = true;
    // d0_sign locks which side of the wall the particle started on.
    // Once set, only apply E-field while the particle stays on that side
    // (matches CPU pusher.cpp:711-718; prevents reverse-field deceleration
    // and outward ejection on overshoot).
    const double d0 = (xcur[0] - sh_srefx) * sh_nx
                    + (xcur[1] - sh_srefy) * sh_ny
                    + (xcur[2] - sh_srefz) * sh_nz;
    sh_d0_sign = (d0 >= 0.0) ? 1.0 : -1.0;
  }

  for (int isub = 0; isub < nsub; isub++) {
    double E[3] = {0.0, 0.0, 0.0};
    double B[3] = {0.0, 0.0, 0.0};

    // Per-particle B point-query (matches CPU dispatch order):
    //   mesh > equilibrium > cell-center fallback.
    bool got_B = false;
    if (oe_has_mesh_b) {
      got_B = MeshKokkos::query_bfield_at_point(
          xcur, oe_dim, oe_axisymmetric,
          d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
          d_oe_mesh_tri_br, d_oe_mesh_tri_bz, d_oe_mesh_tri_bt,
          d_oe_mesh_tri_rmin, d_oe_mesh_tri_rmax,
          d_oe_mesh_tri_zmin, d_oe_mesh_tri_zmax,
          d_oe_hash_offset, d_oe_hash_entries,
          oe_mesh_hash_rmin, oe_mesh_hash_zmin,
          oe_mesh_hash_dr,   oe_mesh_hash_dz,
          oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri,
          B);
    }
    if (!got_B && oe_has_equilibrium) {
      EquilibriumKokkos::query_bfield_at_point(
          xcur, oe_dim, oe_axisymmetric,
          d_oe_equ_r, d_oe_equ_z, d_oe_equ_psi,
          oe_equ_btf, oe_equ_rtf, oe_equ_jm, oe_equ_km,
          B);
      got_B = true;
    }
    if (!got_B && d_oe_plasma_compute.data() && oe_bx_col >= 0) {
      B[0] = d_oe_plasma_compute(icell, oe_bx_col);
      B[1] = d_oe_plasma_compute(icell, oe_by_col);
      B[2] = d_oe_plasma_compute(icell, oe_bz_col);
    }

    // Phase D: per-subcycle sheath E-field via Coulette-Manfredi at the
    // particle's current distance from the wall. Gated on plasma-side
    // start (sh_d0_sign > 0) and physics-derived d_max engagement window.
    if (sh_active && sh_d0_sign > 0.0) {
      const double d_raw = (xcur[0] - sh_srefx) * sh_nx
                         + (xcur[1] - sh_srefy) * sh_ny
                         + (xcur[2] - sh_srefz) * sh_nz;
      if (d_raw > 0.0 && d_raw < sh_dmax) {
        SheathModelsKokkos::BorodkinaSheathResult sr =
          SheathModelsKokkos::coulette_manfredi_sheath_at_distance(
            d_raw, sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha,
            oe_sheath_mD_amu, 0.0);
        // E points INTO the wall (along -n) — unified inward-normal
        // convention from the 2026-04-21 emit/surf overhaul.
        E[0] -= sr.emag_vpm * sh_nx;
        E[1] -= sr.emag_vpm * sh_ny;
        E[2] -= sr.emag_vpm * sh_nz;
      }
    }

    BorisGridKokkos::push_velocity(qm, dt_sub, E, B, vcur);
    xcur[0] += vcur[0] * dt_sub;
    xcur[1] += vcur[1] * dt_sub;
    xcur[2] += vcur[2] * dt_sub;

    // Per-subcycle surface crossing guard (3D only, nsub > 1)
    // Uses existing d_csurfs and d_tris from base SPARTA
    if (nsub > 1) {
      int gcell = icell;
      if (d_cells[icell].nsplit <= 0 && d_cells[icell].isplit >= 0)
        gcell = d_sinfo[d_cells[icell].isplit].icell;

      int nsurf_cell = d_cells[gcell].nsurf;
      if (nsurf_cell > 0) {
        double xold[3] = {xcur[0] - vcur[0]*dt_sub,
                          xcur[1] - vcur[1]*dt_sub,
                          xcur[2] - vcur[2]*dt_sub};
        auto csurfs_begin = d_csurfs.row_map(gcell);
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = d_csurfs.entries(csurfs_begin + m);
          double xc[3], param;
          int side;
          if (GeometryKokkos::line_tri_intersect(
                xold, xcur,
                d_tris[isurf].p1, d_tris[isurf].p2, d_tris[isurf].p3,
                d_tris[isurf].norm, xc, param, side)) {
            // Surface crossing detected — stop subcycling
            v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
            xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];
            return;
          }
        }
      }
    }
  }

  v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
  xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];
}

/* ----------------------------------------------------------------------
   OpenEdge Phase C3: device-callable hybrid Boris/GCA dispatcher.
   Mirrors the no-sheath core of CPU Pusher::push_hybrid_3d:
     - reads B + grad|B| + kappa + curl(b_hat) at particle position
       (equilibrium psi map preferred; mesh and cell-center fall-throughs
       give B-only and so disable the GCA branch for that particle).
     - per-particle decision: GCA when rho_L < L_B / pusher_gca_switch,
       else subcycled Boris fallback.
     - persistent GCA state read/written through C2 device views.
   Sheath E-field stays out of this port — Phase D.
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void UpdateKokkos::oe_hybrid3d(int i, int icell, double dt_full,
                                double *x, double *v, double *xnew,
                                double charge, double mass) const
{
  // Neutrals: pure advection
  if (charge == 0.0) {
    xnew[0] = x[0] + v[0] * dt_full;
    xnew[1] = x[1] + v[1] * dt_full;
    xnew[2] = x[2] + v[2] * dt_full;
    return;
  }

  const double qm = (charge * oe_echarge) / mass;
  const double qm_abs = Kokkos::fabs(qm);

  // --- Read B + (optionally) grad|B|, kappa, curl(b_hat) ---
  double E[3] = {0.0, 0.0, 0.0};
  double B[3] = {0.0, 0.0, 0.0};
  double gradBmag[3] = {0.0, 0.0, 0.0};
  double kappa[3]    = {0.0, 0.0, 0.0};
  double curl_b[3]   = {0.0, 0.0, 0.0};

  bool have_grad = false;
  if (oe_has_equilibrium) {
    have_grad = EquilibriumKokkos::query_bfield_grad_at_point(
        x, oe_dim, oe_axisymmetric,
        d_oe_equ_r, d_oe_equ_z, d_oe_equ_psi,
        oe_equ_btf, oe_equ_rtf, oe_equ_jm, oe_equ_km,
        B, gradBmag, kappa, curl_b);
  }
  bool got_B = have_grad;
  if (!got_B && oe_has_mesh_b) {
    got_B = MeshKokkos::query_bfield_at_point(
        x, oe_dim, oe_axisymmetric,
        d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
        d_oe_mesh_tri_br, d_oe_mesh_tri_bz, d_oe_mesh_tri_bt,
        d_oe_mesh_tri_rmin, d_oe_mesh_tri_rmax,
        d_oe_mesh_tri_zmin, d_oe_mesh_tri_zmax,
        d_oe_hash_offset, d_oe_hash_entries,
        oe_mesh_hash_rmin, oe_mesh_hash_zmin,
        oe_mesh_hash_dr,   oe_mesh_hash_dz,
        oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri,
        B);
  }
  if (!got_B && d_oe_plasma_compute.data() && oe_bx_col >= 0) {
    B[0] = d_oe_plasma_compute(icell, oe_bx_col);
    B[1] = d_oe_plasma_compute(icell, oe_by_col);
    B[2] = d_oe_plasma_compute(icell, oe_bz_col);
  }

  // Phase D: read this cell's sheath cache once. Mirrors the CPU
  // pusher.cpp:1104-1221 hybrid-mode sheath block (which precedes the
  // GCA-vs-Boris switching decision).
  bool sh_active = false;
  double sh_nx = 0.0, sh_ny = 0.0, sh_nz = 0.0;
  double sh_srefx = 0.0, sh_srefy = 0.0, sh_srefz = 0.0;
  double sh_te = 0.0, sh_ti = 0.0, sh_ne = 0.0;
  double sh_bmag = 0.0, sh_alpha = 0.0, sh_dmax = 0.0;
  double sh_d0_sign = 0.0;
  if (oe_has_sheath_spatial && d_oe_sheath_cell.data()
      && icell >= 0 && icell < (int)d_oe_sheath_cell.extent(0)
      && d_oe_sheath_cell(icell, 12) > 0.5) {
    sh_nx    = d_oe_sheath_cell(icell, 0);
    sh_ny    = d_oe_sheath_cell(icell, 1);
    sh_nz    = d_oe_sheath_cell(icell, 2);
    sh_srefx = d_oe_sheath_cell(icell, 3);
    sh_srefy = d_oe_sheath_cell(icell, 4);
    sh_srefz = d_oe_sheath_cell(icell, 5);
    sh_te    = d_oe_sheath_cell(icell, 6);
    sh_ti    = d_oe_sheath_cell(icell, 7);
    sh_ne    = d_oe_sheath_cell(icell, 8);
    sh_bmag  = d_oe_sheath_cell(icell, 9);
    sh_alpha = d_oe_sheath_cell(icell, 10);
    sh_dmax  = d_oe_sheath_cell(icell, 11);
    sh_active = true;
    const double d0 = (x[0] - sh_srefx) * sh_nx
                    + (x[1] - sh_srefy) * sh_ny
                    + (x[2] - sh_srefz) * sh_nz;
    sh_d0_sign = (d0 >= 0.0) ? 1.0 : -1.0;

    // Pre-evaluate sheath E at initial particle position. This is what
    // the GCA branch sees (single E used during the whole RK4 step). The
    // Boris-fallback branch overrides it per subcycle below.
    if (sh_d0_sign > 0.0) {
      const double d_raw = (x[0] - sh_srefx) * sh_nx
                         + (x[1] - sh_srefy) * sh_ny
                         + (x[2] - sh_srefz) * sh_nz;
      if (d_raw > 0.0 && d_raw < sh_dmax) {
        SheathModelsKokkos::BorodkinaSheathResult sr =
          SheathModelsKokkos::coulette_manfredi_sheath_at_distance(
            d_raw, sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha,
            oe_sheath_mD_amu, 0.0);
        E[0] -= sr.emag_vpm * sh_nx;
        E[1] -= sr.emag_vpm * sh_ny;
        E[2] -= sr.emag_vpm * sh_nz;
      }
    }
  }

  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  const double gradBmag_mag = Kokkos::sqrt(gradBmag[0]*gradBmag[0]
                                         + gradBmag[1]*gradBmag[1]
                                         + gradBmag[2]*gradBmag[2]);

  // --- Decide GCA vs Boris fallback per particle ---
  bool use_gca = false;
  if (have_grad && Bmag > 0.0 && qm_abs > 0.0 && oe_has_gca_state) {
    const double bhx = B[0]/Bmag, bhy = B[1]/Bmag, bhz = B[2]/Bmag;
    double v_perp = 0.0;
    if (d_oe_gca_on(i) > 0.5) {
      const double mu_eff = (d_oe_gca_mu(i) > 0.0) ? d_oe_gca_mu(i) : 0.0;
      const double vperp2 = (2.0 * mu_eff * Bmag) / mass;
      v_perp = (vperp2 > 0.0) ? Kokkos::sqrt(vperp2) : 0.0;
    } else {
      const double v_par = v[0]*bhx + v[1]*bhy + v[2]*bhz;
      const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
      double vperp2 = v2 - v_par*v_par;
      if (vperp2 < 0.0) vperp2 = 0.0;
      v_perp = Kokkos::sqrt(vperp2);
    }
    const double rho_L = GCAPusherKokkos::larmor_radius(v_perp, qm_abs, Bmag);
    const double L_B   = GCAPusherKokkos::grad_b_length(Bmag, gradBmag_mag);
    if (rho_L > 0.0)
      use_gca = (rho_L < L_B / oe_pusher_gca_switch);
  }

  if (use_gca) {
    GCAPusherKokkos::GCAState st;
    if (d_oe_gca_on(i) > 0.5) {
      st.X[0]  = d_oe_gca_x(i);
      st.X[1]  = d_oe_gca_y(i);
      st.X[2]  = d_oe_gca_z(i);
      st.v_par = d_oe_gca_vpar(i);
      st.mu    = (d_oe_gca_mu(i) > 0.0) ? d_oe_gca_mu(i) : 0.0;
    } else {
      st = GCAPusherKokkos::init_from_particle(x, v, mass, B);
    }
    GCAPusherKokkos::push_gca_rk4(qm, dt_full, mass, E, B, Bmag,
                                  gradBmag, kappa, curl_b, st);
    d_oe_gca_x(i)    = st.X[0];
    d_oe_gca_y(i)    = st.X[1];
    d_oe_gca_z(i)    = st.X[2];
    d_oe_gca_vpar(i) = st.v_par;
    d_oe_gca_mu(i)   = st.mu;
    d_oe_gca_on(i)   = 1.0;

    // Reconstruct full v from GC state for diagnostics + Boris fallback.
    // Phase derived from particle ID alone (deterministic, no ntimestep
    // capture from device). Slightly different from CPU's id+omega*dt
    // mixing, but adequate for orbit shape + future Boris fallback.
    const double phi_golden = 0.6180339887498949;
    const double pid = static_cast<double>(d_particles(i).id);
    double phase_turns = pid * phi_golden;
    const double rand_u = phase_turns - Kokkos::floor(phase_turns);
    GCAPusherKokkos::gca_to_particle(st, B, mass, rand_u, xnew, v);
  } else {
    if (oe_has_gca_state) d_oe_gca_on(i) = 0.0;
    const int nsub = (oe_pusher_subcycles > 0) ? oe_pusher_subcycles : 1;
    const double dt_sub = dt_full / static_cast<double>(nsub);
    double xcur[3] = {x[0], x[1], x[2]};
    double vcur[3] = {v[0], v[1], v[2]};
    for (int s = 0; s < nsub; s++) {
      // Per-subcycle E: keep whatever was set above (incl. baseline
      // sheath at x), then refresh sheath at the current position.
      double E_sub[3] = {E[0], E[1], E[2]};
      if (sh_active && sh_d0_sign > 0.0) {
        // Overwrite the baseline sheath contribution with one evaluated
        // at xcur. (Baseline above was at x; here we evaluate at xcur
        // each subcycle for the same physical reason as oe_boris3d.)
        // Strip the baseline first by re-zeroing the sheath term, then
        // re-add at xcur. Simpler: re-evaluate from a clean E.
        E_sub[0] = 0.0; E_sub[1] = 0.0; E_sub[2] = 0.0;
        const double d_raw = (xcur[0] - sh_srefx) * sh_nx
                           + (xcur[1] - sh_srefy) * sh_ny
                           + (xcur[2] - sh_srefz) * sh_nz;
        if (d_raw > 0.0 && d_raw < sh_dmax) {
          SheathModelsKokkos::BorodkinaSheathResult sr =
            SheathModelsKokkos::coulette_manfredi_sheath_at_distance(
              d_raw, sh_te, sh_ti, sh_ne, sh_bmag, sh_alpha,
              oe_sheath_mD_amu, 0.0);
          E_sub[0] -= sr.emag_vpm * sh_nx;
          E_sub[1] -= sr.emag_vpm * sh_ny;
          E_sub[2] -= sr.emag_vpm * sh_nz;
        }
      }
      BorisGridKokkos::push_velocity(qm, dt_sub, E_sub, B, vcur);
      xcur[0] += vcur[0] * dt_sub;
      xcur[1] += vcur[1] * dt_sub;
      xcur[2] += vcur[2] * dt_sub;
    }
    v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
    xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];
  }
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::bounce_set(bigint ntimestep)
void UpdateKokkos::tally_set(bigint ntimestep)
{
  Update::tally_set(ntimestep);

  int i;

  if (nboundary_tally > KOKKOS_MAX_BLIST)
    error->all(FLERR,"Kokkos currently only supports two instances of compute boundary");

  if (nboundary_tally) {
    for (i = 0; i < nboundary_tally; i++) {
      ComputeBoundaryKokkos* compute_boundary_kk = (ComputeBoundaryKokkos*)(blist_active[i]);
      compute_boundary_kk->pre_boundary_tally();
      blist_active_copy[i].copy(compute_boundary_kk);
    }
  }

  if (nsurf_tally > KOKKOS_MAX_SLIST)
    error->all(FLERR,"Kokkos currently only supports two instances of compute surface");

  if (nsurf_tally) {
    for (i = 0; i < nsurf_tally; i++) {
      if (strcmp(slist_active[i]->style,"isurf/grid") == 0)
        error->all(FLERR,"Kokkos doesn't yet support compute isurf/grid");
      ComputeSurfKokkos* compute_surf_kk = dynamic_cast<ComputeSurfKokkos*>(slist_active[i]);
      if (!compute_surf_kk)
        error->all(FLERR,"Kokkos does not (yet) support compute surf/collision/tally or compute surf/reaction/tally");
      compute_surf_kk->pre_surf_tally();
      slist_active_copy[i].copy(compute_surf_kk);
    }
  }

  if (ngas_tally)
    error->all(FLERR,"Kokkos does not (yet) support tallying gas/gas collisions or reactions");
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::backup()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  d_particles = particle_kk->k_particles.view_device();
  d_particles_backup = decltype(d_particles)(Kokkos::view_alloc("update:particles_backup",Kokkos::WithoutInitializing),d_particles.extent(0));

  Kokkos::deep_copy(d_particles_backup,d_particles);

  if (surf->nsc > 0) {
    int nspec,ndiff,npist;
    nspec = ndiff = npist = 0;
    for (int n = 0; n < surf->nsc; n++) {
      if (strcmp(surf->sc[n]->style,"specular") == 0) {
        sc_kk_specular_copy[nspec].obj.backup();
        nspec++;
      } else if (strcmp(surf->sc[n]->style,"diffuse") == 0) {
        sc_kk_diffuse_copy[ndiff].obj.backup();
        ndiff++;
      } else if (strcmp(surf->sc[n]->style,"piston") == 0) {
        sc_kk_piston_copy[npist].obj.backup();
        npist++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void UpdateKokkos::restore()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  Kokkos::deep_copy(particle_kk->k_particles.view_device(),d_particles_backup);
  d_particles = particle_kk->k_particles.view_device();

  if (surf->nsc > 0) {
    int nspec,ndiff,npist;
    nspec = ndiff = npist = 0;
    for (int n = 0; n < surf->nsc; n++) {
      if (strcmp(surf->sc[n]->style,"specular") == 0) {
        sc_kk_specular_copy[nspec].obj.restore();
        nspec++;
      } else if (strcmp(surf->sc[n]->style,"diffuse") == 0) {
        sc_kk_diffuse_copy[ndiff].obj.restore();
        ndiff++;
      } else if (strcmp(surf->sc[n]->style,"piston") == 0) {
        sc_kk_piston_copy[npist].obj.restore();
        npist++;
      }
    }
  }

  // deallocate references to reduce memory use

  d_particles_backup = {};
}
