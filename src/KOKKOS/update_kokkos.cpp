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
#include "fix_background.h"
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
#define VAL_3(X) VAL_2(X), VAL_1(X)

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
  slist_active_copy{VAL_3(KKCopy<ComputeSurfKokkos>(sparta))},
  tmp_compute_boundary_kk(sparta),
  tmp_compute_surf_kk(sparta)
{

  // use 1D view for scalars to reduce GPU memory operations

  d_scalars = t_int_14("update:scalars");
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

  tmp_compute_boundary_kk.uncopy = 1;
  tmp_compute_surf_kk.uncopy = 1;

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
  oe_echarge = echarge;
  oe_bx_col = oe_by_col = oe_bz_col = -1;

  // OpenEdge Phase A: equilibrium-based point-query B (defaults off).
  // Actual binding to ComputePlasmaFieldsKokkos's d_equ_* views happens
  // at the same site where d_oe_plasma_compute is bound (see below).
  oe_has_equilibrium = 0;
  oe_has_equ_bmaps = 0;
  oe_plasma_kkbase = NULL;
  oe_equ_jm = oe_equ_km = 0;
  oe_equ_btf = oe_equ_rtf = 0.0;
  oe_dim = domain->dimension;
  oe_axisymmetric = domain->axisymmetric;

  // OpenEdge Phase B: mesh-triangulation B (defaults off).
  oe_has_mesh_b = 0;
  oe_has_mesh_e = 0;
  oe_mesh_ntri = 0;
  oe_mesh_hash_nr = oe_mesh_hash_nz = 0;
  oe_mesh_hash_rmin = oe_mesh_hash_zmin = 0.0;
  oe_mesh_hash_dr = oe_mesh_hash_dz = 1.0;

  // OpenEdge Phase D: sheath spatial-mode data (defaults off).
  oe_sheath_provider = 0;
  oe_sheath_sgroupbit = 0;
  oe_sheath_stamp_n = -1;
  oe_sheath_stamp_id = (cellint) -1;
  oe_sheath_mD_amu = sheath_mD_amu;
  oe_sheath_dmax_user = sheath_dmax;
  oe_col_x0 = oe_col_y0 = 0.0;
  oe_pcache_dev = 0;
  oe_pc_mask = 0;
  oe_pc_csg = 0;
  oe_cd_dev = 0;
  oe_has_mesh_plasma = 0;
  oe_has_mesh_drag = 0;
  oe_has_mesh_gradte = 0;
  oe_has_mesh_gradti = 0;
  oe_has_sheath_customs = 0;

  // Spatial-sheath engagement diagnostics (device twins of the CPU
  // sheath_diag_* counters; `global pusher ... dump yes` enables them).
  oe_sheath_diag = (pusher->pusher_dump_flag &&
                    sheath_flag && !sheath_kick) ? 1 : 0;
  if (oe_sheath_diag) {
    d_oe_shd_counts = DAT::t_int_1d("oe_shd_counts",3);
    d_oe_shd_esum = Kokkos::View<double*,DeviceType>("oe_shd_esum",2);
  }
  // Per-particle sheath trace (parity debugging): OE_SHEATH_TRACE_ID=<id>
  oe_trace_id = getenv("OE_SHEATH_TRACE_ID")
    ? atol(getenv("OE_SHEATH_TRACE_ID")) : -1;

  // OpenEdge: unported pusher/sheath modes fail loudly instead of
  // silently diverging. The old device hybrid/GCA port (oe_hybrid3d)
  // encoded physics since removed from the CPU pusher (pre-selector
  // sheath force; switching without the Boris shell / hysteresis /
  // trial-replay) and was deleted 2026-08-26. Kick/boundary sheath modes
  // fire host-side machinery the Kokkos mover never calls (decision
  // 2026-08-25: spatial is the production mode; do not port).
  if (oe_pusher_subcycles > 0 &&
      pusher->pusher_mode != Pusher::PUSHER_BORIS)
    error->all(FLERR,"Pusher mode hybrid/gca is not supported with Kokkos; "
               "use global pusher mode boris");
  if (sheath_flag && (sheath_kick || sheath_boundary))
    error->all(FLERR,"Sheath kick/boundary modes are not supported with "
               "Kokkos; use sheath spatial");
  // the device Boris dispatch is DIM == 3 only: a 2D/axisymmetric deck
  // with a configured pusher would silently advect ions ballistically
  if (oe_pusher_subcycles > 0 && domain->dimension != 3)
    error->all(FLERR,"The Kokkos pusher is 3D-only in this version; "
               "run 2D/axisymmetric pusher decks on the CPU build");
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
      oe_plasma_kkbase = kk_cp;
      // Column mapping from compute: bx=0, by=1, bz=2 (first 3 values)
      oe_bx_col = 0; oe_by_col = 1; oe_bz_col = 2;
    } else if (cp) {
      // fail loudly instead of moving ions ballistically
      error->all(FLERR,"Kokkos pusher: plasma compute is not "
                 "Kokkos-enabled (use the /kk variant)");
    }

    // Phase A: bind to the device-resident equilibrium psi map (if any)
    // for smooth point-query B inside oe_boris3d. Falls back to cell-center
    // columns when no equilibrium is loaded.
    auto *cp_pf = dynamic_cast<ComputePlasmaFieldsKokkos*>(cp);
    if (cp_pf) {
      oe_col_x0 = cp_pf->plasma_data.column_x0;
      oe_col_y0 = cp_pf->plasma_data.column_y0;
    }
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

  }

  // OpenEdge: fix-background plasma provider (static SOLPS/SOLEDGE3X
  // file, e.g. the WEST monoblock deck): build the device mesh B/E views
  // directly from the fix so the Boris pusher runs on the device for
  // these decks too (previously oe_boris3d was silently skipped and ions
  // moved ballistically)

  else if (oe_pusher_subcycles > 0 && pusher->pusher_plasma_fidx >= 0)
    build_oe_mesh_from_fix();

  // Phase D: build the spatial-mode sheath data once at run() setup
  // (per-element cache for fix provider, per-cell plasma for compute
  // provider). Static-plasma assumption: not refreshed per step.
  // Triggered only when `global pusher ... sheath spatial geom <ID>` is
  // configured. Outside the compute-provider block so fix-background
  // decks get it too.

  if (oe_pusher_subcycles > 0 &&
      sheath_flag && !sheath_kick && sheath_geom_cidx >= 0)
    build_oe_sheath_cache();

  // OpenEdge: per-species pusher bypass (dust grains advect ballistically
  // even when charged — their forces live in the grain fixes; mirrors the
  // pusher_skip_flag check at the top of the CPU Boris kernels)

  d_oe_pusher_skip = DAT::t_int_1d();
  if (oe_pusher_subcycles > 0 && pusher->pusher_skip_flag) {
    const int nspecies = particle->nspecies;
    d_oe_pusher_skip = DAT::t_int_1d("oe_pusher_skip",nspecies);
    auto h_skip = Kokkos::create_mirror_view(d_oe_pusher_skip);
    for (int s = 0; s < nspecies; s++)
      h_skip(s) = pusher->pusher_skip_flag[s];
    Kokkos::deep_copy(d_oe_pusher_skip,h_skip);
  }

  // cellweightflag = 1 if grid-based particle weighting is ON

  int cellweightflag = 0;
  if (grid->cellweightflag) cellweightflag = 1;

  // loop over timesteps


  // OpenEdge gate 9b: decide once per run whether the plasma cache can
  // be filled on the device. The host fill costs a full particle+custom
  // D2H sync every step (the TIME_PCACHE bucket, ~15% at 1000x load).
  oe_pcache_dev = 0;
  oe_pc_csg = (sheath_flag && sheath_geom_cidx >= 0) ? 1 : 0;
  if (plasma_cache_flag) {
    const char *why = nullptr;
    const int sup = PCACHE_TE | PCACHE_NE | PCACHE_TI | PCACHE_NI |
                    PCACHE_VPAR | PCACHE_BFIELD;
    FixBackground *pdc = nullptr;
    if (pusher->pusher_plasma_fidx >= 0)
      pdc = dynamic_cast<FixBackground*>(
                modify->fix[pusher->pusher_plasma_fidx]);
    auto okc = [&](int cidx) {
      return cidx >= 0 && particle->ewhich[cidx] >= 0;
    };
    if (getenv("OE_PCACHE_HOST"))
      why = "OE_PCACHE_HOST env override";
    else if (domain->dimension != 3)
      why = "2D/axisymmetric (device fill is 3D-only)";
    else if (!oe_has_mesh_b || !oe_has_mesh_plasma)
      why = "device mesh B/plasma views not built (fix-provider mesh decks only)";
    else if (pcache_need_mask & ~sup)
      why = "unsupported cache slots (gradients / E-field)";
    else if ((pcache_need_mask & (PCACHE_NI | PCACHE_VPAR)) &&
             !oe_has_mesh_drag)
      why = "mesh ni/upar fields absent";
    else if (pdc && pdc->has_const_bfield())
      why = "constant-B branch (device B chain is mesh/equilibrium only)";
    else if (oe_pc_csg &&
             !(oe_sheath_provider && d_oe_midx_gcell.data()))
      why = "sheath ne correction needs the device sheath cache";
    else if (((pcache_need_mask & PCACHE_TE) && !okc(pc_te_custom)) ||
             ((pcache_need_mask & PCACHE_TI) && !okc(pc_ti_custom)) ||
             ((pcache_need_mask & PCACHE_NE) && !okc(pc_ne_custom)) ||
             ((pcache_need_mask & PCACHE_NI) && !okc(pc_ni_custom)) ||
             ((pcache_need_mask & PCACHE_VPAR) && !okc(pc_vpar_custom)) ||
             ((pcache_need_mask & PCACHE_BFIELD) &&
              !(okc(pc_bx_custom) && okc(pc_by_custom) && okc(pc_bz_custom))))
      why = "pcache custom slots unresolved";
    oe_pcache_dev = (why == nullptr);
    if (comm->me == 0 && screen) {
      if (oe_pcache_dev)
        fprintf(screen,"  [kokkos] pcache: DEVICE fill active (mask 0x%x%s)\n",
                pcache_need_mask,
                oe_pc_csg ? ", sheath ne correction" : "");
      else
        fprintf(screen,"  [kokkos] pcache: host fill (%s)\n",why);
    }
  }

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

    // cache plasma fields at particle positions for host-side consumers
    // (fix volume/chem/adas etc.), same cadence and code path as the CPU
    // mover. The query machinery (fix background mesh interpolation,
    // sheath Boltzmann correction) is host-resident: sync particles to
    // the host, fill the per-particle custom vectors there, and mark
    // them stale on the device. Device-side port = later optimization.

    // OpenEdge BUGFIX (2026-08-26, gate-6 "Bug F"): fix balance
    // re-decomposes the grid mid-run (every 200 steps in the monoblock
    // deck). The spatial-sheath maps built at run() setup are indexed by
    // LOCAL cell id (d_oe_midx_gcell, d_oe_sheath_cellplasma) and went
    // STALE after every rebalance — near-wall ions then looked up wrong
    // elements/geometry, a fraction lost their sheath pull, and a
    // deep-dwelling high-charge population accumulated (+24-42% Np,
    // ladder-tail excess; absent at 1 rank where balance is a no-op,
    // growing with rank count). Rebuild the cache when the
    // decomposition stamps change. The CPU pusher is immune: it reads
    // the live compute per particle-move.
    // rebind the compute-provider per-cell view every step: rebalance
    // can remap local cells and the compute can reallocate
    // d_array_grid mid-run - a view captured at run() setup then
    // dangles (same Bug-F class as the sheath maps)
    if (oe_plasma_kkbase)
      d_oe_plasma_compute = oe_plasma_kkbase->d_array_grid;

    if (oe_sheath_provider) {
      const int nloc_stamp = grid->nlocal;
      const cellint fid_stamp = (nloc_stamp > 0 && grid->cells)
        ? grid->cells[0].id : (cellint) -1;
      // CLAUDE.md MPI trap: the stamps are per-rank and a rebalance can
      // change them on a SUBSET of ranks, but build_oe_sheath_cache()
      // contains collectives — make the rebuild decision collective.
      int need = (nloc_stamp != oe_sheath_stamp_n ||
                  fid_stamp != oe_sheath_stamp_id) ? 1 : 0;
      int need_any = 0;
      MPI_Allreduce(&need,&need_any,1,MPI_INT,MPI_MAX,world);
      if (need_any) build_oe_sheath_cache();
    }

    if (plasma_cache_flag &&
        (pcache_nevery <= 1 || ntimestep % pcache_nevery == 0)) {
      if (oe_pcache_dev) {
        cache_plasma_particles_device();
      } else {
        particle_kk->sync(Host,PARTICLE_MASK|CUSTOM_MASK);
        cache_plasma_particles();
        particle_kk->modify(Host,CUSTOM_MASK);
      }
      timer->stamp(TIME_PCACHE);
    }

    // upload cross-field diffusion displacements (filled on the host by
    // fix cross_field_diffusion at start_of_step) for the device mover

    if (cd_flag && cd_nmax > 0 && !oe_cd_dev) {
      if ((int) d_dx_cd.extent(0) < cd_nmax) {
        d_dx_cd = DAT::t_float_2d_lr(Kokkos::view_alloc("update:dx_cd",
                    Kokkos::WithoutInitializing),cd_nmax,3);
        h_dx_cd = Kokkos::create_mirror_view(d_dx_cd);
      }
      for (int p = 0; p < cd_nmax; p++) {
        h_dx_cd(p,0) = dx_cd[p][0];
        h_dx_cd(p,1) = dx_cd[p][1];
        h_dx_cd(p,2) = dx_cd[p][2];
      }
      Kokkos::deep_copy(d_dx_cd,h_dx_cd);
      timer->stamp(TIME_PCACHE);
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

  modify->post_run();

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

  // OpenEdge: reset the spatial-sheath engagement diagnostics for this step
  if (oe_sheath_diag) {
    Kokkos::deep_copy(d_oe_shd_counts,0);
    Kokkos::deep_copy(d_oe_shd_esum,0.0);
  }

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

      // OpenEdge Phase D: (re)bind the spatial-sheath custom device views
      // each attempt — particle growth (retry, react) reallocates the
      // underlying dual views. The kernel writes bank/phiprev; they are
      // marked device-modified after the particle loop below, and
      // backup()/restore() snapshot them so a react/retry replay does not
      // double-apply the potential impulse.
      oe_has_sheath_customs = 0;
      if (oe_sheath_provider && sheath_bank_custom >= 0 &&
          sheath_phiprev_custom >= 0) {
        particle_kk->sync(Device,CUSTOM_MASK);
        d_oe_sheath_bank = particle_kk->k_edvec.h_view[
            particle->ewhich[sheath_bank_custom]].k_view.d_view;
        d_oe_sheath_phiprev = particle_kk->k_edvec.h_view[
            particle->ewhich[sheath_phiprev_custom]].k_view.d_view;
        oe_has_sheath_customs = 1;
      }

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
    if (oe_has_sheath_customs)
      particle_kk->modify(Device,CUSTOM_MASK);
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

  // OpenEdge: spatial-sheath engagement diagnostics — same format and
  // gating as the CPU print at the end of Update::move().
  if (oe_sheath_diag && (ntimestep % pusher->pusher_dump_every == 0)) {
    auto h_cnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     d_oe_shd_counts);
    auto h_es  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     d_oe_shd_esum);
    long loc[3] = {(long)h_cnt(0),(long)h_cnt(1),(long)h_cnt(2)};
    long glob[3] = {0,0,0};
    double es_loc = h_es(0), es_glob = 0.0;
    double em_loc = h_es(1), em_glob = 0.0;
    MPI_Reduce(loc,glob,3,MPI_LONG,MPI_SUM,0,world);
    MPI_Reduce(&es_loc,&es_glob,1,MPI_DOUBLE,MPI_SUM,0,world);
    MPI_Reduce(&em_loc,&em_glob,1,MPI_DOUBLE,MPI_MAX,0,world);
    if (comm->me == 0) {
      FILE *fp = screen ? screen : logfile;
      if (fp) {
        const double emean = glob[1] > 0 ? es_glob / glob[1] : 0.0;
        fprintf(fp, "  [kk] sheath step " BIGINT_FORMAT " [spatial]: "
                "near-wall=%ld engaged=%ld turnrefl=%ld "
                "|E_sheath| mean=%.3e max=%.3e V/m\n",
                ntimestep, glob[0], glob[1], glob[2], emean, em_glob);
      }
    }
  }

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
  // cross-field diffusion kick bookkeeping, same semantics as update.cpp
  double vkick0 = 0.0, vkick1 = 0.0, vkick2 = 0.0;
  int has_kick = 0;
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
    // OpenEdge: device Boris mover (hybrid/GCA errors out at init)
    if (DIM == 3 && oe_pusher_subcycles > 0 && (d_oe_plasma_compute.data() || oe_has_mesh_b)) {
      const int ispecies = particle_i.ispecies;
      const double charge = d_species[ispecies].charge;
      const double mass = d_species[ispecies].mass;
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

    // apply cross-field diffusion displacement (if active), same
    // policy as the CPU mover: PKEEP only, indices past the buffer get
    // their first kick next step. Kick goes into v AND xnew so the
    // traced chord xnew = x + dtremain*v stays exact; stripped again at
    // the first velocity-transforming event or post-move bookkeeping.

    if (cd_flag && i < cd_nmax && i < (int) d_dx_cd.extent(0)) {
      vkick0 = d_dx_cd(i,0) / dtremain;
      vkick1 = d_dx_cd(i,1) / dtremain;
      v[0] += vkick0;
      v[1] += vkick1;
      xnew[0] += d_dx_cd(i,0);
      xnew[1] += d_dx_cd(i,1);
      if (DIM == 3) {
        vkick2 = d_dx_cd(i,2) / dtremain;
        v[2] += vkick2;
        xnew[2] += d_dx_cd(i,2);
      }
      has_kick = 1;
    }
  } else if (pflag == PINSERT) {
    dtremain = particle_i.dtremain;
    // OpenEdge: same Boris dispatch for newly inserted particles
    if (DIM == 3 && oe_pusher_subcycles > 0 && (d_oe_plasma_compute.data() || oe_has_mesh_b)) {
      const int ispecies = particle_i.ispecies;
      const double charge = d_species[ispecies].charge;
      const double mass = d_species[ispecies].mass;
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

          // strip the cross-field kick BEFORE the collision model so
          // PWI physics never sees the phantom kick velocity; if the
          // stripped velocity no longer points at the surface, treat as
          // a graze: no collision physics, continue with own velocity
          // (same semantics as update.cpp)

          if (has_kick) {
            v[0] -= vkick0;
            v[1] -= vkick1;
            if (DIM == 3) v[2] -= vkick2;
            has_kick = 0;

            const double *nrm_k = (DIM == 3) ? tri->norm : line->norm;
            if (v[0]*nrm_k[0] + v[1]*nrm_k[1] + v[2]*nrm_k[2] >= 0.0) {
              dtremain *= 1.0 - minparam*frac;
              if (minparam == 0.0) stuck_iterate++;
              else stuck_iterate = 0;
              if (stuck_iterate >= MAXSTUCK) {
                particle_i.flag = PDISCARD;
                if (ATOMIC_REDUCTION == 1)
                  Kokkos::atomic_inc(&d_nstuck());
                else if (ATOMIC_REDUCTION == 0)
                  d_nstuck()++;
                else
                  reduce.nstuck++;
                break;
              }
              xnew[0] = x[0] + dtremain*v[0];
              xnew[1] = x[1] + dtremain*v[1];
              if (DIM != 2) xnew[2] = x[2] + dtremain*v[2];
              exclude = minsurf;
              continue;
            }
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
      // strip the cross-field kick before migrating mid-move: the
      // receiver rebuilds xnew = x + dtremain*v and cannot strip later
      if (has_kick) {
        v[0] -= vkick0;
        v[1] -= vkick1;
        if (DIM == 3) v[2] -= vkick2;
        has_kick = 0;
      }
      break;
    }

    // if nsurf < 0, new cell is EMPTY ghost
    // exit with particle flag = PENTRY, so receiver can continue move

    if (d_cells[icell].nsurf < 0) {
      particle_i.flag = PENTRY;
      particle_i.dtremain = dtremain;
      d_entryexit() = 1;
      // same mid-move migration strip as the PEXIT case above
      if (has_kick) {
        v[0] -= vkick0;
        v[1] -= vkick1;
        if (DIM == 3) v[2] -= vkick2;
        has_kick = 0;
      }
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

  // strip the cross-field diffusion kick now that this step's move is
  // complete: the kick models a position random walk, not heating
  // (same as update.cpp post_move_bookkeeping)

  if (has_kick &&
      (particle_i.flag == PKEEP || particle_i.flag == PDONE)) {
    v[0] -= vkick0;
    v[1] -= vkick1;
    if (DIM == 3) v[2] -= vkick2;
  }

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
   OpenEdge: build the device mesh B/E views directly from FixBackground
   (static SOLPS/SOLEDGE3X plasma file). Mirrors the ComputePlasmaFields-
   Kokkos upload: per-tri fields, per-tri bounding boxes, CSR spatial
   hash. E components live per mesh CELL in the fix; flattened here to
   per-tri via mesh_cell_idx so the same device point-query serves both.
   Plasma te/ti/ne are flattened the same way for the spatial-sheath
   per-particle fallback (CPU interp2D's mesh branch is tri-constant via
   mesh_cell_idx, so this reproduces it exactly).
------------------------------------------------------------------------- */

void UpdateKokkos::build_oe_mesh_from_fix()
{
  oe_has_mesh_b = 0;
  oe_has_mesh_e = 0;
  oe_has_mesh_plasma = 0;
  // reset the full flag family, not just b/e/plasma: leftover
  // drag/gradient/equilibrium flags from a previous run would otherwise
  // outlive a plasma source that no longer provides them
  oe_has_mesh_drag = 0;
  oe_has_mesh_gradte = 0;
  oe_has_mesh_gradti = 0;
  oe_has_equilibrium = 0;
  oe_has_equ_bmaps = 0;
  if (pusher->pusher_plasma_fidx < 0) return;
  FixBackground *pd =
    dynamic_cast<FixBackground*>(modify->fix[pusher->pusher_plasma_fidx]);
  const int have_mesh_b = pd && pd->has_mesh && pd->mesh_nvtx > 0 &&
    pd->mesh_ntri > 0 &&
    (int) pd->mesh_tri_br.size() == pd->mesh_ntri;
  if (!have_mesh_b) {
    // fail loudly instead of advecting ions ballistically: the CPU
    // point-query chain still serves B for equilibrium-only and
    // constant-B decks, but the device equ binding below is only built
    // alongside a mesh, so without one the pusher would silently no-op
    if (pd && (!pd->equ_r.empty() || pd->has_const_bfield()))
      error->all(FLERR,"Kokkos pusher: fix background has no mesh B "
                 "(equilibrium-only / constant-B decks are not supported "
                 "on the device mover; use the CPU build)");
    return;
  }
  const int nvtx = pd->mesh_nvtx;
  const int ntri = pd->mesh_ntri;
  // column-axis offset for all device point queries (also set by the
  // sheath-cache builder, but decks without spatial sheath land here)
  oe_col_x0 = pd->column_x0;
  oe_col_y0 = pd->column_y0;

  oe_dim = domain->dimension;
  oe_axisymmetric = domain->axisymmetric;

  d_oe_mesh_vtx_r    = DAT::t_float_1d("oe_mesh_vtx_r",nvtx);
  d_oe_mesh_vtx_z    = DAT::t_float_1d("oe_mesh_vtx_z",nvtx);
  d_oe_mesh_tri      = DAT::t_int_1d("oe_mesh_tri",ntri*3);
  d_oe_mesh_tri_br   = DAT::t_float_1d("oe_mesh_tri_br",ntri);
  d_oe_mesh_tri_bz   = DAT::t_float_1d("oe_mesh_tri_bz",ntri);
  d_oe_mesh_tri_bt   = DAT::t_float_1d("oe_mesh_tri_bt",ntri);
  d_oe_mesh_tri_rmin = DAT::t_float_1d("oe_mesh_tri_rmin",ntri);
  d_oe_mesh_tri_rmax = DAT::t_float_1d("oe_mesh_tri_rmax",ntri);
  d_oe_mesh_tri_zmin = DAT::t_float_1d("oe_mesh_tri_zmin",ntri);
  d_oe_mesh_tri_zmax = DAT::t_float_1d("oe_mesh_tri_zmax",ntri);

  auto h_vtx_r    = Kokkos::create_mirror_view(d_oe_mesh_vtx_r);
  auto h_vtx_z    = Kokkos::create_mirror_view(d_oe_mesh_vtx_z);
  auto h_tri      = Kokkos::create_mirror_view(d_oe_mesh_tri);
  auto h_tri_br   = Kokkos::create_mirror_view(d_oe_mesh_tri_br);
  auto h_tri_bz   = Kokkos::create_mirror_view(d_oe_mesh_tri_bz);
  auto h_tri_bt   = Kokkos::create_mirror_view(d_oe_mesh_tri_bt);
  auto h_tri_rmin = Kokkos::create_mirror_view(d_oe_mesh_tri_rmin);
  auto h_tri_rmax = Kokkos::create_mirror_view(d_oe_mesh_tri_rmax);
  auto h_tri_zmin = Kokkos::create_mirror_view(d_oe_mesh_tri_zmin);
  auto h_tri_zmax = Kokkos::create_mirror_view(d_oe_mesh_tri_zmax);

  double mesh_rmin = 1.0e30, mesh_rmax = -1.0e30;
  double mesh_zmin = 1.0e30, mesh_zmax = -1.0e30;
  for (int i = 0; i < nvtx; i++) {
    h_vtx_r(i) = pd->mesh_vtx_r[i];
    h_vtx_z(i) = pd->mesh_vtx_z[i];
    mesh_rmin = std::min(mesh_rmin,pd->mesh_vtx_r[i]);
    mesh_rmax = std::max(mesh_rmax,pd->mesh_vtx_r[i]);
    mesh_zmin = std::min(mesh_zmin,pd->mesh_vtx_z[i]);
    mesh_zmax = std::max(mesh_zmax,pd->mesh_vtx_z[i]);
  }
  for (int t = 0; t < ntri; t++) {
    const int v0 = pd->mesh_tri[3*t+0];
    const int v1 = pd->mesh_tri[3*t+1];
    const int v2 = pd->mesh_tri[3*t+2];
    h_tri(3*t+0) = v0;
    h_tri(3*t+1) = v1;
    h_tri(3*t+2) = v2;
    h_tri_br(t)  = pd->mesh_tri_br[t];
    h_tri_bz(t)  = pd->mesh_tri_bz[t];
    h_tri_bt(t)  = pd->mesh_tri_bt[t];
    h_tri_rmin(t) = std::min(pd->mesh_vtx_r[v0],
                    std::min(pd->mesh_vtx_r[v1],pd->mesh_vtx_r[v2]));
    h_tri_rmax(t) = std::max(pd->mesh_vtx_r[v0],
                    std::max(pd->mesh_vtx_r[v1],pd->mesh_vtx_r[v2]));
    h_tri_zmin(t) = std::min(pd->mesh_vtx_z[v0],
                    std::min(pd->mesh_vtx_z[v1],pd->mesh_vtx_z[v2]));
    h_tri_zmax(t) = std::max(pd->mesh_vtx_z[v0],
                    std::max(pd->mesh_vtx_z[v1],pd->mesh_vtx_z[v2]));
  }

  Kokkos::deep_copy(d_oe_mesh_vtx_r,h_vtx_r);
  Kokkos::deep_copy(d_oe_mesh_vtx_z,h_vtx_z);
  Kokkos::deep_copy(d_oe_mesh_tri,h_tri);
  Kokkos::deep_copy(d_oe_mesh_tri_br,h_tri_br);
  Kokkos::deep_copy(d_oe_mesh_tri_bz,h_tri_bz);
  Kokkos::deep_copy(d_oe_mesh_tri_bt,h_tri_bt);
  Kokkos::deep_copy(d_oe_mesh_tri_rmin,h_tri_rmin);
  Kokkos::deep_copy(d_oe_mesh_tri_rmax,h_tri_rmax);
  Kokkos::deep_copy(d_oe_mesh_tri_zmin,h_tri_zmin);
  Kokkos::deep_copy(d_oe_mesh_tri_zmax,h_tri_zmax);

  // CSR spatial hash: bin triangles by bounding-box overlap

  const int nr = 256, nz = 256;
  const double dr = (mesh_rmax - mesh_rmin) / nr;
  const double dz = (mesh_zmax - mesh_zmin) / nz;
  if (dr > 0.0 && dz > 0.0) {
    const int nbins = nr*nz;
    std::vector<std::vector<int>> bins(nbins);
    for (int t = 0; t < ntri; t++) {
      int j0 = (int) ((h_tri_rmin(t) - mesh_rmin) / dr);
      int j1 = (int) ((h_tri_rmax(t) - mesh_rmin) / dr);
      int k0 = (int) ((h_tri_zmin(t) - mesh_zmin) / dz);
      int k1 = (int) ((h_tri_zmax(t) - mesh_zmin) / dz);
      j0 = std::max(0,std::min(nr-1,j0));
      j1 = std::max(0,std::min(nr-1,j1));
      k0 = std::max(0,std::min(nz-1,k0));
      k1 = std::max(0,std::min(nz-1,k1));
      for (int k = k0; k <= k1; k++)
        for (int j = j0; j <= j1; j++)
          bins[k*nr + j].push_back(t);
    }
    std::vector<int> offset(nbins+1,0);
    for (int b = 0; b < nbins; b++)
      offset[b+1] = offset[b] + (int) bins[b].size();
    const int ntotal = offset[nbins];

    d_oe_hash_offset  = DAT::t_int_1d("oe_hash_offset",nbins+1);
    d_oe_hash_entries = DAT::t_int_1d("oe_hash_entries",ntotal);
    auto h_offset  = Kokkos::create_mirror_view(d_oe_hash_offset);
    auto h_entries = Kokkos::create_mirror_view(d_oe_hash_entries);
    for (int b = 0; b <= nbins; b++) h_offset(b) = offset[b];
    int k = 0;
    for (int b = 0; b < nbins; b++)
      for (int t : bins[b]) h_entries(k++) = t;
    Kokkos::deep_copy(d_oe_hash_offset,h_offset);
    Kokkos::deep_copy(d_oe_hash_entries,h_entries);

    oe_mesh_hash_rmin = mesh_rmin;
    oe_mesh_hash_zmin = mesh_zmin;
    oe_mesh_hash_dr   = dr;
    oe_mesh_hash_dz   = dz;
    oe_mesh_hash_nr   = nr;
    oe_mesh_hash_nz   = nz;
  } else {
    // NOTE: a degenerate hash disables mesh sampling on device entirely
    // (device tri location returns a miss for hash_nr <= 0, matching the
    // CPU find_mesh_triangle_hash miss) — there is no brute-force scan
    oe_mesh_hash_nr = oe_mesh_hash_nz = 0;
  }

  oe_mesh_ntri = ntri;
  oe_has_mesh_b = 1;

  // E-field: per mesh CELL in the fix (cylindrical E_R, E_Z, E_t),
  // flattened to per-tri via mesh_cell_idx

  const int ncellE = (int) pd->mesh_e_r.size();
  if (ncellE > 0 && (int) pd->mesh_e_z.size() == ncellE &&
      (int) pd->mesh_e_t.size() == ncellE) {
    d_oe_mesh_tri_er = DAT::t_float_1d("oe_mesh_tri_er",ntri);
    d_oe_mesh_tri_ez = DAT::t_float_1d("oe_mesh_tri_ez",ntri);
    d_oe_mesh_tri_et = DAT::t_float_1d("oe_mesh_tri_et",ntri);
    auto h_er = Kokkos::create_mirror_view(d_oe_mesh_tri_er);
    auto h_ez = Kokkos::create_mirror_view(d_oe_mesh_tri_ez);
    auto h_et = Kokkos::create_mirror_view(d_oe_mesh_tri_et);
    const bool have_map = ((int) pd->mesh_cell_idx.size() == ntri);
    for (int t = 0; t < ntri; t++) {
      int c = have_map ? pd->mesh_cell_idx[t] : t;
      // unmapped tri: store 0 (CPU interp2D falls through to the
      // raster / 0 for such tris; substituting cell t was wrong)
      if (c < 0 || c >= ncellE) c = -1;
      h_er(t) = (c < 0) ? 0.0 : pd->mesh_e_r[c];
      h_ez(t) = (c < 0) ? 0.0 : pd->mesh_e_z[c];
      h_et(t) = (c < 0) ? 0.0 : pd->mesh_e_t[c];
    }
    Kokkos::deep_copy(d_oe_mesh_tri_er,h_er);
    Kokkos::deep_copy(d_oe_mesh_tri_ez,h_ez);
    Kokkos::deep_copy(d_oe_mesh_tri_et,h_et);
    oe_has_mesh_e = 1;
  }

  // Plasma te/ti/ne per mesh CELL -> per-tri, for the spatial-sheath
  // per-particle fallback (wall elements whose centroid falls outside
  // the plasma-mesh footprint).

  const int ncellP = (int) pd->mesh_te.size();
  if (ncellP > 0 && (int) pd->mesh_ti.size() == ncellP &&
      (int) pd->mesh_ne.size() == ncellP) {
    d_oe_mesh_tri_te = DAT::t_float_1d("oe_mesh_tri_te",ntri);
    d_oe_mesh_tri_ti = DAT::t_float_1d("oe_mesh_tri_ti",ntri);
    d_oe_mesh_tri_ne = DAT::t_float_1d("oe_mesh_tri_ne",ntri);
    auto h_te = Kokkos::create_mirror_view(d_oe_mesh_tri_te);
    auto h_ti = Kokkos::create_mirror_view(d_oe_mesh_tri_ti);
    auto h_ne = Kokkos::create_mirror_view(d_oe_mesh_tri_ne);
    const bool have_map = ((int) pd->mesh_cell_idx.size() == ntri);
    for (int t = 0; t < ntri; t++) {
      int c = have_map ? pd->mesh_cell_idx[t] : t;
      // unmapped tri: store 0 (CPU interp2D falls through to the
      // raster / 0 for such tris; substituting cell t was wrong)
      if (c < 0 || c >= ncellP) c = -1;
      h_te(t) = (c < 0) ? 0.0 : pd->mesh_te[c];
      h_ti(t) = (c < 0) ? 0.0 : pd->mesh_ti[c];
      h_ne(t) = (c < 0) ? 0.0 : pd->mesh_ne[c];
    }
    Kokkos::deep_copy(d_oe_mesh_tri_te,h_te);
    Kokkos::deep_copy(d_oe_mesh_tri_ti,h_ti);
    Kokkos::deep_copy(d_oe_mesh_tri_ne,h_ne);
    oe_has_mesh_plasma = 1;
  }

  // OpenEdge gate 9: per-cell -> per-tri flattening for the device
  // coulomb drag (ni, upar) and thermal force (grad-T) fixes. Same
  // mesh_cell_idx mapping as te/ti/ne so the device point-query
  // reproduces interp2D's mesh branch exactly.

  {
    auto flatten = [&](const std::vector<double> &src, const char *label,
                       DAT::t_float_1d &dst) -> int {
      const int nc = (int) src.size();
      if (nc <= 0) return 0;
      dst = DAT::t_float_1d(std::string(label),ntri);
      auto h = Kokkos::create_mirror_view(dst);
      const bool have_map = ((int) pd->mesh_cell_idx.size() == ntri);
      for (int t = 0; t < ntri; t++) {
        int c = have_map ? pd->mesh_cell_idx[t] : t;
        if (c < 0 || c >= nc) { h(t) = 0.0; continue; }  // unmapped tri: CPU falls through to raster/0
        h(t) = src[c];
      }
      Kokkos::deep_copy(dst,h);
      return 1;
    };
    const int have_ni   = flatten(pd->mesh_ni,  "oe_mesh_tri_ni",
                                  d_oe_mesh_tri_ni);
    const int have_upar = flatten(pd->mesh_upar,"oe_mesh_tri_upar",
                                  d_oe_mesh_tri_upar);
    oe_has_mesh_drag = (have_ni && have_upar);

    // gradients: PER MESH CELL, direct copy — the host pd_grad consumer
    // indexes mesh_grad_*[cell_mesh_cell[icell]] (SPARTA-cell centroid's
    // mesh cell), so no tri flattening here
    auto upload_cells = [&](const std::vector<double> &src,
                            const char *label,
                            DAT::t_float_1d &dst) -> int {
      const int nc = (int) src.size();
      if (nc <= 0) return 0;
      dst = DAT::t_float_1d(std::string(label),nc);
      auto h = Kokkos::create_mirror_view(dst);
      for (int c = 0; c < nc; c++) h(c) = src[c];
      Kokkos::deep_copy(dst,h);
      return 1;
    };
    oe_has_mesh_gradte =
      upload_cells(pd->mesh_grad_te_r,"oe_meshcell_gter",d_oe_meshcell_gter) &&
      upload_cells(pd->mesh_grad_te_z,"oe_meshcell_gtez",d_oe_meshcell_gtez);
    oe_has_mesh_gradti =
      upload_cells(pd->mesh_grad_ti_r,"oe_meshcell_gtir",d_oe_meshcell_gtir) &&
      upload_cells(pd->mesh_grad_ti_z,"oe_meshcell_gtiz",d_oe_meshcell_gtiz);
  }

  // Equilibrium psi map from the fix: the CPU bfield_at chain is
  // mesh -> equilibrium -> const; the mover's device dispatch already
  // implements mesh-miss -> equilibrium (it was only ever bound from
  // the compute provider before). Binding it here closes a documented
  // divergence: outside-footprint particles moved ballistically on
  // device while the CPU used equilibrium B (gate 9a fthcmp finding).

  if (pd->has_equ && pd->equ_jm > 1 && pd->equ_km > 1 &&
      (int) pd->psirz.size() >= pd->equ_jm * pd->equ_km &&
      (int) pd->equ_r.size() >= pd->equ_jm &&
      (int) pd->equ_z.size() >= pd->equ_km) {
    const int jm = pd->equ_jm, km = pd->equ_km;
    d_oe_equ_r   = DAT::t_float_1d("oe_equ_r",jm);
    d_oe_equ_z   = DAT::t_float_1d("oe_equ_z",km);
    d_oe_equ_psi = DAT::t_float_2d_lr("oe_equ_psi",km,jm);
    auto h_r   = Kokkos::create_mirror_view(d_oe_equ_r);
    auto h_z   = Kokkos::create_mirror_view(d_oe_equ_z);
    auto h_psi = Kokkos::create_mirror_view(d_oe_equ_psi);
    for (int j = 0; j < jm; j++) h_r(j) = pd->equ_r[j];
    for (int k = 0; k < km; k++) h_z(k) = pd->equ_z[k];
    for (int k = 0; k < km; k++)
      for (int j = 0; j < jm; j++)
        h_psi(k,j) = pd->psirz[(size_t)k*jm + j];
    Kokkos::deep_copy(d_oe_equ_r,h_r);
    Kokkos::deep_copy(d_oe_equ_z,h_z);
    Kokkos::deep_copy(d_oe_equ_psi,h_psi);
    oe_equ_btf = pd->btf;
    oe_equ_rtf = pd->rtf;
    oe_equ_jm  = jm;
    oe_equ_km  = km;
    oe_has_equilibrium = 1;

    // native B maps (preferred over psi-derived, slag b05b4687)
    const size_t equ_n = (size_t) jm * km;
    if (pd->equ_br.size() == equ_n && pd->equ_bt.size() == equ_n &&
        pd->equ_bz.size() == equ_n) {
      d_oe_equ_br = DAT::t_float_2d_lr("oe_equ_br",km,jm);
      d_oe_equ_bt = DAT::t_float_2d_lr("oe_equ_bt",km,jm);
      d_oe_equ_bz = DAT::t_float_2d_lr("oe_equ_bz",km,jm);
      auto h_br = Kokkos::create_mirror_view(d_oe_equ_br);
      auto h_bt = Kokkos::create_mirror_view(d_oe_equ_bt);
      auto h_bz = Kokkos::create_mirror_view(d_oe_equ_bz);
      for (int k = 0; k < km; k++)
        for (int j = 0; j < jm; j++) {
          h_br(k,j) = pd->equ_br[(size_t)k*jm + j];
          h_bt(k,j) = pd->equ_bt[(size_t)k*jm + j];
          h_bz(k,j) = pd->equ_bz[(size_t)k*jm + j];
        }
      Kokkos::deep_copy(d_oe_equ_br,h_br);
      Kokkos::deep_copy(d_oe_equ_bt,h_bt);
      Kokkos::deep_copy(d_oe_equ_bz,h_bz);
      oe_has_equ_bmaps = 1;
    }
  }

  if (comm->me == 0 && screen)
    fprintf(screen,"  [kokkos] mesh B/E bound from fix background: "
            "%d tris, E-field %s, plasma %s, drag %s, gradTe %s, "
            "gradTi %s, equ-fallback %s\n",ntri,
            oe_has_mesh_e ? "yes" : "no",
            oe_has_mesh_plasma ? "yes" : "no",
            oe_has_mesh_drag ? "yes" : "no",
            oe_has_mesh_gradte ? "yes" : "no",
            oe_has_mesh_gradti ? "yes" : "no",
            oe_has_equilibrium ? "yes" : "no");
}

/* ----------------------------------------------------------------------
   OpenEdge Phase D (rev 2): build the spatial-mode sheath data on host.

   CPU-parity design (2026-08-26). The CPU pusher engages the sheath per
   PARTICLE: nearest-surf element for the particle's cell from
   compute nearest_surf/grid, refined against the cell's own csurfs by
   particle distance; geometry (raw normal + centroid) comes from the
   CHOSEN element; the plasma-derived Coulette-Manfredi coefficients and
   d_max come from a per-ELEMENT cache built once at the element centroid
   (fix-background static plasma) or per-particle (compute provider).
   The old per-CELL device cache froze geometry + plasma at the
   cell-center-nearest element, used the removed tan(alpha) d_max formula
   and missed the CM fit's 90-alpha convention — the prime suspect for
   the soft PWI impact ladder in gate 6.

   Here:
   - k_oe_midx_gcell: the compute's per-cell nearest element (all
     providers); refined per particle on the device.
   - fix provider: k_oe_sheath_elem, one row per surf element in the
     sheath group, built by calling the CPU builder
     Pusher::build_sheath_cache_entry_3d directly — plasma/B queries and
     coefficient prep are byte-identical to the CPU cache.
   - compute provider: k_oe_sheath_cellplasma (te,ti,ne,br,bt,bz per
     cell); the device derives alpha/d_max/coefficients per particle the
     way the CPU per-particle path does.
------------------------------------------------------------------------- */

void UpdateKokkos::build_oe_sheath_cache()
{
  oe_sheath_provider = 0;
  if (!sheath_flag || sheath_kick) return;
  if (sheath_geom_cidx < 0) return;
  if (domain->dimension != 3) return;   // device mover is 3D-only

  Compute *cg = modify->compute[sheath_geom_cidx];
  auto *csg = dynamic_cast<ComputeNearestSurfGrid*>(cg);
  if (!csg) return;
  // ALWAYS recompute: this builder also runs after a fix-balance
  // re-decomposition, where a stale invoked_flag would hand back
  // pre-balance midx values for the new local cell numbering.
  cg->compute_per_grid();
  cg->invoked_flag |= INVOKED_PER_GRID;
  oe_sheath_sgroupbit = csg->sgroupbit;

  // Per-cell nearest-surf element map. The compute writes parent cells;
  // the device resolves sub-cell -> parent itself, so a straight copy of
  // the local-cell array is enough.
  const int ng = grid->nlocal;
  k_oe_midx_gcell = DAT::tdual_int_1d("oe_midx_gcell",ng);
  {
    auto h_midx = k_oe_midx_gcell.h_view;
    for (int icell = 0; icell < ng; icell++)
      h_midx(icell) = csg->midx_grid[icell];
  }
  k_oe_midx_gcell.modify_host();
  k_oe_midx_gcell.sync_device();
  d_oe_midx_gcell = k_oe_midx_gcell.d_view;

  // Plasma provider resolution, same order as the CPU pusher.
  ComputePlasmaFields *cp = nullptr;
  if (pusher->pusher_plasma_cidx >= 0) {
    Compute *cp_base = modify->compute[pusher->pusher_plasma_cidx];
    cp = dynamic_cast<ComputePlasmaFields*>(cp_base);
    if (cp && !(cp_base->invoked_flag & INVOKED_PER_GRID)) {
      cp_base->compute_per_grid();
      cp_base->invoked_flag |= INVOKED_PER_GRID;
    }
  }
  // Do NOT null cp when plasma_arr/mag_arr are NULL: that happens
  // per-rank (zero local cells) and this function contains
  // collectives - nulling made the early return a SUBSET collective
  // (the gate-6/8 deadlock class). With grid->nlocal == 0 the fill
  // loops below iterate zero times, so keeping cp is safe.

  FixBackground *pd = nullptr;
  if (!cp && pusher->pusher_plasma_fidx >= 0)
    pd = dynamic_cast<FixBackground*>(modify->fix[pusher->pusher_plasma_fidx]);
  if (!cp && !pd) return;

  Grid::ChildCell *cells = grid->cells;

  if (pd) {

    // ---- fix-background provider: per-ELEMENT coefficient cache ----

    if (!pusher->sheath_cache_enabled)
      error->all(FLERR,"OE_NO_SHEATH_CACHE is not supported with Kokkos "
                 "(the device sheath consumes the per-element cache)");

    oe_col_x0 = pd->column_x0;
    oe_col_y0 = pd->column_y0;

    const int nsurf_all = surf->nlocal + surf->nghost;
    const int ncols = 12;
    k_oe_sheath_elem = DAT::tdual_float_2d_lr("oe_sheath_elem",
                                              nsurf_all,ncols);
    auto h_elem = k_oe_sheath_elem.h_view;
    Surf::Tri *tris = surf->tris;

    int n_group = 0, n_active = 0;
    for (int m = 0; m < nsurf_all; m++) {
      for (int c = 0; c < ncols; c++) h_elem(m,c) = 0.0;
      if (!(tris[m].mask & oe_sheath_sgroupbit)) continue;
      n_group++;
      Pusher::SheathElemCache C;
      C.state = 0;
      pusher->build_sheath_cache_entry_3d(m,C);
      if (C.state != 1) continue;
      const SheathModels::SheathEmagCoeffs &cf = C.coeffs;
      h_elem(m,0)  = 1.0;
      h_elem(m,1)  = C.d_max;
      h_elem(m,2)  = C.phi_total;
      h_elem(m,3)  = cf.lambdaD_m;
      h_elem(m,4)  = cf.lmps_m;
      h_elem(m,5)  = cf.inv_lD;
      h_elem(m,6)  = cf.inv_lmps;
      h_elem(m,7)  = cf.K1_scaled;
      h_elem(m,8)  = cf.K2;
      h_elem(m,9)  = cf.phi_cm_slow_eV;
      h_elem(m,10) = cf.phi_cm_fast_eV;
      h_elem(m,11) = cf.e_slow_at_anchor_vpm;
      n_active++;
    }

    k_oe_sheath_elem.modify_host();
    k_oe_sheath_elem.sync_device();
    d_oe_sheath_elem = k_oe_sheath_elem.d_view;
    oe_sheath_provider = 1;

    // CLAUDE.md MPI trap: collectives on every rank, printf gated only.
    // MAX, not SUM: with non-distributed surfs every rank builds the
    // same replicated element list, so a SUM would inflate by nranks.
    int ncnt[2] = {n_group,n_active}, gcnt[2];
    MPI_Allreduce(ncnt,gcnt,2,MPI_INT,MPI_MAX,world);
    if (comm->me == 0 && screen)
      fprintf(screen,"OpenEdge Phase D: per-element sheath cache — "
              "%d of %d group elements active (provider=fix)\n",
              gcnt[1],gcnt[0]);

  } else {

    // ---- compute provider: per-cell raw plasma; derived per particle ----

    oe_col_x0 = cp->plasma_data.column_x0;
    oe_col_y0 = cp->plasma_data.column_y0;

    const int ncols = 6;
    k_oe_sheath_cellplasma = DAT::tdual_float_2d_lr("oe_sheath_cellplasma",
                                                    ng,ncols);
    auto h_cpl = k_oe_sheath_cellplasma.h_view;
    int n_active = 0;
    for (int icell = 0; icell < ng; icell++) {
      int gcell = icell;
      if (cells[icell].nsplit <= 0 && cells[icell].isplit >= 0)
        gcell = grid->sinfo[cells[icell].isplit].icell;
      h_cpl(icell,0) = cp->plasma_arr[gcell].temp_e;
      h_cpl(icell,1) = cp->plasma_arr[gcell].temp_i;
      h_cpl(icell,2) = cp->plasma_arr[gcell].dens_e;
      h_cpl(icell,3) = cp->mag_arr[gcell].br;
      h_cpl(icell,4) = cp->mag_arr[gcell].bt;
      h_cpl(icell,5) = cp->mag_arr[gcell].bz;
      if (h_cpl(icell,0) > 0.0 && h_cpl(icell,2) > 0.0) n_active++;
    }

    k_oe_sheath_cellplasma.modify_host();
    k_oe_sheath_cellplasma.sync_device();
    d_oe_sheath_cellplasma = k_oe_sheath_cellplasma.d_view;
    oe_sheath_provider = 2;

    int ncnt[2] = {ng,n_active}, gcnt[2];
    MPI_Allreduce(ncnt,gcnt,2,MPI_INT,MPI_SUM,world);
    if (comm->me == 0 && screen)
      fprintf(screen,"OpenEdge Phase D: per-cell sheath plasma — "
              "%d of %d cells with te,ne > 0 globally (provider=compute)\n",
              gcnt[1],gcnt[0]);
  }

  // decomposition stamps for the mid-run rebuild check in run()
  oe_sheath_stamp_n = grid->nlocal;
  oe_sheath_stamp_id = (grid->nlocal > 0 && grid->cells)
    ? grid->cells[0].id : (cellint) -1;
}

/* ----------------------------------------------------------------------
   OpenEdge: device-callable Boris 3D pusher — mirror of CPU
   Pusher::push_boris_3d for the device-supported paths (mode boris,
   spatial-mode sheath, mesh/equilibrium/cell-center B, mesh E):

   - B is point-queried ONCE at the start position and frozen across
     subcycles (the CPU caches B the same way).
   - E (mesh -grad phi) is re-queried per subcycle at the current position.
   - Spatial sheath, CPU-parity rev 2 (2026-08-26): per-particle nearest
     element (cell map + csurfs refinement by particle distance), geometry
     from the CHOSEN tri (raw normal + centroid), Coulette-Manfredi
     coefficients from the per-element cache (fix provider) or prepared
     per particle (compute provider / footprint fallback). Applied as the
     CPU's energy-consistent potential impulse AFTER each position update
     — dKE = Z e [phi(d_new) - phi(d_old)], phi clamped at the wall, with
     the per-particle phi reference (sheath_phiprev) and lifetime bank cap
     (sheath_bank) customs and elastic reflection at turning points. The
     old per-subcycle sheath E-force (removed from the CPU as
     non-conservative) is gone.
   - Per-subcycle guard: an in-cell wall hit clips xnew to the
     intersection point; a cell exit bails to the outer move loop — both
     as on the CPU.
   NOTE: cylindrical->Cartesian rotations here are about the origin; the
   CPU rotates about (column_x0, column_y0). The production decks carry
   no column offset (0,0); build_oe_sheath_cache still records
   oe_col_x0/y0 for the compute-provider rotation below.
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void UpdateKokkos::oe_boris3d(int i, int icell, double dt_full,
                               double *x, double *v, double *xnew,
                               double charge, double mass) const
{
  // Dust grains (skip mixture) and neutrals: pure advection — grain
  // forces (gravity, drag, ablation) live in the grain fixes
  if (charge == 0.0 ||
      (d_oe_pusher_skip.data() &&
       d_oe_pusher_skip(d_particles[i].ispecies))) {
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

  // Sub-cell -> parent (the geom compute and csurfs live on the parent)
  int gcell = icell;
  if (d_cells[icell].nsplit <= 0 && d_cells[icell].isplit >= 0)
    gcell = d_sinfo[d_cells[icell].isplit].icell;

  // --- B point-query once at the start position (CPU caches B the same
  //     way); dispatch mesh > equilibrium > cell-center columns ---
  double B_cached[3] = {0.0, 0.0, 0.0};
  {
    // column-axis offset (3D linear-device decks): CPU point queries
    // subtract it via sparta_to_RZ; shift once so R and the
    // cyl->Cartesian rotation both see column coordinates
    double xq[3] = {xcur[0], xcur[1], xcur[2]};
    if (oe_dim == 3 && !oe_axisymmetric) {
      xq[0] -= oe_col_x0; xq[1] -= oe_col_y0;
    }
    bool got_B = false;
    if (oe_has_mesh_b) {
      got_B = MeshKokkos::query_bfield_at_point(
          xq, oe_dim, oe_axisymmetric,
          d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
          d_oe_mesh_tri_br, d_oe_mesh_tri_bz, d_oe_mesh_tri_bt,
          d_oe_mesh_tri_rmin, d_oe_mesh_tri_rmax,
          d_oe_mesh_tri_zmin, d_oe_mesh_tri_zmax,
          d_oe_hash_offset, d_oe_hash_entries,
          oe_mesh_hash_rmin, oe_mesh_hash_zmin,
          oe_mesh_hash_dr,   oe_mesh_hash_dz,
          oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri,
          B_cached);
    }
    if (!got_B && oe_has_equilibrium) {
      // native B maps take precedence over psi-derived B, matching the
      // CPU equ_bfield_at chain (slag b05b4687); no psi fallback when
      // native maps exist (host returns false on stencil failure)
      if (oe_has_equ_bmaps) {
        got_B = EquilibriumKokkos::query_bfield_native_maps(
            xq, oe_dim, oe_axisymmetric,
            d_oe_equ_r, d_oe_equ_z,
            d_oe_equ_br, d_oe_equ_bt, d_oe_equ_bz,
            oe_equ_jm, oe_equ_km, B_cached);
      } else {
        EquilibriumKokkos::query_bfield_at_point(
            xq, oe_dim, oe_axisymmetric,
            d_oe_equ_r, d_oe_equ_z, d_oe_equ_psi,
            oe_equ_btf, oe_equ_rtf, oe_equ_jm, oe_equ_km,
            B_cached);
        got_B = true;
      }
    }
    if (!got_B && d_oe_plasma_compute.data() && oe_bx_col >= 0) {
      B_cached[0] = d_oe_plasma_compute(icell, oe_bx_col);
      B_cached[1] = d_oe_plasma_compute(icell, oe_by_col);
      B_cached[2] = d_oe_plasma_compute(icell, oe_bz_col);
    }
  }

  // --- Sheath prefetch: per-particle element choice + CM coefficients ---
  bool sh_active = false;
  double sh_nx = 0.0, sh_ny = 0.0, sh_nz = 0.0;
  double sh_sref[3] = {0.0, 0.0, 0.0};
  double sh_dmax = 0.0;
  SheathModelsKokkos::CMCoeffs sh_c = {};
  int sh_midx_dbg = -1, sh_cache_dbg = 0;

  if (oe_sheath_provider && gcell >= 0 &&
      gcell < (int) d_oe_midx_gcell.extent(0)) {
    int midx = d_oe_midx_gcell(gcell);

    // Refine against the parent cell's own surfs: nearest plane to the
    // PARTICLE (mirrors CPU pusher.cpp — fixes wrong-face selection when
    // a thin slab intersects one cell).
    const int nsurf_cell = d_cells[gcell].nsurf;
    if (nsurf_cell > 0) {
      auto csurfs_begin = d_csurfs.row_map(gcell);
      double best_d = 1.0e20;
      int best_m = -1;
      for (int m = 0; m < nsurf_cell; m++) {
        const int ms = d_csurfs.entries(csurfs_begin + m);
        if (!(d_tris[ms].mask & oe_sheath_sgroupbit)) continue;
        const double dpl = Kokkos::fabs(
            (x[0]-d_tris[ms].p1[0])*d_tris[ms].norm[0] +
            (x[1]-d_tris[ms].p1[1])*d_tris[ms].norm[1] +
            (x[2]-d_tris[ms].p1[2])*d_tris[ms].norm[2]);
        if (dpl < best_d) { best_d = dpl; best_m = ms; }
      }
      if (best_m >= 0) midx = best_m;
    }

    if (midx >= 0) {
      sh_midx_dbg = midx;
      // Geometry from the chosen element: RAW normal + centroid (raw, not
      // per-cell flipped — avoids sign flips in split cells; CPU-identical)
      sh_nx = d_tris[midx].norm[0];
      sh_ny = d_tris[midx].norm[1];
      sh_nz = d_tris[midx].norm[2];
      const double nmag = Kokkos::sqrt(sh_nx*sh_nx + sh_ny*sh_ny + sh_nz*sh_nz);
      if (nmag > 0.0) { sh_nx /= nmag; sh_ny /= nmag; sh_nz /= nmag; }
      sh_sref[0] = (d_tris[midx].p1[0]+d_tris[midx].p2[0]+d_tris[midx].p3[0])/3.0;
      sh_sref[1] = (d_tris[midx].p1[1]+d_tris[midx].p2[1]+d_tris[midx].p3[1])/3.0;
      sh_sref[2] = (d_tris[midx].p1[2]+d_tris[midx].p2[2]+d_tris[midx].p3[2])/3.0;

      if (oe_sheath_provider == 1 && d_oe_sheath_elem.data() &&
          midx < (int) d_oe_sheath_elem.extent(0) &&
          d_oe_sheath_elem(midx,0) > 0.5) {
        // Per-element cache: coefficients CPU-identical by construction
        // (rows built on host by Pusher::build_sheath_cache_entry_3d).
        sh_dmax           = d_oe_sheath_elem(midx,1);
        sh_c.phi_total_eV = d_oe_sheath_elem(midx,2);
        sh_c.lambdaD_m    = d_oe_sheath_elem(midx,3);
        sh_c.lmps_m       = d_oe_sheath_elem(midx,4);
        sh_c.inv_lD       = d_oe_sheath_elem(midx,5);
        sh_c.inv_lmps     = d_oe_sheath_elem(midx,6);
        sh_c.K1_scaled    = d_oe_sheath_elem(midx,7);
        sh_c.K2           = d_oe_sheath_elem(midx,8);
        sh_c.phi_slow_eV  = d_oe_sheath_elem(midx,9);
        sh_c.phi_fast_eV  = d_oe_sheath_elem(midx,10);
        sh_c.e_anchor_vpm = d_oe_sheath_elem(midx,11);
        sh_active = true;
        sh_cache_dbg = 1;
      } else {
        // Per-particle path: compute provider (cell plasma), or a
        // fix-provider element whose centroid had no plasma — the CPU
        // then queries at the particle position; the mesh scalar query
        // below is its device twin (tri-constant, like interp2D's mesh
        // branch).
        double te = 0.0, ti = 0.0, ne = 0.0;
        double bvec[3] = {0.0, 0.0, 0.0};
        if (oe_sheath_provider == 2 && d_oe_sheath_cellplasma.data() &&
            icell < (int) d_oe_sheath_cellplasma.extent(0)) {
          te = d_oe_sheath_cellplasma(icell,0);
          ti = d_oe_sheath_cellplasma(icell,1);
          ne = d_oe_sheath_cellplasma(icell,2);
          // cell-center cylindrical B rotated at the particle position
          // about the column axis (CPU pusher.cpp compute-provider path)
          const double br = d_oe_sheath_cellplasma(icell,3);
          const double bt = d_oe_sheath_cellplasma(icell,4);
          const double bz = d_oe_sheath_cellplasma(icell,5);
          const double rx = x[0] - oe_col_x0, ry = x[1] - oe_col_y0;
          const double rmag = Kokkos::sqrt(rx*rx + ry*ry);
          if (rmag > 1.0e-20) {
            const double cphi = rx / rmag, sphi = ry / rmag;
            bvec[0] = br * cphi - bt * sphi;
            bvec[1] = br * sphi + bt * cphi;
            bvec[2] = bz;
          } else { bvec[0] = br; bvec[1] = 0.0; bvec[2] = bz; }
        } else if (oe_sheath_provider == 1 && oe_has_mesh_plasma) {
          double P[3];
          double xqs[3] = {x[0], x[1], x[2]};
          if (oe_dim == 3 && !oe_axisymmetric) {
            xqs[0] -= oe_col_x0; xqs[1] -= oe_col_y0;
          }
          if (MeshKokkos::query_scalars_at_point(
                xqs, oe_dim, oe_axisymmetric,
                d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
                d_oe_mesh_tri_te, d_oe_mesh_tri_ti, d_oe_mesh_tri_ne,
                d_oe_hash_offset, d_oe_hash_entries,
                oe_mesh_hash_rmin, oe_mesh_hash_zmin,
                oe_mesh_hash_dr,   oe_mesh_hash_dz,
                oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri, P)) {
            te = P[0]; ti = P[1]; ne = P[2];
          }
          // B at the particle: the start-position point query above —
          // the CPU fallback queries the same point through the fix.
          bvec[0] = B_cached[0]; bvec[1] = B_cached[1]; bvec[2] = B_cached[2];
        }

        if (te > 0.0 && ne > 0.0) {
          const double bmag = Kokkos::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1]
                                         + bvec[2]*bvec[2]);
          double alpha_deg = 90.0;
          if (bmag > 0.0) {
            const double nvec[3] = {sh_nx, sh_ny, sh_nz};
            SheathModelsKokkos::ChoduraMetrics cm =
              SheathModelsKokkos::chodura_metrics(0.0, 1.0, bvec, nvec);
            alpha_deg = cm.alpha_deg;
          }
          sh_dmax = SheathModelsKokkos::auto_dmax(te, ti, ne, bmag, alpha_deg,
                                                  oe_sheath_mD_amu,
                                                  oe_sheath_dmax_user);
          sh_c = SheathModelsKokkos::prepare_coulette_manfredi(
                     te, ti, ne, bmag, alpha_deg, oe_sheath_mD_amu, 0.0);
          // require B > 0 like the CPU fallback and the cache builders:
          // with B = 0, auto_dmax's rho_i blows up and a spurious
          // alpha = 90 sheath would engulf the whole domain
          sh_active = (bmag > 0.0);
        }
      }
    }
  }

  if (oe_sheath_diag && sh_active)
    Kokkos::atomic_inc(&d_oe_shd_counts(0));

  const bool sh_trace =
      (oe_trace_id >= 0 && (long) d_particles[i].id == oe_trace_id);
  if (sh_trace)
    printf("SHTRACE kk  step %lld id %ld pre: active=%d cache=%d midx=%d "
           "dmax=%.9e n=(%.9e,%.9e,%.9e) sref=(%.9e,%.9e,%.9e)\n",
           (long long) ntimestep, oe_trace_id, (int) sh_active, sh_cache_dbg,
           sh_midx_dbg, sh_dmax, sh_nx, sh_ny, sh_nz,
           sh_sref[0], sh_sref[1], sh_sref[2]);

  // Spatial-mode customs: lifetime energy bank (cap) and phi reference
  // (pays element/profile switches between moves). Mirrors CPU
  // pusher.cpp pre-loop block; customs absent -> same as CPU null vecs.
  const bool have_bank =
      oe_has_sheath_customs && i < (int) d_oe_sheath_bank.extent(0);
  const bool have_phiprev =
      oe_has_sheath_customs && i < (int) d_oe_sheath_phiprev.extent(0);
  double sh_phi_tot_sp = 0.0;
  if (sh_active && have_bank)
    sh_phi_tot_sp = SheathModelsKokkos::phi_at_distance(sh_c, 0.0);
  int sh_phi_pending = 0;
  double sh_phi_ref = 0.0;
  if (have_phiprev) {
    if (sh_active) {
      if (d_oe_sheath_phiprev(i) > 0.0) {
        sh_phi_ref = d_oe_sheath_phiprev(i) - 1.0;
        sh_phi_pending = 1;
      }
    } else d_oe_sheath_phiprev(i) = 1.0;   // out of band: known phi = 0
  }

  for (int isub = 0; isub < nsub; isub++) {
    double E[3] = {0.0, 0.0, 0.0};
    double B[3] = {B_cached[0], B_cached[1], B_cached[2]};

    // Background mesh E-field (E = -grad phi from the plasma file), per
    // subcycle at the current position; the query converts cylindrical
    // (E_R, E_Z, E_t) to SPARTA slot order internally. Matches the CPU
    // pusher's per-subcycle query_efield_at_point.
    if (oe_has_mesh_e) {
      double Emesh[3] = {0.0, 0.0, 0.0};
      double xqe[3] = {xcur[0], xcur[1], xcur[2]};
      if (oe_dim == 3 && !oe_axisymmetric) {
        xqe[0] -= oe_col_x0; xqe[1] -= oe_col_y0;
      }
      if (MeshKokkos::query_bfield_at_point(
            xqe, oe_dim, oe_axisymmetric,
            d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
            d_oe_mesh_tri_er, d_oe_mesh_tri_ez, d_oe_mesh_tri_et,
            d_oe_mesh_tri_rmin, d_oe_mesh_tri_rmax,
            d_oe_mesh_tri_zmin, d_oe_mesh_tri_zmax,
            d_oe_hash_offset, d_oe_hash_entries,
            oe_mesh_hash_rmin, oe_mesh_hash_zmin,
            oe_mesh_hash_dr,   oe_mesh_hash_dz,
            oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri,
            Emesh)) {
        E[0] += Emesh[0];
        E[1] += Emesh[1];
        E[2] += Emesh[2];
      }
    }

    // Spatial-mode sheath is applied AFTER the position update below as
    // an energy-consistent potential impulse, not as an E-field force
    // here — same reasoning as the CPU (the old per-subcycle force was
    // non-conservative and pumped energy on gyro-dips behind the plane).

    double xold[3] = {xcur[0], xcur[1], xcur[2]};

    BorisGridKokkos::push_velocity(qm, dt_sub, E, B, vcur);
    xcur[0] += vcur[0] * dt_sub;
    xcur[1] += vcur[1] * dt_sub;
    xcur[2] += vcur[2] * dt_sub;

    // Spatial-mode sheath: exact work of the sheath potential over this
    // subcycle's normal displacement, dKE = Z e [phi(d_new) - phi(d_old)],
    // phi clamped to phi(0) for d <= 0. Outbound ions that cannot climb
    // the remaining potential reflect elastically at the turning point.
    // Line-for-line port of CPU pusher.cpp push_boris_3d.
    if (sh_active) {
      const double d_old =
        (xold[0] - sh_sref[0]) * sh_nx
      + (xold[1] - sh_sref[1]) * sh_ny
      + (xold[2] - sh_sref[2]) * sh_nz;
      const double d_new =
        (xcur[0] - sh_sref[0]) * sh_nx
      + (xcur[1] - sh_sref[1]) * sh_ny
      + (xcur[2] - sh_sref[2]) * sh_nz;
      if (Kokkos::fmin(d_old, d_new) < sh_dmax) {
        const double phi_old_geo = SheathModelsKokkos::phi_at_distance(
            sh_c, Kokkos::fmax(d_old, 0.0));
        const double phi_old = sh_phi_pending ? sh_phi_ref : phi_old_geo;
        sh_phi_pending = 0;
        const double phi_new = SheathModelsKokkos::phi_at_distance(
            sh_c, Kokkos::fmax(d_new, 0.0));
        double dKE_J =
            Kokkos::fabs(charge) * oe_echarge * (phi_new - phi_old);
        if (sh_trace)
          printf("SHTRACE kk  step %lld sub %d eng: d_old=%.9e d_new=%.9e "
                 "phi_old=%.9e phi_new=%.9e dKE=%.9e bank=%.9e\n",
                 (long long) ntimestep, isub, d_old, d_new,
                 phi_old, phi_new, dKE_J,
                 have_bank ? d_oe_sheath_bank(i) : -1.0);
        // lifetime ledger cap: net energy given may never exceed Z e phi_tot
        if (have_bank && dKE_J > 0.0) {
          const double room =
              Kokkos::fabs(charge) * oe_echarge * sh_phi_tot_sp
              - d_oe_sheath_bank(i);
          if (dKE_J > room) dKE_J = (room > 0.0) ? room : 0.0;
        }
        double sh_d_fin = d_new;
        if (dKE_J != 0.0) {
          const double vn = vcur[0]*sh_nx + vcur[1]*sh_ny + vcur[2]*sh_nz;
          const double s2 = vn*vn + 2.0*dKE_J/mass;
          double vn_new;
          if (s2 >= 0.0) {
            vn_new = (vn >= 0.0) ? Kokkos::sqrt(s2) : -Kokkos::sqrt(s2);
            if (have_bank) d_oe_sheath_bank(i) += dKE_J;
          } else {
            // turning point: elastic reflection; bounce the position back
            // to d_old (the climb to d_new was never paid for)
            vn_new = -vn;
            xcur[0] -= (d_new - d_old) * sh_nx;
            xcur[1] -= (d_new - d_old) * sh_ny;
            xcur[2] -= (d_new - d_old) * sh_nz;
            sh_d_fin = d_old;
            if (oe_sheath_diag)
              Kokkos::atomic_inc(&d_oe_shd_counts(2));
          }
          const double dvn = vn_new - vn;
          vcur[0] += dvn * sh_nx;
          vcur[1] += dvn * sh_ny;
          vcur[2] += dvn * sh_nz;
          if (oe_sheath_diag) {
            Kokkos::atomic_inc(&d_oe_shd_counts(1));
            const double emag_diag = SheathModelsKokkos::emag_at_distance(
                sh_c, Kokkos::fmax(Kokkos::fmin(d_old, d_new), 0.0));
            Kokkos::atomic_add(&d_oe_shd_esum(0), emag_diag);
            Kokkos::atomic_max(&d_oe_shd_esum(1), emag_diag);
          }
        }
        // remember phi at the endpoint for next move's reference payment
        if (have_phiprev)
          d_oe_sheath_phiprev(i) = 1.0 + SheathModelsKokkos::phi_at_distance(
              sh_c, Kokkos::fmax(sh_d_fin, 0.0));
      } else if (have_phiprev) d_oe_sheath_phiprev(i) = 1.0;
    }

    // Per-subcycle guards (3D, nsub > 1), mirroring the CPU:
    // (a) in-cell wall hit — clip xnew to the intersection point so the
    //     outer move loop sees a trajectory that touches the wall
    //     exactly (critical for grazing divertor geometry);
    // (b) cell exit — bail so the outer loop handles migration and the
    //     neighbor cell's surface checks on the remaining chord.
    if (nsub > 1) {
      const int nsurf_cell = d_cells[gcell].nsurf;
      if (nsurf_cell > 0) {
        auto csurfs_begin = d_csurfs.row_map(gcell);
        for (int m = 0; m < nsurf_cell; m++) {
          int isurf = d_csurfs.entries(csurfs_begin + m);
          double xc[3], param;
          int side;
          if (GeometryKokkos::line_tri_intersect(
                xold, xcur,
                d_tris[isurf].p1, d_tris[isurf].p2, d_tris[isurf].p3,
                d_tris[isurf].norm, xc, param, side)) {
            v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
            xnew[0] = xc[0]; xnew[1] = xc[1]; xnew[2] = xc[2];
            return;
          }
        }
      }
      // half-open [lo, hi): a particle exactly at hi has left the cell
      const double *clo = d_cells[gcell].lo;
      const double *chi = d_cells[gcell].hi;
      if (xcur[0] < clo[0] || xcur[0] >= chi[0] ||
          xcur[1] < clo[1] || xcur[1] >= chi[1] ||
          xcur[2] < clo[2] || xcur[2] >= chi[2]) {
        v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
        xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];
        return;
      }
    }
  }

  v[0] = vcur[0]; v[1] = vcur[1]; v[2] = vcur[2];
  xnew[0] = xcur[0]; xnew[1] = xcur[1]; xnew[2] = xcur[2];
}

/* ---------------------------------------------------------------------- */

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
  } else {
    for (int i = 0; i < KOKKOS_MAX_BLIST; i++) {

      // use temporary to avoid the copy getting stale leading to an issue
      //  with view reference counting

      blist_active_copy[i].copy(&tmp_compute_boundary_kk);
    }
  }

  if (nsurf_tally > KOKKOS_MAX_SLIST)
    error->all(FLERR,"Kokkos currently only supports three instances of compute surface");

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
  } else {
    for (int i = 0; i < KOKKOS_MAX_SLIST; i++) {

      // use temporary to avoid the copy getting stale leading to an issue
      //  with view reference counting

      slist_active_copy[i].copy(&tmp_compute_surf_kk);
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

  // OpenEdge Phase D: the move kernel writes the spatial-sheath customs
  // (bank/phiprev); snapshot them so a react/retry replay starts from
  // the pre-attempt state instead of double-applying the impulse ledger.
  if (oe_has_sheath_customs) {
    if (d_oe_sheath_bank_backup.extent(0) != d_oe_sheath_bank.extent(0))
      d_oe_sheath_bank_backup = DAT::t_float_1d(
          Kokkos::view_alloc("update:sheath_bank_backup",
                             Kokkos::WithoutInitializing),
          d_oe_sheath_bank.extent(0));
    if (d_oe_sheath_phiprev_backup.extent(0) != d_oe_sheath_phiprev.extent(0))
      d_oe_sheath_phiprev_backup = DAT::t_float_1d(
          Kokkos::view_alloc("update:sheath_phiprev_backup",
                             Kokkos::WithoutInitializing),
          d_oe_sheath_phiprev.extent(0));
    Kokkos::deep_copy(d_oe_sheath_bank_backup,d_oe_sheath_bank);
    Kokkos::deep_copy(d_oe_sheath_phiprev_backup,d_oe_sheath_phiprev);
  }

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

  // OpenEdge Phase D: roll the spatial-sheath customs back with the
  // particles (see backup()).
  if (oe_has_sheath_customs) {
    Kokkos::deep_copy(d_oe_sheath_bank,d_oe_sheath_bank_backup);
    Kokkos::deep_copy(d_oe_sheath_phiprev,d_oe_sheath_phiprev_backup);
  }

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

/* ----------------------------------------------------------------------
   OpenEdge gate 9b: fill the per-particle plasma-cache customs on the
   device. Semantics mirror Update::cache_plasma_particles() for the
   supported mask exactly: tri-constant mesh scalars (miss = 0, the CPU
   empty-structured fallback), B through the mesh -> equilibrium chain
   with the column-axis shift, and the sheath Boltzmann ne correction
   (nearest group element in the parent cell, CM potential at the
   particle's wall distance). Runs instead of the host fill — no
   particle/custom host round-trip.
------------------------------------------------------------------------- */

void UpdateKokkos::cache_plasma_particles_device()
{
  const int nlocal = particle->nlocal;
  if (nlocal == 0) return;

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();

  // rebind the grid views: a mid-run rebalance re-decomposes
  // cells/sinfo/csurfs (the mover rebinds them per iteration; this
  // kernel runs BEFORE the mover and would otherwise index
  // post-balance icell values into pre-balance views — Bug-F class;
  // caught by compute-sanitizer at 10x where fix balance first fires)
  GridKokkos *grid_kk = (GridKokkos *) grid;
  d_cells  = grid_kk->k_cells.view_device();
  d_sinfo  = grid_kk->k_sinfo.view_device();
  d_csurfs = grid_kk->d_csurfs;

  // bind the masked custom slots fresh each call (grow_custom-safe)
  auto edvec = [&](int cidx) -> DAT::t_float_1d {
    return particle_kk->k_edvec.h_view[particle->ewhich[cidx]].k_view.d_view;
  };
  const int m = pcache_need_mask;
  if (m & PCACHE_TE)   d_pc_te   = edvec(pc_te_custom);
  if (m & PCACHE_TI)   d_pc_ti   = edvec(pc_ti_custom);
  if (m & PCACHE_NE)   d_pc_ne   = edvec(pc_ne_custom);
  if (m & PCACHE_NI)   d_pc_ni   = edvec(pc_ni_custom);
  if (m & PCACHE_VPAR) d_pc_vpar = edvec(pc_vpar_custom);
  if (m & PCACHE_BFIELD) {
    d_pc_bx = edvec(pc_bx_custom);
    d_pc_by = edvec(pc_by_custom);
    d_pc_bz = edvec(pc_bz_custom);
  }
  oe_pc_mask = m;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagUpdatePcacheFill>(0,nlocal),*this);
  DeviceType().fence();
  copymode = 0;

  particle_kk->modify(Device,CUSTOM_MASK);
}

KOKKOS_INLINE_FUNCTION
void UpdateKokkos::operator()(TagUpdatePcacheFill, const int &i) const
{
  Particle::OnePart &p = d_particles(i);
  const int mask = oe_pc_mask;

  // column-axis shift for the (R,Z) queries; plane distances below use
  // the raw SPARTA position (surfaces live in SPARTA coordinates)
  double xq[3] = {p.x[0], p.x[1], p.x[2]};
  if (oe_dim == 3 && !oe_axisymmetric) {
    xq[0] -= oe_col_x0; xq[1] -= oe_col_y0;
  }

  // tri-constant plasma scalars (CPU mesh_cell_for at the particle
  // position; a miss = the CPU empty-structured-grid fallback = 0)
  double te = 0.0, ti = 0.0, ne = 0.0, ni = 0.0, vpar = 0.0;
  const int tri = MeshKokkos::locate_tri_at_point(
      xq, oe_dim, oe_axisymmetric,
      d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
      d_oe_hash_offset, d_oe_hash_entries,
      oe_mesh_hash_rmin, oe_mesh_hash_zmin,
      oe_mesh_hash_dr,   oe_mesh_hash_dz,
      oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri);
  if (tri >= 0) {
    te = d_oe_mesh_tri_te(tri);
    ti = d_oe_mesh_tri_ti(tri);
    ne = d_oe_mesh_tri_ne(tri);
    if (oe_has_mesh_drag) {
      ni   = d_oe_mesh_tri_ni(tri);
      vpar = d_oe_mesh_tri_upar(tri);
    }
  }

  // B at the particle when a slot or the ne correction needs it —
  // mesh -> equilibrium chain, Cartesian components (CPU bfield_at +
  // rotation about the column axis)
  double B[3] = {0.0, 0.0, 0.0};
  if ((mask & PCACHE_BFIELD) || oe_pc_csg) {
    bool got_B = false;
    if (oe_has_mesh_b) {
      got_B = MeshKokkos::query_bfield_at_point(
          xq, oe_dim, oe_axisymmetric,
          d_oe_mesh_vtx_r, d_oe_mesh_vtx_z, d_oe_mesh_tri,
          d_oe_mesh_tri_br, d_oe_mesh_tri_bz, d_oe_mesh_tri_bt,
          d_oe_mesh_tri_rmin, d_oe_mesh_tri_rmax,
          d_oe_mesh_tri_zmin, d_oe_mesh_tri_zmax,
          d_oe_hash_offset, d_oe_hash_entries,
          oe_mesh_hash_rmin, oe_mesh_hash_zmin,
          oe_mesh_hash_dr,   oe_mesh_hash_dz,
          oe_mesh_hash_nr, oe_mesh_hash_nz, oe_mesh_ntri, B);
    }
    if (!got_B && oe_has_equilibrium) {
      if (oe_has_equ_bmaps) {
        EquilibriumKokkos::query_bfield_native_maps(
            xq, oe_dim, oe_axisymmetric,
            d_oe_equ_r, d_oe_equ_z,
            d_oe_equ_br, d_oe_equ_bt, d_oe_equ_bz,
            oe_equ_jm, oe_equ_km, B);
      } else {
        EquilibriumKokkos::query_bfield_at_point(
            xq, oe_dim, oe_axisymmetric,
            d_oe_equ_r, d_oe_equ_z, d_oe_equ_psi,
            oe_equ_btf, oe_equ_rtf, oe_equ_jm, oe_equ_km, B);
      }
    }
  }

  if (mask & PCACHE_TE)   d_pc_te(i)   = te;
  if (mask & PCACHE_TI)   d_pc_ti(i)   = ti;
  if (mask & PCACHE_NI)   d_pc_ni(i)   = ni;
  if (mask & PCACHE_VPAR) d_pc_vpar(i) = vpar;
  if (mask & PCACHE_BFIELD) {
    d_pc_bx(i) = B[0];
    d_pc_by(i) = B[1];
    d_pc_bz(i) = B[2];
  }

  // Boltzmann ne correction: ne_local = ne * exp(-phi/Te), phi = the CM
  // sheath potential at the particle's wall distance (CPU
  // cache_plasma_particles tail; element refinement as in the mover)
  double ne_out = ne;
  if (oe_pc_csg && te > 0.0 && ne > 0.0) {
    const int icell = p.icell;
    if (icell >= 0 && icell < (int) d_cells.extent(0)) {
      int gcell = icell;
      if (d_cells[icell].nsplit <= 0 && d_cells[icell].isplit >= 0)
        gcell = d_sinfo[d_cells[icell].isplit].icell;
      if (gcell >= 0 && gcell < (int) d_oe_midx_gcell.extent(0)) {
        int midx = d_oe_midx_gcell(gcell);
        const int nsurf_cell = d_cells[gcell].nsurf;
        if (nsurf_cell > 0) {
          auto csurfs_begin = d_csurfs.row_map(gcell);
          double best_d = 1.0e20;
          int best_m = -1;
          for (int mm = 0; mm < nsurf_cell; mm++) {
            const int ms = d_csurfs.entries(csurfs_begin + mm);
            if (!(d_tris[ms].mask & oe_sheath_sgroupbit)) continue;
            const double dpl = Kokkos::fabs(
                (p.x[0]-d_tris[ms].p1[0])*d_tris[ms].norm[0] +
                (p.x[1]-d_tris[ms].p1[1])*d_tris[ms].norm[1] +
                (p.x[2]-d_tris[ms].p1[2])*d_tris[ms].norm[2]);
            if (dpl < best_d) { best_d = dpl; best_m = ms; }
          }
          if (best_m >= 0) midx = best_m;
        }
        if (midx >= 0) {
          double nx = d_tris[midx].norm[0];
          double ny = d_tris[midx].norm[1];
          double nz = d_tris[midx].norm[2];
          const double nmag = Kokkos::sqrt(nx*nx + ny*ny + nz*nz);
          if (nmag > 0.0) { nx /= nmag; ny /= nmag; nz /= nmag; }
          const double sref0 =
            (d_tris[midx].p1[0]+d_tris[midx].p2[0]+d_tris[midx].p3[0])/3.0;
          const double sref1 =
            (d_tris[midx].p1[1]+d_tris[midx].p2[1]+d_tris[midx].p3[1])/3.0;
          const double sref2 =
            (d_tris[midx].p1[2]+d_tris[midx].p2[2]+d_tris[midx].p3[2])/3.0;
          const double dpart = Kokkos::fabs(
              (p.x[0]-sref0)*nx + (p.x[1]-sref1)*ny + (p.x[2]-sref2)*nz);

          const double bmag =
              Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
          double alpha_deg = 90.0;
          if (bmag > 0.0) {
            const double nvec[3] = {nx, ny, nz};
            SheathModelsKokkos::ChoduraMetrics cm =
              SheathModelsKokkos::chodura_metrics(0.0, 1.0, B, nvec);
            alpha_deg = cm.alpha_deg;
          }
          const double d_max = SheathModelsKokkos::auto_dmax(
              te, ti, ne, bmag, alpha_deg,
              oe_sheath_mD_amu, oe_sheath_dmax_user);
          if (dpart > 0.0 && dpart < d_max) {
            SheathModelsKokkos::CMCoeffs c =
              SheathModelsKokkos::prepare_coulette_manfredi(
                  te, ti, ne, bmag, alpha_deg, oe_sheath_mD_amu, 0.0);
            const double phi = SheathModelsKokkos::phi_at_distance(c, dpart);
            if (phi > 0.0) ne_out = ne * Kokkos::exp(-phi / te);
          }
        }
      }
    }
  }
  if (mask & PCACHE_NE) d_pc_ne(i) = ne_out;
}
