/* ----------------------------------------------------------------------
    OpenEdge: Kokkos/GPU port of the plasma-wall interaction (PWI)
    surface reaction model - host-side setup, table flattening, and
    react/retry rollback. See surf_react_surface_pwi_kokkos.h.

    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2026)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "surf_react_surface_pwi_kokkos.h"
#include "update.h"
#include "comm.h"
#include "domain.h"
#include "surf.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"
#include "random_knuth.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

// channel types of the base class (surf_react_surface_pwi.cpp)
enum{DISSOCIATION,EXCHANGE,RECOMBINATION,TRIM_REFLECT,ABSORB_REEMIT,SPUTTER};

/* ---------------------------------------------------------------------- */

SurfReactSurfacePWIKokkos::SurfReactSurfacePWIKokkos(SPARTA *sparta, int narg,
                                                     char **arg) :
  SurfReactSurfacePWI(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  kokkosable = 1;

  d_scalars = DAT::t_int_1d("surf_react_pwi:scalars",nlist+1);
  d_nsingle = Kokkos::subview(d_scalars,0);
  d_tally_single = Kokkos::subview(d_scalars,std::make_pair(1,nlist+1));

  h_scalars = HAT::t_int_1d("surf_react_pwi:scalars_mirror",nlist+1);

  d_scalars_bak = DAT::t_int_1d("surf_react_pwi:scalars_bak",nlist+1);

  random_backup = NULL;
  pw_slot = -1;
  sigma_on = ehist_on = 0;
  ncols = nbin = nsp = 0;
  emax = fnum_c = evconv = twall_c = rough_c = 0.0;
}

SurfReactSurfacePWIKokkos::SurfReactSurfacePWIKokkos(SPARTA *sparta) :
  SurfReactSurfacePWI(sparta),
  rand_pool(12345 // seed will be copied over
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  copy = 1;
  random_backup = NULL;
}

/* ---------------------------------------------------------------------- */

SurfReactSurfacePWIKokkos::~SurfReactSurfacePWIKokkos()
{
  if (copy) return;

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
  if (random_backup)
    delete random_backup;
#endif
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::init()
{
  SurfReactSurfacePWI::init();

  check_supported();
  init_device_tables();

  Kokkos::deep_copy(d_scalars,0);

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(random);
#endif
}

/* ----------------------------------------------------------------------
   narrow first-pass scope: fail explicitly for every PWI mode that has
   no device implementation yet, instead of silently approximating
------------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::check_supported()
{
  if (twall_attr)
    error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
               "twall_surf (per-surf wall temperature)");
  if (R_attr)
    error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
               "R_surf (per-surf recycling coefficient)");

  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;

    if (r->type == DISSOCIATION || r->type == EXCHANGE ||
        r->type == RECOMBINATION)
      error->all(FLERR,"surf_react surface/pwi/kk supports only T/A/S "
                 "reaction channels (D/E/R not yet ported)");

    if (r->mat_isp >= 0 || r->conc_isp >= 0)
      error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                 "composition-weighted (mat/conc) reaction channels");

    if (r->type == TRIM_REFLECT && r->refl_tbl >= 0)
      error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                 "rtable composition-resolved reflection");

    if (r->type == SPUTTER && r->sp_tbl >= 0) {
      if (sput_tables[r->sp_tbl].NC > 0)
        error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                   "compound-target (3D) sputter tables");
      if (sput_tables[r->sp_tbl].NT > 0)
        error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                   "temperature-dependent sputter tables");
    }

    if (r->type == ABSORB_REEMIT) {
      // CPU semantics: atomic re-emission (prob R*(1-f_mol)) returns the
      // REACTANT species; the product species only matters for the
      // molecular channel (prob R*f_mol/2), which is not ported yet.
      // Reject only decks where that channel could actually fire.
      if (r->energy[0] > 0.0 && r->energy[1] > 0.0 &&
          (r->nproduct != 1 || r->products[0] != r->reactants[0]))
        error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                   "molecular A-channel conversion (R > 0 with f_mol > 0 "
                   "and product != reactant)");
    }

    // device path emits products with erot = evib = 0; reject species
    // where the CPU would resample internal energy at twall
    if (twall > 0.0) {
      for (int j = 0; j < r->nproduct; j++) {
        int sp = r->products[j];
        if (particle->species[sp].rotdof >= 2 ||
            particle->species[sp].vibdof >= 2)
          error->all(FLERR,"surf_react surface/pwi/kk does not yet support "
                     "polyatomic products with twall accommodation");
      }
    }
  }
}

/* ----------------------------------------------------------------------
   flatten reaction lists and reflection/sputter tables onto device views
------------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::init_device_tables()
{
  int nspecies = particle->nspecies;

  // captured scalars

  fnum_c = update->fnum;
  evconv = update->joule2ev * update->mvv2e;
  twall_c = twall;
  rough_c = rough_dm;
  sigma_on = (sindex_custom >= 0);
  ehist_on = (ehist_file != NULL);
  ncols = sigma_ncols;
  nbin = ehist_nbin;
  nsp = ehist_nsp;
  emax = ehist_emax;

  // per-species reaction dispatch

  int nmax = 0;
  d_reactions_n = DAT::t_int_1d("surf_react_pwi:nreact",nspecies);
  auto h_reactions_n = Kokkos::create_mirror_view(d_reactions_n);
  for (int i = 0; i < nspecies; i++) {
    h_reactions_n(i) = reactions[i].n;
    nmax = MAX(nmax,reactions[i].n);
  }
  d_list = DAT::t_int_2d("surf_react_pwi:list",nspecies,MAX(nmax,1));
  auto h_list = Kokkos::create_mirror_view(d_list);
  for (int i = 0; i < nspecies; i++)
    for (int j = 0; j < reactions[i].n; j++)
      h_list(i,j) = reactions[i].list[j];
  Kokkos::deep_copy(d_reactions_n,h_reactions_n);
  Kokkos::deep_copy(d_list,h_list);

  // per-reaction data

  int nl = MAX(nlist_recycle,1);
  d_type = DAT::t_int_1d("surf_react_pwi:type",nl);
  d_prod = DAT::t_int_1d("surf_react_pwi:prod",nl);
  d_trim = DAT::t_int_1d("surf_react_pwi:trim",nl);
  d_sput = DAT::t_int_1d("surf_react_pwi:sput",nl);
  d_prob = DAT::t_float_1d("surf_react_pwi:prob",nl);
  d_Rrec = DAT::t_float_1d("surf_react_pwi:Rrec",nl);
  d_spp  = DAT::t_float_2d_lr("surf_react_pwi:spp",nl,4);

  auto h_type = Kokkos::create_mirror_view(d_type);
  auto h_prod = Kokkos::create_mirror_view(d_prod);
  auto h_trim = Kokkos::create_mirror_view(d_trim);
  auto h_sput = Kokkos::create_mirror_view(d_sput);
  auto h_prob = Kokkos::create_mirror_view(d_prob);
  auto h_Rrec = Kokkos::create_mirror_view(d_Rrec);
  auto h_spp  = Kokkos::create_mirror_view(d_spp);

  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    h_type(m) = r->type;
    h_prod(m) = (r->nproduct > 0) ? r->products[0] : -1;
    h_trim(m) = r->trim_table;
    h_sput(m) = r->sp_tbl;
    h_prob(m) = r->prob;
    h_Rrec(m) = r->energy[0];      // A channel: R coefficient
    h_spp(m,0) = r->sp_Es;
    h_spp(m,1) = r->sp_Eth;
    h_spp(m,2) = r->sp_Q;
    h_spp(m,3) = r->sp_ETF;
  }
  Kokkos::deep_copy(d_type,h_type);
  Kokkos::deep_copy(d_prod,h_prod);
  Kokkos::deep_copy(d_trim,h_trim);
  Kokkos::deep_copy(d_sput,h_sput);
  Kokkos::deep_copy(d_prob,h_prob);
  Kokkos::deep_copy(d_Rrec,h_Rrec);
  Kokkos::deep_copy(d_spp,h_spp);

  // TRIM reflection tables: fixed EIRENE-schema sizes

  using Reflection::NE;
  using Reflection::NTHETA;
  using Reflection::NQ;
  int ntr = MAX((int) trim_tables.size(),1);
  d_tr_E    = DAT::t_float_2d_lr("surf_react_pwi:tr_E",ntr,NE);
  d_tr_th   = DAT::t_float_2d_lr("surf_react_pwi:tr_th",ntr,NTHETA);
  d_tr_raar = DAT::t_float_2d_lr("surf_react_pwi:tr_raar",ntr,NQ);
  d_tr_RN   = DAT::t_float_2d_lr("surf_react_pwi:tr_RN",ntr,NE*NTHETA);
  d_tr_Eq   = DAT::t_float_2d_lr("surf_react_pwi:tr_Eq",ntr,NE*NTHETA*NQ);
  d_tr_Emin = DAT::t_float_2d_lr("surf_react_pwi:tr_Emin",ntr,NE*NTHETA);
  d_tr_Emax = DAT::t_float_2d_lr("surf_react_pwi:tr_Emax",ntr,NE*NTHETA);
  d_tr_cp   = DAT::t_float_2d_lr("surf_react_pwi:tr_cp",ntr,NE*NTHETA*NQ*NQ);
  d_tr_ca   = DAT::t_float_2d_lr("surf_react_pwi:tr_ca",ntr,NE*NTHETA*NQ*NQ*NQ);

  {
    auto h_E    = Kokkos::create_mirror_view(d_tr_E);
    auto h_th   = Kokkos::create_mirror_view(d_tr_th);
    auto h_raar = Kokkos::create_mirror_view(d_tr_raar);
    auto h_RN   = Kokkos::create_mirror_view(d_tr_RN);
    auto h_Eq   = Kokkos::create_mirror_view(d_tr_Eq);
    auto h_Emin = Kokkos::create_mirror_view(d_tr_Emin);
    auto h_Emax = Kokkos::create_mirror_view(d_tr_Emax);
    auto h_cp   = Kokkos::create_mirror_view(d_tr_cp);
    auto h_ca   = Kokkos::create_mirror_view(d_tr_ca);
    for (size_t t = 0; t < trim_tables.size(); t++) {
      const Reflection::Table &tab = trim_tables[t];
      for (int i = 0; i < NE; i++) h_E(t,i) = tab.E_grid[i];
      for (int i = 0; i < NTHETA; i++) h_th(t,i) = tab.theta_grid[i];
      for (int i = 0; i < NQ; i++) h_raar(t,i) = tab.raar[i];
      for (int i = 0; i < NE*NTHETA; i++) {
        h_RN(t,i) = tab.R_N[i];
        h_Emin(t,i) = tab.Eout_min[i];
        h_Emax(t,i) = tab.Eout_max[i];
      }
      for (int i = 0; i < NE*NTHETA*NQ; i++) h_Eq(t,i) = tab.Eout_q[i];
      for (int i = 0; i < NE*NTHETA*NQ*NQ; i++) h_cp(t,i) = tab.cos_polar_q[i];
      for (int i = 0; i < NE*NTHETA*NQ*NQ*NQ; i++) h_ca(t,i) = tab.cos_azim_q[i];
    }
    Kokkos::deep_copy(d_tr_E,h_E);
    Kokkos::deep_copy(d_tr_th,h_th);
    Kokkos::deep_copy(d_tr_raar,h_raar);
    Kokkos::deep_copy(d_tr_RN,h_RN);
    Kokkos::deep_copy(d_tr_Eq,h_Eq);
    Kokkos::deep_copy(d_tr_Emin,h_Emin);
    Kokkos::deep_copy(d_tr_Emax,h_Emax);
    Kokkos::deep_copy(d_tr_cp,h_cp);
    Kokkos::deep_copy(d_tr_ca,h_ca);
  }

  // 2D sputter-yield tables, padded to max dims

  int nsu = MAX((int) sput_tables.size(),1);
  int maxNE = 1, maxNT = 1;
  for (size_t t = 0; t < sput_tables.size(); t++) {
    maxNE = MAX(maxNE,sput_tables[t].NE);
    maxNT = MAX(maxNT,sput_tables[t].NTHETA);
  }
  d_su_NE = DAT::t_int_1d("surf_react_pwi:su_NE",nsu);
  d_su_NT = DAT::t_int_1d("surf_react_pwi:su_NT",nsu);
  d_su_E  = DAT::t_float_2d_lr("surf_react_pwi:su_E",nsu,maxNE);
  d_su_th = DAT::t_float_2d_lr("surf_react_pwi:su_th",nsu,maxNT);
  d_su_Y  = DAT::t_float_2d_lr("surf_react_pwi:su_Y",nsu,maxNE*maxNT);

  {
    auto h_NE = Kokkos::create_mirror_view(d_su_NE);
    auto h_NT = Kokkos::create_mirror_view(d_su_NT);
    auto h_E  = Kokkos::create_mirror_view(d_su_E);
    auto h_th = Kokkos::create_mirror_view(d_su_th);
    auto h_Y  = Kokkos::create_mirror_view(d_su_Y);
    Kokkos::deep_copy(h_NE,0);
    Kokkos::deep_copy(h_NT,0);
    for (size_t t = 0; t < sput_tables.size(); t++) {
      const ProcessLibrary::TrimSputterTable &st = sput_tables[t];
      h_NE(t) = st.NE;
      h_NT(t) = st.NTHETA;
      for (int i = 0; i < st.NE; i++) h_E(t,i) = st.E[i];
      for (int i = 0; i < st.NTHETA; i++) h_th(t,i) = st.theta[i];
      // device layout is [ie*NT + ia], same row-major as the host table
      for (int i = 0; i < st.NE*st.NTHETA; i++) h_Y(t,i) = st.Y[i];
    }
    Kokkos::deep_copy(d_su_NE,h_NE);
    Kokkos::deep_copy(d_su_NT,h_NT);
    Kokkos::deep_copy(d_su_E,h_E);
    Kokkos::deep_copy(d_su_th,h_th);
    Kokkos::deep_copy(d_su_Y,h_Y);
  }

  // areal-density ledger: per-surf area + global ID for local+ghost surfs

  if (sigma_on) {
    int ntally = (int) (sigma_nsurf * (bigint) sigma_ncols);
    d_sigma_delta = DAT::t_float_1d("surf_react_pwi:sigma_delta",ntally);
    d_dep_delta = DAT::t_float_1d("surf_react_pwi:dep_delta",(int) sigma_nsurf);
    h_sigma_delta = Kokkos::create_mirror_view(d_sigma_delta);
    h_dep_delta = Kokkos::create_mirror_view(d_dep_delta);
    d_sigma_bak = DAT::t_float_1d("surf_react_pwi:sigma_bak",ntally);
    d_dep_bak = DAT::t_float_1d("surf_react_pwi:dep_bak",(int) sigma_nsurf);

    int nslocal = surf->nlocal + surf->nghost;
    d_area = DAT::t_float_1d("surf_react_pwi:area",nslocal);
    d_gid0 = DAT::t_int_1d("surf_react_pwi:gid0",nslocal);
    auto h_area = Kokkos::create_mirror_view(d_area);
    auto h_gid0 = Kokkos::create_mirror_view(d_gid0);
    int dim = domain->dimension;
    for (int i = 0; i < nslocal; i++) {
      h_area(i) = sigma_area[i];
      surfint gid;
      if (dim == 2) gid = surf->lines[i].id;
      else gid = surf->tris[i].id;
      h_gid0(i) = (int) (gid - 1);
    }
    Kokkos::deep_copy(d_area,h_area);
    Kokkos::deep_copy(d_gid0,h_gid0);
  }

  // impact-energy histogram deltas

  if (ehist_on) {
    int ntot = 2*nbin + 2*PWI_NANG + nsp*nbin;
    d_ehist_delta = DAT::t_float_1d("surf_react_pwi:ehist_delta",ntot);
    h_ehist_delta = Kokkos::create_mirror_view(d_ehist_delta);
    d_ehist_bak = DAT::t_float_1d("surf_react_pwi:ehist_bak",ntot);
  }
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::tally_reset()
{
  SurfReact::tally_reset();

  Kokkos::deep_copy(d_scalars,0);
}

/* ----------------------------------------------------------------------
   end-of-step: fold device counters and deltas into the host arrays,
   then let the host base class do everything downstream (ntotal update,
   sync_sigma + strata + MPI reductions, histogram file output)
------------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::tally_update()
{
  Kokkos::deep_copy(h_scalars,d_scalars);
  nsingle += h_scalars(0);
  for (int i = 0; i < nlist; i++) tally_single[i] += h_scalars(i+1);
  Kokkos::deep_copy(d_scalars,0);

  if (sigma_on && update->ntimestep % sigma_nevery == 0) fold_sigma();
  if (ehist_on && ehist_every > 0 &&
      update->ntimestep % ehist_every == 0) fold_ehist();

  SurfReactSurfacePWI::tally_update();
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::fold_sigma()
{
  int ntally = (int) (sigma_nsurf * (bigint) sigma_ncols);
  Kokkos::deep_copy(h_sigma_delta,d_sigma_delta);
  for (int i = 0; i < ntally; i++) sigma_delta[i] += h_sigma_delta(i);
  Kokkos::deep_copy(d_sigma_delta,0.0);

  Kokkos::deep_copy(h_dep_delta,d_dep_delta);
  for (int i = 0; i < (int) sigma_nsurf; i++) dep_delta[i] += h_dep_delta(i);
  Kokkos::deep_copy(d_dep_delta,0.0);
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::fold_ehist()
{
  Kokkos::deep_copy(h_ehist_delta,d_ehist_delta);
  for (int i = 0; i < nbin; i++) {
    ehist_all[i] += h_ehist_delta(i);
    ehist_sput[i] += h_ehist_delta(nbin + i);
  }
  for (int i = 0; i < PWI_NANG; i++) {
    ahist_all[i] += h_ehist_delta(2*nbin + i);
    ahist_sput[i] += h_ehist_delta(2*nbin + PWI_NANG + i);
  }
  if (ehist_z) {
    int off = 2*nbin + 2*PWI_NANG;
    for (int k = 0; k < nsp; k++)
      for (int i = 0; i < nbin; i++)
        ehist_z[k][i] += h_ehist_delta(off + k*nbin + i);
  }
  Kokkos::deep_copy(d_ehist_delta,0.0);
}

/* ----------------------------------------------------------------------
   hooks called by the Kokkos surface collider around the move kernel
------------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::pre_react()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species = particle_kk->k_species.view_device();
  custom_ = particle_kk->device_custom();

  // resolve the pweight edvec slot at move time: other modules can add
  // particle customs after init, which shifts ewhich values
  pw_slot = (pweight_ewhich >= 0) ? particle->ewhich[pweight_ewhich] : -1;
}

void SurfReactSurfacePWIKokkos::post_react()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->modify(Device,CUSTOM_MASK);
}

/* ----------------------------------------------------------------------
   react/retry: snapshot every device accumulator that a failed move
   pass could dirty, and restore it exactly before the pass reruns
------------------------------------------------------------------------- */

void SurfReactSurfacePWIKokkos::backup()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  d_particles = particle_kk->k_particles.view_device();
  custom_ = particle_kk->device_custom();
  pw_slot = (pweight_ewhich >= 0) ? particle->ewhich[pweight_ewhich] : -1;

  Kokkos::deep_copy(d_scalars_bak,d_scalars);
  if (sigma_on) {
    Kokkos::deep_copy(d_sigma_bak,d_sigma_delta);
    Kokkos::deep_copy(d_dep_bak,d_dep_delta);
  }
  if (ehist_on) Kokkos::deep_copy(d_ehist_bak,d_ehist_delta);

#ifdef SPARTA_KOKKOS_EXACT
  if (!random_backup)
    random_backup = new RanKnuth(12345 + comm->me);
  memcpy(random_backup,random,sizeof(RanKnuth));
#endif
}

void SurfReactSurfacePWIKokkos::restore()
{
  Kokkos::deep_copy(d_scalars,d_scalars_bak);
  if (sigma_on) {
    Kokkos::deep_copy(d_sigma_delta,d_sigma_bak);
    Kokkos::deep_copy(d_dep_delta,d_dep_bak);
  }
  if (ehist_on) Kokkos::deep_copy(d_ehist_delta,d_ehist_bak);

#ifdef SPARTA_KOKKOS_EXACT
  memcpy(random,random_backup,sizeof(RanKnuth));
#endif
}
