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
#include "timer.h"
#include "comm.h"
#include "domain.h"
#include "surf.h"
#include "particle_kokkos.h"
#include "surf_kokkos.h"
#include "collide.h"
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
  // distinct base seed per OpenEdge pool (coulomb 32345, chem 42345):
  // a shared base made per-thread streams identical across pools
  rand_pool(22345 + comm->me
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
  conc_dev_on_ = conc_dirty_ = 0;
  dep_alias_on_ = 0;
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
  // 2026-09-08: every PWI mode now has a device implementation
  // (T/A/S incl. mat/conc/rtable/T-axis tables and deposit_as; D/E/R
  // channels; molecular A-conversion; twall_surf / R_surf per-surf
  // customs; polyatomic products via erot_dev/evib_dev). Kept as the
  // single place to reject a future host-only mode explicitly.
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
  {
    enum{NONE,DISCRETE,SMOOTH};
    collide_rot_c = (collide && collide->rotstyle != NONE) ? 1 : 0;
    vibstyle_c = collide ? collide->vibstyle : NONE;
  }
  boltz_c = update->boltz;
  twall_surf_on = R_surf_on = 0;
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
  d_yscale = DAT::t_float_1d("surf_react_pwi:yscale",nl);
  d_mat_isp  = DAT::t_int_1d("surf_react_pwi:mat_isp",nl);
  d_conc_isp = DAT::t_int_1d("surf_react_pwi:conc_isp",nl);
  d_refl_tbl = DAT::t_int_1d("surf_react_pwi:refl_tbl",nl);
  d_prod2 = DAT::t_int_1d("surf_react_pwi:prod2",nl);
  d_e0 = DAT::t_float_1d("surf_react_pwi:e0",nl);
  d_e1 = DAT::t_float_1d("surf_react_pwi:e1",nl);

  auto h_type = Kokkos::create_mirror_view(d_type);
  auto h_prod = Kokkos::create_mirror_view(d_prod);
  auto h_trim = Kokkos::create_mirror_view(d_trim);
  auto h_sput = Kokkos::create_mirror_view(d_sput);
  auto h_prob = Kokkos::create_mirror_view(d_prob);
  auto h_Rrec = Kokkos::create_mirror_view(d_Rrec);
  auto h_spp  = Kokkos::create_mirror_view(d_spp);
  auto h_yscale = Kokkos::create_mirror_view(d_yscale);
  Kokkos::deep_copy(h_yscale,1.0);
  auto h_mat  = Kokkos::create_mirror_view(d_mat_isp);
  auto h_conc = Kokkos::create_mirror_view(d_conc_isp);
  auto h_refl = Kokkos::create_mirror_view(d_refl_tbl);
  auto h_prod2 = Kokkos::create_mirror_view(d_prod2);
  auto h_e0 = Kokkos::create_mirror_view(d_e0);
  auto h_e1 = Kokkos::create_mirror_view(d_e1);
  Kokkos::deep_copy(h_prod2,-1);
  Kokkos::deep_copy(h_mat,-1); Kokkos::deep_copy(h_conc,-1); Kokkos::deep_copy(h_refl,-1);

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
    h_yscale(m) = (r->type == SPUTTER) ? r->sp_yscale : 1.0;
    h_mat(m)  = r->mat_isp;
    h_conc(m) = r->conc_isp;
    h_refl(m) = (r->type == TRIM_REFLECT) ? r->refl_tbl : -1;
    h_prod2(m) = (r->nproduct > 1) ? r->products[1] : -1;
    // energy[] is MAXPRODUCT long; A stores (R, f_mol), D (E0, E1), E (E0)
    h_e0(m) = r->energy ? r->energy[0] : 0.0;
    h_e1(m) = r->energy ? r->energy[1] : 0.0;
  }
  Kokkos::deep_copy(d_type,h_type);
  Kokkos::deep_copy(d_prod,h_prod);
  Kokkos::deep_copy(d_trim,h_trim);
  Kokkos::deep_copy(d_sput,h_sput);
  Kokkos::deep_copy(d_prob,h_prob);
  Kokkos::deep_copy(d_Rrec,h_Rrec);
  Kokkos::deep_copy(d_spp,h_spp);
  Kokkos::deep_copy(d_yscale,h_yscale);
  Kokkos::deep_copy(d_mat_isp,h_mat);
  Kokkos::deep_copy(d_conc_isp,h_conc);
  Kokkos::deep_copy(d_refl_tbl,h_refl);
  Kokkos::deep_copy(d_prod2,h_prod2);
  Kokkos::deep_copy(d_e0,h_e0);
  Kokkos::deep_copy(d_e1,h_e1);

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

  // sputter-yield tables (2D, or 3D with a concentration / wall-T lead
  // axis), padded to max dims; also serve as `rtable` reflection tables

  int nsu = MAX((int) sput_tables.size(),1);
  int maxNE = 1, maxNT = 1, maxNL = 1;
  for (size_t t = 0; t < sput_tables.size(); t++) {
    maxNE = MAX(maxNE,sput_tables[t].NE);
    maxNT = MAX(maxNT,sput_tables[t].NTHETA);
    maxNL = MAX(maxNL,sput_tables[t].nlead());
  }
  d_su_NE = DAT::t_int_1d("surf_react_pwi:su_NE",nsu);
  d_su_NT = DAT::t_int_1d("surf_react_pwi:su_NT",nsu);
  d_su_NL = DAT::t_int_1d("surf_react_pwi:su_NL",nsu);
  d_su_kind = DAT::t_int_1d("surf_react_pwi:su_kind",nsu);
  d_su_E  = DAT::t_float_2d_lr("surf_react_pwi:su_E",nsu,maxNE);
  d_su_th = DAT::t_float_2d_lr("surf_react_pwi:su_th",nsu,maxNT);
  d_su_ax = DAT::t_float_2d_lr("surf_react_pwi:su_ax",nsu,maxNL);
  d_su_Y  = DAT::t_float_2d_lr("surf_react_pwi:su_Y",nsu,maxNL*maxNE*maxNT);

  {
    auto h_NE = Kokkos::create_mirror_view(d_su_NE);
    auto h_NT = Kokkos::create_mirror_view(d_su_NT);
    auto h_NL = Kokkos::create_mirror_view(d_su_NL);
    auto h_kind = Kokkos::create_mirror_view(d_su_kind);
    auto h_E  = Kokkos::create_mirror_view(d_su_E);
    auto h_th = Kokkos::create_mirror_view(d_su_th);
    auto h_ax = Kokkos::create_mirror_view(d_su_ax);
    auto h_Y  = Kokkos::create_mirror_view(d_su_Y);
    Kokkos::deep_copy(h_NE,0);
    Kokkos::deep_copy(h_NT,0);
    Kokkos::deep_copy(h_NL,1);
    Kokkos::deep_copy(h_kind,0);
    Kokkos::deep_copy(h_ax,0.0);
    for (size_t t = 0; t < sput_tables.size(); t++) {
      const ProcessLibrary::TrimSputterTable &st = sput_tables[t];
      h_NE(t) = st.NE;
      h_NT(t) = st.NTHETA;
      h_NL(t) = st.nlead();
      h_kind(t) = (st.NC > 0) ? 1 : ((st.NT > 0) ? 2 : 0);
      for (int i = 0; i < st.NE; i++) h_E(t,i) = st.E[i];
      for (int i = 0; i < st.NTHETA; i++) h_th(t,i) = st.theta[i];
      if (st.NC > 0) for (int i = 0; i < st.NC; i++) h_ax(t,i) = st.C[i];
      else if (st.NT > 0) for (int i = 0; i < st.NT; i++) h_ax(t,i) = st.Tax[i];
      // device layout [il*NE*NTHETA + ie*NTHETA + ia] == host Y order
      // (TrimSputterTable::slice_yield: Y[ic*NE*NTHETA + i*NTHETA + j])
      const int ny = st.nlead()*st.NE*st.NTHETA;
      for (int i = 0; i < ny; i++) h_Y(t,i) = st.Y[i];
    }
    Kokkos::deep_copy(d_su_NE,h_NE);
    Kokkos::deep_copy(d_su_NT,h_NT);
    Kokkos::deep_copy(d_su_NL,h_NL);
    Kokkos::deep_copy(d_su_kind,h_kind);
    Kokkos::deep_copy(d_su_E,h_E);
    Kokkos::deep_copy(d_su_th,h_th);
    Kokkos::deep_copy(d_su_ax,h_ax);
    Kokkos::deep_copy(d_su_Y,h_Y);
  }

  // per-surf material concentration for mat/conc weighting and compound
  // tables: device copy of the <attr>_conc custom (local+ghost surfs),
  // refreshed by upload_conc() whenever sync_sigma re-derives it
  conc_dev_on_ = (sigma_feedback && sconc_index >= 0) ? 1 : 0;
  if (conc_dev_on_) {
    int nslocal = surf->nlocal + surf->nghost;
    d_sconc = DAT::t_float_2d_lr("surf_react_pwi:sconc",MAX(nslocal,1),
                                 MAX(sigma_ncols,1));
    conc_dirty_ = 1;
  }

  // deposit_as alias / debit-candidate tables (identity when unused)
  dep_alias_on_ = (!dep_alias_of.empty() && (int) dep_alias_of.size() == sigma_ncols) ? 1 : 0;
  if (dep_alias_on_) {
    int maxdep = 1;
    for (int j = 0; j < sigma_ncols; j++)
      maxdep = MAX(maxdep,(int) dep_cols_of[j].size());
    d_dep_alias = DAT::t_int_1d("surf_react_pwi:dep_alias",sigma_ncols);
    d_dep_ncols = DAT::t_int_1d("surf_react_pwi:dep_ncols",sigma_ncols);
    d_dep_cols  = DAT::t_int_2d("surf_react_pwi:dep_cols",sigma_ncols,maxdep);
    auto h_al = Kokkos::create_mirror_view(d_dep_alias);
    auto h_nc = Kokkos::create_mirror_view(d_dep_ncols);
    auto h_dc = Kokkos::create_mirror_view(d_dep_cols);
    Kokkos::deep_copy(h_dc,-1);
    for (int j = 0; j < sigma_ncols; j++) {
      h_al(j) = dep_alias_of[j];
      h_nc(j) = (int) dep_cols_of[j].size();
      for (int k = 0; k < h_nc(j); k++) h_dc(j,k) = dep_cols_of[j][k];
    }
    Kokkos::deep_copy(d_dep_alias,h_al);
    Kokkos::deep_copy(d_dep_ncols,h_nc);
    Kokkos::deep_copy(d_dep_cols,h_dc);
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

  if (sigma_on && update->ntimestep % sigma_nevery == 0) {
    timer->stamp(TIME_SREACT);
    fold_sigma();                 // device sigma/dep delta D2H -> "Adens"
    timer->stamp(TIME_ADENS);
  }
  if (ehist_on && ehist_every > 0 &&
      update->ntimestep % ehist_every == 0) fold_ehist();

  SurfReactSurfacePWI::tally_update();
  // sync_sigma (inside the base tally_update) re-derives the per-surf
  // conc every sigma_nevery steps; refresh the device copy before the
  // next move
  // The move kernel reaches this object through a per-step COPY
  // (sr_kk_pwi_copy in the collider), so upload_conc() clearing
  // conc_dirty_ on the copy never reached the original and the conc table
  // (nsurf x ncols doubles) was re-uploaded every move pass. Own the flag
  // here, on the original: dirty only on the ledger-sync step.
  if (conc_dev_on_) conc_dirty_ = (update->ntimestep % sigma_nevery == 0) ? 1 : 0;
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

void SurfReactSurfacePWIKokkos::upload_conc()
{
  if (!conc_dev_on_ || !conc_dirty_) return;
  int nslocal = surf->nlocal + surf->nghost;
  if (nslocal > (int) d_sconc.extent(0))
    d_sconc = DAT::t_float_2d_lr("surf_react_pwi:sconc",nslocal,MAX(sigma_ncols,1));
  double **conc = surf->edarray_local[surf->ewhich[sconc_index]];
  auto h = Kokkos::create_mirror_view(d_sconc);
  for (int i = 0; i < nslocal; i++)
    for (int j = 0; j < sigma_ncols; j++) h(i,j) = conc[i][j];
  Kokkos::deep_copy(d_sconc,h);
  conc_dirty_ = 0;
}

void SurfReactSurfacePWIKokkos::pre_react()
{
  upload_conc();

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species = particle_kk->k_species.view_device();
  custom_ = particle_kk->device_custom();

  // resolve the pweight edvec slot at move time: other modules can add
  // particle customs after init, which shifts ewhich values
  pw_slot = (pweight_ewhich >= 0) ? particle->ewhich[pweight_ewhich] : -1;

  // per-surf customs (twall_surf / R_surf): same binding as the diffuse
  // collider's per-surf temperature; sync only moves data when modified
  twall_surf_on = R_surf_on = 0;
  if (tindex_custom >= 0 || rindex_custom >= 0) {
    SurfKokkos *surf_kk = (SurfKokkos *) surf;
    auto h_edvec_local = surf_kk->k_edvec_local.view_host();
    if (tindex_custom >= 0) {
      if (surf->estatus[tindex_custom] == 0) surf->spread_custom(tindex_custom);
      const int ew = surf->ewhich[tindex_custom];
      h_edvec_local[ew].k_view.sync_device();
      d_twall_surf = h_edvec_local[ew].k_view.view_device();
      twall_surf_on = 1;
    }
    if (rindex_custom >= 0) {
      if (surf->estatus[rindex_custom] == 0) surf->spread_custom(rindex_custom);
      const int ew = surf->ewhich[rindex_custom];
      h_edvec_local[ew].k_view.sync_device();
      d_R_surf = h_edvec_local[ew].k_view.view_device();
      R_surf_on = 1;
    }
  }
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

  Kokkos::deep_copy(DeviceType(),d_scalars_bak,d_scalars);
  if (sigma_on) {
    Kokkos::deep_copy(DeviceType(),d_sigma_bak,d_sigma_delta);
    Kokkos::deep_copy(DeviceType(),d_dep_bak,d_dep_delta);
  }
  if (ehist_on) Kokkos::deep_copy(DeviceType(),d_ehist_bak,d_ehist_delta);

#ifdef SPARTA_KOKKOS_EXACT
  if (!random_backup)
    random_backup = new RanKnuth(12345 + comm->me);
  memcpy(random_backup,random,sizeof(RanKnuth));
#endif
}

void SurfReactSurfacePWIKokkos::restore()
{
  Kokkos::deep_copy(DeviceType(),d_scalars,d_scalars_bak);
  if (sigma_on) {
    Kokkos::deep_copy(DeviceType(),d_sigma_delta,d_sigma_bak);
    Kokkos::deep_copy(DeviceType(),d_dep_delta,d_dep_bak);
  }
  if (ehist_on) Kokkos::deep_copy(DeviceType(),d_ehist_delta,d_ehist_bak);

#ifdef SPARTA_KOKKOS_EXACT
  memcpy(random,random_backup,sizeof(RanKnuth));
#endif
}
