/* ----------------------------------------------------------------------
   OpenEdge: ADAS chemistry — Kokkos backend. See the header for scope.
------------------------------------------------------------------------- */

#include "fix_volume_chem_adas_kokkos.h"
#include "particle_kokkos.h"
#include "grid.h"
#include "update.h"
#include "comm.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "sparta_masks.h"
#include "fix_background.h"

#include <cstring>
#include <cmath>

using namespace SPARTA_NS;

// file-local reaction-type enum, must match fix_volume_chem_adas.cpp
enum{IONIZATION,RECOMBINATION,EXCHANGE,DISSOCIATION};
enum{ADAS,JANEV};                // style enum twin of fix_volume_chem_adas.cpp

static constexpr double MY_2PI_LOC = 6.28318530717958647692;

// ---- hydrogen closed-form rate fits (device twins of the CPU statics) ----

KOKKOS_INLINE_FUNCTION double sigma_cx_hh_m2_dev(double E_lab_eV) {
  const double a[9] = {
    -3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02,
     9.205482406462e-03,  2.400266568315e-03, -1.927122311323e-03,
     3.654750340106e-04, -2.788866460622e-05,  7.422296363524e-07
  };
  const double E = (E_lab_eV < 0.1) ? 0.1
                : (E_lab_eV > 5e5) ? 5e5
                : E_lab_eV;
  const double aL = Kokkos::log(E);
  double s = a[8];
  for (int n = 7; n >= 0; --n) s = s * aL + a[n];
  return Kokkos::exp(s) * 1.0e-4;   // cm^2 -> m^2
}
KOKKOS_INLINE_FUNCTION double log10_sigmav_ioniz_amjuel_cm3s_dev(double Te_eV, double ne_m3) {
  const double c[9][9] = {
    {-3.248025330340e+01, -5.440669186583e-02,  9.048888225109e-02,
     -4.054078993576e-02,  8.976513750477e-03, -1.060334011186e-03,
      6.846238436472e-05, -2.242955329604e-06,  2.890437688072e-08},
    { 1.425332391510e+01, -3.594347160760e-02, -2.014729121556e-02,
      1.039773615730e-02, -1.771792153042e-03,  1.237467264294e-04,
     -3.130184159149e-06, -3.051994601527e-08,  1.888148175469e-09},
    {-6.632235026785e+00,  9.255558353174e-02, -5.580210154625e-03,
     -5.902218748238e-03,  1.295609806553e-03, -1.056721622588e-04,
      4.646310029498e-06, -1.479612391848e-07,  2.852251258320e-09},
    { 2.059544135448e+00, -7.562462086943e-02,  1.519595967433e-02,
      5.803498098354e-04, -3.527285012725e-04,  3.201533740322e-05,
     -1.835196889733e-06,  9.474014343303e-08, -2.342505583774e-09},
    {-4.425370331410e-01,  2.882634019199e-02, -7.285771485050e-03,
      4.643389885987e-04,  1.145700685235e-06,  8.493662724988e-07,
     -1.001032516512e-08, -1.476839184318e-08,  6.047700368169e-10},
    { 6.309381861496e-02, -5.788686535780e-03,  1.507382955250e-03,
     -1.201550548662e-04,  6.574487543511e-06, -9.678782818849e-07,
      5.176265845225e-08,  1.291551676860e-09, -9.685157340473e-11},
    {-5.620091829261e-03,  6.329105568040e-04, -1.527777697951e-04,
      8.270124691336e-06,  3.224101773605e-08,  4.377402649057e-08,
     -2.622921686955e-09, -2.259663431436e-10,  1.161438990709e-11},
    { 2.812016578355e-04, -3.564132950345e-05,  7.222726811078e-06,
      1.433018694347e-07, -1.097431215601e-07,  7.789031791949e-09,
     -4.197728680251e-10,  3.032260338723e-11, -8.911076930014e-13},
    {-6.011143453374e-06,  8.089651265488e-07, -1.186212683668e-07,
     -2.381080756307e-08,  6.271173694534e-09, -5.483010244930e-10,
      3.064611702159e-11, -1.355903284487e-12,  2.935080031599e-14}
  };
  const double Te = (Te_eV < 0.1) ? 0.1 : (Te_eV > 2e4 ? 2e4 : Te_eV);
  const double ne_cm3 = ne_m3 * 1.0e-6;
  const double ne_norm = ne_cm3 / 1.0e8;
  const double ne_c = (ne_norm < 1.0) ? 1.0 : (ne_norm > 1.0e8 ? 1.0e8 : ne_norm);
  const double lT = Kokkos::log(Te);
  const double lD = Kokkos::log(ne_c);
  double sum = 0.0;
  for (int i = 8; i >= 0; --i) {
    double row = c[i][8];
    for (int j = 7; j >= 0; --j) row = row * lD + c[i][j];
    sum = sum * lT + row;
  }
  return sum / 2.302585092994046;
}
KOKKOS_INLINE_FUNCTION double log10_sigmav_cx_hh_cm3s_dev(double Ti_eV, double E_atom_eV) {
  const double c[9][9] = {
    {-1.831670498376e+01,  1.650239332070e-01,  5.025740610454e-02,
      5.288358515136e-03, -2.437122342843e-03, -4.461891214720e-04,
      1.731631548110e-04, -1.588434781959e-05,  4.482291414386e-07},
    { 2.143624996483e-01, -1.067658289373e-01, -5.304993033743e-03,
      8.289383645942e-03, -9.698773663345e-05, -4.470180279338e-04,
      7.944326905066e-05, -5.303688417551e-06,  1.235167254501e-07},
    { 5.139117192662e-02,  9.536923957409e-03, -1.306075129405e-02,
     -1.033166370333e-03,  1.280464204775e-03, -8.453294908907e-05,
     -3.040874906105e-05,  4.747888095498e-06, -1.923953750574e-07},
    {-9.896180369559e-04,  6.315097684976e-03,  2.655464630308e-03,
     -1.365781346175e-03, -1.859939123743e-04,  1.237942304972e-04,
     -1.588253432932e-05,  6.603560345800e-07, -1.970606344918e-09},
    {-2.495327546080e-03, -1.265503371044e-03,  7.569269700468e-04,
      2.756946036257e-04, -1.107375149384e-04, -7.217379426085e-06,
      5.769971321188e-06, -6.717311113584e-07,  2.440961351104e-08},
    {-2.417046684097e-05, -6.945512319613e-05, -2.956984088728e-04,
      2.318277483195e-05,  3.704494397140e-05, -6.066558692480e-06,
     -4.951573401626e-07,  1.437520597154e-07, -6.998724470004e-09},
    { 1.177406072793e-04,  3.698501620365e-05,  3.424317896619e-05,
     -9.815693511794e-06, -4.285719813022e-06,  1.169257650609e-06,
     -4.968953461875e-10, -1.618948982477e-08,  9.440094842562e-10},
    {-1.483036457978e-05, -3.348172574417e-06, -1.527018819072e-06,
      8.362050692462e-07,  2.058392726953e-07, -7.463594884928e-08,
      5.924370389093e-10,  1.078208689229e-09, -6.619767848464e-11},
    { 5.351909441226e-07,  9.728230870242e-08,  1.676354786072e-08,
     -2.237567830699e-08, -3.081685803820e-09,  1.450862501121e-09,
      4.434231893204e-11, -3.324377862622e-11,  1.935019679501e-12}
  };
  const double Ti  = (Ti_eV     < 0.1) ? 0.1 : (Ti_eV     > 1e5 ? 1e5 : Ti_eV);
  const double Ea  = (E_atom_eV < 0.1) ? 0.1 : (E_atom_eV > 1e5 ? 1e5 : E_atom_eV);
  const double lT  = Kokkos::log(Ti);
  const double lE  = Kokkos::log(Ea);
  double sum = 0.0;
  for (int i = 8; i >= 0; --i) {
    double row = c[i][8];
    for (int j = 7; j >= 0; --j) row = row * lE + c[i][j];
    sum = sum * lT + row;
  }
  return sum / 2.302585092994046;
}

/* ---------------------------------------------------------------------- */

FixVolumeChemAdasKokkos::FixVolumeChemAdasKokkos(SPARTA *sparta, int narg,
                                                 char **arg) :
  FixVolumeChemAdas(sparta, narg, arg),
  // distinct base seed per OpenEdge pool: with the shared 12345 base the
  // chem event-test and coulomb partner-sampling streams were identical
  // per thread (first-draw correlation on the same particle index)
  rand_pool(42345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(rng_adas);
#endif
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read   = PARTICLE_MASK | SPECIES_MASK | CUSTOM_MASK;
  datamask_modify = PARTICLE_MASK;

  device_ok = 0;
  warned_fallback = 0;
  nn_stamp_n = -1;
  nn_stamp_id = (cellint) -1;
  nn_stamp_gen = -2;
  have_cx_ = 0;
}

/* ---------------------------------------------------------------------- */

FixVolumeChemAdasKokkos::~FixVolumeChemAdasKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::init()
{
  FixVolumeChemAdas::init();

  // ---- decide whether the device fast path covers this configuration ----

  device_ok = 1;
  const char *why = nullptr;

  if (getenv("OE_CHEM_HOST")) {
    device_ok = 0; why = "OE_CHEM_HOST env override";
  } else if (eirene_mode) {
    device_ok = 0; why = "eirene_mode";
  } else if (volume_source_cidx >= 0 || volume_source_fidx >= 0) {
    device_ok = 0; why = "volume_source";
  } else if (tally_units == TALLY_BATCH || tally_units == TALLY_BATCH_FIX) {
    device_ok = 0; why = "batch tally units";
  } else if (rate_cache_mode == 1) {
    device_ok = 0; why = "rate_cache cell (device is per-particle)";
  }

  if (device_ok) {
    for (int i = 0; i < nlist; i++) {
      if (!rlist[i].active) continue;
      if (rlist[i].nproduct < 1 || rlist[i].products[0] < 0) {
        device_ok = 0; why = "active reaction without a product";
        break;
      }
    }
  }

  // device_ok gates the CX need-flag MPI_Allreduce in end_of_step, and
  // getenv() is per-rank state (a launcher can export OE_CHEM_HOST on a
  // subset of ranks) -> make the decision comm-uniform or that
  // collective becomes a subset collective (gate-6/8 deadlock class)
  {
    int ok_min = device_ok;
    MPI_Allreduce(&device_ok, &ok_min, 1, MPI_INT, MPI_MIN, world);
    if (ok_min < device_ok) {
      device_ok = ok_min;
      why = "host fallback forced on another rank";
    }
  }

  if (!device_ok) {
    if (!warned_fallback && comm->me == 0 && screen)
      fprintf(screen, "fix volume/chem/adas/kk: HOST fallback (%s)\n", why);
    warned_fallback = 1;
    return;
  }

  upload_static_tables();

  if (comm->me == 0 && screen)
    fprintf(screen, "fix volume/chem/adas/kk: device path active "
            "(Z=%d, %d reactions)\n", atomic_number, nlist);
}

/* ----------------------------------------------------------------------
   upload the ADAS tables and the reaction topology (static per run)
------------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::upload_static_tables()
{
  auto mit = materials_rate_data.find(atomic_number);
  if (mit == materials_rate_data.end())
    error->all(FLERR,"volume/chem/adas/kk: no rate tables loaded");
  const RateData &rd = mit->second;

  auto up1 = [&](const std::vector<double> &v, const char *name) {
    DAT::t_float_1d d(Kokkos::view_alloc(std::string(name),
                                         Kokkos::WithoutInitializing),
                      v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); i++) h(i) = v[i];
    Kokkos::deep_copy(d,h);
    return d;
  };

  d_ion_coeff = up1(rd.ion_coeff,"chem:ion");
  d_rec_coeff = up1(rd.rec_coeff,"chem:rec");
  d_cx_coeff  = up1(rd.cx_coeff, "chem:cx");
  d_plt_coeff = up1(rd.plt_coeff,"chem:plt");
  d_prb_coeff = up1(rd.prb_coeff,"chem:prb");
  d_gT_ion = up1(rd.gridT_ion,"chem:gT_ion"); d_gD_ion = up1(rd.gridD_ion,"chem:gD_ion");
  d_gT_rec = up1(rd.gridT_rec,"chem:gT_rec"); d_gD_rec = up1(rd.gridD_rec,"chem:gD_rec");
  d_gT_cx  = up1(rd.gridT_cx, "chem:gT_cx");  d_gD_cx  = up1(rd.gridD_cx, "chem:gD_cx");
  d_gT_plt = up1(rd.gridT_plt,"chem:gT_plt"); d_gD_plt = up1(rd.gridD_plt,"chem:gD_plt");
  d_gT_prb = up1(rd.gridT_prb,"chem:gT_prb"); d_gD_prb = up1(rd.gridD_prb,"chem:gD_prb");
  d_ion_pot = up1(rd.ion_potential,"chem:ion_pot");

  ion_nQ_ = rd.ion_nQ; ion_nT_ = rd.ion_nT; ion_nD_ = rd.ion_nD;
  rec_nQ_ = rd.rec_nQ; rec_nT_ = rd.rec_nT; rec_nD_ = rd.rec_nD;
  cx_nQ_  = rd.cx_nQ;  cx_nT_  = rd.cx_nT;  cx_nD_  = rd.cx_nD;
  plt_nQ_ = rd.plt_nQ; plt_nT_ = rd.plt_nT; plt_nD_ = rd.plt_nD;
  prb_nQ_ = rd.prb_nQ; prb_nT_ = rd.prb_nT; prb_nD_ = rd.prb_nD;

  // reaction topology: per-species CSR + per-reaction type/product
  const int nspecies = particle->nspecies;
  int ntot = 0;
  for (int s = 0; s < nspecies; s++) ntot += reactions[s].n;

  DAT::tdual_int_1d k_off("chem:roff",nspecies+1);
  DAT::tdual_int_1d k_lst("chem:rlst",ntot > 0 ? ntot : 1);
  int at = 0;
  for (int s = 0; s < nspecies; s++) {
    k_off.h_view(s) = at;
    for (int i = 0; i < reactions[s].n; i++)
      k_lst.h_view(at++) = reactions[s].list[i];
  }
  k_off.h_view(nspecies) = at;
  k_off.modify_host(); k_off.sync_device();
  k_lst.modify_host(); k_lst.sync_device();
  d_react_offset = k_off.d_view;
  d_react_list   = k_lst.d_view;

  DAT::tdual_int_1d k_type("chem:rtype",nlist);
  DAT::tdual_int_1d k_prod("chem:rprod",nlist);
  for (int i = 0; i < nlist; i++) {
    k_type.h_view(i) = rlist[i].active ? rlist[i].type : -1;
    k_prod.h_view(i) = (rlist[i].nproduct >= 1) ? rlist[i].products[0] : -1;
  }
  k_type.modify_host(); k_type.sync_device();
  k_prod.modify_host(); k_prod.sync_device();
  d_r_type = k_type.d_view;
  d_r_product0 = k_prod.d_view;

  // A1b: styles, product counts, second product, JANEV coefficients
  int maxc = 1;
  for (int i = 0; i < nlist; i++)
    if (rlist[i].ncoeff > maxc) maxc = rlist[i].ncoeff;
  DAT::tdual_int_1d k_style("chem:rstyle",nlist);
  DAT::tdual_int_1d k_np("chem:rnp",nlist);
  DAT::tdual_int_1d k_p1("chem:rp1",nlist);
  DAT::tdual_int_1d k_nc("chem:rnc",nlist);
  d_r_coeff = DAT::t_float_2d_lr(
      Kokkos::view_alloc("chem:rcoeff",Kokkos::WithoutInitializing),
      nlist,maxc);
  auto h_cf = Kokkos::create_mirror_view(d_r_coeff);
  for (int i = 0; i < nlist; i++) {
    k_style.h_view(i) = rlist[i].style;
    k_np.h_view(i)    = rlist[i].nproduct;
    k_p1.h_view(i)    = (rlist[i].nproduct >= 2) ? rlist[i].products[1] : -1;
    const int nc = (rlist[i].coeff && rlist[i].ncoeff > 0)
                     ? rlist[i].ncoeff : 0;
    k_nc.h_view(i) = nc;
    for (int c = 0; c < maxc; c++)
      h_cf(i,c) = (c < nc) ? rlist[i].coeff[c] : 0.0;
  }
  Kokkos::deep_copy(d_r_coeff,h_cf);
  k_style.modify_host(); k_style.sync_device();
  k_np.modify_host();    k_np.sync_device();
  k_p1.modify_host();    k_p1.sync_device();
  k_nc.modify_host();    k_nc.sync_device();
  d_r_style    = k_style.d_view;
  d_r_nproduct = k_np.d_view;
  d_r_product1 = k_p1.d_view;
  d_r_ncoeff   = k_nc.d_view;

  atomic1_ = (atomic_number == 1) ? 1 : 0;
  have_two_ = 0;
  DAT::tdual_int_1d k_sp2("chem:sp2",nspecies);
  for (int sidx = 0; sidx < nspecies; sidx++) {
    int two = 0;
    for (int i = 0; i < reactions[sidx].n; i++) {
      const int ridx = reactions[sidx].list[i];
      if (rlist[ridx].active && rlist[ridx].nproduct >= 2) two = 1;
    }
    k_sp2.h_view(sidx) = two;
    if (two) have_two_ = 1;
  }
  k_sp2.modify_host(); k_sp2.sync_device();
  d_sp_twoprod = k_sp2.d_view;

  have_cx_ = 0;
  for (int i = 0; i < nlist; i++)
    if (rlist[i].active && rlist[i].type == EXCHANGE) have_cx_ = 1;
}

/* ----------------------------------------------------------------------
   per-cell neutral-D density for the impurity CX partner. Decomposition-
   indexed -> stamped and rebuilt after fix balance re-decompositions
   (same pattern as the sheath maps in update_kokkos.cpp).
------------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::build_nn_cell()
{
  const int ng = grid->nlocal;
  d_nn_cell = DAT::t_float_1d(
      Kokkos::view_alloc("chem:nn_cell",Kokkos::WithoutInitializing),
      ng > 0 ? ng : 1);
  auto h = Kokkos::create_mirror_view(d_nn_cell);
  for (int icell = 0; icell < ng; icell++)
    h(icell) = neutral_dens_at_cell(icell);
  Kokkos::deep_copy(d_nn_cell,h);

  nn_stamp_n = ng;
  nn_stamp_id = (ng > 0 && grid->cells) ? grid->cells[0].id : (cellint) -1;
  nn_stamp_gen = nn_pd ? nn_pd->generation : -1;
}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  if (!update->plasma_cache_flag)
    error->all(FLERR,
      "fix volume/chem/adas: per-particle plasma cache not active — "
      "configure sheath / GCA / global bfield_compute so update.cpp "
      "populates Te/ne at particle positions");

  if (!device_ok) {
    // transparent host fallback: ModifyKokkos wrapped this call with
    // datamask sync, and kokkos_flag=1 means no auto_sync — do the
    // host-side sync explicitly, run the base, mark host-modified.
    ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
    nreact_one = 0;
    if (!particle->sorted) particle->sort();
    end_of_step_no_average();
    nreact_running += nreact_one;
    // CUSTOM too: the base path creates particles and writes their custom
    // attributes (pweight etc. via update_custom); marking only PARTICLE
    // lost those writes on the next sync(Device,CUSTOM_MASK) on CUDA
    particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    // ModifyKokkos marks Device-modified after this call returns; push
    // the host-side changes down first so that mark is truthful (matters
    // on CUDA where host/device memories are distinct).
    particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
    return;
  }

  // ---- device fast path (no sort needed: flat per-particle kernel) ----

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  nreact_one = 0;

  // per-cell source-tally array bookkeeping (identical to the base)
  const int ncols = size_per_grid_cols;
  if (grid->maxlocal > maxgrid_src) {
    const int oldmax = maxgrid_src;
    maxgrid_src = grid->maxlocal;
    memory->grow(array_grid, maxgrid_src, ncols,
                 "volume/chem/adas:array_grid(src)");
    if (maxgrid_src > oldmax)
      memset(&array_grid[oldmax][0], 0,
             sizeof(double) * (maxgrid_src - oldmax) * ncols);
  }
  if (tally_units == TALLY_RATE && maxgrid_src > 0)
    memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * ncols);

  // pweight handle (RATE-mode event weighting)
  pweight_index  = particle->find_custom((char *) "pweight");
  pweight_ewhich = (pweight_index >= 0) ? particle->ewhich[pweight_index] : -1;

  // neutral-density cell cache (CX partner) with rebalance stamps
  if (have_cx_) {
    const int nloc = grid->nlocal;
    const cellint fid = (nloc > 0 && grid->cells)
      ? grid->cells[0].id : (cellint) -1;
    // generation covers a between-runs plasma reload (reload() bumps
    // it): the decomposition stamps alone would keep stale dens_n
    const int gen = nn_pd ? nn_pd->generation : -1;
    int need = (nloc != nn_stamp_n || fid != nn_stamp_id ||
                gen != nn_stamp_gen) ? 1 : 0;
    int need_any = 0;
    MPI_Allreduce(&need,&need_any,1,MPI_INT,MPI_MAX,world);
    if (need_any) build_nn_cell();
  }

  // A1b: pre-grow particle storage so the kernel can append dissociation
  // partners without a realloc-retry protocol (at most one partner per
  // eligible parent per step — a guaranteed bound)
  if (have_two_) {
    particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
    auto d_part = particle_kk->k_particles.view_device();
    auto d_sp2  = d_sp_twoprod;
    const int nloc = particle->nlocal;
    int ncap = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nloc),
      KOKKOS_LAMBDA(const int ii, int &c) {
        const int sp = d_part(ii).ispecies;
        if (sp >= 0 && sp < (int) d_sp2.extent(0)) c += d_sp2(sp);
      }, ncap);
    if (particle->nlocal + ncap > particle->maxlocal)
      particle->grow(particle->nlocal + ncap - particle->maxlocal);
    if (!d_new_count.data())
      d_new_count = Kokkos::View<int, DeviceType>("chem:newn");
    Kokkos::deep_copy(d_new_count, particle->nlocal);
    custom_  = particle_kk->device_custom();
    pw_slot_ = (pweight_index >= 0) ? particle->ewhich[pweight_index] : -1;
  }

  // bind device views (fresh each call: growth/realloc safe)
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species   = particle_kk->k_species.d_view;

  auto edvec = [&](int cidx) -> DAT::t_float_1d {
    return particle_kk->k_edvec.h_view[particle->ewhich[cidx]].k_view.d_view;
  };
  d_pc_te = edvec(update->pc_te_custom);
  d_pc_ne = edvec(update->pc_ne_custom);
  have_ti_ = (update->pc_ti_custom >= 0 &&
              particle->ewhich[update->pc_ti_custom] >= 0);
  have_vpar_ = (update->pc_vpar_custom >= 0 &&
                particle->ewhich[update->pc_vpar_custom] >= 0);
  have_b_ = (update->pc_bx_custom >= 0 &&
             particle->ewhich[update->pc_bx_custom] >= 0);
  if (have_ti_)   d_pc_ti   = edvec(update->pc_ti_custom);
  if (have_vpar_) d_pc_vpar = edvec(update->pc_vpar_custom);
  if (have_b_) {
    d_pc_bx = edvec(update->pc_bx_custom);
    d_pc_by = edvec(update->pc_by_custom);
    d_pc_bz = edvec(update->pc_bz_custom);
  }
  have_pweight_ = (pweight_ewhich >= 0);
  if (have_pweight_)
    d_pweight = particle_kk->k_edvec.h_view[pweight_ewhich].k_view.d_view;

  const int nlocal = particle->nlocal;
  nglocal_ = grid->nlocal;
  dt_chem_ = nevery * update->dt;
  fnum_    = update->fnum;
  atomic_number_ = atomic_number;
  tally_units_   = tally_units;
  output_mode_   = output_mode;

  // event buffers: hard bound of one event per particle
  if ((int) d_ev_ridx.extent(0) < nlocal) {
    d_ev_ridx = DAT::t_int_1d(
        Kokkos::view_alloc("chem:ev_ridx",Kokkos::WithoutInitializing),nlocal);
    d_ev_cell = DAT::t_int_1d(
        Kokkos::view_alloc("chem:ev_cell",Kokkos::WithoutInitializing),nlocal);
    d_ev_vals = DAT::t_float_2d_lr(
        Kokkos::view_alloc("chem:ev_vals",Kokkos::WithoutInitializing),nlocal,6);
  }
  if (!d_ev_count.data())
    d_ev_count = Kokkos::View<int,DeviceType>("chem:ev_count");
  Kokkos::deep_copy(d_ev_count,0);

  if (nlocal > 0) {
    // parallel_for copies the functor (this whole fix); copymode stops
    // the copy's destructor from freeing the base-class allocations
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType,TagFixChemAdas>(0,nlocal),*this);
    Kokkos::fence();
    copymode = 0;
  }

  particle_kk->modify(Device,PARTICLE_MASK);
  if (have_two_ && d_new_count.data()) {
    int nnew_total = 0;
    Kokkos::deep_copy(nnew_total, d_new_count);
    if (nnew_total > particle->nlocal) {
      particle->nlocal = nnew_total;
      particle->sorted = 0;                       // CPU drain parity
      particle_kk->modify(Device,CUSTOM_MASK);    // newborn customs
    }
  }

  // ---- apply the rare events to the host tallies ----
  int nev = 0;
  Kokkos::deep_copy(nev,d_ev_count);
  if (nev > 0) {
    auto h_ridx = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, Kokkos::subview(d_ev_ridx,std::make_pair(0,nev)));
    auto h_cell = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, Kokkos::subview(d_ev_cell,std::make_pair(0,nev)));
    auto h_vals = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{},
        Kokkos::subview(d_ev_vals,std::make_pair(0,nev),Kokkos::ALL));

    for (int e = 0; e < nev; e++) {
      const int ridx = h_ridx(e);
      if (ridx < 0 || ridx >= nlist) continue;   // defensive
      tally_reactions[ridx]++;
      nreact_one++;
      const int icell = h_cell(e);
      if (!array_grid || icell < 0 || icell >= maxgrid_src) continue;
      double *row = array_grid[icell];
      if (output_mode == OUT_DETAILED) {
        int rtype_off = -1;
        switch (rlist[ridx].type) {
          case IONIZATION:     rtype_off = 0; break;
          case RECOMBINATION:  rtype_off = 1; break;
          case EXCHANGE:       rtype_off = 2; break;
          case DISSOCIATION:   rtype_off = 3; break;
        }
        if (rtype_off >= 0) {
          row[rtype_off]      += h_vals(e,0);
          row[4  + rtype_off] += h_vals(e,1);
          row[8  + rtype_off] += h_vals(e,2);
          row[12 + rtype_off] += h_vals(e,3);
          row[16 + rtype_off] += h_vals(e,4);
        }
      } else {
        for (int c = 0; c < 6; c++) row[c] += h_vals(e,c);
      }
    }
  }

  // RATE-mode normalization (identical to the base tail)
  if (tally_units == TALLY_RATE && maxgrid_src > 0) {
    const double window_s = nevery * update->dt;
    const double fnum = update->fnum;
    if (window_s > 0.0 && fnum > 0.0) {
      const double num = fnum / window_s;
      Grid::ChildInfo *cinfo = grid->cinfo;
      for (int icell = 0; icell < nglocal_; icell++) {
        const double vol = cinfo[icell].volume;
        double *row = array_grid[icell];
        if (vol <= 0.0) { for (int c = 0; c < ncols; c++) row[c] = 0.0; continue; }
        const double scale = num / vol;
        for (int c = 0; c < ncols; c++) row[c] *= scale;
      }
    }
  }

  nreact_running += nreact_one;
}

/* ----------------------------------------------------------------------
   device kernel: per-particle port of FixVolumeChemAdas::attempt() for
   the supported scope (impurity ioniz/recomb/CX, single product)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixVolumeChemAdasKokkos::operator()(TagFixChemAdas, const int &i) const
{
  auto &p = d_particles(i);
  const int isp = p.ispecies;
  const int r0 = d_react_offset(isp);
  const int n  = d_react_offset(isp+1) - r0;
  if (n == 0) return;

  const int icell = p.icell;
  if (icell < 0 || icell >= nglocal_) return;

  const double Te_eV = Kokkos::fmax(d_pc_te(i), 1e-6);
  const double ne_m3 = Kokkos::fmax(d_pc_ne(i), 0.0);
  if (Te_eV <= 0.0 || ne_m3 <= 0.0) return;
  const double Ti_eV = have_ti_ ? Kokkos::fmax(d_pc_ti(i), 0.0) : 0.0;
  const double vpar  = have_vpar_ ? d_pc_vpar(i) : 0.0;
  const double bx = have_b_ ? d_pc_bx(i) : 0.0;
  const double by = have_b_ ? d_pc_by(i) : 0.0;
  const double bz = have_b_ ? d_pc_bz(i) : 0.0;

  const double logTe    = Kokkos::log10(Te_eV);
  const double logne_cm = Kokkos::log10(Kokkos::fmax(ne_m3 * 1e-6, 1e-99));

  const int q = (int) Kokkos::fmax(0.0, d_species[isp].charge);

  double lambda[16];
  int    ridx_map[16];
  int    nchan = 0;
  double lambda_total = 0.0;

  for (int k = 0; k < n && nchan < 16; k++) {
    const int ridx = d_react_list(r0 + k);
    const int rtype = d_r_type(ridx);
    if (rtype < 0) continue;   // inactive

    double rate_log10 = -INFINITY;
    double dens_partner = ne_m3;

    if (rtype == IONIZATION) {
      if (q >= atomic_number_) continue;
      if (atomic1_ && q == 0)
        // AMJUEL H.4 2.1.5 (D ionization), replaces ADAS SCD89
        rate_log10 = log10_sigmav_ioniz_amjuel_cm3s_dev(Te_eV, ne_m3);
      else
        rate_log10 = interp_class(d_ion_coeff,d_gT_ion,d_gD_ion,
                                  ion_nQ_,ion_nT_,ion_nD_,q,logTe,logne_cm);
    } else if (rtype == RECOMBINATION) {
      if (q == 0) continue;
      rate_log10 = interp_class(d_rec_coeff,d_gT_rec,d_gD_rec,
                                rec_nQ_,rec_nT_,rec_nD_,q-1,logTe,logne_cm);
    } else if (rtype == EXCHANGE) {
      if (atomic1_ && Ti_eV > 0.0) {
        // HYDHEL H.3 3.1.8 (D-D+ resonant CX): partner = D+ (~ne),
        // per-particle E_atom from the lab-frame kinetic energy
        constexpr double eV_to_J_loc = 1.602176634e-19;
        const double m_atom = d_species[isp].mass;
        const double v2p = p.v[0]*p.v[0] + p.v[1]*p.v[1] + p.v[2]*p.v[2];
        const double E_atom_eV = 0.5 * m_atom * v2p / eV_to_J_loc;
        rate_log10 = log10_sigmav_cx_hh_cm3s_dev(Ti_eV, E_atom_eV);
      } else {
        // impurity-H CX (ADAS CCD): partner = NEUTRAL D (mesh/dens_n)
        dens_partner = d_nn_cell(icell);
        if (dens_partner <= 0.0) continue;
        const int cx_row = (q > 0) ? (q - 1) : 0;
        rate_log10 = interp_class(d_cx_coeff,d_gT_cx,d_gD_cx,
                                  cx_nQ_,cx_nT_,cx_nD_,cx_row,logTe,logne_cm);
      }
    } else if (rtype == DISSOCIATION && d_r_style(ridx) == JANEV) {
      // Janev polynomial: ln<sv> = sum_n b_n (ln Te)^n
      const double lnT = Kokkos::log(Te_eV);
      double lnsv = d_r_coeff(ridx,0);
      double lnTn = 1.0;
      const int ncf = d_r_ncoeff(ridx);
      for (int kk = 1; kk < ncf; kk++) {
        lnTn *= lnT;
        lnsv += d_r_coeff(ridx,kk) * lnTn;
      }
      rate_log10 = lnsv / 2.302585092994046;
    } else {
      continue;
    }

    if (!Kokkos::isfinite(rate_log10)) continue;
    const double lam = lambda_from(rate_log10, dt_chem_, dens_partner);
    if (lam <= 0.0) continue;

    lambda[nchan] = lam;
    ridx_map[nchan] = ridx;
    lambda_total += lam;
    nchan++;
  }

  if (nchan == 0 || lambda_total <= 0.0) return;

  rand_type rand_gen = rand_pool.get_state();

  const double P_any = -Kokkos::expm1(-lambda_total);
  const double u = rand_gen.drand();
  if (u > P_any) { rand_pool.free_state(rand_gen); return; }

  int chosen = 0;
  if (nchan > 1) {
    const double v = rand_gen.drand() * lambda_total;
    double cumsum = 0.0;
    for (int k = 0; k < nchan; k++) {
      cumsum += lambda[k];
      if (v <= cumsum) { chosen = k; break; }
    }
  }

  const int best_idx = ridx_map[chosen];
  const int rtype = d_r_type(best_idx);
  if (d_r_product0(best_idx) < 0) { rand_pool.free_state(rand_gen); return; }

  // ---- event tally values (per output mode), scale folded in ----
  double scale = 1.0;
  if (tally_units_ == TALLY_RATE && have_pweight_ && fnum_ > 0.0)
    scale = d_pweight(i) / fnum_;

  const double m   = d_species[isp].mass;
  const double vx0 = p.v[0], vy0 = p.v[1], vz0 = p.v[2];

  double vals[6] = {0,0,0,0,0,0};
  if (output_mode_ == OUT_DETAILED) {
    const double ke = 0.5 * m * (vx0*vx0 + vy0*vy0 + vz0*vz0);
    vals[0] = scale;
    vals[1] = m * vx0 * scale;
    vals[2] = m * vy0 * scale;
    vals[3] = m * vz0 * scale;
    vals[4] = ke * scale;
  } else {
    constexpr double eV_to_J = 1.602176634e-19;
    const double Bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
    double vix = 0.0, viy = 0.0, viz = 0.0;
    if (Bmag > 1e-30) {
      const double invB = 1.0 / Bmag;
      vix = vpar * bx * invB; viy = vpar * by * invB; viz = vpar * bz * invB;
    }
    const double v2  = vx0*vx0 + vy0*vy0 + vz0*vz0;
    const double vi2 = vix*vix + viy*viy + viz*viz;
    const double Ti_J = Ti_eV * eV_to_J;

    double dSp=0, dSmx=0, dSmy=0, dSmz=0, dQe=0, dQi=0;
    if (rtype == IONIZATION) {
      double E_eff_eV = 0.0;
      if ((int) d_ion_pot.extent(0) > q) E_eff_eV = d_ion_pot(q);
      if (plt_nQ_ > 0 && ion_nQ_ > 0) {
        const double plt_log10 = interp_class(d_plt_coeff,d_gT_plt,d_gD_plt,
                                     plt_nQ_,plt_nT_,plt_nD_,q,logTe,logne_cm);
        const double scd_log10 = interp_class(d_ion_coeff,d_gT_ion,d_gD_ion,
                                     ion_nQ_,ion_nT_,ion_nD_,q,logTe,logne_cm);
        if (Kokkos::isfinite(plt_log10) && Kokkos::isfinite(scd_log10)) {
          const double J_per_event = Kokkos::pow(10.0, plt_log10 - scd_log10);
          E_eff_eV += J_per_event / eV_to_J;
        }
      }
      dSp = +1.0;
      dSmx = +m*vx0; dSmy = +m*vy0; dSmz = +m*vz0;
      dQe = -E_eff_eV * eV_to_J;
      dQi = +0.5 * m * v2;
    } else if (rtype == RECOMBINATION) {
      double E_rec_loss_eV = 0.0;
      if (prb_nQ_ > 0 && rec_nQ_ > 0 && q > 0) {
        const int qrow = q - 1;
        const double prb_log10 = interp_class(d_prb_coeff,d_gT_prb,d_gD_prb,
                                     prb_nQ_,prb_nT_,prb_nD_,qrow,logTe,logne_cm);
        const double acd_log10 = interp_class(d_rec_coeff,d_gT_rec,d_gD_rec,
                                     rec_nQ_,rec_nT_,rec_nD_,qrow,logTe,logne_cm);
        if (Kokkos::isfinite(prb_log10) && Kokkos::isfinite(acd_log10)) {
          const double J_per_event = Kokkos::pow(10.0, prb_log10 - acd_log10);
          E_rec_loss_eV = J_per_event / eV_to_J;
        }
      }
      dSp = -1.0;
      dSmx = -m*vix; dSmy = -m*viy; dSmz = -m*viz;
      dQe = -E_rec_loss_eV * eV_to_J;
      dQi = -(0.5 * m * vi2 + 1.5 * Ti_J);
    } else if (rtype == EXCHANGE) {
      dSp = 0.0;
      dSmx = +m*(vx0 - vix); dSmy = +m*(vy0 - viy); dSmz = +m*(vz0 - viz);
      dQe = 0.0;
      dQi = +0.5 * m * (v2 - vi2) - 1.5 * Ti_J;
    } else if (rtype == DISSOCIATION) {
      // D2/H2/T2 bond energy from the electron fluid (CPU parity)
      constexpr double E_diss_D2_eV = 4.478;
      dQe = -E_diss_D2_eV * eV_to_J;
    }
    vals[0]=dSp*scale; vals[1]=dSmx*scale; vals[2]=dSmy*scale;
    vals[3]=dSmz*scale; vals[4]=dQe*scale; vals[5]=dQi*scale;
  }

  const int slot = Kokkos::atomic_fetch_add(&d_ev_count(),1);
  d_ev_ridx(slot) = best_idx;
  d_ev_cell(slot) = icell;
  for (int c = 0; c < 6; c++) d_ev_vals(slot,c) = vals[c];

  // ---- apply the reaction to the particle ----
  // (no eirene_mode, no GC state to invalidate — hybrid errors out under
  // Kokkos; single product only, enforced at init)
  p.ispecies = d_r_product0(best_idx);

  double fc_kick_x = 0.0, fc_kick_y = 0.0, fc_kick_z = 0.0;

  if (rtype == EXCHANGE && Ti_eV > 0.0) {
    constexpr double kB = 1.380649e-23;
    constexpr double eV_to_J = 1.602176634e-19;
    const double Ti_K = Ti_eV * eV_to_J / kB;
    const double m_prod = d_species[d_r_product0(best_idx)].mass;
    const double v_th = (m_prod > 0.0) ? Kokkos::sqrt(kB * Ti_K / m_prod) : 0.0;

    const double Bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
    double vfx = 0.0, vfy = 0.0, vfz = 0.0;
    if (Bmag > 1e-30) {
      const double invB = 1.0 / Bmag;
      vfx = vpar * bx * invB; vfy = vpar * by * invB; vfz = vpar * bz * invB;
    }
    if (atomic1_) {
      // hydrogen resonant CX: sigma*v-weighted rejection sampling of the
      // bulk-ion partner velocity (exact port of the Z==1 branch)
      const double E_lab_factor = 0.5 * m_prod / eV_to_J;
      constexpr double SGCVMX = 1.2e-13;
      double v_post_x = 0.0, v_post_y = 0.0, v_post_z = 0.0;
      int icount = 0;
      while (true) {
        const double u1 = Kokkos::fmax(rand_gen.drand(), 1e-30);
        const double u2 = rand_gen.drand();
        const double u3 = rand_gen.drand();
        const double u4 = Kokkos::fmax(rand_gen.drand(), 1e-30);
        const double g1 = Kokkos::sqrt(-2.0*Kokkos::log(u1)) * Kokkos::cos(MY_2PI_LOC*u2);
        const double g2 = Kokkos::sqrt(-2.0*Kokkos::log(u1)) * Kokkos::sin(MY_2PI_LOC*u2);
        const double g3 = Kokkos::sqrt(-2.0*Kokkos::log(u4)) * Kokkos::cos(MY_2PI_LOC*u3);
        v_post_x = vfx + v_th * g1;
        v_post_y = vfy + v_th * g2;
        v_post_z = vfz + v_th * g3;

        const double dvx = v_post_x - vx0;
        const double dvy = v_post_y - vy0;
        const double dvz = v_post_z - vz0;
        const double vrel_sq = dvx*dvx + dvy*dvy + dvz*dvz;
        const double vrel    = Kokkos::sqrt(vrel_sq);
        const double E_lab   = E_lab_factor * vrel_sq;
        const double sigma   = sigma_cx_hh_m2_dev(E_lab);
        const double sgv     = sigma * vrel;

        if (rand_gen.drand() * SGCVMX < sgv) break;
        if (++icount >= 500) break;
      }
      p.v[0] = v_post_x;
      p.v[1] = v_post_y;
      p.v[2] = v_post_z;
    } else {
      // impurity-H CX: shifted-Maxwellian draw at local Ti + bulk flow
      // (exact port of the Z>=2 branch of attempt())
      const double u1 = Kokkos::fmax(rand_gen.drand(), 1e-30);
      const double u2 = rand_gen.drand();
      const double u3 = rand_gen.drand();
      const double u4 = Kokkos::fmax(rand_gen.drand(), 1e-30);
      const double g1 = Kokkos::sqrt(-2.0*Kokkos::log(u1)) * Kokkos::cos(MY_2PI_LOC*u2);
      const double g2 = Kokkos::sqrt(-2.0*Kokkos::log(u1)) * Kokkos::sin(MY_2PI_LOC*u2);
      const double g3 = Kokkos::sqrt(-2.0*Kokkos::log(u4)) * Kokkos::cos(MY_2PI_LOC*u3);
      p.v[0] = vfx + v_th * g1;
      p.v[1] = vfy + v_th * g2;
      p.v[2] = vfz + v_th * g3;
    }
  } else if (rtype == DISSOCIATION) {
    // isotropic Franck-Condon kick in the parent CM frame (~3 eV/atom,
    // independent of Ti — CPU parity)
    constexpr double E_FC_eV = 3.0;
    constexpr double eV_to_J = 1.602176634e-19;
    const double m_prod = d_species[d_r_product0(best_idx)].mass;
    const double v_FC = (m_prod > 0.0)
                        ? Kokkos::sqrt(2.0 * E_FC_eV * eV_to_J / m_prod) : 0.0;
    const double u1 = rand_gen.drand();
    const double u2 = rand_gen.drand();
    const double costh = 1.0 - 2.0 * u1;
    const double sinth = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - costh*costh));
    const double phi   = MY_2PI_LOC * u2;
    fc_kick_x = v_FC * sinth * Kokkos::cos(phi);
    fc_kick_y = v_FC * sinth * Kokkos::sin(phi);
    fc_kick_z = v_FC * costh;
    p.v[0] = vx0 + fc_kick_x;
    p.v[1] = vy0 + fc_kick_y;
    p.v[2] = vz0 + fc_kick_z;
  }

  // second product (dissociation partner): back-to-back kick in the CM
  // frame keeps p1 + p2 = m_parent * v_parent. Slot reserved atomically;
  // capacity guaranteed by the pre-grow in end_of_step.
  if (d_r_nproduct(best_idx) == 2 && d_new_count.data()) {
    const int sp2 = d_r_product1(best_idx);
    if (sp2 >= 0) {
      double xn[3] = {p.x[0], p.x[1], p.x[2]};
      double vn[3];
      if (rtype == DISSOCIATION) {
        vn[0] = vx0 - fc_kick_x;
        vn[1] = vy0 - fc_kick_y;
        vn[2] = vz0 - fc_kick_z;
      } else {
        vn[0] = p.v[0]; vn[1] = p.v[1]; vn[2] = p.v[2];
      }
      const int id2 = (int)(MAXSMALLINT * rand_gen.drand());
      const int index = Kokkos::atomic_fetch_add(&d_new_count(),1);
      const int rf = ParticleKokkos::add_particle_kokkos(
          d_particles,index,id2,sp2,icell,xn,vn,0.0,0.0);
      if (!rf) {
        custom_.zero_all(index);
        // CPU drain parity: modify->update_custom -> fix particle/weight
        // default (update->fnum) on the zeroed pweight
        if (pw_slot_ >= 0) custom_.set_dvec(pw_slot_, index, fnum_);
      }
    }
  }

  rand_pool.free_state(rand_gen);
}
