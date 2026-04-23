/* ----------------------------------------------------------------------
   OpenEdge: ADAS ionization/recombination — Kokkos device kernel.
   Rate table lookups, Poisson draws, and species changes all on device.
------------------------------------------------------------------------- */

#include "fix_volume_chem_adas_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "compute.h"
#include "modify.h"
#include "update.h"
#include "comm.h"
#include "kokkos_base.h"
#include "sparta_masks.h"
#include "memory.h"

using namespace SPARTA_NS;

enum{IONIZATION,RECOMBINATION,EXCHANGE};

/* ---------------------------------------------------------------------- */

static Kokkos::View<double*, DeviceType>
vec_to_dev(const std::vector<double> &v, const std::string &label)
{
  size_t n = v.size();
  Kokkos::View<double*, DeviceType> d(Kokkos::view_alloc(label), n);
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < n; i++) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

/* ---------------------------------------------------------------------- */

FixVolumeChemAdasKokkos::FixVolumeChemAdasKokkos(SPARTA *sparta, int narg, char **arg) :
  FixVolumeChemAdas(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK | CINFO_MASK;
  datamask_modify = PARTICLE_MASK;
}

FixVolumeChemAdasKokkos::~FixVolumeChemAdasKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

int FixVolumeChemAdasKokkos::resolve_column(const GridSrc &S)
{
  if (S.kind != SRC_COMP) return -1;
  return S.col - 1;
}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::init()
{
  FixVolumeChemAdas::init();

  // copy rate tables to device
  RateData &rd = materials_rate_data[atomic_number];

  d_ion_coeff  = vec_to_dev(rd.ion_coeff, "adas:ion_coeff");
  d_rec_coeff  = vec_to_dev(rd.rec_coeff, "adas:rec_coeff");
  d_gridT_ion  = vec_to_dev(rd.gridT_ion, "adas:gridT_ion");
  d_gridD_ion  = vec_to_dev(rd.gridD_ion, "adas:gridD_ion");
  d_gridT_rec  = vec_to_dev(rd.gridT_rec, "adas:gridT_rec");
  d_gridD_rec  = vec_to_dev(rd.gridD_rec, "adas:gridD_rec");

  kk_ion_nQ = rd.ion_nQ; kk_ion_nT = rd.ion_nT; kk_ion_nD = rd.ion_nD;
  kk_rec_nQ = rd.rec_nQ; kk_rec_nT = rd.rec_nT; kk_rec_nD = rd.rec_nD;

  d_cx_coeff   = vec_to_dev(rd.cx_coeff, "adas:cx_coeff");
  d_gridT_cx   = vec_to_dev(rd.gridT_cx, "adas:gridT_cx");
  d_gridD_cx   = vec_to_dev(rd.gridD_cx, "adas:gridD_cx");
  kk_cx_nQ = rd.cx_nQ; kk_cx_nT = rd.cx_nT; kk_cx_nD = rd.cx_nD;

  // build per-species reaction lookup for device
  int nspecies = particle->nspecies;
  kk_nlist = nlist;
  kk_atomic_number = atomic_number;

  // count total reactions and build offset/count arrays
  auto h_offset = Kokkos::View<int*, Kokkos::HostSpace>(
    Kokkos::view_alloc("h_offset"), nspecies);
  auto h_count = Kokkos::View<int*, Kokkos::HostSpace>(
    Kokkos::view_alloc("h_count"), nspecies);

  int total = 0;
  for (int i = 0; i < nspecies; i++) {
    h_offset(i) = total;
    h_count(i) = reactions[i].n;
    total += reactions[i].n;
  }

  auto h_list = Kokkos::View<int*, Kokkos::HostSpace>(
    Kokkos::view_alloc("h_list"), total > 0 ? total : 1);
  int k = 0;
  for (int i = 0; i < nspecies; i++)
    for (int j = 0; j < reactions[i].n; j++)
      h_list(k++) = reactions[i].list[j];

  d_rxn_offset = t_int_1d_kk(Kokkos::view_alloc("adas:rxn_offset"), nspecies);
  d_rxn_count  = t_int_1d_kk(Kokkos::view_alloc("adas:rxn_count"), nspecies);
  d_rxn_list   = t_int_1d_kk(Kokkos::view_alloc("adas:rxn_list"), total > 0 ? total : 1);
  Kokkos::deep_copy(d_rxn_offset, h_offset);
  Kokkos::deep_copy(d_rxn_count, h_count);
  Kokkos::deep_copy(d_rxn_list, h_list);

  // per-reaction type and product arrays
  auto h_type = Kokkos::View<int*, Kokkos::HostSpace>(
    Kokkos::view_alloc("h_type"), nlist > 0 ? nlist : 1);
  auto h_prod = Kokkos::View<int*, Kokkos::HostSpace>(
    Kokkos::view_alloc("h_prod"), nlist > 0 ? nlist : 1);
  for (int i = 0; i < nlist; i++) {
    h_type(i) = rlist[i].type;
    h_prod(i) = rlist[i].products[0];
  }
  d_rxn_type     = t_int_1d_kk(Kokkos::view_alloc("adas:rxn_type"), nlist > 0 ? nlist : 1);
  d_rxn_product0 = t_int_1d_kk(Kokkos::view_alloc("adas:rxn_product0"), nlist > 0 ? nlist : 1);
  Kokkos::deep_copy(d_rxn_type, h_type);
  Kokkos::deep_copy(d_rxn_product0, h_prod);

  // device tallies
  d_tally_reactions = t_int_1d_kk(Kokkos::view_alloc("adas:tally"), nlist > 0 ? nlist : 1);
  Kokkos::deep_copy(d_tally_reactions, 0);
  d_nreact_one = DAT::t_int_scalar("adas:nreact_one");

  // resolve compute columns
  col_Te = resolve_column(srcTe);
  col_Ne = resolve_column(srcNe);

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(rng_adas);
#endif
}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdasKokkos::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;

  // refresh compute on host
  refresh_compute_src(srcTe);
  refresh_compute_src(srcNe);

  if (grid->nlocal == 0 || particle->nlocal == 0) return;

  // get plasma compute device view
  if (srcTe.kind != SRC_COMP || srcTe.icompute < 0) {
    // no compute source — fall back to host
    particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);
    nreact_one = 0;
    if (!particle->sorted) particle->sort();
    end_of_step_no_average();
    nreact_running += nreact_one;
    particle_kk->modify(Host, PARTICLE_MASK);
    return;
  }

  Compute *cp = modify->compute[srcTe.icompute];
  if (!(cp->invoked_flag & 16)) {  // INVOKED_PER_GRID = 16
    cp->compute_per_grid();
    cp->invoked_flag |= 16;
  }
  KokkosBase *kk_cp = dynamic_cast<KokkosBase*>(cp);
  if (!kk_cp) {
    // fallback to host
    particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);
    nreact_one = 0;
    if (!particle->sorted) particle->sort();
    end_of_step_no_average();
    nreact_running += nreact_one;
    particle_kk->modify(Host, PARTICLE_MASK);
    return;
  }

  // sync to device
  particle_kk->sync(Device, PARTICLE_MASK | SPECIES_MASK);

  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  d_plasma = kk_cp->d_array_grid;
  kk_nlocal = particle->nlocal;

  kk_dt_chem = nevery * update->dt;

  // reset tallies
  Kokkos::deep_copy(d_nreact_one, 0);
  Kokkos::deep_copy(d_tally_reactions, 0);

  // launch per-particle kernel
  copymode = 1;
  Kokkos::parallel_for(kk_nlocal, *this);
  copymode = 0;

  // copy tallies back to host
  auto h_nreact = Kokkos::create_mirror_view(d_nreact_one);
  Kokkos::deep_copy(h_nreact, d_nreact_one);
  nreact_one = h_nreact();
  nreact_running += nreact_one;

  auto h_tally = Kokkos::create_mirror_view(d_tally_reactions);
  Kokkos::deep_copy(h_tally, d_tally_reactions);
  for (int i = 0; i < kk_nlist; i++)
    tally_reactions[i] += h_tally(i);

  particle_kk->modify(Device, PARTICLE_MASK);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixVolumeChemAdasKokkos::operator()(int i) const
{
  int icell = d_particles(i).icell;

  double Te_eV = d_plasma(icell, col_Te);
  double ne_m3 = d_plasma(icell, col_Ne);
  if (Te_eV < 1e-6) Te_eV = 1e-6;
  if (ne_m3 < 0.0) ne_m3 = 0.0;

  rand_type rng = rand_pool.get_state();
  kk_attempt(i, Te_eV, ne_m3, rng);
  rand_pool.free_state(rng);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int FixVolumeChemAdasKokkos::kk_attempt(int idx, double Te_eV, double ne_m3,
                                   rand_type &rng) const
{
  int isp = d_particles(idx).ispecies;
  int nrxn = d_rxn_count(isp);
  if (nrxn == 0) return 0;
  if (Te_eV <= 0.0 || ne_m3 <= 0.0) return 0;

  double charge = d_species[isp].charge;
  int q = static_cast<int>(charge > 0.0 ? charge : 0.0);

  double logTe = log10(Te_eV);
  double logne_cm = log10(ne_m3 * 1e-6 > 1e-99 ? ne_m3 * 1e-6 : 1e-99);

  int offset = d_rxn_offset(isp);

  // compute per-channel Poisson rates
  double lambda[16];
  int ridx_map[16];
  int nchan = 0;
  double lambda_total = 0.0;

  for (int i = 0; i < nrxn && nchan < 16; i++) {
    int ridx = d_rxn_list(offset + i);
    int rtype = d_rxn_type(ridx);

    double rate_log10 = -1e30;  // -infinity substitute

    if (rtype == IONIZATION) {
      if (q >= kk_atomic_number) continue;
      rate_log10 = kk_interp_rate(0, q, logTe, logne_cm);
    } else if (rtype == RECOMBINATION) {
      if (q == 0) continue;
      rate_log10 = kk_interp_rate(1, q - 1, logTe, logne_cm);
    } else if (rtype == EXCHANGE) {
      if (q == 0) continue;
      rate_log10 = kk_interp_rate(2, q - 1, logTe, logne_cm);
    } else {
      continue;
    }

    if (rate_log10 < -30.0) continue;

    // lambda = k * ne * dt
    double k_cm3s = exp(rate_log10 * 2.302585092994046);
    double k_m3s = (k_cm3s > 0.0 ? k_cm3s : 0.0) * 1e-6;
    double lam = k_m3s * ne_m3 * kk_dt_chem;
    if (lam <= 0.0) continue;
    if (lam > 50.0) lam = 50.0;

    lambda[nchan] = lam;
    ridx_map[nchan] = ridx;
    lambda_total += lam;
    nchan++;
  }

  if (nchan == 0 || lambda_total <= 0.0) return 0;

  // P(any) = 1 - exp(-lambda_total)
  double P_any = 1.0 - exp(-lambda_total);
  double u = rng.drand();
  if (u > P_any) return 0;

  // select channel
  int chosen = 0;
  if (nchan > 1) {
    double v = rng.drand() * lambda_total;
    double cumsum = 0.0;
    for (int i = 0; i < nchan; i++) {
      cumsum += lambda[i];
      if (v <= cumsum) { chosen = i; break; }
    }
  }

  int best_idx = ridx_map[chosen];
  Kokkos::atomic_inc(&d_tally_reactions(best_idx));
  Kokkos::atomic_inc(&d_nreact_one());
  d_particles(idx).ispecies = d_rxn_product0(best_idx);
  return 1;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double FixVolumeChemAdasKokkos::kk_interp_rate(int rate_type, int charge_idx,
                                          double logTe, double logne_cm) const
{
  // rate_type: 0=ionization, 1=recombination, 2=charge exchange
  const t_double_1d &gridT = (rate_type == 2) ? d_gridT_cx : (rate_type == 1 ? d_gridT_rec : d_gridT_ion);
  const t_double_1d &gridD = (rate_type == 2) ? d_gridD_cx : (rate_type == 1 ? d_gridD_rec : d_gridD_ion);
  const t_double_1d &coeff = (rate_type == 2) ? d_cx_coeff : (rate_type == 1 ? d_rec_coeff : d_ion_coeff);
  int nT = (rate_type == 2) ? kk_cx_nT : (rate_type == 1 ? kk_rec_nT : kk_ion_nT);
  int nD = (rate_type == 2) ? kk_cx_nD : (rate_type == 1 ? kk_rec_nD : kk_ion_nD);
  int nQ = (rate_type == 2) ? kk_cx_nQ : (rate_type == 1 ? kk_rec_nQ : kk_ion_nQ);

  if (charge_idx < 0 || charge_idx >= nQ) return -1e30;

  int tlo, thi, nlo, nhi;
  if (!kk_bracket(gridT, nT, logTe, tlo, thi)) return -1e30;
  if (!kk_bracket(gridD, nD, logne_cm, nlo, nhi)) return -1e30;

  double x0 = gridT(tlo), x1 = gridT(thi);
  double y0 = gridD(nlo), y1 = gridD(nhi);

  // flat layout: coeff[q * nT * nD + iT * nD + iD]
  int base = charge_idx * nT * nD;
  double f00 = coeff(base + tlo * nD + nlo);
  double f01 = coeff(base + tlo * nD + nhi);
  double f10 = coeff(base + thi * nD + nlo);
  double f11 = coeff(base + thi * nD + nhi);

  return kk_bilinear(x0, x1, y0, y1, f00, f01, f10, f11, logTe, logne_cm);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double FixVolumeChemAdasKokkos::kk_bilinear(
  double x0, double x1, double y0, double y1,
  double f00, double f01, double f10, double f11,
  double x, double y) const
{
  double dx = x1 - x0;
  double dy = y1 - y0;
  if (dx == 0.0 || dy == 0.0) return f00;
  double fx0 = ((x1 - x) * f00 + (x - x0) * f10) / dx;
  double fx1 = ((x1 - x) * f01 + (x - x0) * f11) / dx;
  return ((y1 - y) * fx0 + (y - y0) * fx1) / dy;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int FixVolumeChemAdasKokkos::kk_bracket(const t_double_1d &grid, int n, double x,
                                   int &ilo, int &ihi) const
{
  if (n < 2) return 0;
  if (x <= grid(0))   { ilo = 0; ihi = 1; return 1; }
  if (x >= grid(n-1)) { ihi = n-1; ilo = ihi-1; return 1; }

  // binary search
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (grid(mid) < x) lo = mid + 1;
    else hi = mid;
  }
  if (hi == 0 || hi >= n) return 0;
  ihi = hi;
  ilo = hi - 1;
  return 1;
}
