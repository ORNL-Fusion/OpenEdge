/* ----------------------------------------------------------------------
   OpenEdge: ADAS ionization/recombination chemistry — Kokkos backend.

   Phase-A device port (gate 8). The per-particle Monte Carlo decision
   (rate lookup -> Poisson channel competition -> species swap and CX
   velocity resampling) runs in a device kernel; rare reaction EVENTS
   are appended to a compact device list and applied to the host-side
   tallies (tally_reactions, array_grid) after the kernel — so no
   per-cell device tally arrays and no per-step full-grid D2H traffic.

   SUPPORTED on device (device_ok, checked in init):
     - impurity chains (atomic_number > 1): ADAS SCD/ACD/CCD rate
       lookups + PLT/PRB per-event energy tallies,
     - no volume_source,
     - rate_cache particle mode (the device is per-particle anyway),
     - tally units COUNTS or RATE, both output modes,
     - single-product reactions (ioniz/recomb/CX species swap +
       impurity shifted-Maxwellian CX resampling).
   HOST FALLBACK (base class end_of_step) for everything else:
     hydrogen AMJUEL/HYDHEL channels, dissociation / two-product
     reactions (device particle creation lands with phase A1b),
     batch tallies, volume_source, rate_cache cell.
     A one-time warning names the reason.

   Sampling semantics are IDENTICAL to the CPU path: inputs come from
   the same per-particle pcache custom attributes (pc_te/pc_ne/...),
   already device-resident. RNG uses a rank-offset Kokkos pool, so
   parity with the CPU is STATISTICAL, not bitwise (validated the
   gate-6 way: matched-decomposition A/B at 1 and N ranks).

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(volume/chem/adas/kk,FixVolumeChemAdasKokkos)

#else

#ifndef SPARTA_FIX_VOLUME_CHEM_ADAS_KOKKOS_H
#define SPARTA_FIX_VOLUME_CHEM_ADAS_KOKKOS_H

#include "fix_volume_chem_adas.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include <vector>
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

struct TagFixChemAdas {};

// PLT + PRB power coefficient on the device: twin of
// FixVolumeChemAdas::radiation_coeff (same clamps, same bilinear form).
// Handed to compute grid/weighted/kk for prad_w / pradtot_w.
struct AdasRadTablesKK {
  DAT::t_float_1d plt, prb, gT_plt, gD_plt, gT_prb, gD_prb;
  DAT::t_int_1d rad_q;      // per species: charge row q, -1 = not in the chain
  int plt_nQ = 0, plt_nT = 0, plt_nD = 0;
  int prb_nQ = 0, prb_nT = 0, prb_nD = 0;

  KOKKOS_INLINE_FUNCTION
  double lookup(const DAT::t_float_1d &c, const DAT::t_float_1d &gT,
                const DAT::t_float_1d &gD, int nQ, int nT, int nD,
                int row, double log_te, double log_ne_cm) const {
    if (row < 0 || row >= nQ || nT < 2 || nD < 2) return 0.0;
    const double lt = Kokkos::fmin(Kokkos::fmax(log_te, gT(0)), gT(nT-1));
    const double ln = Kokkos::fmin(Kokkos::fmax(log_ne_cm, gD(0)), gD(nD-1));
    int tlo, thi, nlo, nhi;
    if (lt <= gT(0)) { tlo = 0; thi = 1; }
    else if (lt >= gT(nT-1)) { thi = nT-1; tlo = nT-2; }
    else {
      int lo = 0, hi = nT;
      while (lo < hi) { const int mid = (lo+hi)/2; if (gT(mid) < lt) lo = mid+1; else hi = mid; }
      thi = lo; tlo = lo-1;
    }
    if (ln <= gD(0)) { nlo = 0; nhi = 1; }
    else if (ln >= gD(nD-1)) { nhi = nD-1; nlo = nD-2; }
    else {
      int lo = 0, hi = nD;
      while (lo < hi) { const int mid = (lo+hi)/2; if (gD(mid) < ln) lo = mid+1; else hi = mid; }
      nhi = lo; nlo = lo-1;
    }
    const double x0 = gT(tlo), x1 = gT(thi), y0 = gD(nlo), y1 = gD(nhi);
    const double f00 = c(row*nT*nD + tlo*nD + nlo);
    const double f01 = c(row*nT*nD + tlo*nD + nhi);
    const double f10 = c(row*nT*nD + thi*nD + nlo);
    const double f11 = c(row*nT*nD + thi*nD + nhi);
    const double fx0 = ((x1 - lt) * f00 + (lt - x0) * f10) / (x1 - x0);
    const double fx1 = ((x1 - lt) * f01 + (lt - x0) * f11) / (x1 - x0);
    const double log10_wcm3 = ((y1 - ln) * fx0 + (ln - y0) * fx1) / (y1 - y0);
    return Kokkos::pow(10.0, log10_wcm3) * 1.0e-6;      // W cm^3 -> W m^3
  }

  // [W m^3] of species isp at Te [eV], ne [m^-3]
  KOKKOS_INLINE_FUNCTION
  double coeff(int isp, double te_eV, double ne_m3) const {
    if (isp < 0 || isp >= (int) rad_q.extent(0)) return 0.0;
    const int q = rad_q(isp);
    if (q < 0) return 0.0;
    if (!(te_eV > 0.0) || !(ne_m3 > 0.0)) return 0.0;
    const double lt = Kokkos::log10(te_eV);
    const double ln = Kokkos::log10(ne_m3 * 1.0e-6);
    // PLT row q: line radiation; PRB row q-1: recombination + bremsstrahlung
    return lookup(plt,gT_plt,gD_plt,plt_nQ,plt_nT,plt_nD,q,lt,ln) +
           lookup(prb,gT_prb,gD_prb,prb_nQ,prb_nT,prb_nD,q-1,lt,ln);
  }
};

class FixVolumeChemAdasKokkos : public FixVolumeChemAdas, public KokkosBase {
 public:
#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  FixVolumeChemAdasKokkos(class SPARTA *, int, char **);
  ~FixVolumeChemAdasKokkos();
  void init() override;
  void end_of_step() override;

  // device PLT/PRB tables for compute grid/weighted/kk (built on request,
  // independent of whether the reaction kernel itself runs on the device)
  AdasRadTablesKK rad_tables_kk();

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixChemAdas, const int &i) const;

 private:
  int device_ok;             // 1 = device fast path usable for this config
  int warned_fallback;
  const char *fallback_why_ = nullptr;   // init-time host reason (ledger)

  // ---- static rate tables (uploaded once in init) ----
  // Flat layout matches RateData: coeff[q*nT*nD + it*nD + id], values are
  // log10; grids are log10(Te [eV]) and log10(ne [cm^-3]).
  DAT::t_float_1d d_ion_coeff, d_rec_coeff, d_cx_coeff;
  DAT::t_float_1d d_plt_coeff, d_prb_coeff;
  DAT::t_float_1d d_gT_ion, d_gD_ion, d_gT_rec, d_gD_rec;
  DAT::t_float_1d d_gT_cx,  d_gD_cx,  d_gT_plt, d_gD_plt;
  DAT::t_float_1d d_gT_prb, d_gD_prb;
  DAT::t_float_1d d_ion_pot;
  int ion_nQ_, ion_nT_, ion_nD_;
  int rec_nQ_, rec_nT_, rec_nD_;
  int cx_nQ_,  cx_nT_,  cx_nD_;
  int plt_nQ_, plt_nT_, plt_nD_;
  int prb_nQ_, prb_nT_, prb_nD_;

  // ---- reaction topology (uploaded once in init) ----
  DAT::t_int_1d d_react_offset;   // nspecies+1 CSR offsets
  DAT::t_int_1d d_react_list;     // reaction indices per species
  DAT::t_int_1d d_r_type;         // per reaction: IONIZATION/RECOMBINATION/...
  DAT::t_int_1d d_r_product0;     // first product species index
  // A1b: hydrogen channels + two-product (dissociation) creation
  DAT::t_int_1d d_r_style;        // ADAS / JANEV
  DAT::t_int_1d d_r_nproduct;
  DAT::t_int_1d d_r_product1;     // second product species (-1 = none)
  DAT::t_int_1d d_r_ncoeff;
  DAT::t_float_2d_lr d_r_coeff;   // JANEV ln-polynomial coefficients
  DAT::t_int_1d d_sp_twoprod;     // per species: any active 2-product channel
  int atomic1_;                   // atomic_number == 1 (AMJUEL/HYDHEL fits)
  int host_two_;                  // host twin of have_two_ (fallback pre-grow)
  std::vector<char> h_sp_twoprod_;
  int have_two_;                  // any active two-product reaction

  // ---- per-cell neutral-D density for the CCD CX partner ----
  // decomposition-indexed: rebuilt when the stamps change (fix balance)
  DAT::t_float_1d d_nn_cell;
  int  nn_stamp_n;
  cellint nn_stamp_id;
  int  nn_stamp_gen;              // FixBackground::generation at build
  int  have_cx_;                  // any active EXCHANGE reactions

  // ---- per-call bindings ----
  t_particle_1d d_particles;
  t_species_1d  d_species;
  DAT::t_float_1d d_pc_te, d_pc_ne, d_pc_ti, d_pc_vpar;
  DAT::t_float_1d d_pc_bx, d_pc_by, d_pc_bz;
  DAT::t_float_1d d_pweight;
  int have_ti_, have_vpar_, have_b_, have_pweight_;
  // newborn creation (dissociation second product; PWI device idiom)
  ParticleKokkos::DeviceCustom custom_;
  int gca_valid_slot_;              // edvec slot of gca_valid (-1 = no hybrid pusher)
  int pw_slot_;
  Kokkos::View<int, DeviceType> d_new_count;
  int nglocal_;
  double dt_chem_, fnum_;
  int atomic_number_, tally_units_, output_mode_;

  // ---- compact per-event output ----
  // one slot per particle is a hard upper bound (<= 1 event/particle)
  DAT::t_int_1d   d_ev_ridx;      // chosen reaction index
  DAT::t_int_1d   d_ev_cell;      // event cell
  DAT::t_float_2d_lr d_ev_vals;   // 6 tally values (layout per output_mode)
  Kokkos::View<int, DeviceType> d_ev_count;
  // perf: host-side readback buffers (persistent; see end_of_step)
  Kokkos::View<int[2], Kokkos::HostSpace> h_cnt_;
  Kokkos::View<int*, Kokkos::HostSpace> h_ev_ridx_, h_ev_cell_;
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> h_ev_vals_;

  void upload_static_tables();
  void build_nn_cell();

  // device helpers
  KOKKOS_INLINE_FUNCTION
  bool bracket(const DAT::t_float_1d &g, int n, double x,
               int &ilo, int &ihi) const {
    if (n < 2) return false;
    if (x <= g(0))   { ilo = 0; ihi = 1; return true; }
    if (x >= g(n-1)) { ihi = n-1; ilo = n-2; return true; }
    // binary search: first index with g(idx) >= x  (std::lower_bound twin)
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo+hi)/2; if (g(mid) < x) lo = mid+1; else hi = mid; }
    if (lo == 0 || lo >= n) return false;
    ihi = lo; ilo = lo-1;
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  double bilinear(double x0, double x1, double y0, double y1,
                  double f00, double f01, double f10, double f11,
                  double x, double y) const {
    // exact port of MathExtra::bilinearInterpolate
    const double fx0 = ((x1 - x) * f00 + (x - x0) * f10) / (x1 - x0);
    const double fx1 = ((x1 - x) * f01 + (x - x0) * f11) / (x1 - x0);
    return ((y1 - y) * fx0 + (y - y0) * fx1) / (y1 - y0);
  }

  // rate lookup: log10(rate) at (logTe, logne_cm) for charge row q.
  // Returns 0.0 on a miss, matching interpolateRateData's rate_final=0
  // convention (the caller's isfinite() check then passes and lambda
  // becomes k=1 cm^3/s — SAME as CPU, bug-for-bug).
  KOKKOS_INLINE_FUNCTION
  double interp_class(const DAT::t_float_1d &coeff,
                      const DAT::t_float_1d &gT, const DAT::t_float_1d &gD,
                      int nQ, int nT, int nD, int q,
                      double logTe, double logne_cm) const {
    if (q < 0 || q >= nQ) return 0.0;
    int tlo, thi, nlo, nhi;
    if (!bracket(gT, nT, logTe, tlo, thi)) return 0.0;
    if (!bracket(gD, nD, logne_cm, nlo, nhi)) return 0.0;
    const double f00 = coeff(q*nT*nD + tlo*nD + nlo);
    const double f01 = coeff(q*nT*nD + tlo*nD + nhi);
    const double f10 = coeff(q*nT*nD + thi*nD + nlo);
    const double f11 = coeff(q*nT*nD + thi*nD + nhi);
    return bilinear(gT(tlo), gT(thi), gD(nlo), gD(nhi),
                    f00, f01, f10, f11, logTe, logne_cm);
  }

  KOKKOS_INLINE_FUNCTION
  double lambda_from(double rate_log10_cm3s, double dt, double dens) const {
    // exact port of computeReactionLambda
    if (!(dt > 0.0) || !(dens > 0.0) || !Kokkos::isfinite(rate_log10_cm3s))
      return 0.0;
    const double k_cm3s = Kokkos::exp(rate_log10_cm3s * 2.302585092994046);
    const double k_m3s  = Kokkos::fmax(0.0, k_cm3s) * 1e-6;
    double lam = k_m3s * dens * dt;
    if (!Kokkos::isfinite(lam)) return 50.0;
    return Kokkos::fmin(lam, 50.0);
  }
};

}  // namespace SPARTA_NS

#endif
#endif
