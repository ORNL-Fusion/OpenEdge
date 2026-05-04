/* ----------------------------------------------------------------------
    OpenEdge sheath helper models (EIRENE-style + Chodura metrics)
------------------------------------------------------------------------- */

#include "sheath_models.h"

#include <algorithm>
#include <cmath>

namespace SPARTA_NS {
namespace SheathModels {

namespace {
constexpr double QE = 1.602176634e-19;
constexpr double ME = 9.1093837015e-31;
constexpr double AMU = 1.66053906660e-27;
constexpr double EPS0 = 8.8541878128e-12;
constexpr double PI = 3.14159265358979323846;

double fd_poly_deg(const double a)
{
  const double fd =
    0.98992 + 5.1220e-3 * a - 7.0040e-4 * a * a +
    3.3591e-5 * a * a * a - 8.2917e-7 * std::pow(a,4) +
    9.5856e-9 * std::pow(a,5) - 4.2682e-11 * std::pow(a,6);
  return std::max(0.0, std::min(1.0, fd));
}
}

// 1D effective thermal speed (not Bohm cs) — see header comment.
double vth_d_eff(double te_eV, double ti_eV, double mD_amu)
{
  const double m = std::max(mD_amu * AMU, 1.0e-99);
  return std::sqrt(std::max(te_eV + ti_eV, 0.0) * QE / (2.0 * m));
}

EireneSheathResult eirene_sheath_ev(double te_eV,
                                    const std::vector<double> &dens_m3,
                                    const std::vector<double> &upar_ms,
                                    const std::vector<int> &charge_state_z,
                                    double gamma_see,
                                    double cur_A_m2)
{
  EireneSheathResult out;
  const int n = static_cast<int>(dens_m3.size());
  if (n <= 0 || static_cast<int>(upar_ms.size()) != n ||
      static_cast<int>(charge_state_z.size()) != n || te_eV <= 0.0) {
    out.esheath_eV = 2.8 * std::max(te_eV, 0.0);
    out.fallback = 1;
    return out;
  }

  double de = 0.0;
  for (int i = 0; i < n; ++i) de += std::max(dens_m3[i], 0.0) * std::max(charge_state_z[i], 0);
  de = std::max(de, 1.0e-60);

  double ueff = 0.0;
  for (int i = 0; i < n; ++i) {
    const double zi = static_cast<double>(std::max(charge_state_z[i], 0));
    const double ni = std::max(dens_m3[i], 0.0);
    const double wi = zi * ni / de;
    ueff += wi * std::abs(upar_ms[i]);
  }

  const double ce = std::sqrt(std::max(te_eV, 1.0e-12) * QE / ME);
  const double fac = std::sqrt(2.0 * PI) / std::max(1.0 - gamma_see, 1.0e-12);
  out.arg = fac * (ueff / ce - cur_A_m2 / (QE * de * ce));

  if (out.arg > 0.0) {
    const double lg = std::log(out.arg);
    if (lg < 0.0) {
      out.esheath_eV = -te_eV * lg;
      out.fallback = 0;
      return out;
    }
  }

  out.esheath_eV = 2.8 * te_eV;
  out.fallback = 1;
  return out;
}

ChoduraMetrics chodura_metrics(double upar_ms,
                               double cs_ms,
                               const double b[3],
                               const double n[3])
{
  ChoduraMetrics m;
  const double bmag = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  const double nmag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (bmag <= 0.0 || nmag <= 0.0 || cs_ms <= 0.0) return m;

  const double bdotn = (b[0]*n[0] + b[1]*n[1] + b[2]*n[2]) / (bmag * nmag);
  const double absbn = std::max(0.0, std::min(1.0, std::abs(bdotn)));
  m.bdotn = bdotn;
  m.alpha_deg = std::acos(absbn) * 180.0 / PI;
  m.mach_par = std::abs(upar_ms) / cs_ms;
  m.u_n = std::abs(upar_ms) * absbn;
  m.mach_n = m.u_n / cs_ms;
  return m;
}

BorodkinaSheathResult borodkina_sheath_at_distance(double dist_m,
                                                   double te_eV,
                                                   double ti_eV,
                                                   double ne_m3,
                                                   double bmag_T,
                                                   double alpha_deg,
                                                   double mD_amu,
                                                   double pot_mult)
{
  // Thin wrapper: prepare coefficients and evaluate at `dist_m`.
  // For hot loops (Boris subcycles) call sheath_prepare_borodkina()
  // once outside the loop and sheath_emag_at_distance() inside.
  const SheathEmagCoeffs c = sheath_prepare_borodkina(te_eV, ti_eV, ne_m3,
                                                       bmag_T, alpha_deg,
                                                       mD_amu, pot_mult);
  BorodkinaSheathResult out;
  out.lambdaD_m  = c.lambdaD_m;
  out.lmps_m     = c.lmps_m;
  out.rho_i_m    = c.rho_i_m;
  out.fd         = c.fd;
  out.phi_ds_eV  = c.phi0_ds_eV;   // fd * phi0 at sheath entrance
  out.phi_cs_eV  = c.phi0_mps_eV;  // (1-fd) * phi0 at sheath entrance
  out.emag_vpm   = sheath_emag_at_distance(c, dist_m);
  out.esheath_eV = sheath_phi_at_distance(c, dist_m);
  return out;
}

/* ----------------------------------------------------------------------
   Coulette-Manfredi sheath model (kinetic PIC data fit)

   Two-exponential fit to Vlasov simulation data from:
     Coulette & Manfredi, PPCF 58 025008 (2016)
   Covers full CS+DS transition for α ∈ [2°,90°].

   φ/Te = -[ B1(α) exp(-k1(α)*s) + B2(α) exp(-k2(α)*s) ]
   where s = d/λ_D, and B1,B2,k1,k2 = exp(linear in α_deg).

   For d > cutoff_lambdaD, adds a BK-style MPS tail to capture the
   Chodura layer at large ρ_i/λ_D ratios (fusion-relevant parameters).
------------------------------------------------------------------------- */

BorodkinaSheathResult coulette_manfredi_sheath_at_distance(
    double dist_m, double te_eV, double ti_eV, double ne_m3,
    double bmag_T, double alpha_deg, double mD_amu, double pot_mult)
{
  // Thin wrapper over the fast-evaluation API.
  const SheathEmagCoeffs c = sheath_prepare_coulette_manfredi(
      te_eV, ti_eV, ne_m3, bmag_T, alpha_deg, mD_amu, pot_mult);
  BorodkinaSheathResult out;
  out.lambdaD_m  = c.lambdaD_m;
  out.lmps_m     = c.lmps_m;
  out.rho_i_m    = c.rho_i_m;
  out.fd         = c.fd;
  out.phi_cs_eV  = c.phi_cm_slow_eV;   // slow -> CS
  out.phi_ds_eV  = c.phi_cm_fast_eV;   // fast -> DS
  out.emag_vpm   = sheath_emag_at_distance(c, dist_m);
  out.esheath_eV = sheath_phi_at_distance(c, dist_m);
  return out;
}

/* ----------------------------------------------------------------------
   Fast-evaluation API: prepare once per Boris call, evaluate per subcycle.

   Every field of SheathEmagCoeffs is a function of Te, Ti, ne, B, alpha,
   mD, pot_mult only, so they can be hoisted out of the subcycle loop.
   The per-subcycle call (sheath_emag_at_distance / sheath_phi_at_distance)
   reduces to 2-4 exp() and a handful of multiplies.

   Behavior matches borodkina_sheath_at_distance and
   coulette_manfredi_sheath_at_distance exactly; those functions now
   delegate here.
------------------------------------------------------------------------- */

SheathEmagCoeffs sheath_prepare_borodkina(double te_eV, double ti_eV,
                                          double ne_m3, double bmag_T,
                                          double alpha_deg, double mD_amu,
                                          double pot_mult)
{
  SheathEmagCoeffs c;
  c.kind = SHEATH_BORODKINA;

  const double te = std::max(te_eV, 1.0e-12);
  const double ti = std::max(ti_eV, 0.0);
  const double ne = std::max(ne_m3, 1.0e-60);
  const double bmag = std::max(std::abs(bmag_T), 1.0e-20);
  const double mD = std::max(mD_amu * AMU, 1.0e-99);

  const double lambdaD = std::sqrt(EPS0 * te / (ne * QE));
  // vth_d: 1D effective thermal speed for rho_i (not Bohm cs).
  const double vth_d = std::sqrt((te + ti) * QE / (2.0 * mD));
  const double omega_ci = QE * bmag / mD;
  const double rho_i = vth_d / std::max(std::abs(omega_ci), 1.0e-99);

  const double alpha = std::max(0.0, std::min(90.0, alpha_deg));
  const double alpha_n_rad = alpha * PI / 180.0;
  constexpr double tan_ratio_max = 30.0;
  constexpr double tan_ratio_min = 1.0e-3;
  const double tan_a = std::min(std::abs(std::tan(alpha_n_rad)), tan_ratio_max);
  const double lmps = rho_i * std::max(tan_a, tan_ratio_min);

  const double fd = fd_poly_deg(alpha);
  const double phi0 = std::max(pot_mult, 0.0) * te;

  const double inv_2lD = 1.0 / std::max(2.0 * lambdaD, 1.0e-99);
  const double inv_lmps = 1.0 / std::max(lmps, 1.0e-99);

  c.lambdaD_m = lambdaD;
  c.rho_i_m   = rho_i;
  c.lmps_m    = lmps;
  c.fd        = fd;
  c.phi_total_eV = phi0;
  c.phi0_ds_eV   = phi0 * fd;
  c.phi0_mps_eV  = phi0 * (1.0 - fd);
  c.inv_2lD      = inv_2lD;
  c.inv_lmps     = inv_lmps;
  c.amp_ds_vpm   = c.phi0_ds_eV  * inv_2lD;
  c.amp_mps_vpm  = c.phi0_mps_eV * inv_lmps;
  return c;
}

SheathEmagCoeffs sheath_prepare_coulette_manfredi(double te_eV, double ti_eV,
                                                  double ne_m3, double bmag_T,
                                                  double alpha_deg, double mD_amu,
                                                  double pot_mult)
{
  SheathEmagCoeffs c;
  c.kind = SHEATH_COULETTE_MANFREDI;

  const double te = std::max(te_eV, 1.0e-12);
  const double ti = std::max(ti_eV, 0.0);
  const double ne = std::max(ne_m3, 1.0e-60);
  const double bmag = std::max(std::abs(bmag_T), 1.0e-20);
  const double mD = std::max(mD_amu * AMU, 1.0e-99);

  const double lambdaD = std::sqrt(EPS0 * te / (ne * QE));
  // vth_d: 1D effective thermal speed for rho_i (not Bohm cs).
  const double vth_d = std::sqrt((te + ti) * QE / (2.0 * mD));
  const double omega_ci = QE * bmag / mD;
  const double rho_i = vth_d / std::max(std::abs(omega_ci), 1.0e-99);

  const double alpha_n  = std::max(0.0, std::min(90.0, alpha_deg));
  const double alpha    = 90.0 - alpha_n;   // CM fit uses B-wall angle.

  constexpr double b10 =  0.788600127141, b11 = -0.0140352947024;
  constexpr double b20 = -0.511290440114, b21 =  0.0149209038566;
  constexpr double p10 = -3.57918079952,  p11 =  0.0414523939029;
  constexpr double p20 = -0.705851316996, p21 =  0.000903186346144;

  const double B1 = std::exp(b10 + b11 * alpha);
  const double B2 = std::exp(b20 + b21 * alpha);
  const double K1 = std::exp(p10 + p11 * alpha);
  const double K2 = std::exp(p20 + p21 * alpha);

  const double rho_over_lD = rho_i / std::max(lambdaD, 1.0e-99);
  constexpr double rho_over_lD_ref = 20.0;
  const double K1_scaled = K1 * rho_over_lD_ref / std::max(rho_over_lD, 1.0);

  const double phi_float_mult =
      0.5 * std::log((mD / std::max(2.0 * PI * ME, 1.0e-99)) / (1.0 + ti / te));
  const double phi_total =
      (pot_mult > 0.0) ? (pot_mult * te) : (std::max(phi_float_mult, 0.0) * te);
  const double phi_cm_wall_Te = B1 + B2;
  const double scale = phi_total / std::max(phi_cm_wall_Te * te, 1.0e-99);
  const double phi_cm_slow = scale * B1 * te;
  const double phi_cm_fast = scale * B2 * te;

  constexpr double tan_ratio_max = 30.0;
  constexpr double tan_ratio_min = 1.0e-3;
  const double alpha_n_rad = alpha_n * PI / 180.0;
  const double tan_an = std::min(std::max(std::abs(std::tan(alpha_n_rad)),
                                           tan_ratio_min), tan_ratio_max);
  const double lmps_phys = rho_i * tan_an;

  constexpr double s_blend_start = 60.0;
  constexpr double s_blend_end   = 120.0;

  const double inv_lD   = 1.0 / std::max(lambdaD, 1.0e-99);
  const double inv_lmps = 1.0 / std::max(lmps_phys, 1.0e-99);

  c.lambdaD_m        = lambdaD;
  c.rho_i_m          = rho_i;
  c.lmps_m           = lmps_phys;
  c.phi_total_eV     = phi_total;
  c.fd               = (phi_total > 0.0) ? (phi_cm_fast / phi_total) : 0.0;
  c.phi_cm_slow_eV   = phi_cm_slow;
  c.phi_cm_fast_eV   = phi_cm_fast;
  c.inv_lD           = inv_lD;
  c.inv_lmps         = inv_lmps;
  c.K1_scaled        = K1_scaled;
  c.K2               = K2;
  c.amp_slow_vpm     = phi_cm_slow * K1_scaled * inv_lD;
  c.amp_fast_vpm     = phi_cm_fast * K2        * inv_lD;
  c.s_blend_start    = s_blend_start;
  c.s_blend_end      = s_blend_end;
  c.s_blend_width_inv = 1.0 / (s_blend_end - s_blend_start);
  c.e_slow_at_anchor_vpm =
      c.amp_slow_vpm * std::exp(-K1_scaled * s_blend_start);
  return c;
}

double sheath_emag_at_distance(const SheathEmagCoeffs &c, double dist_m)
{
  const double d = std::max(dist_m, 0.0);

  if (c.kind == SHEATH_BORODKINA) {
    const double e_ds  = c.amp_ds_vpm  * std::exp(-d * c.inv_2lD);
    const double e_mps = c.amp_mps_vpm * std::exp(-d * c.inv_lmps);
    return std::abs(e_ds + e_mps);
  }

  // Coulette-Manfredi with BK Chodura tail for s > s_blend_start.
  const double s = d * c.inv_lD;
  const double e_slow_cm = c.amp_slow_vpm * std::exp(-c.K1_scaled * s);
  const double e_fast    = c.amp_fast_vpm * std::exp(-c.K2        * s);

  if (s <= c.s_blend_start) {
    return std::abs(e_slow_cm + e_fast);
  }

  const double d_past_anchor = (s - c.s_blend_start) * c.lambdaD_m;
  const double e_slow_mps =
      c.e_slow_at_anchor_vpm * std::exp(-d_past_anchor * c.inv_lmps);
  const double blend = (s >= c.s_blend_end) ? 1.0
                      : (s - c.s_blend_start) * c.s_blend_width_inv;
  const double e_slow = (1.0 - blend) * e_slow_cm + blend * e_slow_mps;
  return std::abs(e_slow + e_fast);
}

double sheath_phi_at_distance(const SheathEmagCoeffs &c, double dist_m)
{
  const double d = std::max(dist_m, 0.0);

  if (c.kind == SHEATH_BORODKINA) {
    // phi(d) = phi0 * [fd·exp(-d·inv_2lD) + (1-fd)·exp(-d·inv_lmps)]
    // with phi_total_eV = phi0. Returned as positive potential drop.
    const double phi_ds  = c.phi0_ds_eV  * std::exp(-d * c.inv_2lD);
    const double phi_mps = c.phi0_mps_eV * std::exp(-d * c.inv_lmps);
    return phi_ds + phi_mps;
  }

  const double s = d * c.inv_lD;
  const double phi_slow_cm = c.phi_cm_slow_eV * std::exp(-c.K1_scaled * s);
  const double phi_fast    = c.phi_cm_fast_eV * std::exp(-c.K2        * s);

  if (s <= c.s_blend_start) {
    return phi_slow_cm + phi_fast;
  }

  const double d_past_anchor = (s - c.s_blend_start) * c.lambdaD_m;
  const double phi_slow_mps = c.e_slow_at_anchor_vpm * c.lmps_m *
                              std::exp(-d_past_anchor * c.inv_lmps);
  const double blend = (s >= c.s_blend_end) ? 1.0
                      : (s - c.s_blend_start) * c.s_blend_width_inv;
  const double phi_slow = (1.0 - blend) * phi_slow_cm + blend * phi_slow_mps;
  return phi_slow + phi_fast;
}

}  // namespace SheathModels
}  // namespace SPARTA_NS
