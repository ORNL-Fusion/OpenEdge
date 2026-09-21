/* ----------------------------------------------------------------------
    OpenEdge sheath helper models (Coulette-Manfredi profile + Chodura metrics)
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
  // MPS tail amplitude, chosen phi-consistent: the tail's potential
  // integral E*L_mps must equal the CM slow potential remaining at the
  // anchor, phi_cm_slow*exp(-K1_scaled*s_start). Using the raw CM field
  // amplitude here (amp_slow*exp(-K1_scaled*s_start)) made phi(d) jump
  // ~x17 at the anchor at grazing incidence (E_anchor*L_mps with
  // L_mps = rho_i*tan(88 deg) >> lambda_D vastly overshoots phi_total),
  // which the sheath Boltzmann ne correction exponentiates into a total
  // ionization dead zone 0.2-2 mm off the target.
  c.e_slow_at_anchor_vpm =
      phi_cm_slow * std::exp(-K1_scaled * s_blend_start) * inv_lmps;
  return c;
}

double sheath_emag_at_distance(const SheathEmagCoeffs &c, double dist_m)
{
  const double d = std::max(dist_m, 0.0);

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
