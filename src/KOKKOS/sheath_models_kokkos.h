/* ----------------------------------------------------------------------
   OpenEdge sheath helper models — Kokkos device-callable versions.

   Device twins of the CPU fast-evaluation sheath API in
   sheath_models.cpp: prepare_coulette_manfredi (coefficients hoisted out
   of the subcycle loop), phi_at_distance (the per-subcycle potential,
   incl. the blended Borodkina-style Chodura tail for s >= 60), and
   auto_dmax (the engagement cut-off). Each is a line-for-line port of
   its CPU counterpart — sheath_prepare_coulette_manfredi,
   sheath_phi_at_distance (CM branch) and sheath_auto_dmax — so the
   Kokkos mover reproduces the CPU spatial-mode sheath exactly.
   The Borodkina kind and eirene_sheath_ev stay host-only (the pusher's
   spatial cache always prepares Coulette-Manfredi).

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_MODELS_KOKKOS_H
#define OPENEDGE_SHEATH_MODELS_KOKKOS_H

#include "Kokkos_Core.hpp"
#include <cmath>

namespace SPARTA_NS {
namespace SheathModelsKokkos {

// Physical constants (same as sheath_models.cpp)
KOKKOS_INLINE_FUNCTION constexpr double QE()   { return 1.602176634e-19; }
KOKKOS_INLINE_FUNCTION constexpr double ME()   { return 9.1093837015e-31; }
KOKKOS_INLINE_FUNCTION constexpr double AMU()  { return 1.66053906660e-27; }
KOKKOS_INLINE_FUNCTION constexpr double EPS0() { return 8.8541878128e-12; }
KOKKOS_INLINE_FUNCTION constexpr double PI()   { return 3.14159265358979323846; }

struct ChoduraMetrics {
  double bdotn;
  double alpha_deg;
  double mach_par;
  double mach_n;
  double u_n;
};

// Coulette-Manfredi coefficient bundle: device twin of the CM fields of
// SheathModels::SheathEmagCoeffs (sheath_models.h). One bundle per wall
// element (fix-background cache) or per particle (compute-provider path).
struct CMCoeffs {
  double phi_total_eV;   // phi(0) = phi_slow + phi_fast
  double lambdaD_m;
  double lmps_m;
  double inv_lD;
  double inv_lmps;
  double K1_scaled;
  double K2;
  double phi_slow_eV;
  double phi_fast_eV;
  double e_anchor_vpm;   // MPS tail amplitude (phi-consistent, see CPU)
};

// Blend window of the Chodura tail, in s = d/lambdaD. Must match the
// constexpr values in sheath_prepare_coulette_manfredi (sheath_models.cpp).
KOKKOS_INLINE_FUNCTION constexpr double S_BLEND_START() { return 60.0; }
KOKKOS_INLINE_FUNCTION constexpr double S_BLEND_END()   { return 120.0; }

/* ---------------------------------------------------------------------- */

// 1D effective thermal speed (not Bohm cs); see sheath_models.h.
KOKKOS_INLINE_FUNCTION
double vth_d_eff(double te_eV, double ti_eV, double mD_amu)
{
  const double m = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);
  return Kokkos::sqrt(Kokkos::fmax(te_eV + ti_eV, 0.0) * QE() / (2.0 * m));
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
ChoduraMetrics chodura_metrics(double upar_ms, double cs_ms,
                               const double b[3], const double n[3])
{
  ChoduraMetrics m;
  m.bdotn = 0.0; m.alpha_deg = 90.0; m.mach_par = 0.0; m.mach_n = 0.0; m.u_n = 0.0;

  const double bmag = Kokkos::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  const double nmag = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (bmag <= 0.0 || nmag <= 0.0 || cs_ms <= 0.0) return m;

  const double bdotn = (b[0]*n[0] + b[1]*n[1] + b[2]*n[2]) / (bmag * nmag);
  const double absbn = Kokkos::fmax(0.0, Kokkos::fmin(1.0, Kokkos::fabs(bdotn)));
  m.bdotn = bdotn;
  m.alpha_deg = Kokkos::acos(absbn) * 180.0 / PI();
  m.mach_par = Kokkos::fabs(upar_ms) / cs_ms;
  m.u_n = Kokkos::fabs(upar_ms) * absbn;
  m.mach_n = m.u_n / cs_ms;
  return m;
}

/* ----------------------------------------------------------------------
   Exact port of SheathModels::sheath_auto_dmax (pusher.cpp). MPS
   normal-direction thickness is a few rho_i, roughly angle-independent;
   user_ceiling (global pusher sheath dmax) > 0 sets the extent
   explicitly, 0 = auto.
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double auto_dmax(double te_eV, double ti_eV, double ne_m3,
                 double bmag_T, double alpha_deg,
                 double mD_amu, double user_ceiling)
{
  const double mD_kg = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);
  const double lambdaD = Kokkos::sqrt(EPS0() * Kokkos::fmax(te_eV, 1.0e-12)
                                      / (Kokkos::fmax(ne_m3, 1.0e-60) * QE()));
  const double vth_d = Kokkos::sqrt(Kokkos::fmax(te_eV + ti_eV, 0.0) * QE()
                                    / (2.0 * mD_kg));
  const double omega_ci = QE() * Kokkos::fmax(Kokkos::fabs(bmag_T), 1.0e-20) / mD_kg;
  const double rho_i = vth_d / Kokkos::fmax(omega_ci, 1.0e-99);
  (void)alpha_deg;
  if (user_ceiling > 0.0) return user_ceiling;
  return Kokkos::fmax(5.0 * rho_i, 10.0 * lambdaD);
}

/* ----------------------------------------------------------------------
   Exact port of SheathModels::sheath_prepare_coulette_manfredi
   (sheath_models.cpp), reduced to the fields phi_at_distance consumes.
   NOTE the alpha convention: alpha_deg is the B-to-NORMAL angle; the CM
   fit coefficients use the B-to-WALL angle (90 - alpha_n).
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
CMCoeffs prepare_coulette_manfredi(double te_eV, double ti_eV,
                                   double ne_m3, double bmag_T,
                                   double alpha_deg, double mD_amu,
                                   double pot_mult)
{
  CMCoeffs c;

  const double te = Kokkos::fmax(te_eV, 1.0e-12);
  const double ti = Kokkos::fmax(ti_eV, 0.0);
  const double ne = Kokkos::fmax(ne_m3, 1.0e-60);
  const double bmag = Kokkos::fmax(Kokkos::fabs(bmag_T), 1.0e-20);
  const double mD = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);

  const double lambdaD = Kokkos::sqrt(EPS0() * te / (ne * QE()));
  // vth_d: 1D effective thermal speed for rho_i (not Bohm cs).
  const double vth_d = Kokkos::sqrt((te + ti) * QE() / (2.0 * mD));
  const double omega_ci = QE() * bmag / mD;
  const double rho_i = vth_d / Kokkos::fmax(Kokkos::fabs(omega_ci), 1.0e-99);

  const double alpha_n = Kokkos::fmax(0.0, Kokkos::fmin(90.0, alpha_deg));
  const double alpha   = 90.0 - alpha_n;   // CM fit uses B-wall angle.

  constexpr double b10 =  0.788600127141, b11 = -0.0140352947024;
  constexpr double b20 = -0.511290440114, b21 =  0.0149209038566;
  constexpr double p10 = -3.57918079952,  p11 =  0.0414523939029;
  constexpr double p20 = -0.705851316996, p21 =  0.000903186346144;

  const double B1 = Kokkos::exp(b10 + b11 * alpha);
  const double B2 = Kokkos::exp(b20 + b21 * alpha);
  const double K1 = Kokkos::exp(p10 + p11 * alpha);
  const double K2 = Kokkos::exp(p20 + p21 * alpha);

  const double rho_over_lD = rho_i / Kokkos::fmax(lambdaD, 1.0e-99);
  constexpr double rho_over_lD_ref = 20.0;
  const double K1_scaled = K1 * rho_over_lD_ref / Kokkos::fmax(rho_over_lD, 1.0);

  const double phi_float_mult =
      0.5 * Kokkos::log((mD / Kokkos::fmax(2.0 * PI() * ME(), 1.0e-99))
                        / (1.0 + ti / te));
  const double phi_total =
      (pot_mult > 0.0) ? (pot_mult * te)
                       : (Kokkos::fmax(phi_float_mult, 0.0) * te);
  const double phi_cm_wall_Te = B1 + B2;
  const double scale = phi_total / Kokkos::fmax(phi_cm_wall_Te * te, 1.0e-99);
  const double phi_cm_slow = scale * B1 * te;
  const double phi_cm_fast = scale * B2 * te;

  constexpr double tan_ratio_max = 30.0;
  constexpr double tan_ratio_min = 1.0e-3;
  const double alpha_n_rad = alpha_n * PI() / 180.0;
  const double tan_an = Kokkos::fmin(Kokkos::fmax(
      Kokkos::fabs(Kokkos::tan(alpha_n_rad)), tan_ratio_min), tan_ratio_max);
  const double lmps_phys = rho_i * tan_an;

  const double inv_lD   = 1.0 / Kokkos::fmax(lambdaD, 1.0e-99);
  const double inv_lmps = 1.0 / Kokkos::fmax(lmps_phys, 1.0e-99);

  c.phi_total_eV = phi_total;
  c.lambdaD_m    = lambdaD;
  c.lmps_m       = lmps_phys;
  c.inv_lD       = inv_lD;
  c.inv_lmps     = inv_lmps;
  c.K1_scaled    = K1_scaled;
  c.K2           = K2;
  c.phi_slow_eV  = phi_cm_slow;
  c.phi_fast_eV  = phi_cm_fast;
  // MPS tail amplitude, phi-consistent at the blend anchor (see the CPU
  // comment in sheath_models.cpp for why the raw CM field amplitude is
  // NOT used here).
  c.e_anchor_vpm =
      phi_cm_slow * Kokkos::exp(-K1_scaled * S_BLEND_START()) * inv_lmps;
  return c;
}

/* ----------------------------------------------------------------------
   Exact port of the Coulette-Manfredi branch of
   SheathModels::sheath_phi_at_distance: positive potential drop [V] at
   distance d from the wall, with the blended Chodura tail past s = 60.
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double phi_at_distance(const CMCoeffs &c, double dist_m)
{
  const double d = Kokkos::fmax(dist_m, 0.0);

  const double s = d * c.inv_lD;
  const double phi_slow_cm = c.phi_slow_eV * Kokkos::exp(-c.K1_scaled * s);
  const double phi_fast    = c.phi_fast_eV * Kokkos::exp(-c.K2        * s);

  if (s <= S_BLEND_START()) {
    return phi_slow_cm + phi_fast;
  }

  const double d_past_anchor = (s - S_BLEND_START()) * c.lambdaD_m;
  const double phi_slow_mps = c.e_anchor_vpm * c.lmps_m *
                              Kokkos::exp(-d_past_anchor * c.inv_lmps);
  const double blend = (s >= S_BLEND_END()) ? 1.0
                     : (s - S_BLEND_START()) / (S_BLEND_END() - S_BLEND_START());
  const double phi_slow = (1.0 - blend) * phi_slow_cm + blend * phi_slow_mps;
  return phi_slow + phi_fast;
}

}  // namespace SheathModelsKokkos
}  // namespace SPARTA_NS

#endif  // OPENEDGE_SHEATH_MODELS_KOKKOS_H
