/* ----------------------------------------------------------------------
   OpenEdge sheath helper models — Kokkos device-callable versions.

   These are KOKKOS_INLINE_FUNCTION equivalents of the functions in
   sheath_models.cpp.  Only the two spatial sheath models
   (borodkina, coulette_manfredi), sound_speed_d, and chodura_metrics
   are ported here — they are called in the Boris subcycle inner loop.
   eirene_sheath_ev (uses std::vector) stays host-only.

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

// Reuse the same result structs (POD, device-safe)
struct BorodkinaSheathResult {
  double esheath_eV;
  double emag_vpm;
  double fd;
  double lmps_m;
  double lambdaD_m;
  double rho_i_m;
  double phi_ds_eV;
  double phi_cs_eV;
};

struct ChoduraMetrics {
  double bdotn;
  double alpha_deg;
  double mach_par;
  double mach_n;
  double u_n;
};

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double fd_poly_deg(const double a)
{
  const double fd =
    0.98992 + 5.1220e-3 * a - 7.0040e-4 * a * a +
    3.3591e-5 * a * a * a - 8.2917e-7 * a*a*a*a +
    9.5856e-9 * a*a*a*a*a - 4.2682e-11 * a*a*a*a*a*a;
  return Kokkos::fmax(0.0, Kokkos::fmin(1.0, fd));
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double sound_speed_d(double te_eV, double ti_eV, double mD_amu)
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

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
BorodkinaSheathResult borodkina_sheath_at_distance(
    double dist_m, double te_eV, double ti_eV, double ne_m3,
    double bmag_T, double alpha_deg, double mD_amu, double pot_mult = 2.5)
{
  BorodkinaSheathResult out;
  out.esheath_eV = out.emag_vpm = out.fd = out.lmps_m = 0.0;
  out.lambdaD_m = out.rho_i_m = out.phi_ds_eV = out.phi_cs_eV = 0.0;

  const double d  = Kokkos::fmax(dist_m, 0.0);
  const double te = Kokkos::fmax(te_eV, 1.0e-12);
  const double ti = Kokkos::fmax(ti_eV, 0.0);
  const double ne = Kokkos::fmax(ne_m3, 1.0e-60);
  const double bmag = Kokkos::fmax(Kokkos::fabs(bmag_T), 1.0e-20);
  const double mD = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);

  const double lambdaD = Kokkos::sqrt(EPS0() * te / (ne * QE()));
  out.lambdaD_m = lambdaD;

  const double cs = Kokkos::sqrt((te + ti) * QE() / (2.0 * mD));
  const double omega_ci = QE() * bmag / mD;
  const double rho_i = cs / Kokkos::fmax(Kokkos::fabs(omega_ci), 1.0e-99);
  out.rho_i_m = rho_i;

  const double alpha = Kokkos::fmax(0.0, Kokkos::fmin(90.0, alpha_deg));
  const double alpha_n_rad = alpha * PI() / 180.0;
  const double sin_a = Kokkos::fabs(Kokkos::sin(alpha_n_rad));
  const double sin_floor = 0.05;
  const double lmps = rho_i / Kokkos::fmax(sin_a, sin_floor);
  out.lmps_m = lmps;

  const double fd = fd_poly_deg(alpha);
  out.fd = fd;
  const double phi0 = Kokkos::fmax(pot_mult, 0.0) * te;
  out.phi_ds_eV = fd * phi0;
  out.phi_cs_eV = (1.0 - fd) * phi0;

  const double e_ds  = Kokkos::exp(-d / Kokkos::fmax(2.0 * lambdaD, 1.0e-99));
  const double e_mps = Kokkos::exp(-d / Kokkos::fmax(lmps, 1.0e-99));

  const double phi_eV = -phi0 * (fd * e_ds + (1.0 - fd) * e_mps);
  out.esheath_eV = -phi_eV;

  const double e_ds_vpm  = phi0 * fd * e_ds / Kokkos::fmax(2.0 * lambdaD, 1.0e-99);
  const double e_mps_vpm = phi0 * (1.0 - fd) * e_mps / Kokkos::fmax(lmps, 1.0e-99);
  out.emag_vpm = Kokkos::fabs(e_ds_vpm + e_mps_vpm);
  return out;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
BorodkinaSheathResult coulette_manfredi_sheath_at_distance(
    double dist_m, double te_eV, double ti_eV, double ne_m3,
    double bmag_T, double alpha_deg, double mD_amu, double pot_mult = 0.0)
{
  BorodkinaSheathResult out;
  out.esheath_eV = out.emag_vpm = out.fd = out.lmps_m = 0.0;
  out.lambdaD_m = out.rho_i_m = out.phi_ds_eV = out.phi_cs_eV = 0.0;

  const double d  = Kokkos::fmax(dist_m, 0.0);
  const double te = Kokkos::fmax(te_eV, 1.0e-12);
  const double ti = Kokkos::fmax(ti_eV, 0.0);
  const double ne = Kokkos::fmax(ne_m3, 1.0e-60);
  const double bmag = Kokkos::fmax(Kokkos::fabs(bmag_T), 1.0e-20);
  const double mD = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);

  const double lambdaD = Kokkos::sqrt(EPS0() * te / (ne * QE()));
  out.lambdaD_m = lambdaD;

  const double cs = Kokkos::sqrt((te + ti) * QE() / (2.0 * mD));
  const double omega_ci = QE() * bmag / mD;
  const double rho_i = cs / Kokkos::fmax(Kokkos::fabs(omega_ci), 1.0e-99);
  out.rho_i_m = rho_i;

  const double alpha = Kokkos::fmax(0.0, Kokkos::fmin(90.0, alpha_deg));

  // CM two-exponential fit coefficients
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

  const double s = d / Kokkos::fmax(lambdaD, 1.0e-99);

  const double phi_float_mult =
    0.5 * Kokkos::log((mD / Kokkos::fmax(2.0 * PI() * ME(), 1.0e-99)) / (1.0 + ti / te));
  const double phi_total = (pot_mult > 0.0) ? (pot_mult * te)
                         : (Kokkos::fmax(phi_float_mult, 0.0) * te);

  const double phi_cm_wall_Te = B1 + B2;
  const double scale = phi_total / Kokkos::fmax(phi_cm_wall_Te * te, 1.0e-99);

  const double phi_cm_slow = scale * B1 * te;
  const double phi_cm_fast = scale * B2 * te;

  out.phi_cs_eV = phi_cm_slow;
  out.phi_ds_eV = phi_cm_fast;
  out.fd = (phi_total > 0.0) ? (phi_cm_fast / phi_total) : 0.0;
  out.lmps_m = 1.0 / (K1_scaled / Kokkos::fmax(lambdaD, 1.0e-99));

  const double e_slow = phi_cm_slow * K1_scaled * Kokkos::exp(-K1_scaled * s)
                        / Kokkos::fmax(lambdaD, 1.0e-99);
  const double e_fast = phi_cm_fast * K2 * Kokkos::exp(-K2 * s)
                        / Kokkos::fmax(lambdaD, 1.0e-99);

  out.esheath_eV = phi_cm_slow * Kokkos::exp(-K1_scaled * s)
                 + phi_cm_fast * Kokkos::exp(-K2 * s);
  out.emag_vpm = Kokkos::fabs(e_slow + e_fast);
  return out;
}

}  // namespace SheathModelsKokkos
}  // namespace SPARTA_NS

#endif  // OPENEDGE_SHEATH_MODELS_KOKKOS_H
