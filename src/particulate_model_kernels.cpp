/* ----------------------------------------------------------------------
   Pure DIS particulate-model equation kernels.

   The collection expressions follow Smirnov et al., PPCF 49, 347 (2007),
   Eqs. (5)/(6), the primary source for DIS Appendix A.  Eq. (5) supplies
   the corrected (1-2 W_m) attractive-grain term.  Eq. (6) corrects the
   opposite-sign exponential coefficients that printed DIS A4 collapses
   to one sign.  Both branches agree with an independent exact OML moment.
------------------------------------------------------------------------- */

#include "particulate_model_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace SPARTA_NS {
namespace ParticulateModel {

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

double sqrt_pi()
{
  return std::sqrt(PI);
}
} // namespace

// DIS (B2)/(B3), including analytic stationary limits.
double dis_ion_current_factor(const double U, const double X)
{
  const double sp = sqrt_pi();
  if (U < 1.0e-3)
    return X >= 0.0 ? 2.0 * (1.0 + X) / sp
                    : 2.0 * std::exp(X) / sp;

  const double u2 = U * U;
  if (X >= 0.0)
    return ((1.0 + 2.0 * (u2 + X)) * std::erf(U)
            + 2.0 * U * std::exp(-u2) / sp) / (2.0 * U);

  const double sx = std::sqrt(-X);
  const double up = U + sx;
  const double um = U - sx;
  return ((1.0 + 2.0 * (u2 + X)) * (std::erf(up) + std::erf(um))
          + (2.0 / sp) * (up * std::exp(-um * um)
                          + um * std::exp(-up * up))) / (4.0 * U);
}

// DIS (B4), including the positive-grain branch chi<0.
double dis_electron_current_factor(const double chi)
{
  return chi >= 0.0 ? std::exp(-chi) : 1.0 - chi;
}

// DIS (B5), multiplying the unretarded Richardson-Dushman current.
double dis_thermionic_recollection_factor(const double chi,
                                           const double Te_eV,
                                           const double Td_K,
                                           const double echarge,
                                           const double boltz)
{
  if (chi >= 0.0) return 1.0;
  if (!(Te_eV > 0.0) || !(Td_K > 0.0) || !(echarge > 0.0) ||
      !(boltz > 0.0)) return std::numeric_limits<double>::quiet_NaN();
  const double y = chi * Te_eV * echarge / (boltz * Td_K);
  return (1.0 - y) * std::exp(y);
}

// DIS (C2)/(C3), using erf(U) in C2 as selected by the exact OML oracle.
double dis_ion_energy_factor(const double U, const double X)
{
  const double sp = sqrt_pi();
  if (U < 1.0e-3)
    return X >= 0.0 ? 2.0 * (2.0 + X) / sp
                    : 2.0 * std::exp(X) * (2.0 - X) / sp;

  const double u2 = U * U;
  if (X >= 0.0) {
    const double a = (2.0 / sp) * (5.0 + 2.0 * (u2 + X))
                     * std::exp(-u2);
    const double b = (3.0 + 12.0 * u2 + 4.0 * u2 * u2
                      + 2.0 * X * (1.0 + 2.0 * u2))
                     * std::erf(U) / U;
    return 0.25 * (a + b);
  }

  const double sx = std::sqrt(-X);
  const double up = U + sx;
  const double um = U - sx;
  double value = (2.0 / sp) *
    ((5.0 + 2.0 * u2 - (3.0 + 2.0 * u2) * sx / U)
       * std::exp(-up * up)
     + (5.0 + 2.0 * u2 + (3.0 + 2.0 * u2) * sx / U)
       * std::exp(-um * um));
  value += (3.0 + 12.0 * u2 + 4.0 * u2 * u2
            + 2.0 * X * (1.0 + 2.0 * u2))
           * (std::erf(up) + std::erf(um)) / U;
  return value / 8.0;
}

// DIS (C4): energy per collected electron is T_e times this factor.
double dis_electron_energy_factor(const double chi)
{
  return chi >= 0.0 ? 2.0 + chi : (2.0 - chi) / (1.0 - chi);
}

// Kollath energy law used by Smirnov et al., Sec. 3.1.
double dis_kollath_yield(const double incident_energy_eV,
                         const double delta_m, const double E_m_eV)
{
  if (!(incident_energy_eV >= 0.0) || !(delta_m >= 0.0) ||
      !(E_m_eV > 0.0) || !std::isfinite(incident_energy_eV) ||
      !std::isfinite(delta_m) || !std::isfinite(E_m_eV))
    return std::numeric_limits<double>::quiet_NaN();
  const double ratio = incident_energy_eV / E_m_eV;
  return 2.72 * 2.72 * delta_m * ratio
    * std::exp(-2.0 * std::sqrt(ratio));
}

/* ----------------------------------------------------------------------
   Maxwellian-flux average of the Kollath yield.

   With a=max(-chi,0), alpha=Te/Em, and y=sqrt(x+a), the integral is

     <delta> = C delta_m alpha/(1+a)
       * 2 exp(a+alpha) int_[sqrt(a)+sqrt(alpha)]^inf
         (t-sqrt(alpha))^5 exp(-t^2) dt .

   The remaining moments obey
     J_n(c)=0.5 c^(n-1) exp(-c^2)+(n-1)J_(n-2)/2,
   so this is both faster and more accurate than doing a quadrature inside
   every current-balance iteration.
------------------------------------------------------------------------- */
double dis_kollath_flux_averaged_yield(const double Te_eV,
                                       const double chi,
                                       const double delta_m,
                                       const double E_m_eV,
                                       const int angular_correction)
{
  if (!(Te_eV > 0.0) || !(delta_m >= 0.0) || !(E_m_eV > 0.0) ||
      !std::isfinite(Te_eV) || !std::isfinite(chi) ||
      !std::isfinite(delta_m) || !std::isfinite(E_m_eV))
    return std::numeric_limits<double>::quiet_NaN();
  if (delta_m == 0.0) return 0.0;

  const long double a = chi < 0.0 ? -static_cast<long double>(chi) : 0.0L;
  const long double alpha = static_cast<long double>(Te_eV / E_m_eV);
  const long double b = std::sqrt(alpha);
  const long double c = std::sqrt(a) + b;
  const long double ec = std::exp(-c * c);
  long double moment[6];
  moment[0] = 0.5L * std::sqrt(static_cast<long double>(PI)) * std::erfc(c);
  moment[1] = 0.5L * ec;
  for (int n = 2; n <= 5; ++n)
    moment[n] = 0.5L * std::pow(c, n - 1) * ec
      + 0.5L * (n - 1) * moment[n - 2];

  static const int choose5[6] = {1, 5, 10, 10, 5, 1};
  long double shifted_moment = 0.0L;
  for (int k = 0; k <= 5; ++k)
    shifted_moment += choose5[k] * std::pow(-b, 5 - k) * moment[k];

  long double average = 2.72L * 2.72L * delta_m * alpha
    * 2.0L * std::exp(a + alpha) * shifted_moment / (1.0L + a);
  if (angular_correction) average *= 1.25L;
  if (!std::isfinite(average) || average < 0.0L)
    return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>(average);
}

double dis_secondary_escape_fraction(const double q)
{
  if (!std::isfinite(q)) return std::numeric_limits<double>::quiet_NaN();
  if (q <= 0.0) return 1.0;
  return (1.0 + 3.0 * q) / std::pow(1.0 + q, 3.0);
}

double dis_secondary_energy_per_electron(const double q,
                                          const double work_function_eV)
{
  if (!std::isfinite(q) || !(work_function_eV > 0.0) ||
      !std::isfinite(work_function_eV))
    return std::numeric_limits<double>::quiet_NaN();
  const double qp = std::max(q, 0.0);
  return 3.0 * work_function_eV * (1.0 + qp) * (1.0 + 2.0 * qp)
    / (1.0 + 3.0 * qp);
}

double dis_thermionic_energy_escape_fraction(const double b)
{
  if (!std::isfinite(b)) return std::numeric_limits<double>::quiet_NaN();
  if (b <= 0.0) return 1.0;
  return 0.5 * (2.0 + 2.0 * b + b * b) * std::exp(-b);
}

namespace {

bool valid_dis_parameters(const DISCurrentParameters &p)
{
  return p.Te_eV > 0.0 && p.Ti_eV > 0.0 && p.ne_m3 > 0.0 &&
    p.ni_m3 > 0.0 && p.Td_K > 0.0 && p.radius_m > 0.0 &&
    p.relative_mach >= 0.0 && p.ion_mass_kg > 0.0 &&
    p.ion_charge_state > 0.0 && p.echarge > 0.0 &&
    p.electron_mass > 0.0 && p.boltz > 0.0 &&
    std::isfinite(p.Te_eV) && std::isfinite(p.Ti_eV) &&
    std::isfinite(p.ne_m3) && std::isfinite(p.ni_m3) &&
    std::isfinite(p.Td_K) && std::isfinite(p.radius_m) &&
    std::isfinite(p.relative_mach) && std::isfinite(p.ion_mass_kg) &&
    std::isfinite(p.ion_charge_state);
}

} // namespace

bool dis_current_state_at_chi(const DISCurrentParameters &p,
                              const double chi, DISCurrentState &s)
{
  if (!valid_dis_parameters(p) || !std::isfinite(chi) ||
      p.secondary_model == SECONDARY_DUSTT2005 ||
      p.secondary_model == SECONDARY_YOUNG_DEKKER)
    return false;
  if (p.thermionic_on && (!(p.work_function_eV > 0.0) ||
      !(p.richardson_A > 0.0) || !std::isfinite(p.work_function_eV) ||
      !std::isfinite(p.richardson_A))) return false;
  if (p.secondary_model == SECONDARY_KOLLATH_SMIRNOV2007 &&
      (!(p.work_function_eV > 0.0) || !(p.see_delta_m > 0.0) ||
       !(p.see_E_m_eV > 0.0))) return false;

  const double area_coll = PI * p.radius_m * p.radius_m;
  const double area_total = 4.0 * area_coll;
  const double vte_bar = std::sqrt(8.0 * p.Te_eV * p.echarge /
                                   (PI * p.electron_mass));
  const double vti = std::sqrt(2.0 * p.Ti_eV * p.echarge / p.ion_mass_kg);
  const double X = p.ion_charge_state * chi * p.Te_eV / p.Ti_eV;
  const double Fe = dis_electron_current_factor(chi);
  const double Fi = dis_ion_current_factor(p.relative_mach, X);

  s.chi = chi;
  s.phi_V = -chi * p.Te_eV;
  s.electron_A = -p.echarge * area_coll * p.ne_m3 * vte_bar * Fe;
  s.ion_A = p.ion_charge_state * p.echarge * area_coll * p.ni_m3
    * vti * Fi;
  s.thermionic_unsuppressed_A = 0.0;
  s.thermionic_A = 0.0;
  if (p.thermionic_on) {
    s.thermionic_unsuppressed_A = area_total * p.richardson_A * p.Td_K
      * p.Td_K * std::exp(-p.work_function_eV * p.echarge /
                          (p.boltz * p.Td_K));
    const double fth = dis_thermionic_recollection_factor(
      chi, p.Te_eV, p.Td_K, p.echarge, p.boltz);
    s.thermionic_A = s.thermionic_unsuppressed_A * fth;
  }

  s.secondary_A = 0.0;
  s.secondary_yield = 0.0;
  s.secondary_escape_fraction = 1.0;
  if (p.secondary_model == SECONDARY_KOLLATH_SMIRNOV2007) {
    s.secondary_yield = dis_kollath_flux_averaged_yield(
      p.Te_eV, chi, p.see_delta_m, p.see_E_m_eV,
      p.see_angular_correction);
    const double q = std::max(-chi * p.Te_eV / p.work_function_eV, 0.0);
    s.secondary_escape_fraction = dis_secondary_escape_fraction(q);
    s.secondary_A = std::fabs(s.electron_A) * s.secondary_yield
      * s.secondary_escape_fraction;
  }
  s.residual_A = s.ion_A + s.electron_A + s.thermionic_A + s.secondary_A;
  return std::isfinite(s.ion_A) && std::isfinite(s.electron_A) &&
    std::isfinite(s.thermionic_unsuppressed_A) &&
    std::isfinite(s.thermionic_A) && std::isfinite(s.secondary_A) &&
    std::isfinite(s.secondary_yield) &&
    std::isfinite(s.secondary_escape_fraction) &&
    std::isfinite(s.residual_A);
}

bool dis_solve_current_balance(const DISCurrentParameters &p,
                               DISCurrentState &s)
{
  if (!valid_dis_parameters(p)) return false;

  const double ratio = std::max(p.Te_eV, p.Ti_eV) / p.Te_eV;
  double lo = -20.0 * ratio;
  double hi = 80.0 * ratio;
  DISCurrentState slo, shi, smid;
  if (!dis_current_state_at_chi(p, lo, slo) ||
      !dis_current_state_at_chi(p, hi, shi)) return false;

  int expand = 0;
  while (slo.residual_A * shi.residual_A > 0.0 && expand < 12) {
    lo *= 2.0;
    hi *= 2.0;
    if (!dis_current_state_at_chi(p, lo, slo) ||
        !dis_current_state_at_chi(p, hi, shi)) return false;
    ++expand;
  }
  if (slo.residual_A == 0.0) { s = slo; return true; }
  if (shi.residual_A == 0.0) { s = shi; return true; }
  if (slo.residual_A * shi.residual_A > 0.0) return false;

  for (int iteration = 0; iteration < 160; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    if (!dis_current_state_at_chi(p, mid, smid)) return false;
    const double scale = std::fabs(smid.ion_A) + std::fabs(smid.electron_A)
      + std::fabs(smid.thermionic_A) + std::fabs(smid.secondary_A);
    if (std::fabs(smid.residual_A) <= 1.0e-12 * std::max(scale, 1.0e-300) ||
        std::fabs(hi - lo) <= 1.0e-12 * std::max(1.0, std::fabs(mid))) {
      s = smid;
      return true;
    }
    if (slo.residual_A * smid.residual_A <= 0.0) {
      hi = mid;
      shi = smid;
    } else {
      lo = mid;
      slo = smid;
    }
  }
  s = smid;
  return std::isfinite(s.residual_A);
}

bool dis_heat_flux(const DISCurrentParameters &p,
                   const DISCurrentState &s,
                   const double ion_neutralization_eV,
                   DISHeatFluxState &h)
{
  if (!valid_dis_parameters(p) || !std::isfinite(s.chi) ||
      !(ion_neutralization_eV >= 0.0)) return false;
  const double area = 4.0 * PI * p.radius_m * p.radius_m;
  const double vti = std::sqrt(2.0 * p.Ti_eV * p.echarge / p.ion_mass_kg);
  const double X = p.ion_charge_state * s.chi * p.Te_eV / p.Ti_eV;
  const double Gi = dis_ion_energy_factor(p.relative_mach, X);
  const double Ge = dis_electron_energy_factor(s.chi);
  if (!(Gi >= 0.0) || !(Ge >= 0.0) || !std::isfinite(Gi) ||
      !std::isfinite(Ge)) return false;

  const double gamma_i0 = 0.25 * p.ni_m3 * vti;
  h.ion_kinetic_W_m2 = p.echarge * gamma_i0 * p.Ti_eV * Gi;
  h.electron_kinetic_W_m2 = std::fabs(s.electron_A) / area
    * p.Te_eV * Ge;
  h.ion_neutralization_W_m2 = s.ion_A / area /
    p.ion_charge_state * ion_neutralization_eV;
  h.sheath_W_m2 = (s.ion_A + s.electron_A) / area
    * s.chi * p.Te_eV;

  h.thermionic_cooling_W_m2 = 0.0;
  if (p.thermionic_on && s.thermionic_unsuppressed_A > 0.0) {
    const double b = std::max(-s.chi * p.Te_eV * p.echarge /
                              (p.boltz * p.Td_K), 0.0);
    const double number_fraction = dis_thermionic_recollection_factor(
      s.chi, p.Te_eV, p.Td_K, p.echarge, p.boltz);
    const double energy_fraction = dis_thermionic_energy_escape_fraction(b);
    const double kTd_eV = p.boltz * p.Td_K / p.echarge;
    h.thermionic_cooling_W_m2 = s.thermionic_unsuppressed_A / area
      * (p.work_function_eV * number_fraction
         + 2.0 * kTd_eV * energy_fraction);
  }

  h.secondary_cooling_W_m2 = 0.0;
  if (s.secondary_A > 0.0) {
    const double q = std::max(-s.chi * p.Te_eV / p.work_function_eV, 0.0);
    h.secondary_cooling_W_m2 = s.secondary_A / area
      * dis_secondary_energy_per_electron(q, p.work_function_eV);
  }
  h.net_W_m2 = h.ion_kinetic_W_m2 + h.electron_kinetic_W_m2
    + h.ion_neutralization_W_m2 + h.sheath_W_m2
    - h.thermionic_cooling_W_m2 - h.secondary_cooling_W_m2;
  return std::isfinite(h.ion_kinetic_W_m2) &&
    std::isfinite(h.electron_kinetic_W_m2) &&
    std::isfinite(h.ion_neutralization_W_m2) &&
    std::isfinite(h.sheath_W_m2) &&
    std::isfinite(h.thermionic_cooling_W_m2) &&
    std::isfinite(h.secondary_cooling_W_m2) &&
    std::isfinite(h.net_W_m2);
}

// Smirnov (5)/(6): corrected DIS (A2)/(A4) collection for either sign.
bool dis_ion_drag_collection_factor(const double U, const double X,
                                     double &factor)
{
  factor = std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(U) || !std::isfinite(X) || U < 0.0)
    return false;

  const double sp = sqrt_pi();
  if (X >= 0.0 && U < 1.0e-3) {
    factor = (8.0 + 4.0 * X) / (3.0 * sp);
  } else if (X >= 0.0) {
    const double u2 = U * U;
    const double wp = u2 + X;
    const double wm = u2 - X;
    const double bracket = 1.0 + 2.0 * wp
      - (1.0 - 2.0 * wm) / (2.0 * u2);
    factor = ((1.0 + 2.0 * wp) * std::exp(-u2) / sp
              + U * bracket * std::erf(U)) / (2.0 * u2);
  } else {
    // Smirnov Eq. (6) is finite as U->0 but its direct form is 0/0 and
    // suffers catastrophic cancellation.  Its even small-U expansion is
    // used through O(U^2); it agrees with the exact OML integral while a
    // direct evaluation at 1e-4 loses several digits in double precision.
    if (U < 1.0e-2) {
      const double s2 = -X;
      const double ex = std::exp(-s2);
      const double c0 = 4.0 * (s2 + 2.0) * ex / (3.0 * sp);
      const double c2 = ex / sp
        * (8.0 * s2 * s2 / 15.0 + 4.0 * s2 / 5.0 + 8.0 / 15.0);
      factor = c0 + U * U * c2;
      return std::isfinite(factor) && factor >= 0.0;
    }
    const double u = U;
    const double u2 = u * u;
    const double sx = std::sqrt(-X);
    const double up = u + sx;
    const double um = u - sx;
    const double wp = u2 + X;
    const double wm = u2 - X;
    const double c = (1.0 - 2.0 * u2) * sx / u;
    const double t1 =
      ((1.0 + 2.0 * u2 + c) * std::exp(-up * up)
       + (1.0 + 2.0 * u2 - c) * std::exp(-um * um)) / sp;
    const double t2 =
      u * (1.0 + 2.0 * wp - (1.0 - 2.0 * wm) / (2.0 * u2))
      * (std::erf(up) + std::erf(um));
    factor = (t1 + t2) / (4.0 * u2);
  }
  return std::isfinite(factor) && factor >= 0.0;
}

// DIS (A5), defined for either sign because it depends on X^2.
double dis_ion_drag_scattering_factor(const double U, const double X,
                                       const double lnLambda)
{
  if (!std::isfinite(U) || !std::isfinite(X) ||
      !std::isfinite(lnLambda) || U < 0.0 || lnLambda < 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  const double sp = sqrt_pi();
  if (U < 1.0e-3)
    return X * X * lnLambda * 4.0 / (3.0 * sp);
  const double u2 = U * U;
  return X * X * lnLambda
    * (std::erf(U) - 2.0 * U * std::exp(-u2) / sp) / (U * u2);
}

// DIS (A2)+(A5), multiplying m_i n_i pi a^2 v_T (u_i-v_d).
bool dis_ion_drag_factor(const double U, const double X,
                         const double lnLambda, double &factor)
{
  double coll = 0.0;
  if (!dis_ion_drag_collection_factor(U, X, coll)) {
    factor = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  const double scat = dis_ion_drag_scattering_factor(U, X, lnLambda);
  factor = coll + scat;
  return std::isfinite(factor) && factor >= 0.0;
}

} // namespace ParticulateModel
} // namespace SPARTA_NS
