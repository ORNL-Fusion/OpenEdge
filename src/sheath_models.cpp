/* ----------------------------------------------------------------------
   Borodkina sheath model, Russian Phys. J. 58 (2015) 438-445 and
   Contrib. Plasma Phys. 56 (2016) 640-645.
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
constexpr double KMPS = 2.0;   // L_MPS = k rho_cs sin(alpha); k = 2 as in Borodkina 2015/PYEAD (was 3, Nov 2026)
}

ChoduraMetrics chodura_metrics(double upar_ms, double cs_ms,
                               const double b[3], const double n[3])
{
  ChoduraMetrics m;
  const double bmag = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  const double nmag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (bmag <= 0.0 || nmag <= 0.0 || cs_ms <= 0.0) return m;
  const double bdotn = (b[0]*n[0] + b[1]*n[1] + b[2]*n[2]) / (bmag*nmag);
  const double absbn = std::min(std::max(std::abs(bdotn), 0.0), 1.0);
  m.bdotn = bdotn;
  m.alpha_deg = std::acos(absbn) * 180.0 / PI;
  m.mach_par = std::abs(upar_ms) / cs_ms;
  m.u_n = std::abs(upar_ms) * absbn;
  m.mach_n = m.u_n / cs_ms;
  return m;
}

SheathEmagCoeffs sheath_prepare_borodkina(double te_eV, double ti_eV,
                                          double ne_m3, double bmag_T,
                                          double alpha_normal_deg,
                                          double mD_amu, double pot_mult)
{
  SheathEmagCoeffs c;
  const double te = std::max(te_eV, 1.0e-12);
  const double ti = std::max(ti_eV, 0.0);
  const double ne = std::max(ne_m3, 1.0e-60);
  const double bmag = std::max(std::abs(bmag_T), 1.0e-20);
  const double mD = std::max(mD_amu * AMU, 1.0e-99);
  const double alpha = std::min(std::max(alpha_normal_deg, 0.0), 89.999);
  const double ar = alpha * PI / 180.0;
  const double lambdaD = std::sqrt(EPS0 * te / (ne * QE));
  const double cs = std::sqrt((te + ti) * QE / mD);
  const double rho = cs / (QE * bmag / mD);
  const double phi_float = 0.5 * std::log(
      (mD / (2.0 * PI * ME)) / (1.0 + ti / te));
  const double phi_total = (pot_mult > 0.0) ? pot_mult * te
                                            : std::max(phi_float, 0.0) * te;
  const double lambda_w = -phi_total / te;
  const double cos_astar = std::min(std::max(std::exp(lambda_w), 0.0), 1.0);
  const double alpha_star = std::acos(cos_astar) * 180.0 / PI;
  const double cosa = std::max(std::cos(ar), 1.0e-12);
  const double sina = std::sin(ar);
  double lambda_mps = std::log(cosa);
  bool pure_mps = alpha >= alpha_star;
  double phi_mps = pure_mps ? phi_total : -te * lambda_mps;
  if (phi_mps < 1.0e-14 * std::max(phi_total, 1.0)) {
    phi_mps = 0.0;
    lambda_mps = 0.0;
  }
  double phi_ds = std::max(phi_total - phi_mps, 0.0);
  if (phi_ds < 1.0e-12 * std::max(phi_total, 1.0)) {
    pure_mps = true;
    phi_ds = 0.0;
    phi_mps = phi_total;
  }
  const double lmps = KMPS * rho * sina;

  double a = 0.0;
  double q = 0.0;
  double xi_mps = 0.0;
  if (!pure_mps) {
    const double lmps_D = lmps / lambdaD;
    const double slope_mps = lmps_D > 0.0 ? -2.0 * lambda_mps / lmps_D : 0.0;
    const double beta = 2.0 / (1.0 + ti / te);
    const double c1 = slope_mps * slope_mps - 6.0 * cosa;
    const double ion_root = std::sqrt(std::max(
        1.0 - beta * (lambda_w - lambda_mps), 0.0));
    const double wall_slope = std::sqrt(std::max(
        2.0 * std::exp(lambda_w) + 4.0 * cosa * ion_root + c1, 0.0));
    const double denom = lambda_w - lambda_mps;
    a = (slope_mps - wall_slope) / denom;
    a = std::max(a, 1.0e-12);
    q = wall_slope / a;
    if (lmps_D <= 1.0e-14) {
      q = -denom;
      xi_mps = 20.0 / a;
    } else {
      const double ratio = std::min(std::max((denom + q) / q, 1.0e-300), 1.0);
      xi_mps = -std::log(ratio) / a;
    }
  }

  c.lambdaD_m = lambdaD;
  c.rho_i_m = rho;
  c.lmps_m = lmps;
  c.mps_end_m = pure_mps ? 10.0 * rho * sina
      : std::max(xi_mps * lambdaD + (phi_mps > 0.0 ? 10.0 * rho * sina : 0.0),
                 20.0 * lambdaD);
  c.phi_total_eV = phi_total;
  c.phi_ds_eV = phi_ds;
  c.phi_mps_eV = phi_mps;
  c.te_eV = te;
  c.lambda_w = lambda_w;
  c.lambda_mps = lambda_mps;
  c.a_ds = a;
  c.q_ds = q;
  c.xi_mps = xi_mps;
  c.inv_lambdaD = 1.0 / lambdaD;
  c.inv_mps = lmps > 0.0 ? 2.0 / lmps : 0.0;
  c.fd = phi_total > 0.0 ? phi_ds / phi_total : 0.0;
  c.alpha_star_deg = alpha_star;
  c.pure_mps = pure_mps ? 1 : 0;
  return c;
}

SheathEmagCoeffs sheath_prepare(double te_eV, double ti_eV, double ne_m3,
                                double bmag_T, double alpha_deg, double mD_amu,
                                double pot_mult)
{
  return sheath_prepare_borodkina(te_eV, ti_eV, ne_m3, bmag_T, alpha_deg,
                                  mD_amu, pot_mult);
}

double sheath_phi_at_distance(const SheathEmagCoeffs &c, double dist_m)
{
  const double d = std::max(dist_m, 0.0);
  if (c.pure_mps)
    return c.phi_total_eV * std::exp(-d * c.inv_mps);
  const double xi = d * c.inv_lambdaD;
  if (xi <= c.xi_mps) {
    const double lambda = c.lambda_w + c.q_ds -
                          c.q_ds * std::exp(-c.a_ds * xi);
    return std::max(-c.te_eV * lambda, 0.0);
  }
  const double dmps = d - c.xi_mps * c.lambdaD_m;
  return c.phi_mps_eV * std::exp(-dmps * c.inv_mps);
}

void sheath_turning_step(const SheathEmagCoeffs &c, double q_abs, double mass,
                         double vn, double dt, double d0,
                         double &vn_new, double &dd)
{
  auto phi = [&](double d) { return sheath_phi_at_distance(c, std::max(d, 0.0)); };
  if (!(vn > 0.0) || !(dt > 0.0) || !(mass > 0.0) || !(q_abs > 0.0)) {
    vn_new = vn;
    dd = vn * dt;
    return;
  }
  const double k_over_q = 0.5 * mass * vn * vn / q_abs;      // climb available, in volts
  const double phi0 = phi(d0);
  const double phi_t = phi0 - k_over_q;                      // potential at the turning point
  double lo = d0, hi = d0 + vn * dt;
  if (phi(hi) >= phi_t) {                                     // no turning point in reach (caller
    vn_new = std::sqrt(std::max(vn * vn + 2.0 * q_abs * (phi(hi) - phi0) / mass, 0.0));
    dd = vn * dt;                                             // misjudged); plain exact step
    return;
  }
  for (int it = 0; it < 48; it++) {                           // phi decreases outward
    const double mid = 0.5 * (lo + hi);
    if (phi(mid) > phi_t) lo = mid; else hi = mid;
  }
  const double d_t = 0.5 * (lo + hi);
  const double climb = d_t - d0;
  double a = (climb > 1.0e-12) ? vn * vn / (2.0 * climb)      // mean deceleration over the climb
                               : q_abs * sheath_emag_at_distance(c, d_t) / mass;
  if (!(a > 0.0)) a = q_abs * sheath_emag_at_distance(c, d_t) / mass;
  const double t1 = (a > 0.0) ? vn / a : dt;                  // time to the turning point
  double d_f;
  if (t1 >= dt) {                                             // still climbing at the end of the step
    d_f = d0 + vn * dt - 0.5 * a * dt * dt;
    vn_new = std::sqrt(std::max(vn * vn + 2.0 * q_abs * (phi(d_f) - phi0) / mass, 0.0));
  } else {                                                    // turned, falling back for t2
    const double t2 = dt - t1;
    d_f = d_t - 0.5 * a * t2 * t2;
    vn_new = -std::sqrt(std::max(2.0 * q_abs * (phi(d_f) - phi_t) / mass, 0.0));
  }
  dd = d_f - d0;
}

double sheath_emag_at_distance(const SheathEmagCoeffs &c, double dist_m)
{
  const double d = std::max(dist_m, 0.0);
  if (c.pure_mps)
    return c.phi_total_eV * c.inv_mps * std::exp(-d * c.inv_mps);
  const double xi = d * c.inv_lambdaD;
  if (xi <= c.xi_mps)
    return c.te_eV * c.a_ds * c.q_ds *
           std::exp(-c.a_ds * xi) * c.inv_lambdaD;
  const double dmps = d - c.xi_mps * c.lambdaD_m;
  return c.phi_mps_eV * c.inv_mps * std::exp(-dmps * c.inv_mps);
}

SheathProfile sheath_at_distance(double dist_m, double te_eV, double ti_eV,
                                 double ne_m3, double bmag_T, double alpha_deg,
                                 double mD_amu, double pot_mult)
{
  const SheathEmagCoeffs c = sheath_prepare(te_eV, ti_eV, ne_m3, bmag_T,
                                            alpha_deg, mD_amu, pot_mult);
  SheathProfile out;
  out.esheath_eV = sheath_phi_at_distance(c, dist_m);
  out.emag_vpm = sheath_emag_at_distance(c, dist_m);
  out.lambdaD_m = c.lambdaD_m;
  out.rho_i_m = c.rho_i_m;
  out.extent_m = c.mps_end_m;
  out.phi_total_eV = c.phi_total_eV;
  out.phi_ds_eV = c.phi_ds_eV;
  return out;
}

}  // namespace SheathModels
}  // namespace SPARTA_NS
