/* ----------------------------------------------------------------------
   Device-callable Borodkina sheath model. Mirrors sheath_models.cpp.
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_MODELS_KOKKOS_H
#define OPENEDGE_SHEATH_MODELS_KOKKOS_H

#include "Kokkos_Core.hpp"
#include "update.h"

namespace SPARTA_NS {
namespace SheathModelsKokkos {

KOKKOS_INLINE_FUNCTION constexpr double QE()   { return PhysConst::QE; }
KOKKOS_INLINE_FUNCTION constexpr double ME()   { return PhysConst::ME; }
KOKKOS_INLINE_FUNCTION constexpr double AMU()  { return PhysConst::AMU; }
KOKKOS_INLINE_FUNCTION constexpr double EPS0() { return PhysConst::EPS0; }
KOKKOS_INLINE_FUNCTION constexpr double PI()   { return 3.14159265358979323846; }
KOKKOS_INLINE_FUNCTION constexpr double KMPS() { return 2.0; }   // keep equal to sheath_models.cpp

struct ChoduraMetrics {
  double bdotn;
  double alpha_deg;
  double mach_par;
  double mach_n;
  double u_n;
};

struct SheathCoeffs {
  double lambdaD_m;
  double rho_i_m;
  double lmps_m;
  double mps_end_m;
  double phi_total_eV;
  double phi_ds_eV;
  double phi_mps_eV;
  double te_eV;
  double lambda_w;
  double lambda_mps;
  double a_ds;
  double q_ds;
  double xi_mps;
  double inv_lambdaD;
  double inv_mps;
  double fd;
  double alpha_star_deg;
  int pure_mps;
};

KOKKOS_INLINE_FUNCTION
ChoduraMetrics chodura_metrics(double upar_ms, double cs_ms,
                               const double b[3], const double n[3])
{
  ChoduraMetrics m;
  m.bdotn = 0.0; m.alpha_deg = 90.0; m.mach_par = 0.0;
  m.mach_n = 0.0; m.u_n = 0.0;
  const double bmag = Kokkos::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  const double nmag = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (bmag <= 0.0 || nmag <= 0.0 || cs_ms <= 0.0) return m;
  const double bdotn = (b[0]*n[0] + b[1]*n[1] + b[2]*n[2]) / (bmag*nmag);
  const double absbn = Kokkos::fmin(Kokkos::fmax(Kokkos::fabs(bdotn), 0.0), 1.0);
  m.bdotn = bdotn;
  m.alpha_deg = Kokkos::acos(absbn) * 180.0 / PI();
  m.mach_par = Kokkos::fabs(upar_ms) / cs_ms;
  m.u_n = Kokkos::fabs(upar_ms) * absbn;
  m.mach_n = m.u_n / cs_ms;
  return m;
}

KOKKOS_INLINE_FUNCTION
SheathCoeffs prepare_borodkina(double te_eV, double ti_eV,
                               double ne_m3, double bmag_T,
                               double alpha_normal_deg,
                               double mD_amu, double pot_mult)
{
  SheathCoeffs c;
  const double te = Kokkos::fmax(te_eV, 1.0e-12);
  const double ti = Kokkos::fmax(ti_eV, 0.0);
  const double ne = Kokkos::fmax(ne_m3, 1.0e-60);
  const double bmag = Kokkos::fmax(Kokkos::fabs(bmag_T), 1.0e-20);
  const double mD = Kokkos::fmax(mD_amu * AMU(), 1.0e-99);
  const double alpha = Kokkos::fmin(Kokkos::fmax(alpha_normal_deg, 0.0), 89.999);
  const double ar = alpha * PI() / 180.0;
  const double lambdaD = Kokkos::sqrt(EPS0() * te / (ne * QE()));
  const double cs = Kokkos::sqrt((te + ti) * QE() / mD);
  const double rho = cs / (QE() * bmag / mD);
  const double phi_float = 0.5 * Kokkos::log(
      (mD / (2.0 * PI() * ME())) / (1.0 + ti / te));
  const double phi_total = (pot_mult > 0.0) ? pot_mult * te
      : Kokkos::fmax(phi_float, 0.0) * te;
  const double lambda_w = -phi_total / te;
  const double cos_astar = Kokkos::fmin(Kokkos::fmax(Kokkos::exp(lambda_w), 0.0), 1.0);
  const double alpha_star = Kokkos::acos(cos_astar) * 180.0 / PI();
  const double cosa = Kokkos::fmax(Kokkos::cos(ar), 1.0e-12);
  const double sina = Kokkos::sin(ar);
  double lambda_mps = Kokkos::log(cosa);
  bool pure_mps = alpha >= alpha_star;
  double phi_mps = pure_mps ? phi_total : -te * lambda_mps;
  if (phi_mps < 1.0e-14 * Kokkos::fmax(phi_total, 1.0)) {
    phi_mps = 0.0;
    lambda_mps = 0.0;
  }
  double phi_ds = Kokkos::fmax(phi_total - phi_mps, 0.0);
  if (phi_ds < 1.0e-12 * Kokkos::fmax(phi_total, 1.0)) {
    pure_mps = true;
    phi_ds = 0.0;
    phi_mps = phi_total;
  }
  const double lmps = KMPS() * rho * sina;

  double a = 0.0;
  double q = 0.0;
  double xi_mps = 0.0;
  if (!pure_mps) {
    const double lmps_D = lmps / lambdaD;
    const double slope_mps = lmps_D > 0.0 ? -2.0 * lambda_mps / lmps_D : 0.0;
    const double beta = 2.0 / (1.0 + ti / te);
    const double c1 = slope_mps * slope_mps - 6.0 * cosa;
    const double ion_root = Kokkos::sqrt(Kokkos::fmax(
        1.0 - beta * (lambda_w - lambda_mps), 0.0));
    const double wall_slope = Kokkos::sqrt(Kokkos::fmax(
        2.0 * Kokkos::exp(lambda_w) + 4.0 * cosa * ion_root + c1, 0.0));
    const double denom = lambda_w - lambda_mps;
    a = Kokkos::fmax((slope_mps - wall_slope) / denom, 1.0e-12);
    q = wall_slope / a;
    if (lmps_D <= 1.0e-14) {
      q = -denom;
      xi_mps = 20.0 / a;
    } else {
      const double ratio = Kokkos::fmin(Kokkos::fmax((denom + q) / q, 1.0e-300), 1.0);
      xi_mps = -Kokkos::log(ratio) / a;
    }
  }

  c.lambdaD_m = lambdaD;
  c.rho_i_m = rho;
  c.lmps_m = lmps;
  c.mps_end_m = pure_mps ? 10.0 * rho * sina
      : Kokkos::fmax(xi_mps * lambdaD + (phi_mps > 0.0 ? 10.0 * rho * sina : 0.0),
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

KOKKOS_INLINE_FUNCTION
double phi_at_distance(const SheathCoeffs &c, double dist_m)
{
  const double d = Kokkos::fmax(dist_m, 0.0);
  if (c.pure_mps)
    return c.phi_total_eV * Kokkos::exp(-d * c.inv_mps);
  const double xi = d * c.inv_lambdaD;
  if (xi <= c.xi_mps) {
    const double lambda = c.lambda_w + c.q_ds
        - c.q_ds * Kokkos::exp(-c.a_ds * xi);
    return Kokkos::fmax(-c.te_eV * lambda, 0.0);
  }
  const double dmps = d - c.xi_mps * c.lambdaD_m;
  return c.phi_mps_eV * Kokkos::exp(-dmps * c.inv_mps);
}

KOKKOS_INLINE_FUNCTION
double emag_at_distance(const SheathCoeffs &c, double dist_m)
{
  const double d = Kokkos::fmax(dist_m, 0.0);
  if (c.pure_mps)
    return c.phi_total_eV * c.inv_mps * Kokkos::exp(-d * c.inv_mps);
  const double xi = d * c.inv_lambdaD;
  if (xi <= c.xi_mps)
    return c.te_eV * c.a_ds * c.q_ds
        * Kokkos::exp(-c.a_ds * xi) * c.inv_lambdaD;
  const double dmps = d - c.xi_mps * c.lambdaD_m;
  return c.phi_mps_eV * c.inv_mps * Kokkos::exp(-dmps * c.inv_mps);
}

// Normal motion across a turning point inside one substep, exact in energy.
// Device mirror of SheathModels::sheath_turning_step (sheath_models.cpp);
// keep the two in sync by hand.
KOKKOS_INLINE_FUNCTION
void turning_step(const SheathCoeffs &c, double q_abs, double mass,
                  double vn, double dt, double d0,
                  double &vn_new, double &dd)
{
  if (!(vn > 0.0) || !(dt > 0.0) || !(mass > 0.0) || !(q_abs > 0.0)) {
    vn_new = vn;
    dd = vn * dt;
    return;
  }
  const double k_over_q = 0.5 * mass * vn * vn / q_abs;      // climb available, in volts
  const double phi0 = phi_at_distance(c, Kokkos::fmax(d0, 0.0));
  const double phi_t = phi0 - k_over_q;                      // potential at the turning point
  double lo = d0, hi = d0 + vn * dt;
  const double phi_hi = phi_at_distance(c, Kokkos::fmax(hi, 0.0));
  if (phi_hi >= phi_t) {                                      // no turning point in reach
    vn_new = Kokkos::sqrt(Kokkos::fmax(vn * vn + 2.0 * q_abs * (phi_hi - phi0) / mass, 0.0));
    dd = vn * dt;
    return;
  }
  for (int it = 0; it < 48; it++) {                           // phi decreases outward
    const double mid = 0.5 * (lo + hi);
    if (phi_at_distance(c, Kokkos::fmax(mid, 0.0)) > phi_t) lo = mid; else hi = mid;
  }
  const double d_t = 0.5 * (lo + hi);
  const double climb = d_t - d0;
  double a = (climb > 1.0e-12) ? vn * vn / (2.0 * climb)      // mean deceleration over the climb
                               : q_abs * emag_at_distance(c, d_t) / mass;
  if (!(a > 0.0)) a = q_abs * emag_at_distance(c, d_t) / mass;
  const double t1 = (a > 0.0) ? vn / a : dt;                  // time to the turning point
  double d_f;
  if (t1 >= dt) {                                             // still climbing at the end of the step
    d_f = d0 + vn * dt - 0.5 * a * dt * dt;
    vn_new = Kokkos::sqrt(Kokkos::fmax(vn * vn + 2.0 * q_abs *
               (phi_at_distance(c, Kokkos::fmax(d_f, 0.0)) - phi0) / mass, 0.0));
  } else {                                                    // turned, falling back for t2
    const double t2 = dt - t1;
    d_f = d_t - 0.5 * a * t2 * t2;
    vn_new = -Kokkos::sqrt(Kokkos::fmax(2.0 * q_abs *
               (phi_at_distance(c, Kokkos::fmax(d_f, 0.0)) - phi_t) / mass, 0.0));
  }
  dd = d_f - d0;
}

}  // namespace SheathModelsKokkos
}  // namespace SPARTA_NS

#endif
