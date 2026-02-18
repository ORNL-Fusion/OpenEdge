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

double sound_speed_d(double te_eV, double ti_eV, double mD_amu)
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
  BorodkinaSheathResult out;

  const double d = std::max(dist_m, 0.0);
  const double te = std::max(te_eV, 1.0e-12);
  const double ti = std::max(ti_eV, 0.0);
  const double ne = std::max(ne_m3, 1.0e-60);
  const double bmag = std::max(std::abs(bmag_T), 1.0e-20);
  const double mD = std::max(mD_amu * AMU, 1.0e-99);

  // Keep Debye length fixed at sheath-entrance values.
  const double lambdaD = std::sqrt(EPS0 * te / (ne * QE));
  out.lambdaD_m = lambdaD;

  // c_s = sqrt((Te+Ti)e/(2mD))
  const double cs = std::sqrt((te + ti) * QE / (2.0 * mD));
  const double omega_ci = QE * bmag / mD;
  const double rho_i = cs / std::max(std::abs(omega_ci), 1.0e-99);

  const double alpha = std::max(0.0, std::min(90.0, alpha_deg));
  const double alpha_n_rad = alpha * PI / 180.0;
  const double sin_a = std::abs(std::sin(alpha_n_rad));
  const double sin_floor = 0.05;
  const double lmps = rho_i / std::max(sin_a, sin_floor);
  out.lmps_m = lmps;

  const double fd = fd_poly_deg(alpha);
  out.fd = fd;
  const double phi0 = std::max(pot_mult, 0.0) * te;

  const double e_ds = std::exp(-d / std::max(2.0 * lambdaD, 1.0e-99));
  const double e_mps = std::exp(-d / std::max(lmps, 1.0e-99));

  // phi relative to plasma potential (in eV-equivalent potential drop).
  const double phi_eV = -phi0 * (fd * e_ds + (1.0 - fd) * e_mps);
  out.esheath_eV = -phi_eV;

  const double e_ds_vpm = phi0 * fd * e_ds / std::max(2.0 * lambdaD, 1.0e-99);
  const double e_mps_vpm = phi0 * (1.0 - fd) * e_mps / std::max(lmps, 1.0e-99);
  out.emag_vpm = std::abs(e_ds_vpm + e_mps_vpm);
  return out;
}

}  // namespace SheathModels
}  // namespace SPARTA_NS
