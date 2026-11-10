#include "sheath_iead.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "math_extra.h"
#include "pusher.h"
#include "random_knuth.h"
#include "sheath_models.h"

using namespace SPARTA_NS;

namespace {
constexpr double QE = 1.602176634e-19;
constexpr double AMU = 1.66053906660e-27;
constexpr double PI = 3.14159265358979323846;
constexpr double OMEGA_DT = 0.05;      // gyro-phase advance per substep
constexpr long MAX_STEPS = 4000000;   // hard cap; the transit-time cap below is the real limit
constexpr int NSUB = 16;               // substeps of the wall-crossing step
}

bool SheathIead::Key::operator<(const Key &o) const
{
  if (Z != o.Z) return Z < o.Z;
  if (te != o.te) return te < o.te;
  if (ti != o.ti) return ti < o.ti;
  if (ne != o.ne) return ne < o.ne;
  if (bmag != o.bmag) return bmag < o.bmag;
  if (alpha_deg != o.alpha_deg) return alpha_deg < o.alpha_deg;
  if (mass != o.mass) return mass < o.mass;
  return upar < o.upar;
}

SheathIead::SheathIead(int nmarkers_, int nE_, int nT_, double mD_amu_,
                       int drift_mode_, int seed_)
  : nmarkers(nmarkers_), nE(nE_), nT(nT_), drift_mode(drift_mode_),
    mD_amu(mD_amu_), seed(seed_)
{
  random = new RanKnuth(seed);
}

SheathIead::~SheathIead() { delete random; }

size_t SheathIead::n_high_escape() const
{
  size_t n = 0;
  for (const auto &kv : cache) if (kv.second.escape_frac > 0.5) n++;
  return n;
}

void SheathIead::dump(const char *path) const
{
  FILE *fp = fopen(path, "w");
  if (!fp) return;
  for (const auto &kv : cache) {
    const Key &k = kv.first; const Hist &h = kv.second;
    fprintf(fp, "# te %g ti %g ne %g B %g alpha %g Z %d mass %g upar %g "
            "nimpact %ld escape %g e_mean %g th_mean %g emax %g nE %d nT %d "
            "dmax %g dt %g nstep_max %ld\n",
            k.te, k.ti, k.ne, k.bmag, k.alpha_deg, k.Z, k.mass, k.upar,
            h.nimpact, h.escape_frac, h.e_mean, h.th_mean, h.emax, h.nE, h.nT,
            h.d_launch, h.dt, h.nstep_max);
    for (int ie = 0; ie < h.nE; ie++) {
      for (int it = 0; it < h.nT; it++)
        fprintf(fp, "%s%.6e", it ? " " : "", h.f[static_cast<size_t>(ie) * h.nT + it]);
      fputc('\n', fp);
    }
  }
  fclose(fp);
}

const SheathIead::Hist &SheathIead::get(double te, double ti, double ne,
                                        const double b[3], const double n[3],
                                        double mass, int Z, double upar)
{
  const double bmag = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  const double nmag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  double cosa = 0.0;
  if (bmag > 0.0 && nmag > 0.0)
    cosa = std::fabs(b[0]*n[0] + b[1]*n[1] + b[2]*n[2]) / (bmag * nmag);
  cosa = std::min(1.0, cosa);
  auto q = [](double v) {   // 1 % log quantization, 0 stays 0
    if (!(v > 0.0)) return 0.0;
    return std::exp(std::round(std::log(v) / 0.01) * 0.01);
  };
  Key k;
  k.te = q(te); k.ti = q(ti); k.ne = q(ne); k.bmag = q(bmag);
  k.alpha_deg = std::round(std::acos(cosa) * 180.0 / PI * 4.0) / 4.0;
  k.mass = mass; k.Z = Z;
  k.upar = (drift_mode == 2 && upar > 0.0) ? q(upar) : 0.0;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  const auto t0 = std::chrono::steady_clock::now();
  Hist h = compute(k);
  cpu_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  return cache.emplace(k, std::move(h)).first->second;
}

SheathIead::Hist SheathIead::compute(const Key &k)
{
  Hist h;
  h.nE = nE; h.nT = nT;
  h.f.assign(static_cast<size_t>(nE) * nT, 0.0);

  // same sheath as the pusher (global pusher ... sheath spatial)
  SheathModels::SheathEmagCoeffs c = SheathModels::sheath_prepare(
      k.te, k.ti, k.ne, k.bmag, k.alpha_deg, mD_amu, 0.0);
  const double d_max = (dmax_override > 0.0) ? dmax_override : c.mps_end_m;
  const double q = k.Z * QE;
  const double qm = q / k.mass;
  const double sig = std::sqrt(QE * k.ti / k.mass);      // Maxwellian at Ti
  const double mD = mD_amu * AMU;
  double u;
  if (k.upar > 0.0) u = k.upar;
  else if (drift_mode == 0) u = std::sqrt(QE * (k.te + k.ti) / (2.0 * mD));
  else u = std::sqrt(QE * (k.te + k.ti) / mD);

  // local frame: wall at z = 0, normal +z, B in the x-z plane into the wall
  const double a = k.alpha_deg * PI / 180.0;
  const double bhat[3] = {std::sin(a), 0.0, -std::cos(a)};
  const double B[3] = {k.bmag * bhat[0], 0.0, k.bmag * bhat[2]};
  const double E0[3] = {0.0, 0.0, 0.0};

  // substep: gyro-phase and presheath-scale resolution. The potential
  // impulse is exact in energy for any step, but the impact ANGLE at
  // grazing incidence needs the L_MPS sheet resolved (gyro-only steps
  // make O3+ at 0.3 deg ~8 deg too oblique); the Debye sheath need not be.
  const double vmax = u + 4.0 * sig + std::sqrt(2.0 * q * c.phi_total_eV / k.mass);
  double dt = (k.bmag > 0.0) ? OMEGA_DT / (qm * k.bmag) : 1e300;
  const double scale = (c.lmps_m > 0.0) ? c.lmps_m : d_max;
  if (vmax > 0.0 && scale > 0.0) dt = std::min(dt, scale_frac * scale / vmax);
  if (!(dt > 0.0) || !std::isfinite(dt)) dt = 1e-12;

  h.emax = k.Z * c.phi_total_eV
         + 0.5 * k.mass * (u + 4.0 * sig) * (u + 4.0 * sig) / QE + 8.0 * k.ti;
  const double rho = (k.bmag > 0.0) ? (u + 3.0 * sig) / (qm * k.bmag) : d_max;
  // launch where the potential has decayed (10 L_MPS + 20 lambda_D, as the
  // SHEATH tracker) plus a gyro-orbit margin along the normal, but never
  // beyond the pusher's d_max: at grazing incidence the sheet is microns
  // thick while d_max is 10 rho_s, and markers started there would mostly
  // drift along the wall
  double d_launch = 10.0 * c.lmps_m + 20.0 * c.lambdaD_m + 2.0 * rho * std::cos(a);
  if (!(d_launch > 0.0) || d_launch > d_max) d_launch = d_max;
  const double z_escape = 2.0 * d_launch + 4.0 * rho;
  // an ion that has not reached the wall after 20 gyro-periods plus 10
  // normal transit times never will in this model (it drifts along the
  // wall); count it as not impacting instead of running to MAX_STEPS
  const double u_n = std::max(u * std::cos(a), 0.1 * sig);
  double t_max = 10.0 * d_launch / u_n;
  if (k.bmag > 0.0) t_max += 20.0 * 2.0 * PI / (qm * k.bmag);
  const long nstep_max = std::min(MAX_STEPS, static_cast<long>(t_max / dt) + 1);
  h.d_launch = d_launch; h.dt = dt; h.nstep_max = nstep_max;

  random->reset(seed, 0, 100);   // same sequence for every histogram: rank-independent
  long nimp = 0, nesc = 0;
  double esum = 0.0, thsum = 0.0;
  for (int m = 0; m < nmarkers; m++) {
    double x[3] = {0.0, 0.0, d_launch};
    double v[3] = {sig * random->gaussian() + u * bhat[0],
                   sig * random->gaussian() + u * bhat[1],
                   sig * random->gaussian() + u * bhat[2]};
    bool impacted = false;
    // one Boris substep of length dtl plus the spatial-sheath impulse
    // (pusher.cpp 3-D block); returns true when the wall was reached
    auto advance = [&](double dtl) -> bool {
      const double z_old = x[2];
      BorisGrid::push_velocity(qm, dtl, E0, B, v);
      x[0] += v[0] * dtl; x[1] += v[1] * dtl; x[2] += v[2] * dtl;
      if (std::min(z_old, x[2]) < d_max) {
        const double phi_old = SheathModels::sheath_phi_at_distance(c, std::max(z_old, 0.0));
        const double phi_new = SheathModels::sheath_phi_at_distance(c, std::max(x[2], 0.0));
        const double dKE = q * (phi_new - phi_old);
        if (dKE != 0.0) {
          const double vn = v[2];
          const double s2 = vn*vn + 2.0 * dKE / k.mass;
          double vn_new;
          if (s2 >= 0.0) {
            vn_new = (vn >= 0.0) ? std::sqrt(s2) : -std::sqrt(s2);
          } else {
            double dd;
            SheathModels::sheath_turning_step(c, q, k.mass, vn, dtl, z_old, vn_new, dd);
            x[2] = z_old + dd;
          }
          v[2] = vn_new;
        }
      }
      return x[2] <= 0.0;
    };
    for (long is = 0; is < nstep_max; is++) {
      const double x0[3] = {x[0], x[1], x[2]};
      const double v0[3] = {v[0], v[1], v[2]};
      if (advance(dt)) {
        // the wall was crossed inside this step: redo it in NSUB finer
        // substeps so the impact velocity (angle) is read at the wall, not
        // up to one gyro-step past it
        x[0] = x0[0]; x[1] = x0[1]; x[2] = x0[2];
        v[0] = v0[0]; v[1] = v0[1]; v[2] = v0[2];
        impacted = false;
        for (int js = 0; js < NSUB && !impacted; js++) impacted = advance(dt / NSUB);
        if (!impacted) impacted = true;   // rounding: the full step did cross
        break;
      }
      if (x[2] > z_escape && v[2] > 0.0) break;
    }
    if (!impacted) { nesc++; continue; }
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double e = 0.5 * k.mass * v2 / QE;
    const double th = std::acos(std::min(1.0, std::max(-1.0, -v[2] / std::sqrt(v2)))) * 180.0 / PI;
    int ie = static_cast<int>(e / h.emax * nE);
    int it = static_cast<int>(th / 90.0 * nT);
    ie = std::min(std::max(ie, 0), nE - 1);
    it = std::min(std::max(it, 0), nT - 1);
    h.f[static_cast<size_t>(ie) * nT + it] += 1.0;
    nimp++; esum += e; thsum += th;
  }
  if (nimp > 0) {
    for (double &w : h.f) w /= nimp;
    h.e_mean = esum / nimp; h.th_mean = thsum / nimp;
  }
  h.nimpact = nimp;
  h.escape_frac = static_cast<double>(nesc) / nmarkers;
  return h;
}
