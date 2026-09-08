/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.

   OpenEdge Phase B (2026-09-08): device twins of the GCAPusher helpers in
   pusher.h (init_from_particle, push_gca, gca_perp_basis, gca_to_particle,
   gca_rhs, larmor_radius, grad_b_length) and of the static
   flux_phase_sample in pusher.cpp. Same math, Kokkos math functions, no
   std:: containers. The RK2/RK4 drivers live in UpdateKokkos::oe_hybrid3d
   because the per-stage field resampling needs the mover's views.
------------------------------------------------------------------------- */

#ifndef SPARTA_GCA_KOKKOS_H
#define SPARTA_GCA_KOKKOS_H

#include "kokkos_type.h"

namespace SPARTA_NS {
namespace GCAKokkos {

constexpr double PHASE_GOLDEN = 0.6180339887498949;   // pusher.cpp GCA_PHASE_GOLDEN
constexpr double TWO_PI = 6.283185307179586;

struct State {
  double X[3];
  double v_par;
  double mu;
};

struct Fields {
  double E[3];
  double B[3];
  double Bmag;
  double gradBmag[3];
  double kappa[3];
  double curl_b[3];
  bool derivs_valid;
  bool e_valid;
  KOKKOS_INLINE_FUNCTION Fields() : Bmag(0.0), derivs_valid(false), e_valid(false) {
    for (int k = 0; k < 3; k++) { E[k] = B[k] = gradBmag[k] = kappa[k] = curl_b[k] = 0.0; }
  }
};

struct Rhs {
  double dXdt[3];
  double dvpar_dt;
};

KOKKOS_INLINE_FUNCTION
State init_from_particle(const double x[3], const double v[3],
                         double mass, double qm, const double B[3])
{
  State s;
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  s.X[0] = x[0]; s.X[1] = x[1]; s.X[2] = x[2];
  if (Bmag > 0.0) {
    const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
    const double Om = qm * Bmag;
    if (Kokkos::fabs(Om) > 0.0) {
      s.X[0] += (v[1]*bhat[2] - v[2]*bhat[1]) / Om;
      s.X[1] += (v[2]*bhat[0] - v[0]*bhat[2]) / Om;
      s.X[2] += (v[0]*bhat[1] - v[1]*bhat[0]) / Om;
    }
    s.v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double vperp2 = v2 - s.v_par * s.v_par;
    const double vperp2_safe = (vperp2 > 0.0) ? vperp2 : 0.0;
    s.mu = mass * vperp2_safe / (2.0 * Bmag);
  } else {
    s.v_par = Kokkos::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    s.mu = 0.0;
  }
  return s;
}

KOKKOS_INLINE_FUNCTION
void push_gca(double qm, double dt, double mass,
              const double E[3], const double B[3], const double gradBmag[3],
              State &state)
{
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) {
    state.v_par += qm * Kokkos::sqrt(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]) * dt;
    state.X[0] += state.v_par * dt;
    return;
  }
  const double invB = 1.0 / Bmag;
  const double invB2 = invB * invB;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};
  const double Omega = qm * Bmag;
  if (Kokkos::fabs(Omega) <= 0.0) return;
  double v_ExB[3];
  v_ExB[0] = (E[1]*B[2] - E[2]*B[1]) * invB2;
  v_ExB[1] = (E[2]*B[0] - E[0]*B[2]) * invB2;
  v_ExB[2] = (E[0]*B[1] - E[1]*B[0]) * invB2;
  const double gradB_coeff = state.mu / (mass * Omega);
  double BxgradB[3];
  BxgradB[0] = B[1]*gradBmag[2] - B[2]*gradBmag[1];
  BxgradB[1] = B[2]*gradBmag[0] - B[0]*gradBmag[2];
  BxgradB[2] = B[0]*gradBmag[1] - B[1]*gradBmag[0];
  double v_gradB[3];
  v_gradB[0] = gradB_coeff * BxgradB[0] * invB;
  v_gradB[1] = gradB_coeff * BxgradB[1] * invB;
  v_gradB[2] = gradB_coeff * BxgradB[2] * invB;
  const double bdotgradB = bhat[0]*gradBmag[0] + bhat[1]*gradBmag[1] + bhat[2]*gradBmag[2];
  const double bdotE = bhat[0]*E[0] + bhat[1]*E[1] + bhat[2]*E[2];
  const double dvpar_dt = -(state.mu / mass) * bdotgradB + qm * bdotE;
  const double v_par_half = state.v_par + 0.5 * dvpar_dt * dt;
  state.X[0] += (v_par_half * bhat[0] + v_ExB[0] + v_gradB[0]) * dt;
  state.X[1] += (v_par_half * bhat[1] + v_ExB[1] + v_gradB[1]) * dt;
  state.X[2] += (v_par_half * bhat[2] + v_ExB[2] + v_gradB[2]) * dt;
  state.v_par += dvpar_dt * dt;
}

KOKKOS_INLINE_FUNCTION
void perp_basis(const double bhat[3], double e1[3], double e2[3])
{
  double ref[3];
  if (Kokkos::fabs(bhat[0]) < 0.9) { ref[0] = 1.0; ref[1] = 0.0; ref[2] = 0.0; }
  else                             { ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0; }
  e1[0] = bhat[1]*ref[2] - bhat[2]*ref[1];
  e1[1] = bhat[2]*ref[0] - bhat[0]*ref[2];
  e1[2] = bhat[0]*ref[1] - bhat[1]*ref[0];
  const double m1 = Kokkos::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
  e1[0] /= m1; e1[1] /= m1; e1[2] /= m1;
  e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
  e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
  e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];
}

KOKKOS_INLINE_FUNCTION
void to_particle(const State &state, const double B[3], double mass, double qm,
                 double rand_uniform, double x[3], double v[3])
{
  x[0] = state.X[0]; x[1] = state.X[1]; x[2] = state.X[2];
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) { v[0] = state.v_par; v[1] = 0.0; v[2] = 0.0; return; }
  const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
  const double vperp = Kokkos::sqrt(2.0 * state.mu * Bmag / mass);
  double e1[3], e2[3];
  perp_basis(bhat, e1, e2);
  const double phi = TWO_PI * rand_uniform;
  const double cp = Kokkos::cos(phi), sp = Kokkos::sin(phi);
  v[0] = state.v_par * bhat[0] + vperp * (cp * e1[0] + sp * e2[0]);
  v[1] = state.v_par * bhat[1] + vperp * (cp * e1[1] + sp * e2[1]);
  v[2] = state.v_par * bhat[2] + vperp * (cp * e1[2] + sp * e2[2]);
  const double Om = qm * Bmag;
  if (Kokkos::fabs(Om) > 0.0) {
    x[0] -= (v[1]*bhat[2] - v[2]*bhat[1]) / Om;
    x[1] -= (v[2]*bhat[0] - v[0]*bhat[2]) / Om;
    x[2] -= (v[0]*bhat[1] - v[1]*bhat[0]) / Om;
  }
}

KOKKOS_INLINE_FUNCTION
Rhs rhs(double qm, double mass, double v_par, double mu, const Fields &F)
{
  Rhs r;
  const double *E = F.E; const double *B = F.B; const double Bmag = F.Bmag;
  if (Bmag <= 0.0) { r.dXdt[0] = r.dXdt[1] = r.dXdt[2] = 0.0; r.dvpar_dt = 0.0; return r; }
  const double invB = 1.0 / Bmag;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};
  const double Omega = qm * Bmag;
  const double mvpar_over_q = v_par / qm;
  double Bstar[3];
  Bstar[0] = B[0] + mvpar_over_q * F.curl_b[0];
  Bstar[1] = B[1] + mvpar_over_q * F.curl_b[1];
  Bstar[2] = B[2] + mvpar_over_q * F.curl_b[2];
  const double Bstar_par = bhat[0]*Bstar[0] + bhat[1]*Bstar[1] + bhat[2]*Bstar[2];
  if (Kokkos::fabs(Bstar_par) < 1.0e-30) { r.dXdt[0] = r.dXdt[1] = r.dXdt[2] = 0.0; r.dvpar_dt = 0.0; return r; }
  const double invBstar_par = 1.0 / Bstar_par;
  double ExB[3];
  ExB[0] = E[1]*bhat[2] - E[2]*bhat[1];
  ExB[1] = E[2]*bhat[0] - E[0]*bhat[2];
  ExB[2] = E[0]*bhat[1] - E[1]*bhat[0];
  const double gradB_coeff = mu / (mass * Omega);
  double BxgradB[3];
  BxgradB[0] = B[1]*F.gradBmag[2] - B[2]*F.gradBmag[1];
  BxgradB[1] = B[2]*F.gradBmag[0] - B[0]*F.gradBmag[2];
  BxgradB[2] = B[0]*F.gradBmag[1] - B[1]*F.gradBmag[0];
  for (int k = 0; k < 3; k++)
    r.dXdt[k] = invBstar_par * (v_par * Bstar[k] + ExB[k] + gradB_coeff * BxgradB[k]);
  double force[3];
  for (int k = 0; k < 3; k++) force[k] = -(mu / mass) * F.gradBmag[k] + qm * E[k];
  r.dvpar_dt = invBstar_par * (Bstar[0]*force[0] + Bstar[1]*force[1] + Bstar[2]*force[2]);
  return r;
}

KOKKOS_INLINE_FUNCTION
double larmor_radius(double v_perp, double qm_abs, double Bmag)
{
  if (qm_abs <= 0.0 || Bmag <= 0.0) return 1.0e20;
  return v_perp / (qm_abs * Bmag);
}

KOKKOS_INLINE_FUNCTION
double grad_b_length(double Bmag, double gradBmag_magnitude)
{
  if (gradBmag_magnitude <= 0.0) return 1.0e20;
  return Bmag / gradBmag_magnitude;
}

// pusher.cpp flux_phase_sample: rejection sampling of the gyrophase from
// the first-passage flux weight max(-vn(phi),0), deterministic in u0
KOKKOS_INLINE_FUNCTION
double flux_phase_sample(double a, double cx, double cy, double u0)
{
  const double cmag = Kokkos::sqrt(cx*cx + cy*cy);
  const double wmax = -a + cmag;
  if (wmax <= 0.0) return u0;
  double u = u0;
  for (int trial = 0; trial < 32; trial++) {
    const double ph = u * TWO_PI;
    const double w = -(a + cx*Kokkos::cos(ph) + cy*Kokkos::sin(ph));
    const double ua = Kokkos::fmod(u * 971.0 + 0.372549, 1.0);
    if (w > 0.0 && ua * wmax <= w) return u;
    u = Kokkos::fmod(u + PHASE_GOLDEN, 1.0);
  }
  return u0;
}

}  // namespace GCAKokkos
}  // namespace SPARTA_NS

#endif
