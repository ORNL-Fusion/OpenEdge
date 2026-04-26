/* ----------------------------------------------------------------------
   OpenEdge unified pusher math — Kokkos device-callable kernels.

   Single header containing both pusher kernels, mirroring src/OPENEDGE/pusher.h
   on the GPU side:
     BorisGridKokkos::push_velocity   — Boris kick-rotate-kick.
     BorisGridKokkos::read_field      — read 3-component field from a 2D view.
     GCAPusherKokkos::push_gca        — simplified GCA (no curvature term).
     GCAPusherKokkos::gca_rhs         — full Littlejohn RHS with B* correction.
     GCAPusherKokkos::push_gca_rk4    — RK4 integrator on top of gca_rhs.
     GCAPusherKokkos::init_from_particle / gca_to_particle.

   Phase-1 refactor (consolidated from boris_grid_kokkos.h + gca_pusher_kokkos.h).
   The two old headers are gone — include this one instead.

   References:
     Boris, J.P., 4th Conf. Numerical Simulation of Plasmas, NRL (1970).
     Littlejohn, R.G., J. Plasma Phys. 29 (1983) 111.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifndef SPARTA_PUSHER_KOKKOS_H
#define SPARTA_PUSHER_KOKKOS_H

#include "Kokkos_Core.hpp"
#include "kokkos_type.h"
#include <cmath>

namespace SPARTA_NS {

/* ===================================================================
   Boris kick-rotate-kick + view field extraction.
   =================================================================== */
namespace BorisGridKokkos {

KOKKOS_INLINE_FUNCTION
void read_field(const DAT::t_float_2d_lr &d_arr, int idx, double out[3])
{
  out[0] = d_arr(idx, 0);
  out[1] = d_arr(idx, 1);
  out[2] = d_arr(idx, 2);
}

KOKKOS_INLINE_FUNCTION
void push_velocity(double qm, double dt,
                   const double E[3], const double B[3],
                   double v[3])
{
  // Half E-kick
  double vminus[3] = {
    v[0] + qm * E[0] * 0.5 * dt,
    v[1] + qm * E[1] * 0.5 * dt,
    v[2] + qm * E[2] * 0.5 * dt
  };

  // Rotation vectors
  const double t[3] = {
    qm * B[0] * 0.5 * dt,
    qm * B[1] * 0.5 * dt,
    qm * B[2] * 0.5 * dt
  };
  const double t2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
  const double s[3] = {
    2.0 * t[0] / (1.0 + t2),
    2.0 * t[1] / (1.0 + t2),
    2.0 * t[2] / (1.0 + t2)
  };

  // vminus x t
  double vprime[3];
  vprime[0] = vminus[1]*t[2] - vminus[2]*t[1] + vminus[0];
  vprime[1] = vminus[2]*t[0] - vminus[0]*t[2] + vminus[1];
  vprime[2] = vminus[0]*t[1] - vminus[1]*t[0] + vminus[2];

  // vprime x s
  double vplus[3];
  vplus[0] = vprime[1]*s[2] - vprime[2]*s[1] + vminus[0];
  vplus[1] = vprime[2]*s[0] - vprime[0]*s[2] + vminus[1];
  vplus[2] = vprime[0]*s[1] - vprime[1]*s[0] + vminus[2];

  // Second half E-kick
  v[0] = vplus[0] + qm * E[0] * 0.5 * dt;
  v[1] = vplus[1] + qm * E[1] * 0.5 * dt;
  v[2] = vplus[2] + qm * E[2] * 0.5 * dt;
}

}  // namespace BorisGridKokkos

/* ===================================================================
   Guiding-Center Approximation (GCA) device-callable kernels.
   =================================================================== */
namespace GCAPusherKokkos {

struct GCAState {
  double X[3];
  double v_par;
  double mu;
};

KOKKOS_INLINE_FUNCTION
GCAState init_from_particle(const double x[3], const double v[3],
                             double mass, const double B[3])
{
  GCAState s;
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

  s.X[0] = x[0]; s.X[1] = x[1]; s.X[2] = x[2];

  if (Bmag > 0.0) {
    const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
    s.v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double vperp2 = Kokkos::fmax(v2 - s.v_par * s.v_par, 0.0);
    s.mu = mass * vperp2 / (2.0 * Bmag);
  } else {
    s.v_par = Kokkos::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    s.mu = 0.0;
  }
  return s;
}

struct GCARhs {
  double dXdt[3];
  double dvpar_dt;
};

KOKKOS_INLINE_FUNCTION
GCARhs gca_rhs(double qm, double mass, double v_par, double mu,
                const double E[3], const double B[3],
                double Bmag, const double gradBmag[3],
                const double kappa[3], const double curl_b[3])
{
  GCARhs rhs;
  if (Bmag <= 0.0) {
    rhs.dXdt[0] = rhs.dXdt[1] = rhs.dXdt[2] = 0.0;
    rhs.dvpar_dt = 0.0;
    return rhs;
  }

  const double invB = 1.0 / Bmag;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};
  const double Omega = qm * Bmag;     // signed cyclotron frequency q/m * B
  // Littlejohn B* = B + (m v_par / q) * curl(b̂); v_par/qm has units T·m.
  const double mvpar_over_q = v_par / qm;

  double Bstar[3];
  Bstar[0] = B[0] + mvpar_over_q * curl_b[0];
  Bstar[1] = B[1] + mvpar_over_q * curl_b[1];
  Bstar[2] = B[2] + mvpar_over_q * curl_b[2];

  const double Bstar_par = bhat[0]*Bstar[0] + bhat[1]*Bstar[1] + bhat[2]*Bstar[2];
  if (Kokkos::fabs(Bstar_par) < 1.0e-30) {
    rhs.dXdt[0] = rhs.dXdt[1] = rhs.dXdt[2] = 0.0;
    rhs.dvpar_dt = 0.0;
    return rhs;
  }
  const double invBstar_par = 1.0 / Bstar_par;

  // E x bhat
  double ExB[3];
  ExB[0] = E[1]*bhat[2] - E[2]*bhat[1];
  ExB[1] = E[2]*bhat[0] - E[0]*bhat[2];
  ExB[2] = E[0]*bhat[1] - E[1]*bhat[0];

  // grad-B drift
  const double gradB_coeff = mu / (mass * Omega);
  double BxgradB[3];
  BxgradB[0] = B[1]*gradBmag[2] - B[2]*gradBmag[1];
  BxgradB[1] = B[2]*gradBmag[0] - B[0]*gradBmag[2];
  BxgradB[2] = B[0]*gradBmag[1] - B[1]*gradBmag[0];

  rhs.dXdt[0] = invBstar_par * (v_par * Bstar[0] + ExB[0] + gradB_coeff * BxgradB[0] * invB);
  rhs.dXdt[1] = invBstar_par * (v_par * Bstar[1] + ExB[1] + gradB_coeff * BxgradB[1] * invB);
  rhs.dXdt[2] = invBstar_par * (v_par * Bstar[2] + ExB[2] + gradB_coeff * BxgradB[2] * invB);

  double force[3];
  force[0] = -(mu / mass) * gradBmag[0] + qm * E[0];
  force[1] = -(mu / mass) * gradBmag[1] + qm * E[1];
  force[2] = -(mu / mass) * gradBmag[2] + qm * E[2];

  rhs.dvpar_dt = invBstar_par * (Bstar[0]*force[0] + Bstar[1]*force[1] + Bstar[2]*force[2]);
  return rhs;
}

KOKKOS_INLINE_FUNCTION
void push_gca(double qm, double dt, double mass,
              const double E[3], const double B[3],
              const double gradBmag[3], GCAState &state)
{
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) {
    state.v_par += qm * Kokkos::sqrt(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]) * dt;
    state.X[0] += state.v_par * dt;
    return;
  }

  const double invB = 1.0 / Bmag;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};
  const double Omega = qm * Bmag;
  const double Omega_abs = Kokkos::fabs(Omega);
  if (Omega_abs <= 0.0) return;

  // E x B drift
  const double invB2 = invB * invB;
  double v_ExB[3];
  v_ExB[0] = (E[1]*B[2] - E[2]*B[1]) * invB2;
  v_ExB[1] = (E[2]*B[0] - E[0]*B[2]) * invB2;
  v_ExB[2] = (E[0]*B[1] - E[1]*B[0]) * invB2;

  // grad-B drift
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
void push_gca_rk4(double qm, double dt, double mass,
                   const double E[3], const double B[3],
                   double Bmag, const double gradBmag[3],
                   const double kappa[3], const double curl_b[3],
                   GCAState &state)
{
  double y[4] = {state.X[0], state.X[1], state.X[2], state.v_par};

  // k1
  GCARhs r1 = gca_rhs(qm, mass, y[3], state.mu, E, B, Bmag, gradBmag, kappa, curl_b);
  double k1[4] = {dt*r1.dXdt[0], dt*r1.dXdt[1], dt*r1.dXdt[2], dt*r1.dvpar_dt};

  // k2
  double y2[4] = {y[0]+0.5*k1[0], y[1]+0.5*k1[1], y[2]+0.5*k1[2], y[3]+0.5*k1[3]};
  GCARhs r2 = gca_rhs(qm, mass, y2[3], state.mu, E, B, Bmag, gradBmag, kappa, curl_b);
  double k2[4] = {dt*r2.dXdt[0], dt*r2.dXdt[1], dt*r2.dXdt[2], dt*r2.dvpar_dt};

  // k3
  double y3[4] = {y[0]+0.5*k2[0], y[1]+0.5*k2[1], y[2]+0.5*k2[2], y[3]+0.5*k2[3]};
  GCARhs r3 = gca_rhs(qm, mass, y3[3], state.mu, E, B, Bmag, gradBmag, kappa, curl_b);
  double k3[4] = {dt*r3.dXdt[0], dt*r3.dXdt[1], dt*r3.dXdt[2], dt*r3.dvpar_dt};

  // k4
  double y4[4] = {y[0]+k3[0], y[1]+k3[1], y[2]+k3[2], y[3]+k3[3]};
  GCARhs r4 = gca_rhs(qm, mass, y4[3], state.mu, E, B, Bmag, gradBmag, kappa, curl_b);
  double k4[4] = {dt*r4.dXdt[0], dt*r4.dXdt[1], dt*r4.dXdt[2], dt*r4.dvpar_dt};

  for (int i = 0; i < 3; ++i)
    state.X[i] = y[i] + (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
  state.v_par = y[3] + (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) / 6.0;
}

KOKKOS_INLINE_FUNCTION
void gca_to_particle(const GCAState &state, const double B[3],
                     double mass, double rand_uniform,
                     double x[3], double v[3])
{
  x[0] = state.X[0]; x[1] = state.X[1]; x[2] = state.X[2];

  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) {
    v[0] = state.v_par; v[1] = 0.0; v[2] = 0.0;
    return;
  }

  const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
  const double vperp = Kokkos::sqrt(2.0 * state.mu * Bmag / mass);

  // Build perpendicular basis
  double e1[3], e2[3];
  double ref[3];
  if (Kokkos::fabs(bhat[0]) < 0.9) {
    ref[0] = 1.0; ref[1] = 0.0; ref[2] = 0.0;
  } else {
    ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0;
  }
  e1[0] = bhat[1]*ref[2] - bhat[2]*ref[1];
  e1[1] = bhat[2]*ref[0] - bhat[0]*ref[2];
  e1[2] = bhat[0]*ref[1] - bhat[1]*ref[0];
  const double e1mag = Kokkos::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
  e1[0] /= e1mag; e1[1] /= e1mag; e1[2] /= e1mag;

  e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
  e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
  e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];

  const double PI = 3.14159265358979323846;
  const double phi = 2.0 * PI * rand_uniform;
  const double cp = Kokkos::cos(phi), sp = Kokkos::sin(phi);

  v[0] = state.v_par * bhat[0] + vperp * (cp * e1[0] + sp * e2[0]);
  v[1] = state.v_par * bhat[1] + vperp * (cp * e1[1] + sp * e2[1]);
  v[2] = state.v_par * bhat[2] + vperp * (cp * e1[2] + sp * e2[2]);
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

}  // namespace GCAPusherKokkos

/* ===================================================================
   Equilibrium-based B-field point-query on device.

   Mirrors the equilibrium branch of
       ComputePlasmaFields::query_bfield_at_point() (CPU)
   for use inside the Kokkos Boris/GCA pushers. Computes B at an
   arbitrary particle position (xyz) by bilinear-interpolating
   psi(R,Z) gradients off the equilibrium grid:

     B_R   = -(1/R) ∂psi/∂Z
     B_Z   =  (1/R) ∂psi/∂R
     B_phi =  btf · rtf / R

   Only the three B components are returned (Phase A). Gradients of
   |B| (needed for grad-B / curvature drifts) come in Phase C with the
   GCA-on-device port.

   Slot conventions handled the same as openedge_geom.h:
     axisymmetric == 1 :  x = Z (axis), y = R (radial)
     dim == 2          :  x = R, y = Z (legacy 2D Cartesian)
     dim == 3          :  R = sqrt(x^2 + y^2), Z = z (3D Cartesian)
   The output B[3] is delivered in the same SPARTA slot order as the
   input position (so Boris can apply v×B directly without an extra
   rotation):
     axisymmetric : (B_Z, B_R, B_phi)
     2D Cartesian : (B_R, B_Z, B_phi)
     3D Cartesian : (B_R cos φ - B_phi sin φ,  B_R sin φ + B_phi cos φ,  B_Z)
   =================================================================== */
namespace EquilibriumKokkos {

KOKKOS_INLINE_FUNCTION
void query_bfield_at_point(
    const double xyz[3], int dim, int axisymmetric,
    const Kokkos::View<double*,  DeviceType> &equ_r,
    const Kokkos::View<double*,  DeviceType> &equ_z,
    const Kokkos::View<double**, DeviceType> &equ_psi,
    double btf, double rtf, int jm, int km,
    double B[3])
{
  B[0] = 0.0; B[1] = 0.0; B[2] = 0.0;
  if (jm < 3 || km < 3) return;

  // SPARTA-slot → cylindrical (R, Z)
  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }
  if (R < 1.0e-10) return;

  const double dr = equ_r(1) - equ_r(0);
  const double dz = equ_z(1) - equ_z(0);
  if (dr <= 0.0 || dz <= 0.0) return;

  const double fj = (R - equ_r(0)) / dr;
  const double fk = (Z - equ_z(0)) / dz;
  int jc = static_cast<int>(Kokkos::round(fj));
  int kc = static_cast<int>(Kokkos::round(fk));
  if (jc < 1) jc = 1;  else if (jc > jm - 2) jc = jm - 2;
  if (kc < 1) kc = 1;  else if (kc > km - 2) kc = km - 2;

  const double dR = equ_r(jc+1) - equ_r(jc-1);
  const double dZ = equ_z(kc+1) - equ_z(kc-1);
  const double dpsi_dR = (equ_psi(kc, jc+1) - equ_psi(kc, jc-1)) / dR;
  const double dpsi_dZ = (equ_psi(kc+1, jc) - equ_psi(kc-1, jc)) / dZ;

  const double invR = 1.0 / R;
  const double bR   = -dpsi_dZ * invR;
  const double bZ   =  dpsi_dR * invR;
  const double bphi = (btf * rtf) * invR;

  // Pack into SPARTA slot order
  if (axisymmetric)  { B[0] = bZ;  B[1] = bR;  B[2] = bphi; }
  else if (dim == 2) { B[0] = bR;  B[1] = bZ;  B[2] = bphi; }
  else {
    const double rxy = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
    double cphi = 1.0, sphi = 0.0;
    if (rxy > 1.0e-20) { cphi = xyz[0] / rxy; sphi = xyz[1] / rxy; }
    B[0] = bR * cphi - bphi * sphi;
    B[1] = bR * sphi + bphi * cphi;
    B[2] = bZ;
  }
}

}  // namespace EquilibriumKokkos
}  // namespace SPARTA_NS

#endif  // SPARTA_PUSHER_KOKKOS_H
