/* ----------------------------------------------------------------------
   OpenEdge Guiding Center Approximation (GCA) pusher
   Hybrid Boris/GCA scheme inspired by ERO2.0.
   When the Larmor radius is small compared to the B gradient scale length,
   GCA tracks only the drift motion (averaging over gyration), eliminating
   the need for expensive Boris subcycling.

   References:
     - Littlejohn, J. Plasma Phys. 29 (1983) 111
     - Borodkina & Igitkhanov, PSI ERO2.0 approach
------------------------------------------------------------------------- */

#ifndef SPARTA_GCA_PUSHER_H
#define SPARTA_GCA_PUSHER_H

#include <cmath>

namespace SPARTA_NS {
namespace GCAPusher {

struct GCAState {
  double X[3];      // guiding center position (Cartesian)
  double v_par;     // parallel velocity (along b)
  double mu;        // magnetic moment (adiabatic invariant) [J/T]
};

/* ---------------------------------------------------------------------- */
// Initialize GCA state from full particle state
// B[3] is the local magnetic field in Cartesian coords
/* ---------------------------------------------------------------------- */
inline GCAState init_from_particle(const double x[3], const double v[3],
                                   double mass, const double B[3])
{
  GCAState s;
  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

  s.X[0] = x[0];
  s.X[1] = x[1];
  s.X[2] = x[2];

  if (Bmag > 0.0) {
    const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};
    s.v_par = v[0]*bhat[0] + v[1]*bhat[1] + v[2]*bhat[2];

    // v_perp^2 = |v|^2 - v_par^2
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double vperp2 = v2 - s.v_par * s.v_par;
    const double vperp2_safe = (vperp2 > 0.0) ? vperp2 : 0.0;

    // mu = m * v_perp^2 / (2 * B)
    s.mu = mass * vperp2_safe / (2.0 * Bmag);
  } else {
    s.v_par = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    s.mu = 0.0;
  }
  return s;
}

/* ---------------------------------------------------------------------- */
// Single GCA timestep: advance guiding center position and v_par
//
// qm       = charge * e / mass  (charge-to-mass ratio in SI)
// dt       = timestep
// mass     = particle mass [kg]
// E[3]     = electric field at guiding center [V/m] (Cartesian)
// B[3]     = magnetic field [T] (Cartesian)
// gradBmag[3] = gradient of |B| [T/m] (Cartesian)
// state    = GCA state (modified in place)
//
// mu is conserved (not updated).
/* ---------------------------------------------------------------------- */
inline void push_gca(double qm, double dt, double mass,
                     const double E[3], const double B[3],
                     const double gradBmag[3],
                     GCAState &state)
{
  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) {
    // No B-field: pure E acceleration (fallback)
    state.v_par += qm * std::sqrt(E[0]*E[0] + E[1]*E[1] + E[2]*E[2]) * dt;
    state.X[0] += state.v_par * dt;
    return;
  }

  const double invB = 1.0 / Bmag;
  const double invB2 = invB * invB;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};

  // Cyclotron frequency
  const double Omega = qm * Bmag;  // signed
  const double Omega_abs = std::fabs(Omega);
  if (Omega_abs <= 0.0) return;  // uncharged particle shouldn't be here

  // --- Drift velocities ---

  // 1. E x B drift: v_ExB = (E x B) / B^2
  double v_ExB[3];
  v_ExB[0] = (E[1]*B[2] - E[2]*B[1]) * invB2;
  v_ExB[1] = (E[2]*B[0] - E[0]*B[2]) * invB2;
  v_ExB[2] = (E[0]*B[1] - E[1]*B[0]) * invB2;

  // 2. Grad-B drift: v_gradB = (mu / (m * Omega)) * (B x gradB) / B
  //    = (mu / (q * B^2)) * (B x gradBmag)
  const double gradB_coeff = state.mu / (mass * Omega);  // mu/(m*Omega) = mu*e/(q_C * B)
  double BxgradB[3];
  BxgradB[0] = B[1]*gradBmag[2] - B[2]*gradBmag[1];
  BxgradB[1] = B[2]*gradBmag[0] - B[0]*gradBmag[2];
  BxgradB[2] = B[0]*gradBmag[1] - B[1]*gradBmag[0];

  double v_gradB[3];
  v_gradB[0] = gradB_coeff * BxgradB[0] * invB;
  v_gradB[1] = gradB_coeff * BxgradB[1] * invB;
  v_gradB[2] = gradB_coeff * BxgradB[2] * invB;

  // 3. Curvature drift: v_curv = (v_par^2 / Omega) * (b x kappa)
  //    where kappa = (b . grad)b ≈ -grad_perp(B)/B for low-beta plasma
  //    grad_perp(B) = gradBmag - (b . gradBmag) * b
  const double bdotgradB = bhat[0]*gradBmag[0] + bhat[1]*gradBmag[1] + bhat[2]*gradBmag[2];
  double gradBperp[3];
  gradBperp[0] = gradBmag[0] - bdotgradB * bhat[0];
  gradBperp[1] = gradBmag[1] - bdotgradB * bhat[1];
  gradBperp[2] = gradBmag[2] - bdotgradB * bhat[2];

  // kappa ≈ -gradBperp / B
  double kappa[3];
  kappa[0] = -gradBperp[0] * invB;
  kappa[1] = -gradBperp[1] * invB;
  kappa[2] = -gradBperp[2] * invB;

  // b x kappa
  double bxkappa[3];
  bxkappa[0] = bhat[1]*kappa[2] - bhat[2]*kappa[1];
  bxkappa[1] = bhat[2]*kappa[0] - bhat[0]*kappa[2];
  bxkappa[2] = bhat[0]*kappa[1] - bhat[1]*kappa[0];

  const double curv_coeff = state.v_par * state.v_par / Omega;
  double v_curv[3];
  v_curv[0] = curv_coeff * bxkappa[0];
  v_curv[1] = curv_coeff * bxkappa[1];
  v_curv[2] = curv_coeff * bxkappa[2];

  // --- Parallel acceleration ---
  // dv_par/dt = -(mu/m) * (b . gradBmag) + (q/m) * (b . E)
  const double bdotE = bhat[0]*E[0] + bhat[1]*E[1] + bhat[2]*E[2];
  const double dvpar_dt = -(state.mu / mass) * bdotgradB + qm * bdotE;

  // --- Integration (leapfrog / Euler) ---
  // Half-step v_par for 2nd-order accuracy
  const double v_par_half = state.v_par + 0.5 * dvpar_dt * dt;

  // Total guiding center velocity
  double dXdt[3];
  dXdt[0] = v_par_half * bhat[0] + v_ExB[0] + v_gradB[0] + v_curv[0];
  dXdt[1] = v_par_half * bhat[1] + v_ExB[1] + v_gradB[1] + v_curv[1];
  dXdt[2] = v_par_half * bhat[2] + v_ExB[2] + v_gradB[2] + v_curv[2];

  // Advance position
  state.X[0] += dXdt[0] * dt;
  state.X[1] += dXdt[1] * dt;
  state.X[2] += dXdt[2] * dt;

  // Full-step v_par
  state.v_par += dvpar_dt * dt;
}

/* ---------------------------------------------------------------------- */
// Convert GCA state back to full particle velocity
// Reconstructs v_perp with a random gyrophase angle.
// Requires a uniform random number phi in [0, 2*pi).
/* ---------------------------------------------------------------------- */
inline void gca_to_particle(const GCAState &state, const double B[3],
                            double mass, double rand_uniform,
                            double x[3], double v[3])
{
  x[0] = state.X[0];
  x[1] = state.X[1];
  x[2] = state.X[2];

  const double Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag <= 0.0) {
    v[0] = state.v_par;
    v[1] = 0.0;
    v[2] = 0.0;
    return;
  }

  const double bhat[3] = {B[0]/Bmag, B[1]/Bmag, B[2]/Bmag};

  // v_perp from mu: v_perp = sqrt(2 * mu * B / m)
  const double vperp = std::sqrt(2.0 * state.mu * Bmag / mass);

  // Build two perpendicular unit vectors e1, e2 to bhat
  double e1[3], e2[3];
  // Pick a vector not parallel to bhat
  double ref[3];
  if (std::fabs(bhat[0]) < 0.9) {
    ref[0] = 1.0; ref[1] = 0.0; ref[2] = 0.0;
  } else {
    ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0;
  }
  // e1 = bhat x ref (then normalize)
  e1[0] = bhat[1]*ref[2] - bhat[2]*ref[1];
  e1[1] = bhat[2]*ref[0] - bhat[0]*ref[2];
  e1[2] = bhat[0]*ref[1] - bhat[1]*ref[0];
  const double e1mag = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
  e1[0] /= e1mag; e1[1] /= e1mag; e1[2] /= e1mag;

  // e2 = bhat x e1
  e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
  e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
  e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];

  // Random gyrophase
  const double phi = 2.0 * M_PI * rand_uniform;
  const double cp = std::cos(phi), sp = std::sin(phi);

  v[0] = state.v_par * bhat[0] + vperp * (cp * e1[0] + sp * e2[0]);
  v[1] = state.v_par * bhat[1] + vperp * (cp * e1[1] + sp * e2[1]);
  v[2] = state.v_par * bhat[2] + vperp * (cp * e1[2] + sp * e2[2]);
}

/* ---------------------------------------------------------------------- */
// Right-hand side of the full Littlejohn GCA equations with B* correction.
//
// B* = B + (m v_par / (q B)) curl(b̂)
// B*_par = b̂ · B*
//
// dX/dt = (v_par B* + (1/B*_par)[E×b̂ + (μ/qB)(B×∇B)]) / B*_par ... wait
// Simplified: we compute the full GCA velocity including all drifts.
//
// Inputs in Cartesian (XYZ).
// kappa[3], curl_b[3], gradBmag[3] are exact from equilibrium.
/* ---------------------------------------------------------------------- */

struct GCARhs {
  double dXdt[3];   // guiding center velocity
  double dvpar_dt;  // parallel acceleration
};

inline GCARhs gca_rhs(double qm, double mass, double v_par, double mu,
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
  const double invB2 = invB * invB;
  const double bhat[3] = {B[0]*invB, B[1]*invB, B[2]*invB};

  const double Omega = qm * Bmag;  // signed cyclotron frequency q/m * B

  // B* = B + (v_par / Omega) * curl(b̂)   [since m*v_par/(q*B) = v_par/Omega]
  const double vpar_over_Omega = v_par / Omega;
  double Bstar[3];
  Bstar[0] = B[0] + vpar_over_Omega * curl_b[0];
  Bstar[1] = B[1] + vpar_over_Omega * curl_b[1];
  Bstar[2] = B[2] + vpar_over_Omega * curl_b[2];

  // B*_par = b̂ · B*
  const double Bstar_par = bhat[0]*Bstar[0] + bhat[1]*Bstar[1] + bhat[2]*Bstar[2];
  if (std::fabs(Bstar_par) < 1.0e-30) {
    rhs.dXdt[0] = rhs.dXdt[1] = rhs.dXdt[2] = 0.0;
    rhs.dvpar_dt = 0.0;
    return rhs;
  }
  const double invBstar_par = 1.0 / Bstar_par;

  // E × b̂ drift
  double ExB[3];
  ExB[0] = E[1]*bhat[2] - E[2]*bhat[1];
  ExB[1] = E[2]*bhat[0] - E[0]*bhat[2];
  ExB[2] = E[0]*bhat[1] - E[1]*bhat[0];

  // (μ / (m Ω)) * (B × ∇|B|) / B = (μ / (q B²)) * (B × ∇|B|)
  const double gradB_coeff = mu / (mass * Omega);
  double BxgradB[3];
  BxgradB[0] = B[1]*gradBmag[2] - B[2]*gradBmag[1];
  BxgradB[1] = B[2]*gradBmag[0] - B[0]*gradBmag[2];
  BxgradB[2] = B[0]*gradBmag[1] - B[1]*gradBmag[0];

  // dX/dt = (1/B*_par) * [v_par * B* + E×b̂ + (μ/(mΩ)) * (B×∇B)/B * B_par_correction]
  // More precisely: dX/dt = v_par * B*/B*_par + (1/B*_par) * [E×b̂ + (μ/(mΩ))(B×∇|B|)/B]
  // where the last term reduces to the grad-B drift.
  rhs.dXdt[0] = invBstar_par * (v_par * Bstar[0] + ExB[0] + gradB_coeff * BxgradB[0] * invB);
  rhs.dXdt[1] = invBstar_par * (v_par * Bstar[1] + ExB[1] + gradB_coeff * BxgradB[1] * invB);
  rhs.dXdt[2] = invBstar_par * (v_par * Bstar[2] + ExB[2] + gradB_coeff * BxgradB[2] * invB);

  // dv_par/dt = (B*/B*_par) · [-(μ/m)∇|B| + (q/m)E]
  // = (1/B*_par) * B* · [-(μ/m)∇|B| + qm*E]
  double force[3];
  force[0] = -(mu / mass) * gradBmag[0] + qm * E[0];
  force[1] = -(mu / mass) * gradBmag[1] + qm * E[1];
  force[2] = -(mu / mass) * gradBmag[2] + qm * E[2];

  rhs.dvpar_dt = invBstar_par * (Bstar[0]*force[0] + Bstar[1]*force[1] + Bstar[2]*force[2]);

  return rhs;
}

/* ---------------------------------------------------------------------- */
// RK4 integrator for the full GCA equations with B* correction.
//
// E, B, gradBmag, kappa, curl_b are assumed constant over dt
// (valid since fields are cell-based and don't change within a timestep).
// Only X and v_par evolve through the RK stages.
/* ---------------------------------------------------------------------- */

inline void push_gca_rk4(double qm, double dt, double mass,
                          const double E[3], const double B[3],
                          double Bmag, const double gradBmag[3],
                          const double kappa[3], const double curl_b[3],
                          GCAState &state)
{
  // y = (X[0], X[1], X[2], v_par) — 4-component state
  double y[4] = {state.X[0], state.X[1], state.X[2], state.v_par};

  auto eval_rhs = [&](const double yy[4]) -> GCARhs {
    // Fields are constant over the cell, so just use the same E, B, etc.
    return gca_rhs(qm, mass, yy[3], state.mu, E, B, Bmag, gradBmag, kappa, curl_b);
  };

  // k1
  GCARhs r1 = eval_rhs(y);
  double k1[4] = {dt * r1.dXdt[0], dt * r1.dXdt[1], dt * r1.dXdt[2], dt * r1.dvpar_dt};

  // k2
  double y2[4] = {y[0]+0.5*k1[0], y[1]+0.5*k1[1], y[2]+0.5*k1[2], y[3]+0.5*k1[3]};
  GCARhs r2 = eval_rhs(y2);
  double k2[4] = {dt * r2.dXdt[0], dt * r2.dXdt[1], dt * r2.dXdt[2], dt * r2.dvpar_dt};

  // k3
  double y3[4] = {y[0]+0.5*k2[0], y[1]+0.5*k2[1], y[2]+0.5*k2[2], y[3]+0.5*k2[3]};
  GCARhs r3 = eval_rhs(y3);
  double k3[4] = {dt * r3.dXdt[0], dt * r3.dXdt[1], dt * r3.dXdt[2], dt * r3.dvpar_dt};

  // k4
  double y4[4] = {y[0]+k3[0], y[1]+k3[1], y[2]+k3[2], y[3]+k3[3]};
  GCARhs r4 = eval_rhs(y4);
  double k4[4] = {dt * r4.dXdt[0], dt * r4.dXdt[1], dt * r4.dXdt[2], dt * r4.dvpar_dt};

  // Combine: y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
  for (int i = 0; i < 3; ++i)
    state.X[i] = y[i] + (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
  state.v_par = y[3] + (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) / 6.0;
}

/* ---------------------------------------------------------------------- */
// Compute Larmor radius: rho_L = m * v_perp / (|Z| * e * B)
// Here we use qm = |Z|*e/m, so rho_L = v_perp / (|qm| * B)
/* ---------------------------------------------------------------------- */
inline double larmor_radius(double v_perp, double qm_abs, double Bmag)
{
  if (qm_abs <= 0.0 || Bmag <= 0.0) return 1.0e20;
  return v_perp / (qm_abs * Bmag);
}

/* ---------------------------------------------------------------------- */
// Compute characteristic B gradient length: L_B = B / |grad B|
/* ---------------------------------------------------------------------- */
inline double grad_b_length(double Bmag, double gradBmag_magnitude)
{
  if (gradBmag_magnitude <= 0.0) return 1.0e20;
  return Bmag / gradBmag_magnitude;
}

}  // namespace GCAPusher
}  // namespace SPARTA_NS

#endif  // SPARTA_GCA_PUSHER_H
