/* ----------------------------------------------------------------------
   OpenEdge unified pusher math — Kokkos device-callable kernels.

   Single header containing the device pusher kernels, mirroring
   src/OPENEDGE/pusher.h on the GPU side:
     BorisGridKokkos::push_velocity   — Boris kick-rotate-kick.
     BorisGridKokkos::read_field      — read 3-component field from a 2D view.
     EquilibriumKokkos / MeshKokkos   — B/E/scalar point-queries.

   The GCAPusherKokkos namespace (old device GCA integrator) was removed
   2026-08-26 together with UpdateKokkos::oe_hybrid3d: it encoded a CPU
   hybrid-pusher generation superseded by the Boris shell / trial-replay
   design in pusher.cpp. Hybrid/GCA mode now errors out under Kokkos.

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
    const DAT::t_float_1d &equ_r,
    const DAT::t_float_1d &equ_z,
    const DAT::t_float_2d_lr &equ_psi,
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

/* ===================================================================
   Native equilibrium B maps (slag b05b4687): when the plasma file
   carries equ_br/bt/bz on the same [z][r] grid, the CPU equ_bfield_at
   prefers BILINEAR interpolation of those maps over psi-derived B.
   Device twin: cell-based upper-bound stencil with edge clamping,
   exactly mirroring make_regular_grid_stencil + sample_regular_map.
   =================================================================== */

KOKKOS_INLINE_FUNCTION
double bilinear_regular_map(const DAT::t_float_2d_lr &f, int k, int j,
                            double fr, double fz)
{
  const double lower = (1.0 - fr) * f(k,   j) + fr * f(k,   j+1);
  const double upper = (1.0 - fr) * f(k+1, j) + fr * f(k+1, j+1);
  return (1.0 - fz) * lower + fz * upper;
}

KOKKOS_INLINE_FUNCTION
bool query_bfield_native_maps(
    const double xyz[3], int dim, int axisymmetric,
    const DAT::t_float_1d &equ_r,
    const DAT::t_float_1d &equ_z,
    const DAT::t_float_2d_lr &equ_br,
    const DAT::t_float_2d_lr &equ_bt,
    const DAT::t_float_2d_lr &equ_bz,
    int jm, int km,
    double B[3])
{
  B[0] = 0.0; B[1] = 0.0; B[2] = 0.0;
  if (jm < 2 || km < 2) return false;

  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }
  if (R < 1.0e-10) return false;
  if (equ_r(jm-1) <= equ_r(0) || equ_z(km-1) <= equ_z(0)) return false;

  const double Rc = R < equ_r(0) ? equ_r(0)
                    : (R > equ_r(jm-1) ? equ_r(jm-1) : R);
  const double Zc = Z < equ_z(0) ? equ_z(0)
                    : (Z > equ_z(km-1) ? equ_z(km-1) : Z);

  // std::upper_bound twin: first index with value > coordinate, minus 1
  int lo = 0, hi = jm;
  while (lo < hi) { const int mid = (lo+hi)/2;
                    if (equ_r(mid) <= Rc) lo = mid+1; else hi = mid; }
  int j = lo - 1; if (j < 0) j = 0; if (j > jm-2) j = jm-2;
  lo = 0; hi = km;
  while (lo < hi) { const int mid = (lo+hi)/2;
                    if (equ_z(mid) <= Zc) lo = mid+1; else hi = mid; }
  int k = lo - 1; if (k < 0) k = 0; if (k > km-2) k = km-2;

  const double drs = equ_r(j+1) - equ_r(j);
  const double dzs = equ_z(k+1) - equ_z(k);
  if (drs <= 0.0 || dzs <= 0.0) return false;
  const double fr = (Rc - equ_r(j)) / drs;
  const double fz = (Zc - equ_z(k)) / dzs;

  const double bR   = bilinear_regular_map(equ_br, k, j, fr, fz);
  const double bphi = bilinear_regular_map(equ_bt, k, j, fr, fz);
  const double bZ   = bilinear_regular_map(equ_bz, k, j, fr, fz);

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
  return true;
}

/* ---------------------------------------------------------------------- */
// Full point-query: B + grad|B| + κ + curl(b̂) at particle position from
// the equilibrium ψ map. Currently unused (the old device hybrid/GCA
// pusher was removed 2026-08-26); kept for a future re-port of the
// current CPU hybrid. All outputs are in SPARTA slot order matching the
// input position.
//
// Math is identical to ComputePlasmaFields::query_bfield_at_point (CPU,
// equilibrium branch lines 1976–2037 of compute_plasma_fields.cpp) +
// the κ / curl(b̂) cylindrical-decomposition block in
// pusher.cpp::push_hybrid_3d.
//
// Returns true on success, false when point falls outside the equilibrium
// grid or no equilibrium is loaded — caller should fall back to plain
// Boris with B-only point-query.
/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
bool query_bfield_grad_at_point(
    const double xyz[3], int dim, int axisymmetric,
    const DAT::t_float_1d &equ_r,
    const DAT::t_float_1d &equ_z,
    const DAT::t_float_2d_lr &equ_psi,
    double btf, double rtf, int jm, int km,
    double B[3], double gradBmag[3],
    double kappa[3], double curl_b[3])
{
  for (int k = 0; k < 3; k++) {
    B[k] = 0.0; gradBmag[k] = 0.0; kappa[k] = 0.0; curl_b[k] = 0.0;
  }
  if (jm < 3 || km < 3) return false;

  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }
  if (R < 1.0e-10) return false;

  const double dr = equ_r(1) - equ_r(0);
  const double dz = equ_z(1) - equ_z(0);
  if (dr <= 0.0 || dz <= 0.0) return false;

  const double fj = (R - equ_r(0)) / dr;
  const double fk = (Z - equ_z(0)) / dz;
  int jc = static_cast<int>(Kokkos::round(fj));
  int kc = static_cast<int>(Kokkos::round(fk));
  if (jc < 1) jc = 1;  else if (jc > jm - 2) jc = jm - 2;
  if (kc < 1) kc = 1;  else if (kc > km - 2) kc = km - 2;

  // Centred-difference psi derivatives + 2nd derivatives + mixed partial
  const double dRsum = equ_r(jc+1) - equ_r(jc-1);
  const double dZsum = equ_z(kc+1) - equ_z(kc-1);
  const double dpsi_dR  = (equ_psi(kc, jc+1) - equ_psi(kc, jc-1)) / dRsum;
  const double dpsi_dZ  = (equ_psi(kc+1, jc) - equ_psi(kc-1, jc)) / dZsum;

  const double dR1 = equ_r(jc+1) - equ_r(jc);
  const double dR0 = equ_r(jc)   - equ_r(jc-1);
  const double dZ1 = equ_z(kc+1) - equ_z(kc);
  const double dZ0 = equ_z(kc)   - equ_z(kc-1);
  const double d2psi_dR2 = 2.0 * (equ_psi(kc, jc+1) / (dR1*(dR1+dR0))
                                 - equ_psi(kc, jc)   / (dR1*dR0)
                                 + equ_psi(kc, jc-1) / (dR0*(dR1+dR0)));
  const double d2psi_dZ2 = 2.0 * (equ_psi(kc+1, jc) / (dZ1*(dZ1+dZ0))
                                 - equ_psi(kc, jc)   / (dZ1*dZ0)
                                 + equ_psi(kc-1, jc) / (dZ0*(dZ1+dZ0)));
  const double d2psi_dRdZ = (equ_psi(kc+1, jc+1) - equ_psi(kc+1, jc-1)
                            - equ_psi(kc-1, jc+1) + equ_psi(kc-1, jc-1))
                           / (dRsum * dZsum);

  const double invR  = 1.0 / R;
  const double invR2 = invR * invR;

  // Cylindrical B and B-component derivatives
  const double bR   = -dpsi_dZ * invR;
  const double bZ   =  dpsi_dR * invR;
  const double bphi = (btf * rtf) * invR;

  const double dBr_dr =  dpsi_dZ * invR2 - d2psi_dRdZ * invR;
  const double dBr_dz = -d2psi_dZ2 * invR;
  const double dBz_dr = -dpsi_dR * invR2 + d2psi_dR2  * invR;
  const double dBz_dz =  d2psi_dRdZ * invR;
  const double dBt_dr = -btf * rtf * invR2;
  const double dBt_dz =  0.0;

  const double Bmag = Kokkos::sqrt(bR*bR + bphi*bphi + bZ*bZ);
  if (Bmag <= 0.0) return false;
  const double invBm = 1.0 / Bmag;

  const double dBmag_dr = (bR*dBr_dr + bphi*dBt_dr + bZ*dBz_dr) * invBm;
  const double dBmag_dz = (bR*dBr_dz + bphi*dBt_dz + bZ*dBz_dz) * invBm;

  // Unit b̂ derivatives
  const double bhR = bR * invBm, bhP = bphi * invBm, bhZ = bZ * invBm;
  const double dbR_dR  = invBm * (dBr_dr - bhR * dBmag_dr);
  const double dbR_dZ  = invBm * (dBr_dz - bhR * dBmag_dz);
  const double dbP_dR  = invBm * (dBt_dr - bhP * dBmag_dr);
  const double dbP_dZ  = invBm * (dBt_dz - bhP * dBmag_dz);
  const double dbZ_dR  = invBm * (dBz_dr - bhZ * dBmag_dr);
  const double dbZ_dZ  = invBm * (dBz_dz - bhZ * dBmag_dz);

  // κ = (b̂·∇)b̂ in cylindrical (axisymmetric, ∂/∂φ = 0)
  const double kR  = bhR * dbR_dR  + bhZ * dbR_dZ - bhP * bhP * invR;
  const double kP  = bhR * dbP_dR  + bhZ * dbP_dZ + bhR * bhP * invR;
  const double kZ  = bhR * dbZ_dR  + bhZ * dbZ_dZ;

  // curl(b̂) in cylindrical (axisymmetric, ∂/∂φ = 0)
  const double cR  = -dbP_dZ;
  const double cP  =  dbR_dZ - dbZ_dR;
  const double cZ  =  bhP * invR + dbP_dR;

  // Pack all outputs into SPARTA slot order
  if (axisymmetric) {
    B[0] = bZ;   B[1] = bR;   B[2] = bphi;
    gradBmag[0] = dBmag_dz; gradBmag[1] = dBmag_dr; gradBmag[2] = 0.0;
    kappa[0]    = kZ;       kappa[1]    = kR;       kappa[2]    = kP;
    curl_b[0]   = cZ;       curl_b[1]   = cR;       curl_b[2]   = cP;
  } else if (dim == 2) {
    B[0] = bR;   B[1] = bZ;   B[2] = bphi;
    gradBmag[0] = dBmag_dr; gradBmag[1] = dBmag_dz; gradBmag[2] = 0.0;
    kappa[0]    = kR;       kappa[1]    = kZ;       kappa[2]    = kP;
    curl_b[0]   = cR;       curl_b[1]   = cZ;       curl_b[2]   = cP;
  } else {
    const double rxy = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
    double cphi = 1.0, sphi = 0.0;
    if (rxy > 1.0e-20) { cphi = xyz[0] / rxy; sphi = xyz[1] / rxy; }
    B[0] = bR * cphi - bphi * sphi;
    B[1] = bR * sphi + bphi * cphi;
    B[2] = bZ;
    gradBmag[0] = dBmag_dr * cphi;
    gradBmag[1] = dBmag_dr * sphi;
    gradBmag[2] = dBmag_dz;
    kappa[0]    = kR * cphi - kP * sphi;
    kappa[1]    = kR * sphi + kP * cphi;
    kappa[2]    = kZ;
    curl_b[0]   = cR * cphi - cP * sphi;
    curl_b[1]   = cR * sphi + cP * cphi;
    curl_b[2]   = cZ;
  }
  return true;
}

}  // namespace EquilibriumKokkos

/* ===================================================================
   Mesh-triangulation B-field point-query on device.

   Mirrors the mesh branch of
       ComputePlasmaFields::query_bfield_at_point() (CPU)
   for SOLPS / SOLEDGE3X plasmas where B is carried as
   /mesh/vtx_b{r,t,z} (vertex values, vertex-averaged per triangle on
   ingest). Locates the triangle containing (R,Z) via the CSR-flat
   spatial hash uploaded by ComputePlasmaFieldsKokkos::sync_mesh_to_device,
   then reads B[3] from that triangle's stored values. No interpolation
   inside the triangle — matches CPU semantics. Returns true on hit,
   false if the point falls outside the meshed footprint.
   =================================================================== */
namespace MeshKokkos {

KOKKOS_INLINE_FUNCTION
bool query_bfield_at_point(
    const double xyz[3], int dim, int axisymmetric,
    const DAT::t_float_1d &mesh_vtx_r,
    const DAT::t_float_1d &mesh_vtx_z,
    const DAT::t_int_1d &mesh_tri,
    const DAT::t_float_1d &mesh_tri_br,
    const DAT::t_float_1d &mesh_tri_bz,
    const DAT::t_float_1d &mesh_tri_bt,
    const DAT::t_float_1d &mesh_tri_rmin,
    const DAT::t_float_1d &mesh_tri_rmax,
    const DAT::t_float_1d &mesh_tri_zmin,
    const DAT::t_float_1d &mesh_tri_zmax,
    const DAT::t_int_1d &hash_offset,
    const DAT::t_int_1d &hash_entries,
    double hash_rmin, double hash_zmin,
    double hash_dr,   double hash_dz,
    int hash_nr, int hash_nz, int ntri,
    double B[3])
{
  B[0] = 0.0; B[1] = 0.0; B[2] = 0.0;
  if (ntri <= 0) return false;

  // SPARTA-slot → cylindrical (R,Z)
  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }

  // Find triangle containing (R,Z) via spatial hash (O(1)).
  int tri = -1;
  if (hash_nr > 0 && hash_nz > 0 && hash_dr > 0.0 && hash_dz > 0.0) {
    const int ir = static_cast<int>((R - hash_rmin) / hash_dr);
    const int iz = static_cast<int>((Z - hash_zmin) / hash_dz);
    if (ir >= 0 && ir < hash_nr && iz >= 0 && iz < hash_nz) {
      const int b   = iz * hash_nr + ir;
      const int beg = hash_offset(b);
      const int end = hash_offset(b + 1);
      for (int k = beg; k < end; k++) {
        const int t = hash_entries(k);
        const int v0 = mesh_tri(3*t+0);
        const int v1 = mesh_tri(3*t+1);
        const int v2 = mesh_tri(3*t+2);
        const double r0 = mesh_vtx_r(v0), z0 = mesh_vtx_z(v0);
        const double r1 = mesh_vtx_r(v1), z1 = mesh_vtx_z(v1);
        const double r2 = mesh_vtx_r(v2), z2 = mesh_vtx_z(v2);
        const double d  = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
        if (Kokkos::fabs(d) < 1e-30) continue;
        const double a  = ((R-r0)*(z2-z0) - (r2-r0)*(Z-z0)) / d;
        const double bb = ((r1-r0)*(Z-z0) - (R-r0)*(z1-z0)) / d;
        if (a >= -1e-10 && bb >= -1e-10 && (a+bb) <= 1.0+1e-10) {
          tri = t; break;
        }
      }
    }
  }

  if (tri < 0) return false;

  const double bR   = mesh_tri_br(tri);
  const double bZ   = mesh_tri_bz(tri);
  const double bphi = mesh_tri_bt(tri);

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
  return true;
}

/* ===================================================================
   Scalar variant of query_bfield_at_point: same triangle location, but
   the three per-tri fields are SCALARS (e.g. te, ti, ne) — no
   cylindrical-to-slot rotation of the result. Device twin of the mesh
   branch of FixBackground::interp2D (tri-constant via mesh_cell_idx).
   =================================================================== */

KOKKOS_INLINE_FUNCTION
bool query_scalars_at_point(
    const double xyz[3], int dim, int axisymmetric,
    const DAT::t_float_1d &mesh_vtx_r,
    const DAT::t_float_1d &mesh_vtx_z,
    const DAT::t_int_1d &mesh_tri,
    const DAT::t_float_1d &mesh_tri_f1,
    const DAT::t_float_1d &mesh_tri_f2,
    const DAT::t_float_1d &mesh_tri_f3,
    const DAT::t_int_1d &hash_offset,
    const DAT::t_int_1d &hash_entries,
    double hash_rmin, double hash_zmin,
    double hash_dr,   double hash_dz,
    int hash_nr, int hash_nz, int ntri,
    double out[3])
{
  out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
  if (ntri <= 0) return false;

  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }

  int tri = -1;
  if (hash_nr > 0 && hash_nz > 0 && hash_dr > 0.0 && hash_dz > 0.0) {
    const int ir = static_cast<int>((R - hash_rmin) / hash_dr);
    const int iz = static_cast<int>((Z - hash_zmin) / hash_dz);
    if (ir >= 0 && ir < hash_nr && iz >= 0 && iz < hash_nz) {
      const int b   = iz * hash_nr + ir;
      const int beg = hash_offset(b);
      const int end = hash_offset(b + 1);
      for (int k = beg; k < end; k++) {
        const int t = hash_entries(k);
        const int v0 = mesh_tri(3*t+0);
        const int v1 = mesh_tri(3*t+1);
        const int v2 = mesh_tri(3*t+2);
        const double r0 = mesh_vtx_r(v0), z0 = mesh_vtx_z(v0);
        const double r1 = mesh_vtx_r(v1), z1 = mesh_vtx_z(v1);
        const double r2 = mesh_vtx_r(v2), z2 = mesh_vtx_z(v2);
        const double d  = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
        if (Kokkos::fabs(d) < 1e-30) continue;
        const double a  = ((R-r0)*(z2-z0) - (r2-r0)*(Z-z0)) / d;
        const double bb = ((r1-r0)*(Z-z0) - (R-r0)*(z1-z0)) / d;
        if (a >= -1e-10 && bb >= -1e-10 && (a+bb) <= 1.0+1e-10) {
          tri = t; break;
        }
      }
    }
  }

  if (tri < 0) return false;

  out[0] = mesh_tri_f1(tri);
  out[1] = mesh_tri_f2(tri);
  out[2] = mesh_tri_f3(tri);
  return true;
}

/* ===================================================================
   Bare triangle locator: same CSR-hash containment search as the two
   query functions above, returning the TRI INDEX so callers can fetch
   arbitrary per-tri fields (gate 9: coulomb drag + thermal force read
   te/ti/ne/ni/upar and the grad-T fields at the particle position).
   Returns -1 outside the meshed footprint — callers treat that as the
   CPU's "fall through to the (empty) structured grid" = 0.
   =================================================================== */

KOKKOS_INLINE_FUNCTION
int locate_tri_at_point(
    const double xyz[3], int dim, int axisymmetric,
    const DAT::t_float_1d &mesh_vtx_r,
    const DAT::t_float_1d &mesh_vtx_z,
    const DAT::t_int_1d &mesh_tri,
    const DAT::t_int_1d &hash_offset,
    const DAT::t_int_1d &hash_entries,
    double hash_rmin, double hash_zmin,
    double hash_dr,   double hash_dz,
    int hash_nr, int hash_nz, int ntri)
{
  if (ntri <= 0) return -1;

  double R, Z;
  if (axisymmetric)      { Z = xyz[0]; R = xyz[1]; }
  else if (dim == 2)     { R = xyz[0]; Z = xyz[1]; }
  else                   { R = Kokkos::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);
                           Z = xyz[2]; }

  if (hash_nr <= 0 || hash_nz <= 0 || hash_dr <= 0.0 || hash_dz <= 0.0)
    return -1;
  const int ir = static_cast<int>((R - hash_rmin) / hash_dr);
  const int iz = static_cast<int>((Z - hash_zmin) / hash_dz);
  if (ir < 0 || ir >= hash_nr || iz < 0 || iz >= hash_nz) return -1;

  const int b   = iz * hash_nr + ir;
  const int beg = hash_offset(b);
  const int end = hash_offset(b + 1);
  for (int k = beg; k < end; k++) {
    const int t = hash_entries(k);
    const int v0 = mesh_tri(3*t+0);
    const int v1 = mesh_tri(3*t+1);
    const int v2 = mesh_tri(3*t+2);
    const double r0 = mesh_vtx_r(v0), z0 = mesh_vtx_z(v0);
    const double r1 = mesh_vtx_r(v1), z1 = mesh_vtx_z(v1);
    const double r2 = mesh_vtx_r(v2), z2 = mesh_vtx_z(v2);
    const double d  = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
    if (Kokkos::fabs(d) < 1e-30) continue;
    const double a  = ((R-r0)*(z2-z0) - (r2-r0)*(Z-z0)) / d;
    const double bb = ((r1-r0)*(Z-z0) - (R-r0)*(z1-z0)) / d;
    if (a >= -1e-10 && bb >= -1e-10 && (a+bb) <= 1.0+1e-10) return t;
  }
  return -1;
}

}  // namespace MeshKokkos
}  // namespace SPARTA_NS

#endif  // SPARTA_PUSHER_KOKKOS_H
