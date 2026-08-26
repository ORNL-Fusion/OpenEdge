#ifndef OPENEDGE_GEOM_H
#define OPENEDGE_GEOM_H

#include <cmath>

// Make the helpers callable from device kernels when Kokkos is in scope.
// Falls back to plain `inline` on CPU-only builds, so this header has no
// hard dependency on Kokkos.
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace OpenEdge {

// Override: when true, force the axisymmetric (x=Z, y=R) R/Z convention even
// in planar 2D mode. Set by `fix background ... rz_axes axi` so a Cartesian
// (planar) run can reuse an axi-convention deck unchanged (same wall_fine.surf
// and plasma.h5, x=Z/y=R) instead of the default planar R=x/Z=y. Default false
// leaves axi and legacy-Cartesian behavior untouched. Host-side global (the
// KOKKOS axi/planar path is separate and already flagged unsupported).
extern bool oe_force_axi_rz;

// nvcc emits both host and device variants of KOKKOS_INLINE_FUNCTION helpers.
// The override above intentionally exists only on the host; referencing it in
// the device variant is ill-formed.  Device paths use their explicit
// axisymmetric flag, matching the documented Kokkos limitation.
KOKKOS_INLINE_FUNCTION
bool use_axi_rz(bool axisymmetric) {
#if defined(__CUDA_ARCH__)
  return axisymmetric;
#else
  return axisymmetric || oe_force_axi_rz;
#endif
}

// Maps a SPARTA position (xyz from particle->x or surf line endpoint) to
// physical cylindrical (R, Z). Three SPARTA modes:
//   2D Cartesian (legacy):  x[0]=R, x[1]=Z              (R, Z) = (x, y)
//   2D axisymmetric:        x[0]=Z (axis), x[1]=R       (R, Z) = (y, x)
//   3D Cartesian:           x[0..2] = (X, Y, Z)         (R, Z) = (sqrt((X-x0)^2+(Y-y0)^2), Z)
//
// The 3D Cartesian path accepts an optional axis offset (x0, y0) for
// linear-device cases (MPEX, proto-lite) whose simulation box is not
// centered on the plasma column. Default (0, 0) preserves SOLPS /
// SOLEDGE3X axisymmetric behavior. Offset is ignored in 2D and axi modes.
//
// The 2D-axisymmetric layout is enforced by SPARTA: `boundary o a p` requires
// the `a` token at ylo, with boxlo[1] == 0 and y the radial direction
// (domain.cpp:174-182, create_box.cpp:55).
KOKKOS_INLINE_FUNCTION
void sparta_to_RZ(const double *xyz, int dim, bool axisymmetric,
                  double &R, double &Z,
                  double x0 = 0.0, double y0 = 0.0) {
  if (use_axi_rz(axisymmetric)) {   // x = Z (axis), y = R (radial)
    Z = xyz[0];
    R = xyz[1];
  } else if (dim == 2) {         // 2D Cartesian (legacy)
    R = xyz[0];
    Z = xyz[1];
  } else {                       // 3D
    const double dx = xyz[0] - x0;
    const double dy = xyz[1] - y0;
    R = std::sqrt(dx*dx + dy*dy);
    Z = xyz[2];
  }
}

// Decompose a physical cylindrical force (F_R, F_Z, F_phi) onto SPARTA's
// (Fx, Fy, Fz) slots. In 3D the azimuth `phi = atan2(y, x)` of the particle
// is needed to project R/phi onto Cartesian X/Y; pass 0 when irrelevant.
KOKKOS_INLINE_FUNCTION
void RZphi_force_to_sparta(double FR, double FZ, double Fphi,
                           int dim, bool axisymmetric, double phi,
                           double &fx, double &fy, double &fz) {
  if (use_axi_rz(axisymmetric)) {   // SPARTA slots: x=Z, y=R, z=phi
    fx = FZ;
    fy = FR;
    fz = Fphi;
  } else if (dim == 2) {         // legacy 2D Cartesian: x=R, y=Z, z=phi
    fx = FR;
    fy = FZ;
    fz = Fphi;
  } else {                       // 3D Cartesian
    const double cosp = std::cos(phi);
    const double sinp = std::sin(phi);
    fx = FR * cosp - Fphi * sinp;
    fy = FR * sinp + Fphi * cosp;
    fz = FZ;
  }
}

// Inverse: SPARTA velocity slots back to cylindrical (vR, vZ, vphi). Mirror
// of RZphi_force_to_sparta. Used when reading particle->v in physics code.
KOKKOS_INLINE_FUNCTION
void sparta_v_to_RZphi(const double *v, int dim, bool axisymmetric,
                       double phi, double &vR, double &vZ, double &vphi) {
  if (use_axi_rz(axisymmetric)) {
    vZ = v[0];
    vR = v[1];
    vphi = v[2];
  } else if (dim == 2) {
    vR = v[0];
    vZ = v[1];
    vphi = v[2];
  } else {
    const double cosp = std::cos(phi);
    const double sinp = std::sin(phi);
    vR = v[0] * cosp + v[1] * sinp;
    vphi = -v[0] * sinp + v[1] * cosp;
    vZ = v[2];
  }
}

}  // namespace OpenEdge

#endif
