/* ----------------------------------------------------------------------
   OpenEdge: device twins of FixBackground::interp2D (bilinear sample on the
   regular (R,Z) raster, clamped, index iz*nr+ir) and of the constant-B
   slot conversion (UpdateKokkos::oe_const_bfield_slot). Shared by the
   pcache fill and the coulomb / thermal / cross-field-diffusion device fixes
   so regular-grid plasma files (e.g. rfpie) run on the device.
------------------------------------------------------------------------- */
#ifndef SPARTA_RASTER_KOKKOS_H
#define SPARTA_RASTER_KOKKOS_H
#include "kokkos_type.h"
#include "openedge_geom.h"
namespace SPARTA_NS {
namespace RasterKokkos {
KOKKOS_INLINE_FUNCTION
void rz_of(const double *xq, int dim, int axisym, double &R, double &Z)
{
  if (dim == 3) { R = sqrt(xq[0]*xq[0] + xq[1]*xq[1]); Z = xq[2]; }
  else if (axisym) { Z = xq[0]; R = xq[1]; }
  else { R = xq[0]; Z = xq[1]; }
}
KOKKOS_INLINE_FUNCTION
double sample(const DAT::t_float_1d &f, double r0, double dr, int nr,
              double z0, double dz, int nz, double R, double Z)
{
  if (f.extent(0) < (size_t)(nr*nz) || nr < 2 || nz < 2) return 0.0;
  const double rmax = r0 + dr*(nr-1), zmax = z0 + dz*(nz-1);
  const double Rc = Kokkos::fmin(Kokkos::fmax(R, r0), rmax);
  const double Zc = Kokkos::fmin(Kokkos::fmax(Z, z0), zmax);
  const double fi = (Rc - r0) / dr, fj = (Zc - z0) / dz;
  int ir0 = (int) fi; if (ir0 < 0) ir0 = 0; if (ir0 > nr-2) ir0 = nr-2;
  int iz0 = (int) fj; if (iz0 < 0) iz0 = 0; if (iz0 > nz-2) iz0 = nz-2;
  const double s = Kokkos::fmin(Kokkos::fmax(fi - ir0, 0.0), 1.0);
  const double t = Kokkos::fmin(Kokkos::fmax(fj - iz0, 0.0), 1.0);
  return (1-s)*(1-t)*f(iz0*nr+ir0) + s*(1-t)*f(iz0*nr+ir0+1)
       + (1-s)*t*f((iz0+1)*nr+ir0) + s*t*f((iz0+1)*nr+ir0+1);
}
}  // namespace RasterKokkos
namespace ConstBKokkos {
// mode 0 none, 1 cylindrical (br,bz,bt), 2 Cartesian bcart; xq column-shifted
KOKKOS_INLINE_FUNCTION
bool slot(int mode, int dim, int axisym, const double *xq,
          double br, double bz, double bt, const double *bcart, double *B)
{
  if (!mode) return false;
  if (mode == 2) {
    if (dim == 3) { B[0] = bcart[0]; B[1] = bcart[1]; B[2] = bcart[2]; }
    else OpenEdge::RZphi_force_to_sparta(bcart[0], bcart[1], bcart[2], dim, axisym != 0, 0.0, B[0], B[1], B[2]);
    return true;
  }
  double phi = 0.0;
  if (dim == 3 && !axisym) phi = Kokkos::atan2(xq[1], xq[0]);
  OpenEdge::RZphi_force_to_sparta(br, bz, bt, dim, axisym != 0, phi, B[0], B[1], B[2]);
  return true;
}
}  // namespace ConstBKokkos
}  // namespace SPARTA_NS
#endif
