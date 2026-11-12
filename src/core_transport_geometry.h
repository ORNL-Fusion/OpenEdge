// Refined, axis-connected toroidal flux volumes. The equilibrium is
// interpolated with a local tensor cubic; quadratic equilibria are exact.
#ifndef OPENEDGE_CORE_TRANSPORT_GEOMETRY_H
#define OPENEDGE_CORE_TRANSPORT_GEOMETRY_H
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace SPARTA_NS {
// psi at (R,Z): the local tensor cubic used for the flux volumes below
// (psi indexed [iz*nr + ir]); points outside the grid are clamped to it
inline double core_psi_cubic(const std::vector<double> &r, const std::vector<double> &z,
                             const std::vector<double> &psi, double R, double Z)
{
  const int nr = r.size(), nz = z.size();
  R = std::min(std::max(R,r.front()),r.back());
  Z = std::min(std::max(Z,z.front()),z.back());
  auto weights = [](const std::vector<double> &axis, double x, int &base, double *w) {
    base = std::max(0,std::min((int) axis.size()-4,
      (int) (std::upper_bound(axis.begin(),axis.end(),x)-axis.begin())-2));
    for (int i = 0; i < 4; ++i) {
      w[i] = 1.0;
      for (int j = 0; j < 4; ++j) if (j != i)
        w[i] *= (x-axis[base+j])/(axis[base+i]-axis[base+j]);
    }
  };
  int ir, iz; double wr[4], wz[4];
  weights(r,R,ir,wr); weights(z,Z,iz,wz);
  double v = 0.0;
  for (int b = 0; b < 4; ++b) for (int a = 0; a < 4; ++a)
    v += wz[b]*wr[a]*psi[(size_t) (iz+b)*nr+ir+a];
  (void) nz;
  return v;
}

// refinement of the equilibrium grid for the flux volumes: about 4e6
// points, at least 8 per cell (pixel error of the volume inside a contour)
inline int core_flux_refine(int nr, int nz)
{
  return std::max(8,std::min(32,(int) sqrt(4.0e6/((double) nr*nz))));
}

struct CoreFluxPoint { double R, Z, psi; };
struct CoreFluxGeometry {
  double raxis;
  std::vector<CoreFluxPoint> points;
  std::vector<double> cumulative;

  CoreFluxGeometry(const std::vector<double> &r, const std::vector<double> &z,
                   const std::vector<double> &psi, double pa, double pb,
                   double cut, int refine = 8, int bins = 1000)
  {
    const int nr = r.size(), nz = z.size();
    if (nr < 4 || nz < 4 || psi.size() != (size_t) nr*nz ||
        fabs(pb-pa) < 1e-30 || refine < 1)
      throw std::runtime_error("core transport: unusable equilibrium grid");
    const int nx = (nr-1)*refine+1, ny = (nz-1)*refine+1;
    const double dr = (r.back()-r.front())/(nx-1);
    const double dz = (z.back()-z.front())/(ny-1);
    auto weights = [](const std::vector<double> &axis, double x, int &base, double *w) {
      base = std::max(0,std::min((int) axis.size()-4,
        (int) (std::upper_bound(axis.begin(),axis.end(),x)-axis.begin())-2));
      for (int i = 0; i < 4; ++i) {
        w[i] = 1.0;
        for (int j = 0; j < 4; ++j) if (j != i)
          w[i] *= (x-axis[base+j])/(axis[base+i]-axis[base+j]);
      }
    };
    std::vector<double> p((size_t) nx*ny);
    int axis = 0;
    for (int k = 0; k < ny; ++k) {
      int iz; double wz[4]; weights(z,z.front()+k*dz,iz,wz);
      for (int j = 0; j < nx; ++j) {
        int ir; double wr[4]; weights(r,r.front()+j*dr,ir,wr);
        double v = 0.0;
        for (int b = 0; b < 4; ++b) for (int a = 0; a < 4; ++a)
          v += wz[b]*wr[a]*psi[(size_t) (iz+b)*nr+ir+a];
        int c = k*nx+j;
        p[c] = (v-pa)/(pb-pa);
        if (!std::isfinite(p[c])) throw std::runtime_error("core transport: non-finite equilibrium");
        if (p[c] < p[axis]) axis = c;
      }
    }
    if (p[axis] >= cut) throw std::runtime_error("core transport: no axis inside cut");
    raxis = r.front()+(axis%nx)*dr;
    std::vector<double> volume(bins,0.0);
    std::vector<char> seen(p.size(),0);
    std::queue<int> q; q.push(axis); seen[axis] = 1;
    while (!q.empty()) {
      int c = q.front(); q.pop(); int j = c%nx, k = c/nx;
      if (j == 0 || j == nx-1 || k == 0 || k == ny-1)
        throw std::runtime_error("core transport: cut contour is not closed inside equilibrium");
      double R = r.front()+j*dr, Z = z.front()+k*dz;
      points.push_back({R,Z,p[c]});
      int b = std::min(bins-1,std::max(0,(int) (p[c]/cut*bins)));
      volume[b] += 6.283185307179586*R*fabs(dr*dz);
      const int neighbor[] = {c-1,c+1,c-nx,c+nx};
      for (int n : neighbor) if (!seen[n] && p[n] < cut) { seen[n] = 1; q.push(n); }
    }
    cumulative.assign(bins+1,0.0);
    for (int b = 0; b < bins; ++b) cumulative[b+1] = cumulative[b]+volume[b];
  }
};
}
#endif
