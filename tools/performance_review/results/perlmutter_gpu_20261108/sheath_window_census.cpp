// clamp census of prepare_borodkina over the mesh plasma: which branch produces the unbounded window
#include <Kokkos_Core.hpp>
#include "sheath_models_kokkos.h"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace SPARTA_NS;
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "r"); double mD = 2.0;
  std::vector<double> te, ti, ne, b; double a1,a2,a3,a4;
  while (fscanf(f, "%lf,%lf,%lf,%lf", &a1,&a2,&a3,&a4) == 4) { te.push_back(a1); ti.push_back(a2); ne.push_back(a3); b.push_back(a4); }
  fclose(f);
  const double alphas[9] = {0.0,15.0,30.0,45.0,60.0,70.0,80.0,85.0,89.0};
  printf("%6s %8s %8s %8s %8s %8s %8s\n","alpha","pure_mps","disc<0","a_clamp","big>1m","med_win","p99_win");
  for (int ia = 0; ia < 9; ia++) {
    long n=0, npure=0, ndisc=0, naclamp=0, nbig=0; std::vector<double> w;
    for (size_t t = 0; t < te.size(); t++) {
      if (!(te[t] > 0 && ne[t] > 0)) continue; n++;
      auto c = SheathModelsKokkos::prepare_borodkina(te[t], ti[t], ne[t], b[t], alphas[ia], mD, 0.0);
      w.push_back(c.mps_end_m);
      if (c.mps_end_m > 1.0) nbig++;
      // re-derive the branch quantities exactly as the function does
      const double tev = std::max(te[t],1e-12), tiv = std::max(ti[t],0.0);
      const double ar = std::min(std::max(alphas[ia],0.0),89.999)*M_PI/180.0;
      const double phi_float = 0.5*std::log((mD*1.66053906660e-27/(2.0*M_PI*9.1093837015e-31))/(1.0+tiv/tev));
      const double phi_total = std::max(phi_float,0.0)*tev; const double lambda_w = -phi_total/tev;
      const double alpha_star = std::acos(std::min(std::max(std::exp(lambda_w),0.0),1.0))*180.0/M_PI;
      const double cosa = std::max(std::cos(ar),1e-12); double lambda_mps = std::log(cosa);
      bool pure = (std::min(std::max(alphas[ia],0.0),89.999) >= alpha_star);
      if (pure) { npure++; continue; }
      const double lambdaD = std::sqrt(8.8541878128e-12*tev/(std::max(ne[t],1e-60)*1.602176634e-19));
      const double cs = std::sqrt((tev+tiv)*1.602176634e-19/(mD*1.66053906660e-27));
      const double rho = cs/(1.602176634e-19*std::max(b[t],1e-20)/(mD*1.66053906660e-27));
      const double lmps = 3.0*rho*std::sin(ar), lmps_D = lmps/lambdaD;
      const double slope_mps = lmps_D > 0 ? -2.0*lambda_mps/lmps_D : 0.0;
      const double beta = 2.0/(1.0+tiv/tev), c1 = slope_mps*slope_mps - 6.0*cosa;
      const double ion_root = std::sqrt(std::max(1.0 - beta*(lambda_w - lambda_mps),0.0));
      const double disc = 2.0*std::exp(lambda_w) + 4.0*cosa*ion_root + c1;
      if (disc < 0) ndisc++;
      const double wall_slope = std::sqrt(std::max(disc,0.0)); const double denom = lambda_w - lambda_mps;
      const double a_raw = (slope_mps - wall_slope)/denom;
      if (a_raw < 1e-12) naclamp++;
    }
    std::sort(w.begin(), w.end());
    printf("%6.1f %8.3f %8.3f %8.3f %8.3f %8.2e %8.2e\n", alphas[ia], double(npure)/n, double(ndisc)/n, double(naclamp)/n, double(nbig)/n, w[w.size()/2], w[w.size()*99/100]);
  }
  return 0;
}
