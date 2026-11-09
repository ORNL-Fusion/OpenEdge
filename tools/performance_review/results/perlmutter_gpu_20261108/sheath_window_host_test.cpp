// host evaluation of the Borodkina sheath window over the plasma mesh: which (te,ti,ne,B,alpha) blow it up?
#include <Kokkos_Core.hpp>
#include "sheath_models_kokkos.h"
#include <cstdio>
#include <vector>
#include <algorithm>
using namespace SPARTA_NS;
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "r"); double mD = 2.0;
  std::vector<double> te, ti, ne, b; double a1,a2,a3,a4;
  while (fscanf(f, "%lf,%lf,%lf,%lf", &a1,&a2,&a3,&a4) == 4) { te.push_back(a1); ti.push_back(a2); ne.push_back(a3); b.push_back(a4); }
  fclose(f);
  const double alphas[10] = {0.0,10.0,20.0,30.0,45.0,60.0,75.0,85.0,89.0,89.999};
  printf("cells %zu\n", te.size());
  for (int a = 0; a < 10; a++) {
    std::vector<double> w; w.reserve(te.size());
    for (size_t t = 0; t < te.size(); t++) {
      if (!(te[t] > 0 && ne[t] > 0)) continue;
      auto c = SheathModelsKokkos::prepare_borodkina(te[t], ti[t], ne[t], b[t], alphas[a], mD, 0.0);
      w.push_back(c.mps_end_m);
    }
    std::sort(w.begin(), w.end());
    size_t n = w.size();
    printf("alpha %7.3f : n %zu  median %.3e  p90 %.3e  p99 %.3e  p999 %.3e  max %.3e m\n", alphas[a], n, w[n/2], w[n*9/10], w[n*99/100], w[n*999/1000], w.back());
  }
  // the max case
  double worst = 0; size_t wt = 0; int wa = 0;
  for (size_t t = 0; t < te.size(); t++) { if (!(te[t] > 0 && ne[t] > 0)) continue; for (int a = 0; a < 10; a++) { auto c = SheathModelsKokkos::prepare_borodkina(te[t], ti[t], ne[t], b[t], alphas[a], mD, 0.0); if (c.mps_end_m > worst) { worst = c.mps_end_m; wt = t; wa = a; } } }
  auto c = SheathModelsKokkos::prepare_borodkina(te[wt], ti[wt], ne[wt], b[wt], alphas[wa], mD, 0.0);
  printf("worst: cell %zu te %.3g ti %.3g ne %.3g B %.3g alpha %.3f -> mps_end %.3e lambdaD %.3e rho %.3e lmps %.3e\n", wt, te[wt], ti[wt], ne[wt], b[wt], alphas[wa], c.mps_end_m, c.lambdaD_m, c.rho_i_m, c.lmps_m);
  return 0;
}
