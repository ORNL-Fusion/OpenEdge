#include "core_transport_1d.h"
#include "core_transport_geometry.h"
#include "core_transport_reservoir.h"
#include <cstdio>
#include <stdexcept>
#include <algorithm>
using namespace SPARTA_NS;
void require(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
int main() {
  const int nr = 12, nz = 3, nn = nr*nz;
  CoreTransport1D ct(nr,nz);
  std::vector<double> rr(nr), D(nn,0.3), v(nn,0), s(nn,0), r(nn,0), n(nn,0), src(nn,0);
  for (int i = 0; i < nr; ++i) rr[i] = i*0.005;
  ct.set_grid(rr.data());
  for (int i = 0; i < nr; ++i) { s[i] = 3e6; s[nr+i] = 1e6; r[i] = 2e6; r[nr+i] = 4e6; }
  n[nr-1] = 1e12; src[nr+nr-1] = 2e16;
  double initial = ct.content(0,n.data()), dt = 2e-5;
  ct.advance(dt,D.data(),v.data(),s.data(),r.data(),nullptr,src.data(),nullptr,0.0025,true,n.data());
  double total = 0, out = 0, source = 0;
  for (int z = 0; z < nz; ++z) {
    total += ct.content(z,n.data()); out += ct.outflux(z);
    for (int i = 0; i < nr; ++i) source += src[z*nr+i]*ct.cell_volume(i);
  }
  require(ct.substeps() > 100,"stiff chemistry did not subcycle");
  require(fabs(total-initial-dt*(source-out)) < 1e-11*(initial+dt*source),"substep ledger failed");
  for (double x : n) require(x >= 0 && std::isfinite(x),"negative stiff density");
  // Transport alone must also constrain a step, with an edge pulse.
  std::fill(s.begin(),s.end(),0); std::fill(r.begin(),r.end(),0);
  std::fill(n.begin(),n.end(),0); n[nr-1] = 1;
  ct.advance(0.001,D.data(),v.data(),s.data(),r.data(),nullptr,nullptr,nullptr,0.0025,true,n.data());
  require(ct.substeps() > 10,"transport was omitted from substep bound");
  n[0] = -1;
  bool rejected = false;
  try { ct.advance(dt,D.data(),v.data(),s.data(),r.data(),nullptr,nullptr,nullptr,0.0025,true,n.data()); }
  catch (const std::exception &) { rejected = true; }
  require(rejected,"negative initial density accepted");
  puts("  ok   stiff chemistry + transport positivity and conservative substeps");

  std::vector<double> R(33), Z(49), psi(R.size()*Z.size());
  for (int j = 0; j < 33; ++j) R[j] = 1.2+j*0.05;
  for (int k = 0; k < 49; ++k) Z[k] = -1.2+k*0.05;
  for (int k = 0; k < 49; ++k) for (int j = 0; j < 33; ++j)
    psi[k*33+j] = pow((R[j]-2)/0.55,2)+pow(Z[k]/0.8,2);
  const double pi = acos(-1.0), exact = 2*pi*pi*2*0.55*0.8*0.2;
  CoreFluxGeometry coarse(R,Z,psi,0,1,0.2,1), fine(R,Z,psi,0,1,0.2,8);
  require(fabs(fine.cumulative.back()/exact-1) < 0.003,"refined torus volume inaccurate");
  require(fabs(fine.cumulative.back()-exact) < fabs(coarse.cumulative.back()-exact),"volume refinement did not improve");
  printf("  ok   analytic elliptical torus: refined volume error %.3g\n",fabs(fine.cumulative.back()/exact-1));

  const int count = 20000;
  std::vector<CoreEntry> whole(count), a(count), b(count);
  double x[] = {1,2,3};
  core_entry_offer(whole,123,1,x,1); core_entry_offer(whole,456,2,x,9);
  core_entry_offer(a,123,1,x,1); core_entry_offer(b,456,2,x,9);
  int heavy = 0;
  for (int i = 0; i < count; ++i) {
    CoreEntry merged = a[i].key < b[i].key ? a[i] : b[i];
    require(merged.key == whole[i].key && merged.surf == whole[i].surf,"reservoir depends on decomposition");
    heavy += merged.surf == 2;
  }
  require(fabs((double) heavy/count-0.9) < 0.015,"reservoir not weighted by physical flux");
  printf("  ok   reservoir merge matches serial; weight-9 entry fraction %.4f\n",(double) heavy/count);
}
