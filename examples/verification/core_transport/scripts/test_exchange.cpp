// C++ core transport + auxiliary-layer exchange vs a full-domain reference, diffusion only.
// Geometry: straight elliptical cylinder (elongation kappa; kappa = 1 is the circular cylinder).
// Flux label rho in [0,1], r_vol = rho sqrt(kappa). Markers carry a physical diffusivity D and are
// mixed over the flux surface every step, so their radial motion is the flux-surface averaged one;
// the core uses D_eff = g2 D with g2 = (kappa^2+1)/(2 kappa). The interface conductance comes from
// the same integrated flux operator: G = 2 pi r_I D_eff / l (per unit length), no extra area factor.
// Core: CoreTransport1D on [0, r_I] with a CLOSED outer boundary; CoreExchange owns the transfer.
// Markers: unequal weights (log-uniform over a factor SPREAD), four states without reactions:
//   0 pulse in the core, 1 pulse on the marker side, 2 few heavy markers far from the interface
//   (core and layers start empty), 3 uniform equilibrium.
// usage: test_exchange aux|brownian kappa seed nranks  -> one JSON line
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core_transport_1d.h"
#include "core_transport_exchange.h"
using namespace SPARTA_NS;
typedef std::vector<double> V;
struct P { double rho, w; uint64_t id; int z; };
static const int NS = 4;
static const double PI = 3.141592653589793, RHOI = 0.5, D = 1.0, T = 0.05, SPREAD = 50.0;
static double dens(int z, double rho) {
  if (z == 0) return exp(-0.5*pow((rho - 0.30)/0.06, 2));
  if (z == 1) return exp(-0.5*pow((rho - 0.70)/0.06, 2));
  if (z == 2) return exp(-0.5*pow((rho - 0.85)/0.04, 2));
  return 1.0;
}

int main(int argc, char **argv)
{
  const bool full = !strcmp(argv[1], "brownian");
  const double kappa = atof(argv[2]); const uint64_t seed = atoll(argv[3]); const int R = atoi(argv[4]);
  const double sk = sqrt(kappa), g2 = (kappa*kappa + 1)/(2*kappa), Deff = g2*D;
  const double rI = RHOI*sk, a = sk;
  const int NC = 50, NF = 400, KC = 2;
  // grids (CoreTransport1D convention: centres i dr, outer face at the domain edge)
  auto grid = [](int n, double edge, V &rr) { double dr = edge/(n - 0.5); rr.resize(n); for (int i = 0; i < n; i++) rr[i] = i*dr; return dr; };
  V rc, rf; const double drc = grid(NC, rI, rc), drf = grid(NF, a, rf);
  CoreTransport1D core(NC, NS), ref(NF, NS); core.set_grid(rc.data()); ref.set_grid(rf.data());
  V nc((size_t) NS*NC), nf((size_t) NS*NF);
  for (int z = 0; z < NS; z++) {
    for (int i = 0; i < NC; i++) nc[z*NC+i] = dens(z, rc[i]/sk);
    for (int i = 0; i < NF; i++) nf[z*NF+i] = dens(z, rf[i]/sk);
  }
  V Dc((size_t) NS*NC, Deff), Df((size_t) NS*NF, Deff), Zc((size_t) NS*NC, 0.0), Zf((size_t) NS*NF, 0.0);
  // layers and conductance
  const double hc = KC*drc, hm = hc, hmrho = hm/sk;
  double Vc = 0; for (int i = NC-KC; i < NC; i++) Vc += core.cell_volume(i);
  const double Vm = PI*(pow(rI + hm, 2) - rI*rI), ell = 0.5*(hc + hm), G = 2*PI*rI*Deff/ell;
  const double tau = 0.5*hc*hc/Deff;
  const int nwin = (int) ceil(T/tau); const double tw = T/nwin;
  const int nsub = std::max(1, (int) lround(tw/(pow(0.3*hmrho, 2)/(2*D)))); const double dtp = tw/nsub, sig = sqrt(2*D*dtp);
  CoreExchange ex(NS); ex.set_interval(G/Vc, G/Vm, tw);

  // markers: positions from n(rho) rho on [lo,1], weights log-uniform, rescaled to the exact inventory
  const double lo = full ? 0.0 : RHOI;
  const int scale = getenv("NMARK_SCALE") ? atoi(getenv("NMARK_SCALE")) : 100;
  const int nmark[NS] = {1500*scale, 1500*scale, 20*scale, 1500*scale};
  std::vector<std::vector<P>> rk(R);
  uint64_t next_id = 0; double wref[NS], inv0[NS], tot0[NS];
  for (int z = 0; z < NS; z++) {
    const int M = 4000; V cdf(M+1, 0.0);
    for (int k = 0; k < M; k++) { double r = lo + (k + 0.5)*(1 - lo)/M; cdf[k+1] = cdf[k] + dens(z, r)*r; }
    const double inv = cdf[M]*(1 - lo)/M*2*PI*kappa;            // per unit length
    std::vector<P> tmp(nmark[z]); double sw = 0;
    for (auto &p : tmp) {
      p.id = next_id++; p.z = z;
      double u = CoreExchange::uniform(p.id, seed, 101)*cdf[M];
      int k = (int) (std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin()) - 1; k = std::min(M-1, std::max(0, k));
      p.rho = lo + (k + (u - cdf[k])/(cdf[k+1] - cdf[k]))*(1 - lo)/M;
      p.w = exp(CoreExchange::uniform(p.id, seed, 102)*log(SPREAD)); sw += p.w;
    }
    for (auto &p : tmp) { p.w *= inv/sw; rk[p.id % R].push_back(p); }
    double c0 = 0; if (!full) c0 = core.content(z, nc.data());
    inv0[z] = inv + c0; tot0[z] = ref.content(z, nf.data());
    // emitted markers carry a weight set by the state's whole inventory, not by how the
    // marker side happened to be sampled at t = 0 (a pulse in the core leaves only a
    // faint tail there, whose tiny weights would turn the outflow into millions of markers)
    wref[z] = inv0[z]/nmark[z];
  }
  // reference marker-side inventory: cumulative interpolation at r_I
  auto ref_side = [&](int z) {
    double s = 0, edge = 0;
    for (int i = 0; i < NF; i++) {
      double hi = (i < NF-1) ? 0.5*(rf[i] + rf[i+1]) : a, c = nf[z*NF+i]*ref.cell_volume(i);
      if (edge >= rI) s += c; else if (hi > rI) s += c*(hi*hi - rI*rI)/(hi*hi - edge*edge);
      edge = hi;
    }
    return s;
  };
  double terr[NS] = {0, 0, 0, 0}, umin = 0.0, pendmax = 0.0;
  const int every = std::max(1, nwin/20);
  if (getenv("EX_DEBUG")) fprintf(stderr, "nwin %d nsub %d tw %.3e dtp %.3e hc %.4f pcm %.4f pmc %.4f\n", nwin, nsub, tw, dtp, hc, ex.p_core_to_markers(), ex.p_markers_to_core());
  for (int w = 1; w <= nwin; w++) {
    for (int s = 0; s < nsub; s++) {
      const uint64_t step = (uint64_t) (w - 1)*nsub + s + 1;
      for (auto &v : rk) for (auto &p : v) {
        double th = 2*PI*CoreExchange::uniform(p.id, step, seed*16 + 1);
        double u1 = std::max(1e-300, CoreExchange::uniform(p.id, step, seed*16 + 2)), u2 = CoreExchange::uniform(p.id, step, seed*16 + 3);
        double g = sig*sqrt(-2*log(u1));
        double x = p.rho*cos(th) + g*cos(2*PI*u2), zz = kappa*p.rho*sin(th) + g*sin(2*PI*u2);
        double r = sqrt(x*x + zz*zz/(kappa*kappa));
        for (int k = 0; k < 4; k++) { if (r < lo) r = 2*lo - r; if (r > 1) r = 2 - r; }
        p.rho = r;
      }
    }
    if (!full) {
      core.advance(tw, Dc.data(), Zc.data(), Zc.data(), Zc.data(), nullptr, nullptr, nullptr, 0.0, true, nc.data());
      if (getenv("EX_DEBUG") && w <= 2) fprintf(stderr, "core substeps %d\n", core.substeps());
      double layer[NS], removed[NS] = {0, 0, 0, 0}, change[NS]; int nemit[NS];
      for (auto &v : rk) {                                   // per rank: take layer markers
        double part[NS] = {0, 0, 0, 0}; size_t k = 0;
        for (size_t i = 0; i < v.size(); i++) {
          if (v[i].rho < RHOI + hmrho && ex.take(v[i].id, w)) part[v[i].z] += v[i].w; else v[k++] = v[i];
        }
        v.resize(k);
        for (int z = 0; z < NS; z++) removed[z] += part[z];  // the reduction
      }
      for (int z = 0; z < NS; z++) { layer[z] = 0; for (int i = NC-KC; i < NC; i++) layer[z] += nc[z*NC+i]*core.cell_volume(i); }
      ex.transfer(layer, removed, wref, change, nemit);
      for (int z = 0; z < NS; z++) {
        for (int i = NC-KC; i < NC; i++) nc[z*NC+i] = nc[z*NC+i]*(1 - ex.p_core_to_markers()) + removed[z]/Vc;
        double after = 0; for (int i = NC-KC; i < NC; i++) after += nc[z*NC+i]*core.cell_volume(i);
        if (fabs((after - layer[z]) - change[z]) > 1e-12*(fabs(layer[z]) + fabs(removed[z]) + 1e-300)) {
          fprintf(stderr, "core_change mismatch: applied %.17g returned %.17g\n", after - layer[z], change[z]); return 3; }
        for (int k = 0; k < nemit[z]; k++) {
          P p; p.id = next_id++; p.z = z; p.w = wref[z];
          p.rho = sqrt(RHOI*RHOI + CoreExchange::uniform(p.id, w, 11)*(pow(RHOI + hmrho, 2) - RHOI*RHOI));
          rk[p.id % R].push_back(p);
        }
        pendmax = std::max(pendmax, ex.pending(z)/wref[z]);
      }
      for (double v : nc) umin = std::min(umin, v);
    }
    if (getenv("EX_DEBUG") && w <= 2) fprintf(stderr, "(ref substeps before this window %d)\n", ref.substeps());
    ref.advance(tw, Df.data(), Zf.data(), Zf.data(), Zf.data(), nullptr, nullptr, nullptr, 0.0, true, nf.data());
    if (w % every == 0 || w == nwin) {
      double side[NS] = {0, 0, 0, 0};
      for (auto &v : rk) for (auto &p : v) if (p.rho >= RHOI) side[p.z] += p.w;
      for (int z = 0; z < NS; z++) terr[z] = std::max(terr[z], fabs(side[z] - ref_side(z))/tot0[z]);
    }
  }
  // final profiles on 25 + 25 bins of r_vol
  const int NB = 25; double cons[NS], drms[NS], jump = 0, checksum = 0;
  for (int z = 0; z < NS; z++) {
    V prof(2*NB, 0.0), rp(2*NB, 0.0), vol(2*NB);
    for (int b = 0; b < 2*NB; b++) { double e0 = rI*b/NB, e1 = rI*(b + 1)/NB; vol[b] = PI*(e1*e1 - e0*e0); }
    double mk = 0;
    for (auto &v : rk) for (auto &p : v) if (p.z == z) { mk += p.w; int b = std::min(2*NB-1, (int) (p.rho*sk/rI*NB)); prof[b] += p.w/vol[b]; }
    auto rebin = [&](const V &n, int N, const V &rr, const CoreTransport1D &ct, double edge, V &out, int nb) {
      double e = 0; for (int i = 0; i < N; i++) {
        double hi = (i < N-1) ? 0.5*(rr[i] + rr[i+1]) : edge;
        for (int b = 0; b < nb; b++) { double b0 = rI*b/NB, b1 = rI*(b + 1)/NB, l = std::max(e, b0), h = std::min(hi, b1); if (h > l) out[b] += n[i]*PI*(h*h - l*l)/vol[b]; }
        e = hi; }
    };
    if (!full) { V cz(nc.begin() + z*NC, nc.begin() + (z+1)*NC), o(2*NB, 0.0); rebin(cz, NC, rc, core, rI, o, NB); for (int b = 0; b < NB; b++) prof[b] = o[b]; }
    V fz(nf.begin() + z*NF, nf.begin() + (z+1)*NF); rebin(fz, NF, rf, ref, a, rp, 2*NB);
    double e = 0, sc = 0; for (int b = 0; b < 2*NB; b++) { e += pow(prof[b] - rp[b], 2); sc += fabs(rp[b]); }
    drms[z] = sqrt(e/(2*NB))/(sc/(2*NB));
    double tot = mk + (full ? 0.0 : core.content(z, nc.data()) + ex.pending(z));
    cons[z] = fabs(tot - inv0[z])/inv0[z]; checksum += tot*(z + 1) + mk;
    if (z == 3) jump = (prof[NB] + prof[NB+1] + prof[NB+2] - prof[NB-1] - prof[NB-2] - prof[NB-3])/3.0;
  }
  size_t nm = 0; for (auto &v : rk) nm += v.size();
  printf("{\"mode\":\"%s\",\"kappa\":%g,\"seed\":%llu,\"ranks\":%d,\"g2\":%.6f,\"Dtau_h2\":%.3f,\"disp_h\":%.3f,\"nsub\":%d,\"nwin\":%d,"
         "\"terr\":[%.6f,%.6f,%.6f,%.6f],\"drms\":[%.5f,%.5f,%.5f,%.5f],\"cons\":[%.2e,%.2e,%.2e,%.2e],\"jump\":%.5f,"
         "\"umin\":%.3g,\"pending_over_w\":%.4f,\"markers\":%zu,\"checksum\":%.15e}\n",
         argv[1], kappa, (unsigned long long) seed, R, g2, Deff*tw/(hc*hc), sig/hmrho, nsub, nwin,
         terr[0], terr[1], terr[2], terr[3], drms[0], drms[1], drms[2], drms[3], cons[0], cons[1], cons[2], cons[3], jump, umin, pendmax, nm, checksum);
  return 0;
}
