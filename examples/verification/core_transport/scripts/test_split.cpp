// Implicit split solver (CoreTransport1D::advance_split) checks:
//   relax    no transport, closed chain: BE steady state = ionization equilibrium, nuclei conserved
//   conv     make_inputs "source" case: error vs dt against a fine substepped reference (order ~ 1),
//            ledger closure, positivity at a very large step
//   pulse    SPARC pulse case (core_scan format): retained fraction and wall time, advance vs split
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "core_transport_1d.h"
using namespace SPARTA_NS;
typedef std::vector<double> V;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv)
{
  int fails = 0;
  auto check = [&](bool ok, const char *msg, double v) { printf("  %s %s (%.3g)\n", ok ? "ok  " : "FAIL", msg, v); fails += !ok; };
  if (!strcmp(argv[1], "relax")) {
    const int ir = 12, nion = 8; const size_t nn = (size_t) ir*nion;
    V rr(ir), D(nn, 0.0), C(nn, 0.0), S(nn, 0.0), R(nn, 0.0), n(nn, 0.0);
    for (int i = 0; i < ir; i++) rr[i] = 0.01*i;
    for (int z = 0; z < nion-1; z++) for (int i = 0; i < ir; i++) {
      S[z*ir+i] = 1e4*(1 + z)*(1 + 0.1*i); R[z*ir+i] = 3e3*(2 + 0.5*z)*(1 + 0.05*i);
    }
    for (int i = 0; i < ir; i++) n[i] = 1e18*(1 + i);
    CoreTransport1D ct(ir, nion); ct.set_grid(rr.data());
    double n0 = 0; for (int z = 0; z < nion; z++) n0 += ct.content(z, n.data());
    for (int k = 0; k < 400; k++) ct.advance_split(1e-3, D.data(), C.data(), S.data(), R.data(), nullptr, nullptr, 0.0, n.data());
    double n1 = 0, worst = 0, mn = 1e300;
    for (int z = 0; z < nion; z++) n1 += ct.content(z, n.data());
    for (int i = 0; i < ir; i++) for (int z = 0; z < nion-1; z++) {
      worst = std::max(worst, fabs(n[(z+1)*ir+i]*R[z*ir+i]/(n[z*ir+i]*S[z*ir+i]) - 1.0));
      mn = std::min(mn, n[z*ir+i]);
    }
    check(worst < 1e-9, "relaxes to n_{z+1}/n_z = S_z/R_z in every cell", worst);
    check(fabs(n1/n0 - 1) < 1e-12, "nuclei conserved through charge changes", fabs(n1/n0 - 1));
    check(mn >= 0, "non-negative", mn);
  } else if (!strcmp(argv[1], "conv")) {
    std::ifstream in(argv[2]);
    int ir, nion, nsteps, ievolve, isrc, in0, ipl; double dt, dlen;
    in >> ir >> nion >> nsteps >> dt >> dlen >> ievolve >> isrc >> in0 >> ipl;
    const size_t nn = (size_t) ir*nion;
    auto rd = [&](V &v, size_t m) { v.resize(m); for (auto &x : v) in >> x; };
    V rr, n0, pl, D, C, S, R, src, ni;
    rd(rr, ir); rd(n0, ir); rd(pl, ir); rd(D, nn); rd(C, nn); rd(S, nn); rd(R, nn); rd(src, nn); rd(ni, nn);
    CoreTransport1D ct(ir, nion); ct.set_grid(rr.data());
    const double T = 2e-3;
    V ref = ni;
    for (int k = 0; k < 2000; k++) ct.advance(T/2000, D.data(), C.data(), S.data(), R.data(), nullptr, src.data(), pl.data(), dlen, true, ref.data());
    double nrm = 0; for (double v : ref) nrm = std::max(nrm, fabs(v));
    double prev = 0, order = 0;
    for (int N : {20, 40, 80, 160, 320}) {
      V n = ni; double led = 0;
      for (int z = 0; z < nion; z++) led += ct.content(z, n.data());
      for (int k = 0; k < N; k++) {
        ct.advance_split(T/N, D.data(), C.data(), S.data(), R.data(), src.data(), pl.data(), dlen, n.data());
        for (int z = 0; z < nion; z++) {
          led -= (ct.outflux(z) + ct.parloss(z))*T/N;
          for (int i = 0; i < ir; i++) led += src[z*ir+i]*ct.cell_volume(i)*T/N;
        }
      }
      double e = 0, c = 0; for (size_t k = 0; k < nn; k++) e = std::max(e, fabs(n[k] - ref[k]));
      for (int z = 0; z < nion; z++) c += ct.content(z, n.data());
      e /= nrm;
      if (prev > 0) order = log2(prev/e);
      printf("  dt = T/%-4d  max error %.3e  order %.2f  ledger %.1e\n", N, e, order, fabs(c/led - 1));
      if (N == 320) { check(fabs(c/led - 1) < 1e-11, "ledger: content = initial + sources - outflux - parallel loss", fabs(c/led - 1));
                      check(e < 5e-3, "error at the finest step", e); }
      prev = e;
    }
    check(order > 0.9 && order < 1.2, "first-order temporal convergence", order);
    V n = ni; double mn = 1e300;
    ct.advance_split(1.0, D.data(), C.data(), S.data(), R.data(), src.data(), pl.data(), dlen, n.data());
    for (double v : n) mn = std::min(mn, v);
    check(mn >= 0, "non-negative after one step of 1 s (dt x max rate ~ 1e4)", mn);
  } else if (!strcmp(argv[1], "study")) {            // study case [h_accept]
    // time-resolved accuracy of the split step against the substepped reference: cumulative
    // return and its charge distribution at 10 us, 100 us, 1 ms, as a function of the
    // internal step h (a coupling window w with nsub steps has h = w/nsub)
    std::ifstream in(argv[2]);
    int ir, nion, nwin; double dtw, dlen;
    in >> ir >> nion >> nwin >> dtw >> dlen;
    const size_t nn = (size_t) ir*nion;
    V rr(ir), D(nn), C(nn, 0.0), S(nn), R(nn), ni(nn);
    for (auto &x : rr) in >> x; for (auto &x : D) in >> x; for (auto &x : S) in >> x; for (auto &x : R) in >> x; for (auto &x : ni) in >> x;
    CoreTransport1D ct(ir, nion); ct.set_grid(rr.data());
    double c0 = 0; for (int z = 0; z < nion; z++) c0 += ct.content(z, ni.data());
    const double tt[3] = {1e-5, 1e-4, 1e-3};
    auto runit = [&](double h, bool split, std::vector<V> &cum) {
      V n = ni, c(nion, 0.0); cum.assign(3, V(nion, 0.0));
      long nst = lround(tt[2]/h); int it = 0;
      for (long k = 1; k <= nst; k++) {
        if (split) ct.advance_split(h, D.data(), C.data(), S.data(), R.data(), nullptr, nullptr, dlen, n.data());
        else ct.advance(h, D.data(), C.data(), S.data(), R.data(), nullptr, nullptr, nullptr, dlen, true, n.data());
        for (int z = 0; z < nion; z++) c[z] += ct.outflux(z)*h/c0;
        while (it < 3 && k*h >= tt[it]*(1 - 1e-9)) cum[it++] = c;
      }
    };
    auto sum = [&](const V &c) { double s = 0; for (double v : c) s += v; return s; };
    auto zm = [&](const V &c) { double s = 0, w = 0; for (int z = 0; z < nion; z++) { s += z*c[z]; w += c[z]; } return w > 0 ? s/w : 0.0; };
    std::vector<V> ref, cur;
    runit(1e-6, false, ref);
    printf("  reference (substepped, 1 us windows): returned %.5f / %.5f / %.5f at 10 us / 100 us / 1 ms, <Z> of return %.3f / %.3f / %.3f\n",
           sum(ref[0]), sum(ref[1]), sum(ref[2]), zm(ref[0]), zm(ref[1]), zm(ref[2]));
    printf("  internal step   return error (10us 100us 1ms)     charge-distribution L1 error      <Z> error\n");
    const double haccept = argc > 3 ? atof(argv[3]) : 0.0;
    for (double h : {1e-4, 1e-5, 2.5e-6, 6.25e-7, 2e-7, 5e-8}) {
      runit(h, true, cur);
      double er[3], el[3], ez[3];
      for (int k = 0; k < 3; k++) {
        er[k] = sum(cur[k]) - sum(ref[k]); el[k] = 0;
        for (int z = 0; z < nion; z++) el[k] += fabs(cur[k][z] - ref[k][z]);
        ez[k] = zm(cur[k]) - zm(ref[k]);
      }
      bool defined0 = h <= tt[0]*(1 + 1e-9), defined1 = h <= tt[1]*(1 + 1e-9);
      printf("  %8.2e   %+8.4f %+8.4f %+8.4f     %7.4f %7.4f %7.4f     %+7.3f %+7.3f %+7.3f%s\n", h,
             defined0 ? er[0] : NAN, defined1 ? er[1] : NAN, er[2], defined0 ? el[0] : NAN, defined1 ? el[1] : NAN, el[2],
             defined0 ? ez[0] : NAN, defined1 ? ez[1] : NAN, ez[2], "");
      if (haccept > 0 && fabs(h/haccept - 1) < 1e-6) {
        double mr = std::max({fabs(er[0]), fabs(er[1]), fabs(er[2])}), ml = std::max({el[0], el[1], el[2]});
        check(mr <= 0.005, "cumulative return within 0.005 of the pulse at 10 us, 100 us, 1 ms", mr);
        check(ml <= 0.010, "returned inventory by charge state within 0.010 (L1)", ml);
      }
    }
  } else {                                           // pulse case nwin_max
    std::ifstream in(argv[2]);
    int ir, nion, nwin; double dtw, dlen;
    in >> ir >> nion >> nwin >> dtw >> dlen;
    nwin = std::min(nwin, atoi(argv[3]));
    const size_t nn = (size_t) ir*nion;
    V rr(ir), D(nn), C(nn, 0.0), S(nn), R(nn), ni(nn);
    for (auto &x : rr) in >> x; for (auto &x : D) in >> x; for (auto &x : S) in >> x; for (auto &x : R) in >> x; for (auto &x : ni) in >> x;
    CoreTransport1D ct(ir, nion); ct.set_grid(rr.data());
    auto content = [&](V &n) { double s = 0; for (int z = 0; z < nion; z++) s += ct.content(z, n.data()); return s; };
    auto zmean = [&](V &n) { double s = 0, w = 0; for (int z = 0; z < nion; z++) { double c = ct.content(z, n.data()); s += z*c; w += c; } return s/w; };
    const double c0 = content(ni);
    V n = ni; double t0 = now();
    for (int w = 0; w < nwin; w++) ct.advance(dtw, D.data(), C.data(), S.data(), R.data(), nullptr, nullptr, nullptr, dlen, true, n.data());
    double tref = now() - t0, rref = content(n)/c0, zref = zmean(n);
    printf("  %d cells, %d windows of %.0e s\n", ir, nwin, dtw);
    printf("  substepped advance : retained %.5f  <Z> %.3f  %.2f s  (%d substeps/window)\n", rref, zref, tref, ct.substeps());
    for (int nsub : {1, 4, 16, 64}) {
      n = ni; t0 = now(); double ret = 0;
      for (int w = 0; w < nwin; w++) {
        ct.advance_split(dtw, D.data(), C.data(), S.data(), R.data(), nullptr, nullptr, dlen, n.data(), nsub);
        for (int z = 0; z < nion; z++) ret += ct.outflux(z)*dtw;
      }
      double t = now() - t0, rr_ = content(n)/c0;
      printf("  split, %2d per window: retained %.5f (diff %+.5f)  <Z> %.3f (diff %+.3f)  %.3f s  speedup %.0fx  ledger %.1e\n",
             nsub, rr_, rr_ - rref, zmean(n), zmean(n) - zref, t, tref/t, fabs(rr_ + ret/c0 - 1));
    }
  }
  if (strcmp(argv[1], "pulse")) printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
