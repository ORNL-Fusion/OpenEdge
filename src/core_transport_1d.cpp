/* ----------------------------------------------------------------------
   OpenEdge: 1-D impurity transport, Linder scheme (see header)
------------------------------------------------------------------------- */

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include "core_transport_1d.h"

using namespace SPARTA_NS;

void CoreTransport1D::advance(double dt, const double *diff, const double *conv,
                              const double *s, const double *r, const double *n0,
                              const double *src, const double *pl, double dlen,
                              bool neutrals, double *n)
{
  if (!(dt > 0.0) || !std::isfinite(dt))
    throw std::runtime_error("core transport: invalid elapsed time");
  const size_t nn = (size_t) ir_*nion_;
  auto positive = [](double v) { return std::isfinite(v) && v >= 0.0; };
  for (size_t k = 0; k < nn; ++k)
    if (!positive(n[k]) || !positive(diff[k]) || !std::isfinite(conv[k]) ||
        !positive(s[k]) || !positive(r[k]) || (src && !positive(src[k])))
      throw std::runtime_error("core transport: negative or non-finite density/coefficient/source");
  for (int i = 0; i < ir_; ++i)
    if ((pl && !positive(pl[i])) || (n0 && !positive(n0[i])))
      throw std::runtime_error("core transport: invalid neutral source or parallel loss");

  // assemble(dt=1) has diagonal 1 + L/2, where L is the local
  // transport loss rate. Limit the FULL substep by L + S + R; this
  // leaves a margin in both explicit half-step right-hand sides.
  std::vector<double> zero(ir_,0.0), out(nion_,0.0), loss(nion_,0.0);
  double rate = 0.0;
  for (int z = neutrals ? 0 : 1; z < nion_; ++z) {
    size_t k = (size_t) z*ir_;
    assemble(zero.data(),diff+k,conv+k,1.0,dlen,pl);
    if (!positive(edge_coef_))
      throw std::runtime_error("core transport: outer boundary has negative flux coefficient");
    for (int i = 0; i < ir_; ++i) {
      double rec = z > 0 ? r[k-ir_+i] : 0.0;
      double v = 2.0*(b_[i]-1.0) + s[k+i] + rec;
      if (!positive(v) || a_[i] > 0.0 || c_[i] > 0.0)
        throw std::runtime_error("core transport: transport operator is not monotone");
      rate = std::max(rate,v);
    }
  }
  double count = std::max(1.0,ceil(dt*rate/0.5));
  if (!std::isfinite(count) || count > 10000000)
    throw std::runtime_error("core transport: excessive substep count");
  substeps_ = (int) count;
  double h = dt/substeps_;
  for (int j = 0; j < substeps_; ++j) {
    step(h,diff,conv,s,r,n0,src,pl,dlen,neutrals,n,true);
    for (size_t k = 0; k < nn; ++k)
      if (!positive(n[k]))
        throw std::runtime_error("core transport: negative or non-finite density after substep");
    for (int z = 0; z < nion_; ++z) {
      if (!positive(outflux_[z]) || !positive(parloss_[z]))
        throw std::runtime_error("core transport: negative or non-finite outgoing flux");
      out[z] += outflux_[z]/substeps_;
      loss[z] += parloss_[z]/substeps_;
    }
  }
  outflux_.swap(out); parloss_.swap(loss);
}

/* ---------------------------------------------------------------------- */

CoreTransport1D::CoreTransport1D(int ir, int nion) :
  ir_(ir), nion_(nion), rr_(ir), vol_(ir), a_(ir), b_(ir), c_(ir), d_(ir),
  outflux_(nion, 0.0), parloss_(nion, 0.0), edge_coef_(0.0) {}

/* ---------------------------------------------------------------------- */

void CoreTransport1D::set_grid(const double *rr)
{
  for (int i = 0; i < ir_; i++) rr_[i] = rr[i];
  // cell faces at the midpoints, axis cell from r = 0, outer cell to
  // rr[ir-1] + drm/2: volumes per unit length pi (rp^2 - rm^2)
  const double pi = 4.0 * atan(1.0);
  double rm = 0.0;
  for (int i = 0; i < ir_; i++) {
    double rp = (i < ir_ - 1) ? 0.5 * (rr_[i] + rr_[i+1])
                              : rr_[i] + 0.5 * (rr_[i] - rr_[i-1]);
    vol_[i] = pi * (rp*rp - rm*rm);
    rm = rp;
  }
}

/* ---------------------------------------------------------------------- */

double CoreTransport1D::content(int z, const double *n) const
{
  double s = 0.0;
  const double *nz = n + (size_t) z * ir_;
  for (int i = 0; i < ir_; i++) s += nz[i] * vol_[i];
  return s;
}

/* ----------------------------------------------------------------------
   transport matrix of one state for a half step dt: Crank-Nicolson
   diffusion, advection with the Linder flux limiter (kap), zero flux at
   the axis, decay-length outer boundary, parallel loss
------------------------------------------------------------------------- */

void CoreTransport1D::assemble(const double *rnt, const double *dimp,
                               const double *vimp, double dt, double dlen,
                               const double *par_loss)
{
  const int ir = ir_;
  const double *rr = rr_.data();
  double *a = a_.data(), *b = b_.data(), *c = c_.data(), *d = d_.data();
  for (int i = 0; i < ir; i++) { a[i] = 0.0; b[i] = 1.0; c[i] = 0.0; d[i] = rnt[i]; }

  auto limiter = [](double v, double D, double dr) {
    if (fabs(v) <= 2.0*D/dr) return 0.0;
    double k = std::max(0.0, 1.0 - 2.0*D/(dr*fabs(v)));
    return (v < 0.0) ? -k : k;
  };

  // axis cell: dn/dr = 0, v(0) = 0
  double drp = rr[1] - rr[0];
  double rp = 0.5*(rr[0] + rr[1]);
  double gi = 2.0/(rp*rp);
  double Dp = 0.5*(dimp[0] + dimp[1]);
  double coefp = 0.5*dt*gi*rp*Dp/drp;
  b[0] += coefp; c[0] -= coefp; d[0] += coefp*(rnt[1] - rnt[0]);
  double vp = 0.5*(vimp[0] + vimp[1]);
  double kap = limiter(vp, Dp, drp);
  coefp = 0.25*dt*gi*rp*vp;
  b[0] += coefp*(1.0 + kap); c[0] += coefp*(1.0 - kap);
  d[0] -= coefp*((1.0 + kap)*rnt[0] + (1.0 - kap)*rnt[1]);

  // interior cells
  double drm, rm, Dm, vm, kam, coefm;
  for (int i = 1; i < ir - 1; i++) {
    drm = drp; drp = rr[i+1] - rr[i];
    rm = rp; rp = 0.5*(rr[i] + rr[i+1]);
    gi = 2.0/(rp*rp - rm*rm);
    Dm = Dp; Dp = 0.5*(dimp[i] + dimp[i+1]);
    coefm = 0.5*dt*gi*rm*Dm/drm;
    coefp = 0.5*dt*gi*rp*Dp/drp;
    a[i] -= coefm; b[i] += coefm + coefp; c[i] -= coefp;
    d[i] += -coefm*(rnt[i] - rnt[i-1]) + coefp*(rnt[i+1] - rnt[i]);
    vm = vp; vp = 0.5*(vimp[i] + vimp[i+1]);
    kam = limiter(vm, Dm, drm);
    kap = limiter(vp, Dp, drp);
    coefm = 0.25*dt*gi*rm*vm;
    coefp = 0.25*dt*gi*rp*vp;
    a[i] -= coefm*(1.0 + kam);
    b[i] += -coefm*(1.0 - kam) + coefp*(1.0 + kap);
    c[i] += coefp*(1.0 - kap);
    d[i] += coefm*((1.0 + kam)*rnt[i-1] + (1.0 - kam)*rnt[i])
          - coefp*((1.0 + kap)*rnt[i] + (1.0 - kap)*rnt[i+1]);
  }

  // outer cell: face at rr[ir-1] + drm/2 (drp keeps the last interior
  // spacing)
  const int o = ir - 1;
  drm = drp; rm = rp; rp = rr[o] + 0.5*drm;
  gi = 2.0/(rp*rp - rm*rm);
  Dm = Dp; Dp = dimp[o];
  coefm = 0.5*dt*gi*rm*Dm/drm;
  coefp = 0.5*dt*gi*rp*Dp;
  a[o] -= coefm; b[o] += coefm; d[o] -= coefm*(rnt[o] - rnt[o-1]);
  if (dlen > 0.0) { b[o] += coefp/dlen; d[o] -= coefp/dlen*rnt[o]; }
  vm = vp; vp = std::max(0.0, vimp[o]);
  kam = limiter(vm, Dm, drm);
  kap = limiter(vp, Dp, drp);
  coefm = 0.25*dt*gi*rm*vm;
  coefp = 0.25*dt*gi*rp*vp;
  a[o] -= coefm*(1.0 + kam); b[o] -= coefm*(1.0 - kam);
  d[o] += coefm*((1.0 + kam)*rnt[o-1] + (1.0 - kam)*rnt[o]);
  edge_coef_ = 0.0;
  if (dlen > 0.0) {
    double f = 2.0 - (1.0 - kap)*drp/dlen;
    b[o] += coefp*f; d[o] -= coefp*f*rnt[o];
    edge_coef_ = rp*(Dp/dlen + 0.5*vp*f);
  }

  // parallel loss: interior cells, outer cell only with the open boundary
  if (par_loss) {
    for (int i = 1; i < ir - 1; i++) {
      coefm = 0.5*dt*par_loss[i];
      b[i] += coefm; d[i] -= coefm*rnt[i];
    }
    if (dlen > 0.0) {
      coefm = 0.5*dt*par_loss[o];
      b[o] += coefm; d[o] -= coefm*rnt[o];
    }
  }
}

/* ---------------------------------------------------------------------- */

void CoreTransport1D::tdma(double *sol)
{
  const int n = ir_;
  const double *a = a_.data(), *b = b_.data(), *c = c_.data(), *d = d_.data();
  std::vector<double> bp(n), dp(n);
  bp[0] = b[0]; dp[0] = d[0];
  for (int i = 1; i < n; i++) {
    bp[i] = b[i] - a[i]*c[i-1]/bp[i-1];
    dp[i] = d[i] - a[i]*dp[i-1]/bp[i-1];
  }
  sol[n-1] = dp[n-1]/bp[n-1];
  for (int i = n - 2; i >= 0; i--) sol[i] = (dp[i] - c[i]*sol[i+1])/bp[i];
}

/* ----------------------------------------------------------------------
   losses of one state over one half step, from the assembled boundary
   and sink terms (exactly what the matrix removed)
------------------------------------------------------------------------- */

void CoreTransport1D::tally(int z, const double *n_old, const double *n_new,
                            double dt, double dlen, const double *par_loss)
{
  const double pi = 4.0 * atan(1.0);
  const int o = ir_ - 1;
  outflux_[z] += 2.0*pi*edge_coef_*0.5*(n_new[o] + n_old[o])*dt;
  if (!par_loss) return;
  double s = 0.0;
  for (int i = 1; i < ir_ - 1; i++) s += vol_[i]*par_loss[i]*0.5*(n_new[i] + n_old[i]);
  if (dlen > 0.0) s += vol_[o]*par_loss[o]*0.5*(n_new[o] + n_old[o]);
  parloss_[z] += s*dt;
}

/* ---------------------------------------------------------------------- */

void CoreTransport1D::step(double det, const double *diff, const double *conv,
                           const double *s_rates, const double *r_rates,
                           const double *n0, const double *src,
                           const double *par_loss, double dlen,
                           bool evolve_neutrals, double *n, bool check)
{
  const int ir = ir_, nion = nion_;
  const double dt = 0.5*det;
  std::vector<double> ra((size_t) nion*ir), old(ir);
  std::copy(n, n + (size_t) nion*ir, ra.begin());
  std::fill(outflux_.begin(), outflux_.end(), 0.0);
  std::fill(parloss_.begin(), parloss_.end(), 0.0);
  auto A = [&](const double *p, int z) { return p + (size_t) z*ir; };
  auto N = [&](int z) { return n + (size_t) z*ir; };
  auto validate = [&](int z) {
    if (check) for (int i = 0; i < ir; ++i)
      if (!std::isfinite(N(z)[i]) || N(z)[i] < 0.0)
        throw std::runtime_error("core transport: negative or non-finite half-step density");
  };
  double *d = d_.data(), *b = b_.data();
  const int ns = evolve_neutrals ? 0 : 1;
  if (ns == 1) for (int i = 0; i < ir; i++) n[i] = n0 ? n0[i] : 0.0;

  // explicit sources of state z (volume rates); n0 feeds the neutrals
  auto sources = [&](int z) {
    if (src) for (int i = 0; i < ir; i++) d[i] += dt*src[(size_t) z*ir+i];
    if (z == 0 && n0) for (int i = 0; i < ir; i++) d[i] += dt*n0[i];
  };

  // first half step, charge states upward: ionization loss implicit,
  // ionization gain from the already updated lower state, recombination
  // explicit from the old densities
  for (int z = ns; z < nion; z++) {
    assemble(A(ra.data(),z), A(diff,z), A(conv,z), dt, dlen, par_loss);
    const double *sz = A(s_rates,z), *rz = A(r_rates,z);
    for (int i = 0; i < ir; i++) {
      b[i] += dt*sz[i];
      if (z > 0) d[i] += dt*(n[(size_t)(z-1)*ir+i]*s_rates[(size_t)(z-1)*ir+i]
                             - ra[(size_t) z*ir+i]*r_rates[(size_t)(z-1)*ir+i]);
      if (z < nion-1) d[i] += dt*ra[(size_t)(z+1)*ir+i]*rz[i];
    }
    sources(z);
    tdma(N(z));
    validate(z);
    tally(z, A(ra.data(),z), N(z), dt, dlen, par_loss);
  }

  // second half step, charge states downward: recombination loss
  // implicit, recombination gain from the updated upper state,
  // ionization explicit
  for (int z = nion - 1; z >= ns; z--) {
    std::copy(N(z), N(z) + ir, old.begin());
    assemble(old.data(), A(diff,z), A(conv,z), dt, dlen, par_loss);
    const double *sz = A(s_rates,z), *rz = A(r_rates,z);
    for (int i = 0; i < ir; i++) {
      d[i] -= dt*old[i]*sz[i];
      if (z > 0) {
        b[i] += dt*r_rates[(size_t)(z-1)*ir+i];
        d[i] += dt*n[(size_t)(z-1)*ir+i]*s_rates[(size_t)(z-1)*ir+i];
      }
      if (z < nion-1) d[i] += dt*n[(size_t)(z+1)*ir+i]*rz[i];
    }
    sources(z);
    tdma(N(z));
    validate(z);
    tally(z, old.data(), N(z), dt, dlen, par_loss);
  }

  for (int z = 0; z < nion; z++) { outflux_[z] /= det; parloss_[z] /= det; }
}

/* ----------------------------------------------------------------------
   implicit split: transport (backward Euler, same monotone operator as
   assemble) then charge chain (backward Euler per cell)
------------------------------------------------------------------------- */

void CoreTransport1D::advance_split(double dt, const double *diff, const double *conv,
                                    const double *s, const double *r, const double *src,
                                    const double *pl, double dlen, double *n, int nsub)
{
  if (!(dt > 0.0) || !std::isfinite(dt) || nsub < 1)
    throw std::runtime_error("core transport: invalid elapsed time or substep count");
  const int ir = ir_, nion = nion_;
  const size_t nn = (size_t) ir*nion;
  auto positive = [](double v) { return std::isfinite(v) && v >= 0.0; };
  for (size_t k = 0; k < nn; ++k)
    if (!positive(n[k]) || !positive(diff[k]) || !std::isfinite(conv[k]) ||
        !positive(s[k]) || !positive(r[k]) || (src && !positive(src[k])))
      throw std::runtime_error("core transport: negative or non-finite density/coefficient/source");
  if (pl) for (int i = 0; i < ir; ++i)
    if (!positive(pl[i])) throw std::runtime_error("core transport: invalid parallel loss");

  const double pi = 4.0*atan(1.0), h = dt/nsub;
  const int o = ir - 1;
  std::vector<double> out(nion,0.0), loss(nion,0.0), lo(nion), di(nion), up(nion), rhs(nion);
  for (int j = 0; j < nsub; ++j) {
    // transport: assemble(.,2h) leaves (I + h L) in a,b,c; right-hand side = old + h src
    for (int z = 0; z < nion; ++z) {
      double *nz = n + (size_t) z*ir;
      assemble(nz,diff+(size_t) z*ir,conv+(size_t) z*ir,2.0*h,dlen,pl);
      for (int i = 0; i < ir; ++i) {
        if (a_[i] > 0.0 || c_[i] > 0.0)
          throw std::runtime_error("core transport: transport operator is not monotone");
        d_[i] = nz[i] + (src ? h*src[(size_t) z*ir+i] : 0.0);
      }
      tdma(nz);
      out[z] += 2.0*pi*edge_coef_*nz[o]*h;
      if (pl) {
        double sl = 0.0;
        for (int i = 1; i < ir - 1; ++i) sl += vol_[i]*pl[i]*nz[i];
        if (dlen > 0.0) sl += vol_[o]*pl[o]*nz[o];
        loss[z] += sl*h;
      }
    }
    // charge chain per cell: (I - h K) x = n, K columns sum to zero
    for (int i = 0; i < ir; ++i) {
      for (int z = 0; z < nion; ++z) {
        double sz = (z < nion-1) ? s[(size_t) z*ir+i] : 0.0;
        double rdn = (z > 0) ? r[(size_t)(z-1)*ir+i] : 0.0;
        di[z] = 1.0 + h*(sz + rdn);
        lo[z] = (z > 0) ? -h*s[(size_t)(z-1)*ir+i] : 0.0;
        up[z] = (z < nion-1) ? -h*r[(size_t) z*ir+i] : 0.0;
        rhs[z] = n[(size_t) z*ir+i];
      }
      for (int z = 1; z < nion; ++z) {
        double m = lo[z]/di[z-1];
        di[z] -= m*up[z-1]; rhs[z] -= m*rhs[z-1];
      }
      n[(size_t)(nion-1)*ir+i] = rhs[nion-1]/di[nion-1];
      for (int z = nion-2; z >= 0; --z)
        n[(size_t) z*ir+i] = (rhs[z] - up[z]*n[(size_t)(z+1)*ir+i])/di[z];
    }
    for (size_t k = 0; k < nn; ++k)
      if (!positive(n[k]))
        throw std::runtime_error("core transport: negative or non-finite density after split step");
  }
  for (int z = 0; z < nion; ++z) {
    if (!positive(out[z]) || !positive(loss[z]))
      throw std::runtime_error("core transport: negative or non-finite outgoing flux");
    outflux_[z] = out[z]/dt; parloss_[z] = loss[z]/dt;
  }
  substeps_ = nsub;
}
