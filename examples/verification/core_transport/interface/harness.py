#!/usr/bin/env python3
"""1-D slab interface harness: continuum core on [0, xI], weighted Brownian markers on [xI, L].

Diffusion only, constant D, reflecting walls at 0 and L, nondimensional (L = D = 1).
Couplings:
  aux     two-sided auxiliary layers: both descriptions reflect at xI; a core layer (width hc)
          and a marker layer (width hm) exchange with conductance G = D/l, l = centre distance,
          directional rates G n_c and G n_m, exact two-box update over the coupling interval.
          The inventory owed to the marker side leaves the continuum at once and waits in a
          per-label bank (part of the ledger, always < one marker weight after emission), so
          rounding to whole markers can never overdraw the core layer.
  legacy  diffusion ANALOGUE of the present fix (not the fix itself): markers crossing xI are
          absorbed and deposited with an exponential depth kernel before the continuum step,
          the core edge is absorbing (zero density half a cell beyond the last centre), its
          outflux is re-emitted as Brownian markers at xI. The fix spreads the influx over
          the window and gives emitted markers a velocity; numbers here are not SPARC numbers.
Counters: endpoint transfers (where inventory ends after each coupling interval; crossings
that return within the interval are not counted) and the expected number of actual jumps
N_c k tau for comparison.
References: full-domain finite volume (fine grid) and full-domain Brownian markers.
Tests: equilibrium, mode (imposed gradient, exact decay), pulse_core, pulse_mc, mixing.
"""
import argparse, json, sys
import numpy as np
from scipy.linalg import solve_banded

L, D, XI = 1.0, 1.0, 0.5


# ---------------------------------------------------------------- continuum
class FV:
    """Crank-Nicolson finite volume on [0, xmax], zero flux at 0; right end closed or absorbing."""
    def __init__(self, n, xmax, absorbing=False):
        self.n, self.dx = n, xmax / n
        self.x = (np.arange(n) + 0.5) * self.dx
        self.absorbing = absorbing
        self._dt = None

    def _build(self, dt):
        n, r = self.n, D * dt / self.dx**2
        lo = np.full(n, -r); up = np.full(n, -r); di = np.full(n, 2 * r)
        di[0] = r
        di[-1] = 3 * r if self.absorbing else r           # ghost value -u at half a cell
        self.A = np.zeros((3, n))
        self.A[0, 1:] = 0.5 * up[:-1]; self.A[1] = 1 + 0.5 * di; self.A[2, :-1] = 0.5 * lo[1:]
        self.di, self.r, self._dt = di, r, dt

    def step(self, u, tau):
        """advance u[..., n] by tau; returns the inventory lost through an absorbing end"""
        nsub = max(1, int(np.ceil(D * tau / self.dx**2 / 0.5)))   # r <= 0.5: positive Crank-Nicolson
        dt = tau / nsub
        if self._dt != dt: self._build(dt)
        lost = np.zeros(u.shape[:-1])
        for _ in range(nsub):
            before = u.sum(-1)
            rhs = u - 0.5 * self.di * u
            rhs[..., 1:] += 0.5 * self.r * u[..., :-1]
            rhs[..., :-1] += 0.5 * self.r * u[..., 1:]
            u[...] = solve_banded((1, 1), self.A, rhs.T).T
            lost += (before - u.sum(-1)) * self.dx
        return lost


def initial(test, x):
    """two label fields a, b on points x (total = a + b)"""
    one = np.ones_like(x)
    if test == "equilibrium": return np.stack([one, 0 * one])
    if test == "mode": return np.stack([one + 0.5 * np.cos(np.pi * x / L), 0 * one])
    if test == "pulse_core": return np.stack([np.exp(-0.5 * ((x - 0.30) / 0.03) ** 2), 0 * one])
    if test == "pulse_mc": return np.stack([np.exp(-0.5 * ((x - 0.70) / 0.03) ** 2), 0 * one])
    if test == "mixing": return np.stack([(x < XI) * one, (x >= XI) * one])
    raise ValueError(test)


def reference(test, T, nout, n=2000):
    """fine full-domain FV: label profiles at T, and A-inventory on the marker side vs time"""
    fv = FV(n, L); u = initial(test, fv.x); t = np.linspace(0, T, nout + 1)
    right = fv.x >= XI; side = [u[0, right].sum() * fv.dx]
    for _ in range(nout):
        fv.step(u, T / nout); side.append(u[0, right].sum() * fv.dx)
    return fv.x, u, t, np.array(side)


# ---------------------------------------------------------------- markers
def sample_markers(test, n, lo, hi, rng):
    """equal-weight markers with labels drawn from the initial fields on [lo, hi]"""
    xs = np.linspace(lo, hi, 4001); xm = 0.5 * (xs[1:] + xs[:-1]); f = initial(test, xm)
    dens = f.sum(0); inv = dens.sum() * (xs[1] - xs[0])
    cdf = np.concatenate([[0], np.cumsum(dens)]); cdf /= cdf[-1]
    x = np.interp(rng.random(n), cdf, xs)
    k = np.clip(np.searchsorted(xs, x) - 1, 0, len(xm) - 1)
    lab = (rng.random(n) < f[1, k] / np.maximum(dens[k], 1e-300)).astype(np.int8)
    return x, lab, inv


def brownian(x, dt, lo, hi, rng):
    x += np.sqrt(2 * D * dt) * rng.standard_normal(x.size)
    for _ in range(3):                                   # reflect (steps are << hi - lo)
        np.abs(x - lo, out=x); x += lo
        x[:] = hi - np.abs(hi - x)
    return x


# ---------------------------------------------------------------- runs
def run(test, mode="aux", T=0.1, nc=50, dtp=2e-5, tau=2e-4, h=None, hm=None, depth=0.0,
        nmark=200000, seed=1, nout=50):
    """returns dict with profiles at T, marker-side A inventory vs time, conservation, counters"""
    rng = np.random.default_rng(seed)
    full = mode == "brownian"
    lo = 0.0 if full else XI
    fv = None if full else FV(nc, XI, absorbing=(mode == "legacy"))
    u = None if full else initial(test, fv.x)
    # marker weight from the full-domain inventory so that both sides share one w0
    xs = np.linspace(0, L, 8001); xm = 0.5 * (xs[1:] + xs[:-1])
    inv_total = initial(test, xm).sum() * (xs[1] - xs[0])
    xq = xm[xm >= lo]; inv_m = initial(test, xq).sum() * (xs[1] - xs[0])
    w0 = inv_total / nmark
    x, lab, _ = sample_markers(test, int(round(inv_m / w0)), lo, L, rng)
    w0 = inv_m / x.size if x.size else w0                # exact initial marker inventory
    inv0 = inv_m + (0.0 if full else u.sum() * fv.dx)
    inv0_lab = w0 * np.bincount(lab, minlength=2) + (0.0 if full else u.sum(1) * fv.dx)
    umin_run = 0.0 if full else float(u.min())
    bank = np.zeros(2); jumps_cm = 0.0; ratios = {}
    if mode == "aux":
        hc = h if h else fv.dx; hmm = hm if hm else hc
        kc = max(1, int(round(hc / fv.dx))); hc = kc * fv.dx
        ell = 0.5 * (hc + hmm); G = D / ell
        ratios = {"Dtau_hc2": D * tau / hc**2, "Dtau_hm2": D * tau / hmm**2,
                  "step_hm": float(np.sqrt(D * dtp) / hmm), "disp_hm": float(np.sqrt(2 * D * dtp) / hmm), "hc": hc, "hm": hmm, "dx": fv.dx}
    if mode == "legacy":
        if depth <= 0: depo = np.zeros(nc); depo[-1] = 1.0
        else: depo = np.exp(-(XI - fv.x) / depth); depo /= depo.sum()
    nwin = int(round(T / tau)); nsub = max(1, int(round(tau / dtp))); dt = tau / nsub
    every = max(1, nwin // nout)
    nb = 50; edges = np.linspace(0, L, 2 * nb + 1); xc = 0.5 * (edges[1:] + edges[:-1])
    pacc, nacc = np.zeros((2, 2 * nb)), 0

    def snapshot():
        """label profiles on the common grid (equilibrium: averaged over the second half)"""
        p = np.zeros((2, 2 * nb))
        for l in (0, 1):
            p[l] = w0 * np.histogram(x[lab == l], edges)[0] / (edges[1] - edges[0])
            if not full:                                   # conservative rebin of the core cells
                cum = np.concatenate([[0], np.cumsum(u[l]) * fv.dx])
                p[l, :nb] = np.diff(np.interp(edges[:nb + 1], np.linspace(0, XI, nc + 1), cum)) / (edges[1] - edges[0])
        return p
    t_hist, side = [0.0], [w0 * np.count_nonzero((lab == 0) & (x >= XI))]
    gross_cm = gross_mc = 0.0; net_cm = np.zeros(2)
    for iw in range(1, nwin + 1):
        if mode == "legacy":
            absorbed = np.zeros(2)
            for _ in range(nsub):
                x += np.sqrt(2 * D * dt) * rng.standard_normal(x.size)
                x[:] = L - np.abs(L - x)
                hit = x < XI
                if hit.any():
                    absorbed += w0 * np.bincount(lab[hit], minlength=2)
                    x, lab = x[~hit], lab[~hit]
            u += absorbed[:, None] * depo / fv.dx
            lost = fv.step(u, tau)
            bank += lost
            nem = np.floor(bank / w0).astype(int); bank -= nem * w0
            for l in (0, 1):
                if nem[l]:
                    x = np.concatenate([x, np.full(nem[l], XI + 1e-12)])
                    lab = np.concatenate([lab, np.full(nem[l], l, np.int8)])
            gross_mc += absorbed.sum(); gross_cm += (nem * w0).sum(); net_cm += nem * w0 - absorbed
        else:
            for _ in range(nsub): brownian(x, dt, lo, L, rng)
            if not full: fv.step(u, tau)
        if mode == "aux":
            # exact two-box exchange over tau: rates k_c = G/hc (core -> markers), k_m = G/hm
            kcr, kmr = G / hc, G / hmm
            e = 1.0 - np.exp(-(kcr + kmr) * tau)
            p_cm, p_mc = kcr / (kcr + kmr) * e, kmr / (kcr + kmr) * e
            inl = np.nonzero(x < XI + hmm)[0]
            go = inl[rng.random(inl.size) < p_mc]
            into_core = w0 * np.bincount(lab[go], minlength=2)
            layer = u[:, -kc:]
            tot = layer.sum(1) * fv.dx
            jumps_cm += tot.sum() * kcr * tau
            # the owed inventory leaves the continuum now (fraction p_cm < 1: never overdrawn)
            # and waits in the bank until it makes whole markers
            bank += tot * p_cm
            u[:, -kc:] = layer * (1.0 - p_cm) + into_core[:, None] / hc
            nem = np.floor(bank / w0).astype(int); bank -= nem * w0
            out_inv = nem * w0
            keep = np.ones(x.size, bool); keep[go] = False
            x, lab = x[keep], lab[keep]
            for l in (0, 1):
                if nem[l]:
                    x = np.concatenate([x, XI + hmm * rng.random(nem[l])])
                    lab = np.concatenate([lab, np.full(nem[l], l, np.int8)])
            gross_cm += out_inv.sum(); gross_mc += into_core.sum(); net_cm += out_inv - into_core
        if not full: umin_run = min(umin_run, float(u.min()))
        if iw % every == 0 or iw == nwin:
            t_hist.append(iw * tau); side.append(w0 * np.count_nonzero((lab == 0) & (x >= XI)))
        if iw == nwin or (test == "equilibrium" and iw > nwin // 2 and iw % max(1, nwin // 200) == 0):
            pacc += snapshot(); nacc += 1
    prof = pacc / nacc
    inv_lab = w0 * np.bincount(lab, minlength=2) + (0.0 if full else u.sum(1) * fv.dx) + bank
    finite = bool(np.isfinite(x).all() and (full or np.isfinite(u).all()))
    return dict(x=xc, prof=prof, t=np.array(t_hist), side=np.array(side), inv0=inv0, inv=inv_lab.sum(),
                inv0_lab=inv0_lab, inv_lab=inv_lab, endpoint_cm=gross_cm, endpoint_mc=gross_mc,
                jumps_cm=jumps_cm, net_cm=net_cm, w0=w0, nmark=x.size, bank=bank, ratios=ratios,
                finite=finite, umin=(None if full else umin_run))


def metrics(test, res, T):
    """density, flux/transfer, conservation and mixing errors against the fine FV reference"""
    xr, ur, tr, sr = reference(test, T, 200)
    nb = res["x"].size; dxr = xr[1] - xr[0]
    ref = np.array([[ur[l, (xr >= a) & (xr < b)].mean() for a, b in zip(np.linspace(0, L, nb + 1)[:-1], np.linspace(0, L, nb + 1)[1:])] for l in (0, 1)])
    tot, rtot = res["prof"].sum(0), ref.sum(0)
    scale = rtot.mean()
    m = {}
    m["dens_rms"] = float(np.sqrt(np.mean((tot - rtot) ** 2)) / scale)
    k = nb // 2
    m["jump"] = float((tot[k:k + 3].mean() - tot[k - 3:k].mean() - (rtot[k:k + 3].mean() - rtot[k - 3:k].mean())) / scale)
    m["side_core_err"] = float((tot[:k].mean() - rtot[:k].mean()) / scale)
    sref = np.interp(res["t"], tr, sr)
    m["transfer_err"] = float(np.max(np.abs(res["side"] - sref)) / max(ur[0].sum() * dxr, 1e-300))
    m["cons"] = float(np.max(np.abs(res["inv_lab"] - res["inv0_lab"])) / res["inv0"])   # per label
    m["endpoint_over_inv"] = float(res["endpoint_cm"] / res["inv0"])
    m["jumps_over_inv"] = float(res["jumps_cm"] / res["inv0"])
    m["umin"] = res["umin"]; m["finite"] = res["finite"]; m["bank_over_w0"] = float(res["bank"].max() / res["w0"])
    m.update(res["ratios"])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test"); ap.add_argument("--mode", default="aux")
    ap.add_argument("--T", type=float, default=0.1); ap.add_argument("--nc", type=int, default=50)
    ap.add_argument("--dtp", type=float, default=2e-5); ap.add_argument("--tau", type=float, default=2e-4)
    ap.add_argument("--h", type=float, default=None); ap.add_argument("--hm", type=float, default=None)
    ap.add_argument("--depth", type=float, default=0.0)
    ap.add_argument("--nmark", type=int, default=200000); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    r = run(a.test, a.mode, a.T, a.nc, a.dtp, a.tau, a.h, a.hm, a.depth, a.nmark, a.seed)
    print(json.dumps(metrics(a.test, r, a.T)))
