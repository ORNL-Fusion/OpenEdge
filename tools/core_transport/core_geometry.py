#!/usr/bin/env python3
"""Metric factors that map physical cross-field coefficients onto the 1-D r_vol operator.

Flux-surface averaged transport of n(r), r = r_vol, with a physical diffusivity D_perp and a
physical pinch speed v_perp along the surface normal:

    dn/dt = 1/V' d/dr [ V' ( <|grad r|^2> D_perp dn/dr - <|grad r|> v_perp n ) ],   V' = 4 pi^2 R0 r

The solver's operator is (1/r) d/dr [ r (D_eff dn/dr - v_eff n) ], therefore

    D_eff(r) = g2(r) D_perp,   g2 = <|grad r_vol|^2>
    v_eff(r) = g1(r) v_perp,   g1 = <|grad r_vol|>

(D_perp, v_perp taken constant on a flux surface; <.> is the volume average over the shell
between neighbouring surfaces.)  The particle flux through a surface is
-V' (g2 D_perp dn/dr - g1 v_perp n), and its area is S = V' g1.

    python tools/core_transport/core_geometry.py --case <case dir>              # SPARC plasma.h5 -> input/core_geometry.dat
    python tools/core_transport/core_geometry.py --case <case dir> --selftest   # circular and elliptical tori vs closed forms
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import label


AXIS_PSI = 0.08


def metric(r, z, psin, psimax=0.95, refine=16, nbin=30):
    """g1, g2 vs r_vol for the closed region around the axis of psin[z, r] (normalised)."""
    sp = RectBivariateSpline(z, r, psin, kx=3, ky=3)
    rf = np.linspace(r[0], r[-1], (len(r) - 1) * refine + 1)
    zf = np.linspace(z[0], z[-1], (len(z) - 1) * refine + 1)
    p = sp(zf, rf); pr = sp(zf, rf, dy=1); pz = sp(zf, rf, dx=1)
    ka, ja = np.unravel_index(np.argmin(p), p.shape)
    lab, _ = label(p < psimax); core = lab == lab[ka, ja]
    R = np.broadcast_to(rf, p.shape)
    dv = 2 * np.pi * R * (rf[1] - rf[0]) * (zf[1] - zf[0])
    raxis = rf[ja]
    order = np.argsort(p[core]); ps = p[core][order]; vc = np.cumsum(dv[core][order])
    rv_of_p = lambda q: np.sqrt(np.interp(q, ps, vc) / (2 * np.pi**2 * raxis))
    # shell averages of |grad psi| and |grad psi|^2, then chain rule with dr_vol/dpsi
    edges = np.linspace(ps[0], psimax, nbin + 1); pc = 0.5 * (edges[1:] + edges[:-1])
    k = np.clip(np.digitize(p[core], edges) - 1, 0, nbin - 1)
    w = dv[core]; g = np.hypot(pr, pz)[core]
    ws = np.bincount(k, w, nbin)
    a1 = np.bincount(k, w * g, nbin) / ws; a2 = np.bincount(k, w * g * g, nbin) / ws
    drdp = (rv_of_p(edges[1:]) - rv_of_p(edges[:-1])) / np.diff(edges)
    rv = rv_of_p(pc)
    g1, g2 = a1 * drdp, a2 * drdp**2
    # the innermost shells hold too few grid points for a shell average: carry the first
    # resolved value (psi_N >= AXIS_PSI) to the axis, where the factors tend to a constant
    inner = pc < AXIS_PSI
    if inner.any() and (~inner).any():
        j = np.argmax(~inner); g1[inner], g2[inner] = g1[j:j + 3].mean(), g2[j:j + 3].mean()
    return dict(rvol=rv, psin=pc, g1=g1, g2=g2, raxis=raxis,
                vprime=4 * np.pi**2 * raxis * rv, shell_volume=ws, dpsi=np.diff(edges))


def torus(kappa, R0=100.0, a=1.0, n=161):
    """psin = x^2/a^2 + z^2/(kappa a)^2 around R0: large aspect ratio -> closed forms"""
    r = np.linspace(R0 - 1.3 * a, R0 + 1.3 * a, n); z = np.linspace(-1.3 * kappa * a, 1.3 * kappa * a, n)
    Z, Rr = np.meshgrid(z, r, indexing="ij")
    return r, z, ((Rr - R0) / a) ** 2 + (Z / (kappa * a)) ** 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--case", default=".", help="case directory holding input/plasma.h5 (default: current directory)")
    a = ap.parse_args()
    if a.selftest:
        ok = True
        for kappa in (1.0, 1.7, 2.5):
            r, z, p = torus(kappa)
            m = metric(r, z, p, psimax=1.0, refine=12)
            sel = (m["psin"] > 0.1) & (m["psin"] < 0.9)
            # straight elliptical cylinder limit: r_vol = rho sqrt(kappa), <|grad rho|^2> = (1 + 1/kappa^2)/2
            g2x = (kappa**2 + 1) / (2 * kappa)
            e2 = np.max(np.abs(m["g2"][sel] / g2x - 1))
            # <|grad r|> = sqrt(kappa) * (2/pi) E(1 - 1/kappa^2)  (complete elliptic integral)
            from scipy.special import ellipe
            g1x = np.sqrt(kappa) * 2 / np.pi * ellipe(1 - 1 / kappa**2)
            e1 = np.max(np.abs(m["g1"][sel] / g1x - 1))
            good = e2 < 5e-3 and e1 < 5e-3; ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} kappa {kappa}: g2 {m['g2'][sel].mean():.4f} (exact {g2x:.4f}, err {e2:.1e}), "
                  f"g1 {m['g1'][sel].mean():.4f} (exact {g1x:.4f}, err {e1:.1e})")
        print("PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
    import h5py
    P = h5py.File(Path(a.case) / "input/plasma.h5", "r")
    E = P["equilibrium"]
    psin = (E["psi"][:] - E["psi_axis"][()]) / (E["psib"][()] - E["psi_axis"][()])
    m = metric(E["r"][:], E["z"][:], psin)
    out = Path(a.case) / "input/core_geometry.dat"
    np.savetxt(out, np.column_stack([np.sqrt(m["psin"]), m["rvol"], m["g1"], m["g2"]]), fmt="%.6e",
               header="metric factors for the r_vol operator: D_eff = g2 D_perp, v_eff = g1 v_perp\n"
                      "(D_perp constant on a flux surface; v_perp = pinch speed along the surface normal)\n"
                      f"axis_extrapolated_below_rho_N {np.sqrt(AXIS_PSI):.6f}   (first resolved shell carried to the axis; not a shell measurement)\n"
                      "rho_N r_vol(m) g1=<|grad r_vol|> g2=<|grad r_vol|^2>")
    for pv in (0.05, 0.2, 0.5, 0.9):
        i = np.argmin(np.abs(m["psin"] - pv))
        print(f"psi_N {m['psin'][i]:.3f}: r_vol {m['rvol'][i]:.4f} m, g1 {m['g1'][i]:.4f}, g2 {m['g2'][i]:.4f}, "
              f"surface area {m['vprime'][i] * m['g1'][i]:.3f} m2")
    print("wrote", out)
