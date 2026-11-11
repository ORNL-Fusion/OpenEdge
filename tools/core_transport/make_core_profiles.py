#!/usr/bin/env python3
"""Core plasma profiles for fix core/transport (keyword `profiles`).

The SOLPS-WG background stops at its core cut (psi_N ~ 0.2).  This script flux-surface
averages n_e, T_e, T_i of input/plasma.h5 against the volume radius r_vol, fits value and
slope at the cut (least squares over the innermost FITW metres of SOLPS data) and
continues the profiles to the axis:

  parabolic  f = f_c + s (r^2 - r_c^2) / (2 r_c)   value + slope at the cut, zero gradient on axis (default)
  linear     f = f_c + s (r - r_c)
  flat       f = f_c

Output: input/core_profiles.dat, columns rho_N = sqrt(psi_N), n_e (m-3), T_e (eV), T_i (eV), r_vol (m).

    python tools/core_transport/make_core_profiles.py --case <case dir> [--mode parabolic|linear|flat] [--case DIR]
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

FITW = 0.03          # m of r_vol outside the cut used for the slope fit
REFINE = 8           # equilibrium grid refinement for the flux-surface volumes
FLOOR = 0.05         # extrapolated values never drop below FLOOR x value at the cut


def flux_volumes(P, psimax=1.0):
    """r_vol(psi_N) of the closed region around the axis, from the refined psi map."""
    r, z = P["equilibrium/r"][:], P["equilibrium/z"][:]
    pa, pb = float(P["equilibrium/psi_axis"][()]), float(P["equilibrium/psib"][()])
    psin = (P["equilibrium/psi"][:] - pa) / (pb - pa)                 # [z, r]
    f = RegularGridInterpolator((z, r), psin, method="cubic")
    rf = np.linspace(r[0], r[-1], (len(r) - 1) * REFINE + 1)
    zf = np.linspace(z[0], z[-1], (len(z) - 1) * REFINE + 1)
    Z, R = np.meshgrid(zf, rf, indexing="ij")
    pf = f(np.stack([Z, R], axis=-1))
    ka, ja = np.unravel_index(np.argmin(pf), pf.shape)
    lab, _ = label(pf < psimax)
    core = lab == lab[ka, ja]                                        # connected to the axis (no PFR)
    dv = 2 * np.pi * R * (rf[1] - rf[0]) * (zf[1] - zf[0])
    order = np.argsort(pf[core])
    ps, vc = pf[core][order], np.cumsum(dv[core][order])
    raxis, zaxis = rf[ja], zf[ka]
    rvol = np.sqrt(vc / (2 * np.pi**2 * raxis))
    return f, ps, rvol, raxis, zaxis, (rf, zf, core)


def solps_profiles(P, f, ps, rvol, mask_grid, nbin=60):
    """Volume-weighted flux-surface averages of the SOLPS cells inside the closed region."""
    vr, vz, tri = P["mesh/vtx_r"][:], P["mesh/vtx_z"][:], P["mesh/triangles"][:]
    tc = P["mesh/cell_index"][:]
    rc, zc = vr[tri].mean(1), vz[tri].mean(1)
    a = 0.5 * np.abs((vr[tri[:, 1]] - vr[tri[:, 0]]) * (vz[tri[:, 2]] - vz[tri[:, 0]])
                     - (vr[tri[:, 2]] - vr[tri[:, 0]]) * (vz[tri[:, 1]] - vz[tri[:, 0]]))
    w = a * rc
    rf, zf, core = mask_grid
    inside = RegularGridInterpolator((zf, rf), core.astype(float), bounds_error=False, fill_value=0)(
        np.stack([zc, rc], axis=-1)) > 0.5
    pt = f(np.stack([zc, rc], axis=-1))
    rv = np.interp(pt, ps, rvol)
    out = {}
    edges = np.linspace(rv[inside].min(), rv[inside].max(), nbin + 1)
    k = np.clip(np.digitize(rv, edges) - 1, 0, nbin - 1)
    ws = np.bincount(k[inside], w[inside], nbin)
    ok = ws > 0
    out["rvol"] = (np.bincount(k[inside], (w * rv)[inside], nbin) / np.where(ok, ws, 1))[ok]
    for name, key in (("ne", "dens_e"), ("te", "temp_e"), ("ti", "temp_i")):
        v = P["mesh/" + key][:][tc]
        out[name] = (np.bincount(k[inside], (w * v)[inside], nbin) / np.where(ok, ws, 1))[ok]
    return out


def extend(r, y, mode, n=40):
    """Continue y(r) from the innermost SOLPS radius to the axis."""
    rc = r[0]
    sel = r <= rc + FITW
    s, b = np.polyfit(r[sel], y[sel], 1)
    yc = s * rc + b
    ri = np.linspace(0.0, rc, n, endpoint=False)
    if mode == "parabolic":
        yi = yc + s * (ri**2 - rc**2) / (2 * rc)
    elif mode == "linear":
        yi = yc + s * (ri - rc)
    else:
        yi = np.full_like(ri, yc)
    return ri, np.maximum(yi, FLOOR * yc), yc, s


def build(case, mode="parabolic"):
    P = h5py.File(Path(case) / "input/plasma.h5", "r")
    f, ps, rvol, raxis, zaxis, mg = flux_volumes(P)
    sp = solps_profiles(P, f, ps, rvol, mg)
    res = {"solps": sp, "raxis": raxis, "zaxis": zaxis, "psi_of_rvol": (rvol, ps), "fit": {}}
    r_all = None
    for name in ("ne", "te", "ti"):
        ri, yi, yc, s = extend(sp["rvol"], sp[name], mode)
        res["fit"][name] = (yc, s, yi[0])
        r_all = np.concatenate([ri, sp["rvol"]])
        res[name] = np.concatenate([yi, sp[name]])
    res["rvol"] = r_all
    res["rho"] = np.sqrt(np.clip(np.interp(r_all, rvol, ps), 0, None))
    res["rcut"] = sp["rvol"][0]
    res["psicut"] = float(np.interp(res["rcut"], rvol, ps))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="parabolic", choices=("parabolic", "linear", "flat"))
    ap.add_argument("--case", default=".", help="case directory holding input/plasma.h5 (default: current directory)")
    a = ap.parse_args()
    res = build(a.case, a.mode)
    out = Path(a.case) / "input/core_profiles.dat"
    hdr = (f"core profiles for fix core/transport, mode {a.mode}; SOLPS outside r_vol = {res['rcut']:.4f} m "
           f"(psi_N {res['psicut']:.3f})\nrho_N ne(m-3) Te(eV) Ti(eV) r_vol(m)")
    np.savetxt(out, np.column_stack([res["rho"], res["ne"], res["te"], res["ti"], res["rvol"]]),
               fmt="%.6e", header=hdr)
    print(f"R_axis {res['raxis']:.3f} m, SOLPS inner edge r_vol {res['rcut']:.4f} m (psi_N {res['psicut']:.3f})")
    for k, u in (("ne", "m-3"), ("te", "eV"), ("ti", "eV")):
        yc, s, y0 = res["fit"][k]
        print(f"  {k}: cut {yc:.4g} {u}, slope {s:.4g} /m, axis {y0:.4g} {u}")
    print("wrote", out)
