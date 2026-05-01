#!/usr/bin/env python3
"""2D maps of OpenEdge plasma-frame source moments resampled onto the SOLPS
B2 mesh.

Produces a 1x4 figure: Sp, Sm, Qe, Qi (per m^3 / N / W densities) with
OpenEdge data nearest-cell interpolated onto SOLPS-EIRENE B2 quads, plus
the wall outline overlaid.

Reads:
  --dump       SPARTA grid dump (axi layout: xc=Z, yc=R)
  --balance-nc SOLPS-ITER balance.nc (provides crx/cry B2 cell corners)
  --wall       wall.surf (axi layout: file col1=Z, col2=R)
  --out        output path

Resampling: nearest-cell from OpenEdge cell centroids to SOLPS quad centroids
(KDTree), so values shown on each B2 quad come from the nearest OpenEdge
adaptive-grid cell. This gives a 1:1 cell-by-cell visual match with whatever
SOLPS plot you put alongside.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import SymLogNorm
from scipy.spatial import cKDTree
import netCDF4 as nc

NCOLS = 16
COL_XLO, COL_YLO, COL_XHI, COL_YHI = 3, 4, 5, 6
COL_SP, COL_SMX, COL_SMY, COL_SMZ, COL_QE, COL_QI = 10, 11, 12, 13, 14, 15


def read_last_snapshot(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    starts = [i for i, L in enumerate(lines) if L.startswith("ITEM: TIMESTEP")]
    last = starts[-1]
    ncell = int(lines[last + 3])
    for i in range(last, len(lines)):
        if lines[i].startswith("ITEM: CELLS"):
            ds = i + 1
            break
    rows = lines[ds:ds + ncell]
    return np.fromstring(" ".join(rows), sep=" ").reshape(ncell, NCOLS)


def openedge_centroids_RZ(arr: np.ndarray):
    # Axi layout: SPARTA x = Z, y = R.  Plot in (R, Z).
    Z_lo = arr[:, COL_XLO]; R_lo = arr[:, COL_YLO]
    Z_hi = arr[:, COL_XHI]; R_hi = arr[:, COL_YHI]
    Rc = 0.5 * (R_lo + R_hi)
    Zc = 0.5 * (Z_lo + Z_hi)
    return Rc, Zc


def solps_quads(path: Path):
    with nc.Dataset(path) as f:
        crx = f.variables['crx'][:]   # (4, ny, nx) cell-corner R
        cry = f.variables['cry'][:]   # (4, ny, nx) cell-corner Z
    ny, nx = crx.shape[1], crx.shape[2]
    nq = ny * nx
    quads = np.empty((nq, 4, 2))
    k = 0
    # SOLPS corner indices: 0=LL, 1=LR, 2=UL, 3=UR -> wind LL,LR,UR,UL
    for iy in range(ny):
        for ix in range(nx):
            quads[k, 0] = (crx[0, iy, ix], cry[0, iy, ix])
            quads[k, 1] = (crx[1, iy, ix], cry[1, iy, ix])
            quads[k, 2] = (crx[3, iy, ix], cry[3, iy, ix])
            quads[k, 3] = (crx[2, iy, ix], cry[2, iy, ix])
            k += 1
    return quads


def parse_wall_segments(path: Path):
    pts, segs, mode = [], [], None
    for L in path.read_text().splitlines():
        s = L.strip()
        if not s: continue
        if s == "Points": mode = "P"; continue
        if s == "Lines":  mode = "L"; continue
        p = s.split()
        if not p[0].lstrip("-").isdigit(): continue
        if mode == "P" and len(p) >= 3:
            pts.append((float(p[2]), float(p[1])))   # axi: (R, Z)
        elif mode == "L" and len(p) >= 3:
            segs.append((int(p[1]) - 1, int(p[2]) - 1))
    pts = np.array(pts)
    return [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs]


def symlog_norm(v, pct=99.5, lin_frac=0.03):
    finite = v[np.isfinite(v) & (v != 0)]
    if len(finite) == 0:
        return None
    absmax = float(np.nanpercentile(np.abs(finite), pct))
    return SymLogNorm(linthresh=max(absmax * lin_frac, 1e-30),
                      vmin=-absmax, vmax=absmax, base=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--balance-nc", type=Path, required=True,
                    help="SOLPS-ITER balance.nc, used only for B2 cell corners.")
    ap.add_argument("--wall", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True)
    args = ap.parse_args()

    arr = read_last_snapshot(args.dump)
    oe_R, oe_Z = openedge_centroids_RZ(arr)
    Sp = arr[:, COL_SP]
    Sm = arr[:, COL_SMX] + arr[:, COL_SMY] + arr[:, COL_SMZ]
    Qe = arr[:, COL_QE]
    Qi = arr[:, COL_QI]

    sp_quads = solps_quads(args.balance_nc)
    sp_R = sp_quads[:, :, 0].mean(axis=1)
    sp_Z = sp_quads[:, :, 1].mean(axis=1)

    # Nearest-cell resample: each SOLPS quad takes the value of the nearest
    # OpenEdge cell centroid.
    tree = cKDTree(np.column_stack([oe_R, oe_Z]))
    _, idx = tree.query(np.column_stack([sp_R, sp_Z]))
    Sp_b2 = Sp[idx]; Sm_b2 = Sm[idx]; Qe_b2 = Qe[idx]; Qi_b2 = Qi[idx]

    wall = parse_wall_segments(args.wall)

    panels = [
        (Sp_b2, r"$S_p$ (m$^{-3}$ s$^{-1}$)"),
        (Sm_b2, r"$S_m$ (kg m$^{-2}$ s$^{-2}$)"),
        (Qe_b2, r"$Q_e$ (W m$^{-3}$)"),
        (Qi_b2, r"$Q_i$ (W m$^{-3}$)"),
    ]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.labelsize": 12,
    })
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True,
                             constrained_layout=True)
    for ax, (v, label) in zip(axes, panels):
        pc = PolyCollection(sp_quads, array=v, cmap="RdBu_r",
                            norm=symlog_norm(v), edgecolors="none")
        ax.add_collection(pc)
        ax.add_collection(LineCollection(wall, colors="black", linewidths=0.6))
        R_all = sp_quads[:, :, 0]; Z_all = sp_quads[:, :, 1]
        ax.set_xlim(R_all.min(), R_all.max())
        ax.set_ylim(Z_all.min(), Z_all.max())
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$ (m)")
        cb = fig.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label(label)
    axes[0].set_ylabel(r"$Z$ (m)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
