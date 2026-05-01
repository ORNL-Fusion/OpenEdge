#!/usr/bin/env python3
"""2D maps of plasma-frame source moments from the last dump snapshot.

Produces a 2x2 figure: Sp (particle), Sm (momentum), Qe (electron energy),
Qi (ion energy) per m^3 [s^-1 / N / W], with the wall outline overlaid.

Reads the SPARTA grid dump produced by `dump d1 grid all <Nevery> ...
f_fden[1..3] f_frate[1..6]` in the axi deck. Column layout (16 cols):
    id xc yc xlo ylo xhi yhi  n_D2 n_D n_Dp  Sp Smx Smy Smz Qe Qi

Axi convention in this deck: SPARTA x = Z (axial), y = R (radial).

Usage:
    python3 analysis/plot_sources_2d.py \\
        --dump diii_d_neutrals_axi.grid \\
        --wall wall.surf \\
        --out  plots/sources_2d.png
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


def cell_quads_RZ(arr: np.ndarray) -> np.ndarray:
    Z_lo, R_lo = arr[:, COL_XLO], arr[:, COL_YLO]
    Z_hi, R_hi = arr[:, COL_XHI], arr[:, COL_YHI]
    quads = np.empty((len(arr), 4, 2))
    quads[:, 0] = np.column_stack([R_lo, Z_lo])
    quads[:, 1] = np.column_stack([R_hi, Z_lo])
    quads[:, 2] = np.column_stack([R_hi, Z_hi])
    quads[:, 3] = np.column_stack([R_lo, Z_hi])
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
    ap.add_argument("--wall", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True)
    args = ap.parse_args()

    arr = read_last_snapshot(args.dump)
    quads = cell_quads_RZ(arr)
    wall = parse_wall_segments(args.wall)

    Sm = arr[:, COL_SMX] + arr[:, COL_SMY] + arr[:, COL_SMZ]
    panels = [
        (arr[:, COL_SP], r"$S_p$ (m$^{-3}$ s$^{-1}$)",  "particle source"),
        (Sm,             r"$S_m$ (kg m$^{-2}$ s$^{-2}$)", "momentum source"),
        (arr[:, COL_QE], r"$Q_e$ (W m$^{-3}$)",          "electron energy"),
        (arr[:, COL_QI], r"$Q_i$ (W m$^{-3}$)",          "ion energy"),
    ]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 13,
        "axes.labelsize": 14, "axes.titlesize": 14,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    for ax, (v, label, title) in zip(axes.flat, panels):
        pc = PolyCollection(quads, array=v, cmap="RdBu_r",
                            norm=symlog_norm(v), edgecolors="none")
        ax.add_collection(pc)
        ax.add_collection(LineCollection(wall, colors="black", linewidths=0.6))
        ax.set_xlim(0.95, 2.45); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$ (m)"); ax.set_ylabel(r"$Z$ (m)")
        ax.set_title(title)
        cb = fig.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label(label)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
