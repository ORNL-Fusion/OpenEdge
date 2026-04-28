#!/usr/bin/env python3
"""3-panel (R, Z) heatmaps for WEST tungsten transport: total W, W,
W+. Reads the SPARTA grid dump produced by `dump dwgrid` in in.west.

Dump column order (matches in.west `dump dwgrid` line):
    id xc yc xlo ylo xhi yhi vol  f_fwgrid[1] f_fwgrid[2] f_fwgrid[3]
                                      total_W       W^0        W^+

In the axisymmetric layout: SPARTA x = Z, y = R. The plot stores
points as (R, Z) for the standard tokamak orientation.

Usage:
    cd test_west_axi
    python3 analysis/make_w_2d_plots.py \\
        --dump output/grid.dens.A.west \\
        --wall input/wall.surf \\
        --out  w_2d.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm


mpl.rcParams.update({
    "font.family":     "serif",
    "mathtext.fontset": "cm",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
})


# Dump schema (0-based)
COL_XC, COL_YC                     = 1, 2
COL_XLO, COL_YLO, COL_XHI, COL_YHI = 3, 4, 5, 6
COL_TOTAL, COL_W0, COL_WP          = 8, 9, 10
NCOLS = 11


def read_last_snapshot(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    starts = [i for i, L in enumerate(lines) if L.startswith("ITEM: TIMESTEP")]
    if not starts:
        raise SystemExit(f"{path}: no snapshots")
    last = starts[-1]
    ncell = int(lines[last + 3])
    for i in range(last, len(lines)):
        if lines[i].startswith("ITEM: CELLS"):
            ds = i + 1
            break
    rows = lines[ds:ds + ncell]
    return np.fromstring(" ".join(rows), sep=" ").reshape(ncell, NCOLS)


def parse_wall(path: Path):
    """Return list of (R, Z) line segments for plotting."""
    pts, segs = [], []
    mode = None
    for L in path.read_text().splitlines():
        s = L.strip()
        if s == "Points": mode = "P"; continue
        if s == "Lines":  mode = "L"; continue
        p = s.split()
        if not p or not p[0].lstrip("-").isdigit(): continue
        if mode == "P" and len(p) >= 3:
            # axi: file col1=Z, col2=R -> (R, Z)
            pts.append((float(p[2]), float(p[1])))
        elif mode == "L" and len(p) >= 3:
            segs.append((int(p[1]) - 1, int(p[2]) - 1))
    pts = np.array(pts)
    return [(pts[a], pts[b]) for a, b in segs]


def cell_quads_RZ(arr: np.ndarray):
    """Build (R, Z) quad for each AMR cell. SPARTA axi: x=Z, y=R."""
    Z_lo, Z_hi = arr[:, COL_XLO], arr[:, COL_XHI]
    R_lo, R_hi = arr[:, COL_YLO], arr[:, COL_YHI]
    quads = np.empty((len(arr), 4, 2))
    quads[:, 0] = np.column_stack([R_lo, Z_lo])
    quads[:, 1] = np.column_stack([R_hi, Z_lo])
    quads[:, 2] = np.column_stack([R_hi, Z_hi])
    quads[:, 3] = np.column_stack([R_lo, Z_hi])
    return quads


def panel(ax, quads, values, title, wall_segs, vmin=None, vmax=None):
    finite = values[np.isfinite(values) & (values > 0)]
    if vmin is None:
        vmin = max(finite.min(), 1e-3) if len(finite) else 1.0
    if vmax is None:
        vmax = finite.max() if len(finite) else 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)

    pc = PolyCollection(quads, array=values, cmap="magma", norm=norm,
                        edgecolors="none", zorder=1)
    ax.add_collection(pc)

    if wall_segs:
        ax.add_collection(LineCollection(wall_segs, colors="black",
                                          linewidths=0.6, zorder=3))

    R_all = quads[:, :, 0]
    Z_all = quads[:, :, 1]
    ax.set_xlim(R_all.min(), R_all.max())
    ax.set_ylim(Z_all.min(), Z_all.max())
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R \; (\mathrm{m})$")
    ax.set_ylabel(r"$Z \; (\mathrm{m})$")
    ax.set_title(title)
    cb = plt.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(r"$n \; (\mathrm{m}^{-3})$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--wall", type=Path, default=None)
    ap.add_argument("--out",  type=Path, required=True)
    args = ap.parse_args()

    arr = read_last_snapshot(args.dump)
    quads = cell_quads_RZ(arr)
    wall_segs = parse_wall(args.wall) if args.wall else []

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    panel(axes[0], quads, arr[:, COL_TOTAL], r"Total tungsten ($\sum_q n_{W^{q+}}$)", wall_segs)
    panel(axes[1], quads, arr[:, COL_W0],    r"$W^{0}$ (neutral)",                    wall_segs)
    panel(axes[2], quads, arr[:, COL_WP],    r"$W^{+}$ (singly ionized)",             wall_segs)

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
