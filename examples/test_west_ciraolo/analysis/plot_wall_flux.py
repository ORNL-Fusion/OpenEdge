#!/usr/bin/env python3
"""Tungsten sputtered flux along the WEST wall.

Reads SPARTA `dump dwflux surf` produced by in.west:
    id v1x v1y v2x v2y c_cpmiO_rate
where v1x,v1y / v2x,v2y are line endpoints in (R, Z) (Cartesian deck).

Two panels:
  (a) flux vs poloidal arc length s [m] starting at the outer-midplane,
      walking the wall in order
  (b) wall polyline in (R, Z) coloured by flux

Usage:
    cd test_west_ciraolo
    python3 analysis/plot_wall_flux.py \
        --dump output/wall.flux.A.west \
        --out  output/wall_flux.png
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm


mpl.rcParams.update({
    "font.family":     "serif",
    "mathtext.fontset": "cm",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   13,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.4,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
})

NCOLS = 6  # id v1x v1y v2x v2y c_cpmiO_rate


def read_last_snapshot(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    starts = [i for i, L in enumerate(lines) if L.startswith("ITEM: TIMESTEP")]
    if not starts:
        raise SystemExit(f"{path}: no snapshots")
    last = starts[-1]
    nrow = int(lines[last + 3])
    for i in range(last, len(lines)):
        if lines[i].startswith("ITEM: SURFS"):
            ds = i + 1
            break
    rows = lines[ds:ds + nrow]
    return np.fromstring(" ".join(rows), sep=" ").reshape(nrow, NCOLS)


def order_along_wall(arr: np.ndarray):
    """Walk segments in connectivity order, starting from the segment
    closest to the outer midplane (max R, Z=0). Returns indices into arr
    in walk order, plus an arclength array s (same order)."""
    v1 = arr[:, 1:3]
    v2 = arr[:, 3:5]
    n = len(arr)

    # endpoint-to-segment map for connectivity. Two segments share an
    # endpoint if their (rounded) coords match.
    key = lambda p: (round(p[0], 9), round(p[1], 9))
    starts = {}
    ends   = {}
    for i in range(n):
        starts.setdefault(key(v1[i]), []).append(i)
        ends.setdefault(key(v2[i]), []).append(i)

    # outer-midplane seed: maximize R among segments with |Z|<=Zmax/4
    Zmax = max(np.abs(v1[:, 1]).max(), np.abs(v2[:, 1]).max())
    midmask = (np.abs(0.5 * (v1[:, 1] + v2[:, 1])) <= 0.25 * Zmax)
    Rmid = 0.5 * (v1[:, 0] + v2[:, 0])
    Rmid_masked = np.where(midmask, Rmid, -np.inf)
    seed = int(np.argmax(Rmid_masked))

    order = [seed]
    visited = {seed}
    cur_end = key(v2[seed])
    for _ in range(n - 1):
        nxt = None
        for j in starts.get(cur_end, []):
            if j not in visited:
                nxt = j; break
        if nxt is None:
            for j in ends.get(cur_end, []):
                if j not in visited:
                    # Reverse this segment
                    v1[j], v2[j] = v2[j].copy(), v1[j].copy()
                    nxt = j; break
        if nxt is None:
            break
        order.append(nxt)
        visited.add(nxt)
        cur_end = key(v2[nxt])

    order = np.array(order, dtype=int)
    seglen = np.linalg.norm(v2[order] - v1[order], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])[:-1] + 0.5 * seglen
    return order, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True)
    args = ap.parse_args()

    arr = read_last_snapshot(args.dump)
    order, s = order_along_wall(arr)
    flux = arr[order, 5]
    v1 = arr[order, 1:3]
    v2 = arr[order, 3:5]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(s, flux, color="C3")
    ax.set_xlabel(r"Poloidal arc $s$ (m), CCW from outer midplane")
    ax.set_ylabel(r"$\Gamma_{W,\mathrm{sputter}}$ (m$^{-2}$ s$^{-1}$)")
    if (flux > 0).any():
        ax.set_yscale("log")
        ax.set_ylim(max(flux[flux > 0].min(), 1e6), flux.max() * 2.0)
    ax.set_title("W sputter flux along wall")

    ax = axes[1]
    segs = np.stack([v1, v2], axis=1)
    pos = flux.copy()
    pos_min = pos[pos > 0].min() if (pos > 0).any() else 1e8
    pos_max = pos.max() if pos.max() > 0 else 1e10
    norm = LogNorm(vmin=max(pos_min, 1e6), vmax=pos_max)
    lc = LineCollection(segs, cmap="inferno", norm=norm, linewidths=2.0)
    lc.set_array(flux)
    ax.add_collection(lc)
    ax.set_xlim(v1[:, 0].min() - 0.05, v1[:, 0].max() + 0.05)
    ax.set_ylim(v1[:, 1].min() - 0.05, v1[:, 1].max() + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title("Wall flux map")
    cb = plt.colorbar(lc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(r"$\Gamma_{W}$ (m$^{-2}$ s$^{-1}$)")

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
