#!/usr/bin/env python3
"""One-slide DIII-D neutrals summary: 1x4 panels.

Panel layout (left -> right):
    1. integral n_D  dV   atom inventory vs time  (atoms)
    2. integral n_D2 dV   molecule inventory vs time  (molecules)
    3. S_p (R, Z)         particle source 2D map  [m^-3 s^-1]
    4. S_m (R, Z)         momentum source 2D map  [N m^-3]

Time-series from EVERY snapshot, maps from the LAST snapshot.

Usage:
    python3 analysis/plot_summary.py \\
        --dump diii_d_neutrals_axi.grid \\
        --wall wall.surf \\
        --out  plots/summary.png \\
        --dt   5e-8
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
COL_ND2, COL_ND = 7, 8
COL_SP, COL_SMX, COL_SMY, COL_SMZ = 10, 11, 12, 13


def iter_snapshots(path: Path):
    lines = path.read_text().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            ncell = int(lines[i + 3])
            j = i + 4
            while j < n and not lines[j].startswith("ITEM: CELLS"):
                j += 1
            ds = j + 1
            de = ds + ncell
            if de > n:
                break
            vals = np.fromstring(" ".join(lines[ds:de]), sep=" ")
            if vals.size != ncell * NCOLS:
                break
            yield step, vals.reshape(ncell, NCOLS)
            i = de
        else:
            i += 1


def cell_volume_axi(arr: np.ndarray) -> np.ndarray:
    dZ   = arr[:, COL_XHI] - arr[:, COL_XLO]
    R_lo = arr[:, COL_YLO]; R_hi = arr[:, COL_YHI]
    return dZ * np.pi * (R_hi * R_hi - R_lo * R_lo)


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


def draw_map(ax, quads, v, label, title, wall):
    pc = PolyCollection(quads, array=v, cmap="RdBu_r",
                        norm=symlog_norm(v), edgecolors="none")
    ax.add_collection(pc)
    ax.add_collection(LineCollection(wall, colors="black", linewidths=0.5))
    ax.set_xlim(0.95, 2.45); ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ (m)"); ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title, pad=4)
    cb = plt.colorbar(pc, ax=ax, fraction=0.06, pad=0.02)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=9)


def draw_series(ax, t_ms, y, label, title):
    ax.plot(t_ms, y, color="tab:blue", lw=1.8)
    if len(y) >= 4:
        tail = y[len(y)//2:].mean()
        ax.axhline(tail, color="tab:red", lw=1.0, ls="--", label="tail mean")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(label)
    ax.set_title(title, pad=4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                        useMathText=True)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--wall", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True)
    ap.add_argument("--dt",   type=float, default=5e-8,
                    help="physical timestep in s (default 5e-8 = deck dt)")
    args = ap.parse_args()

    steps, intD, intD2, last = [], [], [], None
    for step, arr in iter_snapshots(args.dump):
        V = cell_volume_axi(arr)
        steps.append(step)
        intD.append((arr[:, COL_ND]  * V).sum())
        intD2.append((arr[:, COL_ND2] * V).sum())
        last = arr
    if last is None:
        raise SystemExit(f"no snapshots in {args.dump}")
    t_ms = np.asarray(steps) * args.dt * 1e3

    quads = cell_quads_RZ(last)
    wall  = parse_wall_segments(args.wall)
    Sm    = last[:, COL_SMX] + last[:, COL_SMY] + last[:, COL_SMZ]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig = plt.figure(figsize=(16, 4.0))
    # Width ratios: time series get wider panels (squarer), maps stay
    # narrower since they're taller-than-wide in (R, Z).
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 1.0, 1.0],
                          wspace=0.55)
    ax_nd = fig.add_subplot(gs[0, 0])
    ax_n2 = fig.add_subplot(gs[0, 1])
    ax_sp = fig.add_subplot(gs[0, 2])
    ax_sm = fig.add_subplot(gs[0, 3])

    draw_series(ax_nd, t_ms, np.asarray(intD),
                r"$\int n_D\,dV$ (atoms)", "atom inventory")
    draw_series(ax_n2, t_ms, np.asarray(intD2),
                r"$\int n_{D_2}\,dV$ (molecules)", "molecule inventory")
    draw_map(ax_sp, quads, last[:, COL_SP],
             r"$S_p$ (m$^{-3}$ s$^{-1}$)", "particle source", wall)
    draw_map(ax_sm, quads, Sm,
             r"$S_m$ (kg m$^{-2}$ s$^{-2}$)", "momentum source", wall)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
