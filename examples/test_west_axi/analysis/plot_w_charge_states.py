#!/usr/bin/env python3
"""W charge-state domain density vs time, parsed from log.openedge.

Mirror of test_ionization_recombination/plot_charged_states.py for the
WEST tungsten case. The deck registers per-W-charge-state reductions
(c_cw0 ... c_cw10) in the stats line so the SPARTA log carries one
column per charge state per stats step. This plots the 11 curves on a
log y-axis to show the approach to coronal equilibrium.

Usage:
    cd test_west_axi
    python3 analysis/plot_w_charge_states.py [--log log.openedge] \\
                                             [--out w_charge_states.png] \\
                                             [--dt 5e-9]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family":     "serif",
    "mathtext.fontset": "cm",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.6,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
})


COLS = [f"c_cw{k}" for k in range(11)]   # cw0=W, cw1=W+, ..., cw10=W10+
LABELS = [r"$W^{0}$"] + [fr"$W^{{{k}+}}$" for k in range(1, 11)]


def parse_stats(path: Path):
    text = path.read_text(errors="ignore").splitlines()
    rows = []
    seen_steps = set()
    i = 0
    while i < len(text):
        toks = text[i].split()
        if "Step" in toks and all(c in toks for c in COLS):
            header = toks
            idx = {c: header.index(c) for c in ["Step"] + COLS}
            i += 1
            while i < len(text):
                row_toks = text[i].split()
                if not row_toks:
                    break
                try:
                    row = {k: float(row_toks[v]) for k, v in idx.items()}
                except (ValueError, IndexError):
                    break
                if row["Step"] not in seen_steps:
                    seen_steps.add(row["Step"])
                    rows.append(row)
                i += 1
        else:
            i += 1
    if not rows:
        raise SystemExit(f"no stats block with {COLS} found in {path}")
    step = np.array([r["Step"] for r in rows])
    data = {c: np.array([r[c] for r in rows]) for c in COLS}
    return step, data


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean. window<=1 returns y unchanged."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path,
                    default=Path("log.openedge"))
    ap.add_argument("--out", type=Path,
                    default=Path("w_charge_states.png"))
    ap.add_argument("--dt", type=float, default=5e-9,
                    help="Physical timestep [s] (default 5e-9; matches "
                         "in.west `variable dt equal 5.0e-09`)")
    ap.add_argument("--smooth", type=int, default=51,
                    help="Rolling-mean window (steps) to suppress MC noise. "
                         "Use 1 to disable. Default 51 ~ 4 us at dt=5e-9 "
                         "with stats=1; with stats=100 try 5-11.")
    ap.add_argument("--xmin", type=float, default=None,
                    help="Start the x-axis at this many us (default: 0). Use "
                         "to skip the noisy warm-up transient — e.g. --xmin "
                         "1.8 starts after the first sputter pulse.")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Truncate the x-axis at this many us (default: full "
                         "range from the log).")
    ap.add_argument("--ymin", type=float, default=None,
                    help="Floor the y-axis at this density (m^-3). Default: "
                         "auto-pick the smallest non-zero across all curves.")
    ap.add_argument("--floor", type=float, default=None,
                    help="Mask data points below this density (m^-3) as NaN, "
                         "so curves dropping below the noise threshold "
                         "disappear instead of drawing a spurious line. "
                         "Recommended: 10x the smallest non-zero c_cw* value "
                         "in the log (e.g. 1e9 for typical WEST runs).")
    args = ap.parse_args()

    step, data = parse_stats(args.log)
    t_us = step * args.dt * 1e6   # microseconds

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    cmap = plt.get_cmap("viridis", 11)
    for k, col in enumerate(COLS):
        y = smooth(data[col], args.smooth)
        if args.floor is not None:
            y = np.where(y < args.floor, np.nan, y)
        ax.plot(t_us, y, color=cmap(k), label=LABELS[k])
    ax.set_xlabel(r"$t \; (\mathrm{\mu s})$")
    ax.set_ylabel(r"$\langle n_{W^{q+}} \rangle \; (\mathrm{m}^{-3})$")
    ax.set_yscale("log")
    xlo = args.xmin if args.xmin is not None else 0.0
    xhi = args.xmax if args.xmax is not None else t_us[-1]
    ax.set_xlim(xlo, xhi)
    if args.ymin is not None:
        ymax = max(arr.max() for arr in data.values()) * 3
        ax.set_ylim(args.ymin, ymax)
    ax.set_title(r"WEST: tungsten charge-state evolution toward coronal equilibrium")
    ax.legend(ncol=4, loc="lower right", framealpha=0.9)

    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(t_us)} time samples, "
          f"t = {t_us[0]:.2f} -> {t_us[-1]:.2f} us, "
          f"smooth={args.smooth})")


if __name__ == "__main__":
    main()
