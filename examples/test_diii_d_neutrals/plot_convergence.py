#!/usr/bin/env python3
"""Domain-integrated convergence plot from SPARTA multi-snapshot grid dump.

Unlike `plot_stats.py` (which parses SPARTA's stats log and shows the
sum of per-cell densities -- dimensionally m^-3 s^-1, not a physical
total), this tool reads every snapshot in the `dump d1 grid` file and
computes volume-weighted integrals of the source moments over the
whole domain, giving totals in proper SI units:

  integral n_D  dV  [atoms]
  integral n_D2 dV  [molecules]
  integral S_p  dV  [1/s]   -- ions created per second
  integral |S_m| dV [N]     -- net momentum injection rate
  integral Q_e  dV  [W]     -- electron energy sink
  integral Q_i  dV  [W]     -- ion energy source

Each curve's time axis is `step * dt` in milliseconds.

Usage:
  python3 tools/plot_convergence.py \\
      --dump examples/test_diii_d_neutrals/output/diii_d_neutrals.grid \\
      --out  plots/convergence.png \\
      --dt   5e-9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dump column layout (matches make_neutral_plots.py):
#   0 id  1 xc  2 yc  3 xlo  4 ylo  5 xhi  6 yhi
#   7 n_D2  8 n_D  9 n_D+  10 Sp  11 Smx  12 Smy  13 Smz  14 Qe  15 Qi
NCOLS  = 16
COL_XC, COL_YC                     = 1, 2
COL_XLO, COL_YLO, COL_XHI, COL_YHI = 3, 4, 5, 6
COL_ND2, COL_ND, COL_NDP           = 7, 8, 9
COL_SP                             = 10
COL_SMX, COL_SMY, COL_SMZ          = 11, 12, 13
COL_QE, COL_QI                     = 14, 15


def iter_snapshots(path: Path):
    """Yield (step, arr) for every snapshot in a SPARTA grid dump."""
    lines = path.read_text().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            ncell = int(lines[i + 3])
            # Scan forward for "ITEM: CELLS" header.
            j = i + 4
            while j < n and not lines[j].startswith("ITEM: CELLS"):
                j += 1
            data_start = j + 1
            data_end   = data_start + ncell
            if data_end > n:
                break
            rows = lines[data_start:data_end]
            arr = np.fromstring(" ".join(rows), sep=" ").reshape(ncell, NCOLS)
            yield step, arr
            i = data_end
        else:
            i += 1


def axi_cell_volume(arr: np.ndarray) -> np.ndarray:
    """Return per-cell axisymmetric volume: dZ * pi * (R_hi^2 - R_lo^2).

    With our SPARTA axi slot map: x = Z, y = R, so dZ = xhi - xlo and
    R_lo = ylo, R_hi = yhi.
    """
    dZ    = arr[:, COL_XHI] - arr[:, COL_XLO]
    R_lo  = arr[:, COL_YLO]
    R_hi  = arr[:, COL_YHI]
    return dZ * np.pi * (R_hi * R_hi - R_lo * R_lo)


def integrate_snapshot(arr: np.ndarray) -> dict:
    """Volume-weighted domain integrals of each tracked field.

    Units convention matches SOLPS / SOLEDGE3X:
      n_D, n_D2    [particles]         = sum(n * V_cell)
      S_p          [1/s]               = sum(S_p * V_cell)
      |S_m|        [N·m]               = sum(|S_m| * R_cell * V_cell)
                                         (toroidal angular-momentum rate;
                                         the R factor reflects the axi
                                         geometry convention used by SOLPS)
      Q_e, Q_i     [W]                 = sum(Q * V_cell)
    """
    V = axi_cell_volume(arr)
    R = arr[:, COL_YC]           # axi: y = R
    Sm_total = (arr[:, COL_SMX] + arr[:, COL_SMY] + arr[:, COL_SMZ])
    return {
        "n_D":   (arr[:, COL_ND]  * V).sum(),
        "n_D2":  (arr[:, COL_ND2] * V).sum(),
        "S_p":   (arr[:, COL_SP]  * V).sum(),
        "S_m":   (Sm_total * R * V).sum(),      # signed; R * V for N·m
        "Q_e":   (arr[:, COL_QE]  * V).sum(),
        "Q_i":   (arr[:, COL_QI]  * V).sum(),
    }


# Journal-style display info: label, unit (SOLPS convention)
PANELS = [
    ("n_D",   r"$\int n_D\,dV$",     "particles"),
    ("n_D2",  r"$\int n_{D_2}\,dV$", "molecules"),
    ("S_p",   r"$\int S_p\,dV$",     r"s$^{-1}$"),
    ("S_m",   r"$\int R\,S_m\,dV$",  r"N$\,$m"),
    ("Q_e",   r"$\int Q_e\,dV$",     r"W"),
    ("Q_i",   r"$\int Q_i\,dV$",     r"W"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--dt",   type=float, default=5e-9,
                   help="Physical timestep [s] (default 5e-9; used to "
                        "convert step -> time in ms on the x-axis)")
    args = p.parse_args(argv)

    steps = []
    totals = {k: [] for k, _, _ in PANELS}
    for step, arr in iter_snapshots(args.dump):
        d = integrate_snapshot(arr)
        steps.append(step)
        for k in totals:
            totals[k].append(d[k])
    steps = np.asarray(steps, dtype=float)
    if len(steps) < 2:
        sys.exit(f"only {len(steps)} snapshot(s) in {args.dump}; need at "
                 "least 2 to plot a time series")
    t_ms = steps * args.dt * 1e3

    print(f"parsed {len(steps)} snapshots from {args.dump}", file=sys.stderr)
    print(f"final-snapshot domain totals:", file=sys.stderr)
    for k in totals:
        print(f"  {k:>5} = {totals[k][-1]:+.3e}", file=sys.stderr)

    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       13,
        "axes.labelsize":  13,
        "axes.titlesize":  13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth":  0.8,
        "lines.linewidth": 1.5,
    })
    ncols = 2
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.8 * ncols, 3.4 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, (key, label, unit) in zip(axes, PANELS):
        y = np.asarray(totals[key])
        ax.plot(t_ms, y, color="tab:blue", lw=1.6)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.axhline(0, color="0.5", lw=0.5, alpha=0.6)
        # Tail-mean reference line (no annotation text, just the line)
        if len(y) >= 4:
            half = len(y) // 2
            ax.axhline(y[half:].mean(), color="tab:red", lw=1.1,
                       linestyle="--")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                            useMathText=True)
        ax.tick_params(axis="both", direction="in", length=4)

    # Bottom-row x-labels only
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"time $t$ [ms]")
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
