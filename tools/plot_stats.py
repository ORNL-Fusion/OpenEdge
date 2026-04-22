#!/usr/bin/env python3
"""Parse SPARTA stats lines from a log file (or stdin) and plot each
column vs step as a convergence-monitor view.

The SPARTA log's stats block is delimited by the header row starting
with "Step" followed by numeric rows, terminated by a blank line or
any non-numeric line.  This script locates the block, extracts columns
by the header keywords, and writes a PNG with one subplot per column.

Usage:
  python3 tools/plot_stats.py --log run.log --out plots/convergence.png
  mpirun ... | python3 tools/plot_stats.py --out plots/convergence.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_FLOAT_RE = re.compile(r"^\s*-?\d")  # line starts with optional '-' then digit

# Mapping from SPARTA stats column name to (display label, unit).
# Used to render journal-style y-axis labels instead of the raw c_*/f_*
# identifiers from the deck.
DISPLAY = {
    "Step":   ("step",                      None),
    "CPU":    ("CPU time",                  "s"),
    "Np":     (r"$N_p$ (simulation particles)", None),
    "c_rsp":  (r"$\sum_\mathrm{cells} S_p$",     r"m$^{-3}$ s$^{-1}$"),
    "c_rsmx": (r"$\sum_\mathrm{cells} S_{m,x}$", r"kg m$^{-2}$ s$^{-2}$"),
    "c_rsmy": (r"$\sum_\mathrm{cells} S_{m,y}$", r"kg m$^{-2}$ s$^{-2}$"),
    "c_rsmz": (r"$\sum_\mathrm{cells} S_{m,z}$", r"kg m$^{-2}$ s$^{-2}$"),
    "c_rqe":  (r"$\sum_\mathrm{cells} Q_e$",     r"W m$^{-3}$"),
    "c_rqi":  (r"$\sum_\mathrm{cells} Q_i$",     r"W m$^{-3}$"),
}


def parse_stats(text: str) -> tuple[list[str], np.ndarray]:
    """Locate the first SPARTA stats table in the text, return
    (column_names, values[nrows, ncols]).  Raises if no table found."""
    lines = text.splitlines()
    header_idx = None
    for i, L in enumerate(lines):
        s = L.strip()
        # SPARTA prints the header as: "Step CPU Np ..." -- starts with
        # a capital S and contains no digits anywhere.
        if s.startswith("Step ") and not any(c.isdigit() for c in s):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("no 'Step ...' header found in log")
    names = lines[header_idx].split()

    rows = []
    for L in lines[header_idx + 1:]:
        s = L.strip()
        if not s:
            break
        if not _FLOAT_RE.match(L):
            break
        parts = s.split()
        if len(parts) != len(names):
            break
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            break
    if not rows:
        raise RuntimeError("stats table found but no data rows")
    return names, np.array(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--log", type=Path, default=None,
                   help="Path to SPARTA log file (default: read stdin)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path")
    p.add_argument("--x", default="Step",
                   help="Column name for x-axis (default Step)")
    p.add_argument("--skip", nargs="*", default=["Step", "CPU"],
                   help="Column names to skip when plotting")
    p.add_argument("--dt", type=float, default=None,
                   help="If given, also plot time = Step*dt as x-axis label")
    args = p.parse_args(argv)

    text = args.log.read_text() if args.log else sys.stdin.read()
    names, data = parse_stats(text)
    if args.x not in names:
        sys.exit(f"column {args.x!r} not found; have {names}")
    x_idx = names.index(args.x)
    x_vals = data[:, x_idx]

    # Convert step -> time[ms] if dt supplied and x is Step.
    use_time_ms = (args.dt is not None and args.x == "Step")
    if use_time_ms:
        x_plot = x_vals * args.dt * 1e3       # seconds -> milliseconds
        xlabel = r"time $t$ [ms]"
    else:
        x_plot = x_vals
        xlabel = DISPLAY.get(args.x, (args.x, None))[0]

    plot_names = [n for n in names if n not in args.skip]
    n_panels = len(plot_names)
    if n_panels == 0:
        sys.exit("nothing to plot after --skip filter")

    # Journal-style defaults: serif font, larger base size, thicker lines,
    # no grid clutter, hairline axes.
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

    ncols_plot = min(2, n_panels)
    nrows_plot = (n_panels + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(5.8 * ncols_plot, 3.6 * nrows_plot),
                             sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, plot_names):
        y = data[:, names.index(name)]
        disp_name, disp_unit = DISPLAY.get(name, (name, None))
        ylabel = disp_name if disp_unit is None else f"{disp_name} [{disp_unit}]"
        ax.plot(x_plot, y, color="tab:blue", lw=1.5)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="0.5", lw=0.5, alpha=0.6)

        # Running-mean overlay (last 50% of the trace)
        n = len(y)
        if n >= 10:
            half = n // 2
            tail_mean = y[half:].mean()
            tail_std  = y[half:].std()
            ax.axhline(tail_mean, color="tab:red", lw=1.1, linestyle="--",
                       label=f"$\\langle \\cdot \\rangle_\\mathrm{{tail}} "
                             f"= {tail_mean:.2e}$")
            # Shade ±1 sigma band
            ax.axhspan(tail_mean - tail_std, tail_mean + tail_std,
                       color="tab:red", alpha=0.08)
            ax.legend(loc="best", framealpha=0.9, frameon=True,
                      edgecolor="0.7", facecolor="white")

        # Offset formatter for large numbers
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                            useMathText=True)
        # Trim tick padding
        ax.tick_params(axis="both", direction="in", length=4)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Only label x-axis on bottom row
    if nrows_plot > 1:
        for ax in axes[-ncols_plot:]:
            ax.set_xlabel(xlabel)
    else:
        for ax in axes:
            ax.set_xlabel(xlabel)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(plot_names)} panels, {len(x_vals)} points)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
