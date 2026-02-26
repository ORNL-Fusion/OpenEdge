#!/usr/bin/env python3
"""
Plot Ion Energy-Angle Distribution (IEAD) from SPARTA fix ave/time output.

The IEAD compute produces a 2-D histogram:
  rows = angle bins  (0 to Amax degrees, 0° = normal incidence)
  cols = energy bins (0 to Emax eV)

Written by fix ave/time in mode vector with a global array, the file
format is:

  # header lines ...
  <timestep> <nrows>
  1  val1 val2 ... val_ncols
  2  val1 val2 ... val_ncols
  ...

Usage examples:
  python plot_iead.py --file output/iead.dat --emax 200 --amax 90
  python plot_iead.py --file output/iead.dat --emax 200 --amax 90 --log
  python plot_iead.py --file output/iead.dat --emax 200 --amax 90 --marginals
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_iead_file(path: Path):
    """Parse a fix ave/time vector-mode file containing the IEAD array.

    Returns the last snapshot as a 2-D numpy array (rows=angle, cols=energy).
    """
    lines = path.read_text().splitlines()
    i = 0
    n = len(lines)
    last_block = None

    while i < n:
        line = lines[i].strip()
        # Skip blank lines and comment/header lines
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        # A section header is: <timestep> <nrows>
        if len(parts) == 2:
            try:
                timestep = int(parts[0])
                nrows = int(parts[1])
            except ValueError:
                i += 1
                continue

            rows = []
            for j in range(nrows):
                i += 1
                if i >= n:
                    break
                row_parts = lines[i].split()
                # First token is the row index, rest are values
                if len(row_parts) < 2:
                    continue
                vals = [float(v) for v in row_parts[1:]]
                rows.append(vals)

            if rows:
                last_block = (timestep, np.array(rows))
            i += 1
        else:
            i += 1

    if last_block is None:
        raise RuntimeError(f"No IEAD data blocks found in {path}")
    return last_block


def main():
    ap = argparse.ArgumentParser(description="Plot IEAD from SPARTA output")
    ap.add_argument("--file", "-f", required=True, help="Path to iead.dat")
    ap.add_argument("--emax", type=float, default=None,
                    help="Max energy [eV] (auto-detected from bins if omitted)")
    ap.add_argument("--amax", type=float, default=None,
                    help="Max angle [deg] (auto-detected from bins if omitted)")
    ap.add_argument("--log", action="store_true", help="Log color scale")
    ap.add_argument("--marginals", action="store_true",
                    help="Also plot 1-D energy and angle distributions")
    ap.add_argument("--out", "-o", default=None,
                    help="Save figure to file (e.g. iead.png)")
    ap.add_argument("--show", action="store_true", help="Show interactive plot")
    ap.add_argument("--cmap", default="inferno", help="Colormap (default: inferno)")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    fpath = Path(args.file)
    timestep, hist = parse_iead_file(fpath)
    bins_a, bins_e = hist.shape
    print(f"Loaded IEAD: timestep={timestep}, {bins_a} angle bins x {bins_e} energy bins")
    print(f"Total counts: {hist.sum():.0f}")

    emax = args.emax if args.emax else float(bins_e)
    amax = args.amax if args.amax else float(bins_a)

    # Bin edges
    energy_edges = np.linspace(0, emax, bins_e + 1)
    angle_edges = np.linspace(0, amax, bins_a + 1)
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    if args.marginals:
        fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_etop = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_aside = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis("off")
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6), dpi=150, constrained_layout=True)
        ax_etop = ax_aside = ax_info = None

    # --- Main 2-D heatmap ---
    E, A = np.meshgrid(energy_edges, angle_edges)
    if args.log:
        hplot = hist.copy()
        hplot[hplot <= 0] = np.nan
        vmin = np.nanmin(hplot[hplot > 0]) if np.any(hplot > 0) else 1
        vmax = np.nanmax(hplot) if np.any(hplot > 0) else 10
        im = ax_main.pcolormesh(E, A, hplot, cmap=args.cmap,
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                                shading="flat")
    else:
        im = ax_main.pcolormesh(E, A, hist, cmap=args.cmap, shading="flat")

    cb = fig.colorbar(im, ax=ax_main, pad=0.02)
    cb.set_label("Counts (weighted)")
    ax_main.set_xlabel("Energy [eV]")
    ax_main.set_ylabel("Angle from normal [deg]")

    if args.marginals:
        # Energy marginal (sum over angles)
        e_marginal = hist.sum(axis=0)
        ax_etop.bar(energy_centers, e_marginal,
                    width=energy_edges[1] - energy_edges[0],
                    color="steelblue", alpha=0.8, edgecolor="none")
        ax_etop.set_ylabel("Counts")
        ax_etop.set_title("Energy distribution")
        plt.setp(ax_etop.get_xticklabels(), visible=False)

        # Angle marginal (sum over energies)
        a_marginal = hist.sum(axis=1)
        ax_aside.barh(angle_centers, a_marginal,
                      height=angle_edges[1] - angle_edges[0],
                      color="coral", alpha=0.8, edgecolor="none")
        ax_aside.set_xlabel("Counts")
        ax_aside.set_title("Angle distribution")
        plt.setp(ax_aside.get_yticklabels(), visible=False)

        # Info box
        total = hist.sum()
        peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_e = energy_centers[peak_idx[1]]
        peak_a = angle_centers[peak_idx[0]]
        info = (f"Total hits: {total:.0f}\n"
                f"Peak energy: {peak_e:.1f} eV\n"
                f"Peak angle: {peak_a:.1f} deg\n"
                f"Timestep: {timestep}")
        ax_info.text(0.1, 0.5, info, transform=ax_info.transAxes,
                     fontsize=11, verticalalignment="center",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    title = args.title or f"IEAD (step {timestep}, {hist.sum():.0f} hits)"
    fig.suptitle(title, fontsize=13)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()
