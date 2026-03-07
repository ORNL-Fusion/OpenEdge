#!/usr/bin/env python3
"""
Plot Ion Energy-Angle Distribution (IEAD) from surf_react_pmi CSV impact logs.

The PMI impact log (enabled with `log yes file <path>`) writes per-rank CSV
files with columns:
  timestep,rank,surf_id,species,E_in_eV,theta_deg,outcome,E_out_eV,
  x,y,z,vx,vy,vz,nx,ny,nz

This script reads all rank files matching the pattern <base>.rank*, merges
them, and builds a 2-D energy-angle histogram.

Usage examples:
  python plot_iead.py --file output/pmi_impacts.csv --emax 200 --amax 90
  python plot_iead.py --file output/pmi_impacts.csv --emax 200 --amax 90 --log
  python plot_iead.py --file output/pmi_impacts.csv --outcome R --species W+
  python plot_iead.py --file output/pmi_impacts.csv --marginals --out iead.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_impact_csvs(base_path: str) -> pd.DataFrame:
    """Load and concatenate all per-rank CSV files matching base_path.rank*."""
    pattern = f"{base_path}.rank*"
    files = sorted(glob.glob(pattern))
    if not files:
        # Try the base path itself (single-rank run)
        if Path(base_path).exists():
            files = [base_path]
        else:
            raise FileNotFoundError(
                f"No impact CSV files found matching {pattern} or {base_path}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} impacts from {len(files)} rank file(s)")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Plot IEAD from surf_react_pmi CSV impact logs")
    ap.add_argument("--file", "-f", required=True,
                    help="Base path to impact CSV (e.g. output/pmi_impacts.csv)")
    ap.add_argument("--emax", type=float, default=500.0,
                    help="Max energy [eV] for histogram (default: 500)")
    ap.add_argument("--amax", type=float, default=90.0,
                    help="Max angle [deg] for histogram (default: 90)")
    ap.add_argument("--nbins-e", type=int, default=200,
                    help="Number of energy bins (default: 200)")
    ap.add_argument("--nbins-a", type=int, default=90,
                    help="Number of angle bins (default: 90)")
    ap.add_argument("--outcome", default=None, choices=["R", "S", "A"],
                    help="Filter by outcome: R=reflect, S=sputter, A=absorb")
    ap.add_argument("--species", default=None,
                    help="Filter by incident species (e.g. W, W+, W2+)")
    ap.add_argument("--log", action="store_true", help="Log color scale")
    ap.add_argument("--marginals", action="store_true",
                    help="Also plot 1-D energy and angle distributions")
    ap.add_argument("--out", "-o", default=None,
                    help="Save figure to file (e.g. iead.png)")
    ap.add_argument("--show", action="store_true", help="Show interactive plot")
    ap.add_argument("--cmap", default="inferno", help="Colormap (default: inferno)")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    df = load_impact_csvs(args.file)

    # Apply filters
    if args.outcome:
        df = df[df["outcome"] == args.outcome]
        print(f"  After outcome={args.outcome} filter: {len(df)} impacts")
    if args.species:
        df = df[df["species"] == args.species]
        print(f"  After species={args.species} filter: {len(df)} impacts")

    if len(df) == 0:
        print("No impacts match the filters. Nothing to plot.")
        return

    energy = df["E_in_eV"].values
    angle = df["theta_deg"].values

    # Build 2-D histogram
    hist, e_edges, a_edges = np.histogram2d(
        energy, angle,
        bins=[args.nbins_e, args.nbins_a],
        range=[[0, args.emax], [0, args.amax]])
    # Transpose so rows=angle, cols=energy (convention)
    hist = hist.T

    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])

    print(f"IEAD: {args.nbins_a} angle bins x {args.nbins_e} energy bins, "
          f"total counts: {hist.sum():.0f}")

    if args.marginals:
        fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_etop = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_aside = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis("off")
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6), dpi=150,
                                     constrained_layout=True)
        ax_etop = ax_aside = ax_info = None

    # --- Main 2-D heatmap ---
    E, A = np.meshgrid(e_edges, a_edges)
    if args.log:
        hplot = hist.copy().astype(float)
        hplot[hplot <= 0] = np.nan
        vmin = np.nanmin(hplot[hplot > 0]) if np.any(hplot > 0) else 1
        vmax = np.nanmax(hplot) if np.any(hplot > 0) else 10
        im = ax_main.pcolormesh(E, A, hplot, cmap=args.cmap,
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                                shading="flat")
    else:
        im = ax_main.pcolormesh(E, A, hist, cmap=args.cmap, shading="flat")

    cb = fig.colorbar(im, ax=ax_main, pad=0.02)
    cb.set_label("Counts")
    ax_main.set_xlabel("Energy [eV]")
    ax_main.set_ylabel("Angle from normal [deg]")

    if args.marginals:
        # Energy marginal (sum over angles)
        e_marginal = hist.sum(axis=0)
        ax_etop.bar(e_centers, e_marginal,
                    width=e_edges[1] - e_edges[0],
                    color="steelblue", alpha=0.8, edgecolor="none")
        ax_etop.set_ylabel("Counts")
        ax_etop.set_title("Energy distribution")
        plt.setp(ax_etop.get_xticklabels(), visible=False)

        # Angle marginal (sum over energies)
        a_marginal = hist.sum(axis=1)
        ax_aside.barh(a_centers, a_marginal,
                      height=a_edges[1] - a_edges[0],
                      color="coral", alpha=0.8, edgecolor="none")
        ax_aside.set_xlabel("Counts")
        ax_aside.set_title("Angle distribution")
        plt.setp(ax_aside.get_yticklabels(), visible=False)

        # Info box
        total = hist.sum()
        peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_e = e_centers[peak_idx[1]]
        peak_a = a_centers[peak_idx[0]]
        filters = []
        if args.outcome:
            filters.append(f"outcome={args.outcome}")
        if args.species:
            filters.append(f"species={args.species}")
        info = (f"Total hits: {total:.0f}\n"
                f"Peak energy: {peak_e:.1f} eV\n"
                f"Peak angle: {peak_a:.1f} deg")
        if filters:
            info += f"\nFilter: {', '.join(filters)}"
        ax_info.text(0.1, 0.5, info, transform=ax_info.transAxes,
                     fontsize=11, verticalalignment="center",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    title = args.title or f"IEAD ({hist.sum():.0f} hits)"
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
