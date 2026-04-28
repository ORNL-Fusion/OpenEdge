#!/usr/bin/env python3
"""
Plot Tsurf_lm and Gamma_D_lm on the full wall.surf geometry. Each line
segment is colored by its custom value, with surfs that are not LM-treated
(Tsurf_lm = 0, no group membership) drawn in light grey.

Usage:
    python3 plot_wall_tsurf.py --sparta output/surf_fluxes.txt
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
})


def read_sparta_surf_dump(path):
    with open(path) as f:
        text = f.read()
    blocks = re.split(r"ITEM: TIMESTEP\s*\n\d+\s*\n", text)
    block = blocks[-1]
    m = re.search(r"ITEM: SURFS([^\n]*)\n(.*)", block, flags=re.DOTALL)
    cols = m.group(1).split()
    rows = [line.split()[:len(cols)]
            for line in m.group(2).strip().splitlines() if line.strip()]
    return cols, np.array(rows, dtype=float)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparta", required=True)
    ap.add_argument("--out", default="wall_tsurf.png")
    args = ap.parse_args(argv)

    cols, sp = read_sparta_surf_dump(args.sparta)
    idx = {c: i for i, c in enumerate(cols)}

    v1x = sp[:, idx["v1x"]]   # axi: x = Z
    v1y = sp[:, idx["v1y"]]   # axi: y = R
    v2x = sp[:, idx["v2x"]]
    v2y = sp[:, idx["v2y"]]
    Ts = sp[:, idx["s_Tsurf_lm"]]
    Gd = sp[:, idx["s_Gamma_D_lm"]]

    # axi convention: SPARTA axes (x=Z, y=R). Plot in (R, Z) — natural
    # poloidal cross-section.
    R1, Z1 = v1y, v1x
    R2, Z2 = v2y, v2x

    # build segment list
    segs = np.array([[(R1[i], Z1[i]), (R2[i], Z2[i])]
                     for i in range(len(sp))])

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)

    # mask LM-treated surfs (Tsurf_lm > 0)
    treated = Ts > 0.0
    untreated = ~treated

    # data limits for both panels (LineCollection doesn't update them)
    R_all = np.concatenate([R1, R2])
    Z_all = np.concatenate([Z1, Z2])
    pad_R = 0.05 * (R_all.max() - R_all.min())
    pad_Z = 0.05 * (Z_all.max() - Z_all.min())
    xlim = (R_all.min() - pad_R, R_all.max() + pad_R)
    ylim = (Z_all.min() - pad_Z, Z_all.max() + pad_Z)

    # --- Panel 1: Tsurf ----
    ax = axes[0]
    if untreated.any():
        lc_off = LineCollection(segs[untreated], colors="0.78",
                                linewidths=1.5)
        ax.add_collection(lc_off)
    if treated.any():
        lc = LineCollection(segs[treated], cmap="inferno",
                            linewidths=3.5)
        lc.set_array(Ts[treated])
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, shrink=0.8)
        cb.set_label(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$R\ (\mathrm{m})$")
    ax.set_ylabel(r"$Z\ (\mathrm{m})$")
    ax.set_aspect("equal")
    ax.set_title(r"$T_\mathrm{surf,\,LM}$ on the wall")
    ax.grid(True, alpha=0.25)

    # --- Panel 2: Gamma_D ----
    ax = axes[1]
    if untreated.any():
        lc_off = LineCollection(segs[untreated], colors="0.78",
                                linewidths=1.5)
        ax.add_collection(lc_off)
    if treated.any() and Gd[treated].max() > 0:
        # log-color for the wide dynamic range typical of Bohm flux
        Gd_pos_min = Gd[treated][Gd[treated] > 0].min() if (Gd[treated] > 0).any() else 1.0
        Gd_pos = np.where(Gd[treated] > 0, Gd[treated], Gd_pos_min)
        lc = LineCollection(segs[treated], cmap="viridis",
                            linewidths=3.5,
                            norm=matplotlib.colors.LogNorm(
                                vmin=max(Gd_pos.min(), 1.0),
                                vmax=Gd_pos.max()))
        lc.set_array(Gd_pos)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, shrink=0.8)
        cb.set_label(r"$\Gamma_{D^+}\ (\mathrm{m^{-2}\,s^{-1}})$")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$R\ (\mathrm{m})$")
    ax.set_aspect("equal")
    ax.set_title(r"$\Gamma_{D^+}\ \mathrm{(SOLPS\ Bohm)}$")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Wall layout — {os.path.basename(args.sparta)}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
