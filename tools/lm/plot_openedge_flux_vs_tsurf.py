#!/usr/bin/env python3
"""Plot Li source fluxes vs T_surf from OpenEdge surf-dump output.

Companion to `plotter_flux_vs_surf_temp.py` (which plots analytic curves):
this script plots the *simulated* per-surf fluxes against the per-surf
surface temperature from one or more SPARTA dump files.

Each dump row contributes one data point per surface flux component:
  - c_cpmiLi  (physical sputter flux,         m^-2 s^-1)
  - c_cevap   (thermal evaporation flux,      m^-2 s^-1)
  - c_cadat   (ad-atom desorption flux,       m^-2 s^-1)
  - s_Tsurf_lm (liquid-metal surface T, deg C)

Pass several --sparta files to overlay multiple LM operating points
(different inlet flow / heat-flux conditions → different T_surf bands),
which is the way to span a Tsurf range with the static-mode Smolentsev
strip model (each run gives one nearly-uniform T_surf per leg).

Usage:
    python3 plot_openedge_flux_vs_tsurf.py \
        --sparta path/to/surf_fluxes.txt [more files ...] \
        --id-range 140 239 \
        --out flux_vs_tsurf.png
"""

import argparse
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})


def read_sparta_surf_dump(path):
    """Return (col_index_dict, data_array) from the LAST snapshot of a dump."""
    with open(path) as f:
        text = f.read()
    blocks = re.split(r"ITEM: TIMESTEP\s*\n\d+\s*\n", text)
    block = blocks[-1]
    m = re.search(r"ITEM: SURFS([^\n]*)\n(.*)", block, flags=re.DOTALL)
    cols = m.group(1).split()
    rows = [line.split()[:len(cols)]
            for line in m.group(2).strip().splitlines() if line.strip()]
    idx = {name: i for i, name in enumerate(cols)}
    return idx, np.array(rows, dtype=float)


def filter_dump(idx, data, sid_lo, sid_hi):
    sid = data[:, idx["id"]].astype(int)
    mask = (sid >= sid_lo) & (sid <= sid_hi)
    if not mask.any():
        return None
    sub = data[mask]
    return {
        "Tsurf": sub[:, idx["s_Tsurf_lm"]],
        "evp":   sub[:, idx["c_cevap"]],
        "ad":    sub[:, idx["c_cadat"]],
        "sput":  sub[:, idx["c_cpmiLi"]],
    }


def main(argv):
    ap = argparse.ArgumentParser(
        description="Plot OpenEdge per-surf Li source fluxes vs T_surf.")
    ap.add_argument("--sparta", nargs="+", required=True,
                    help="One or more SPARTA surf dump files (output/surf_fluxes.txt).")
    ap.add_argument("--label", nargs="+", default=None,
                    help="Optional per-file labels (must match # of --sparta files).")
    ap.add_argument("--id-range", nargs=2, type=int, default=[140, 239],
                    help="Surf id range (lo hi) to include (default 140 239).")
    ap.add_argument("--ymin", type=float, default=1.0e16,
                    help="y-axis lower limit (default 1e16)")
    ap.add_argument("--ymax", type=float, default=1.0e25,
                    help="y-axis upper limit (default 1e25)")
    ap.add_argument("--out", default="openedge_flux_vs_tsurf.png",
                    help="Output PNG path")
    args = ap.parse_args(argv)

    if args.label and len(args.label) != len(args.sparta):
        sys.exit("--label count must match --sparta count")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    cmap = plt.get_cmap("viridis")
    n = len(args.sparta)

    all_T = []
    for k, path in enumerate(args.sparta):
        idx, data = read_sparta_surf_dump(path)
        leg = filter_dump(idx, data, args.id_range[0], args.id_range[1])
        if leg is None:
            print(f"warning: no surfs in id range {args.id_range} for {path}")
            continue
        col = cmap(0.15 + 0.7 * (k / max(n - 1, 1)))
        lab = args.label[k] if args.label else f"file {k + 1}"
        # Sort by Tsurf for clean line connection within a single file
        order = np.argsort(leg["Tsurf"])
        T = leg["Tsurf"][order]
        all_T.append(T)
        evp  = leg["evp"][order]
        ad   = leg["ad"][order]
        sput = leg["sput"][order]
        # Use circle, triangle, square markers for evp / ad / sput (Islam Fig 7 motif)
        ax.semilogy(T, evp,  color="C0", marker="o", ms=6, ls="-",  alpha=0.9,
                    label=rf"$\Gamma_\mathrm{{Evp}}$ — {lab}" if k == 0 else None)
        ax.semilogy(T, ad,   color="C1", marker="^", ms=7, ls="--", alpha=0.9,
                    label=rf"$\Gamma_\mathrm{{Ad}}$ — {lab}"  if k == 0 else None)
        ax.semilogy(T, sput, color="C2", marker="s", ms=6, ls=":",  alpha=0.9,
                    label=rf"$\Gamma_\mathrm{{Sput}}$ — {lab}" if k == 0 else None)

    if not all_T:
        sys.exit("no data found in any input file for the given id range")

    Tcat = np.concatenate(all_T)
    ax.set_xlabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax.set_ylabel(r"emitted Li flux $(\mathrm{m^{-2}\,s^{-1}})$")
    pad = 0.05 * max(Tcat.max() - Tcat.min(), 1.0)
    ax.set_xlim(Tcat.min() - pad, Tcat.max() + pad)
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("OpenEdge LM per-surf fluxes — surface dump")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
