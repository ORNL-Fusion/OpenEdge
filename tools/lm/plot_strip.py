#!/usr/bin/env python3
"""
Paper-ready 4-panel verification plot for tools/lm/run_strip output.

Reads the CSV produced by run_strip and shows Tsurf, film thickness,
evaporation flux, and adatom flux against the SOLPS target arc length.

Usage:
    python3 plot_strip.py outer.csv [--solps SOLPS_DIR] [--out fig.png]

If --solps is given, the input Wtot column from ld_tg_o.dat is overlaid
on the Tsurf panel (twin axis) for sanity-checking what the strip saw.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,             # mathtext is enough for these labels
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})


def read_run_strip_csv(path):
    cols = np.genfromtxt(path, delimiter=",", names=True, comments="#")
    return cols


def read_solps_wtot(solps_dir, leg="o"):
    p = os.path.join(solps_dir, f"ld_tg_{leg}.dat")
    if not os.path.isfile(p):
        return None, None
    data = []
    with open(p) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            data.append([float(parts[0]), float(parts[4])])
    if not data:
        return None, None
    a = np.asarray(data)
    return a[:, 0], a[:, 1]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--solps", default=None,
                    help="SOLPS b2pl.exe.dir for Wtot overlay")
    ap.add_argument("--leg", default="o", choices=["o", "i"],
                    help="outer (o) or inner (i) target")
    ap.add_argument("--out", default="strip_verification.png")
    args = ap.parse_args(argv)

    d = read_run_strip_csv(args.csv)
    x = d["x_m"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    (axTs, axEvap), (axH, axAd) = axes

    # --- Tsurf with optional Wtot overlay ----------------------------------
    axTs.plot(x, d["Tsurf_C"], color="C3")
    axTs.set_ylabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    axTs.grid(True, alpha=0.3)
    axTs.axhline(180.5, color="0.6", ls=":", lw=1)
    axTs.text(x[-1], 180.5, " Li melt", color="0.4", va="center", fontsize=10)
    if args.solps is not None:
        sx, sw = read_solps_wtot(args.solps, args.leg)
        if sx is not None:
            ax2 = axTs.twinx()
            ax2.plot(sx, sw / 1e6, color="0.4", lw=1.5, ls="--",
                     label=r"$W_\mathrm{tot}$ (SOLPS)")
            ax2.set_ylabel(r"$W_\mathrm{tot}\ (\mathrm{MW/m^2})$",
                           color="0.4")
            ax2.tick_params(axis="y", labelcolor="0.4")

    # --- evap flux (log) ---------------------------------------------------
    pos = d["evap_flux"] > 0
    axEvap.semilogy(x[pos], d["evap_flux"][pos], color="C0")
    axEvap.set_ylabel(r"$\Gamma_\mathrm{evap}\ (\mathrm{atoms\,m^{-2}\,s^{-1}})$")
    axEvap.grid(True, which="both", alpha=0.3)

    # --- film thickness ----------------------------------------------------
    axH.plot(x, d["h_film_m"] * 1e3, color="C2")
    axH.set_ylabel(r"$h_\mathrm{film}\ (\mathrm{mm})$")
    axH.set_xlabel(r"target arc length $x\ (\mathrm{m})$")
    axH.grid(True, alpha=0.3)

    # --- adatom flux (log) -------------------------------------------------
    pos = d["adatom_flux"] > 0
    axAd.semilogy(x[pos], d["adatom_flux"][pos], color="C1")
    axAd.set_ylabel(r"$\Gamma_\mathrm{adatom}\ (\mathrm{atoms\,m^{-2}\,s^{-1}})$")
    axAd.set_xlabel(r"target arc length $x\ (\mathrm{m})$")
    axAd.grid(True, which="both", alpha=0.3)

    # mark strike point (peak Wtot)
    if args.solps is not None and sx is not None:
        x_peak = sx[np.argmax(sw)]
        for ax in (axTs, axEvap, axH, axAd):
            ax.axvline(x_peak, color="0.7", ls=":", lw=1)

    fig.suptitle(
        f"LM strip verification — {os.path.basename(args.csv)}",
        fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
