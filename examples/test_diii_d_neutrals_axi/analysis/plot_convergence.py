#!/usr/bin/env python3
"""Domain-integrated convergence plot from a multi-snapshot SPARTA grid dump.

Reads every snapshot in `dump d1 grid all <Nevery> ...` and computes
volume-weighted integrals of each source moment over the whole axi torus,
giving totals in proper SI units versus physical time:

    integral n_D  dV   (atoms)
    integral n_D2 dV   (molecules)
    integral S_p  dV   (s^-1)         ions created per second
    integral R Sm dV   (N m)          toroidal angular-momentum rate
    integral Q_e  dV   (W)            electron energy source
    integral Q_i  dV   (W)            ion energy source

Cell volume in axi: dV = pi * (R_hi^2 - R_lo^2) * dZ.

Usage:
    python3 analysis/plot_convergence.py \\
        --dump diii_d_neutrals_axi.grid \\
        --out  plots/convergence.png \\
        --dt   5e-8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NCOLS = 16
COL_YC = 2
COL_XLO, COL_YLO, COL_XHI, COL_YHI = 3, 4, 5, 6
COL_ND2, COL_ND = 7, 8
COL_SP, COL_SMX, COL_SMY, COL_SMZ, COL_QE, COL_QI = 10, 11, 12, 13, 14, 15


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


def integrate(arr: np.ndarray) -> dict:
    dZ   = arr[:, COL_XHI] - arr[:, COL_XLO]
    R_lo = arr[:, COL_YLO]; R_hi = arr[:, COL_YHI]
    V = dZ * np.pi * (R_hi * R_hi - R_lo * R_lo)
    R = arr[:, COL_YC]
    Sm = arr[:, COL_SMX] + arr[:, COL_SMY] + arr[:, COL_SMZ]
    return {
        "n_D":  (arr[:, COL_ND]  * V).sum(),
        "n_D2": (arr[:, COL_ND2] * V).sum(),
        "S_p":  (arr[:, COL_SP]  * V).sum(),
        "S_m":  (Sm * R * V).sum(),
        "Q_e":  (arr[:, COL_QE]  * V).sum(),
        "Q_i":  (arr[:, COL_QI]  * V).sum(),
    }


PANELS = [
    ("n_D",  r"$\int n_D\,dV$",      "atoms"),
    ("n_D2", r"$\int n_{D_2}\,dV$",  "molecules"),
    ("S_p",  r"$\int S_p\,dV$",      r"s$^{-1}$"),
    ("S_m",  r"$\int R\,S_m\,dV$",   r"N$\,$m"),
    ("Q_e",  r"$\int Q_e\,dV$",      "W"),
    ("Q_i",  r"$\int Q_i\,dV$",      "W"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True)
    ap.add_argument("--dt",   type=float, default=5e-8,
                    help="physical timestep in s (default 5e-8 = deck dt)")
    args = ap.parse_args()

    steps, totals = [], {k: [] for k, _, _ in PANELS}
    for step, arr in iter_snapshots(args.dump):
        d = integrate(arr)
        steps.append(step)
        for k in totals:
            totals[k].append(d[k])
    if len(steps) < 2:
        raise SystemExit(f"need >=2 snapshots in {args.dump}, got {len(steps)}")
    t_ms = np.asarray(steps) * args.dt * 1e3

    plt.rcParams.update({
        "font.family": "serif", "font.size": 13,
        "axes.labelsize": 14, "axes.titlesize": 13,
    })
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10), sharex=True)
    for ax, (key, label, unit) in zip(axes.flat, PANELS):
        y = np.asarray(totals[key])
        ax.plot(t_ms, y, color="tab:blue", lw=1.8)
        if len(y) >= 4:
            ax.axhline(y[len(y)//2:].mean(), color="tab:red", lw=1.0, ls="--",
                       label="tail mean")
            ax.legend(loc="best", frameon=False, fontsize=10)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_ylabel(f"{label} ({unit})")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                            useMathText=True)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"time $t$ (ms)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
