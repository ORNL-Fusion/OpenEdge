#!/usr/bin/env python3
"""Strong-scaling plot from a timings CSV produced by collect_timings.py.

Usage:
    python3 plot_scaling.py results/timings.csv -o results/strong_scaling.png

Two panels:
  (left)  loop time vs ranks, log-log, with ideal 1/N line.
  (right) parallel efficiency = T(n_ref) * n_ref / (T(n) * n).
"""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.grid": True,
    "grid.alpha": 0.3, "lines.linewidth": 1.8, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r.get("nranks") and r.get("loop_time_s")]
    rows.sort(key=lambda r: int(r["nranks"]))
    n = np.array([int(r["nranks"]) for r in rows])
    t = np.array([float(r["loop_time_s"]) for r in rows])
    return n, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    n, t = load(args.csv)
    if len(n) < 2:
        raise SystemExit(f"need >=2 rank counts in {args.csv}")

    n_ref, t_ref = n[0], t[0]
    ideal = t_ref * n_ref / n
    eff = (t_ref * n_ref) / (t * n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    ax1.loglog(n, t, "o-", label="measured")
    ax1.loglog(n, ideal, "--", color="0.5",
               label=fr"ideal $1/N$ (ref: $N={n_ref}$)")
    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("loop time [s]")
    ax1.set_title("Strong scaling: time-to-solution")
    ax1.legend()

    ax2.semilogx(n, 100 * eff, "o-")
    ax2.axhline(100, color="0.5", ls="--", lw=0.9)
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("parallel efficiency [%]")
    ax2.set_title("Strong scaling: efficiency")
    ax2.set_ylim(0, 110)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
