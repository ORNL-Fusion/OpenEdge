#!/usr/bin/env python3
"""Plot droplet emission distributions f(E) and f(theta) from a SPARTA
particle dump.

Reads output/particles.txt, filters by particle type (default = 1, mist),
and produces:

  (a) f(E/E_max)   — kinetic energy normalized by the launch-cap energy
                     E_max = 0.5 * m * vmax^2 (vmax from CLI, matches the
                     fix droplet/emit `vmax` setting).
  (b) f(theta)     — polar angle of the velocity from +y (vertical) [deg]

Use the latest dump snapshot, or pass --all-snapshots to merge every
recorded timestep (better statistics when emission rate is low).

Usage:
    python3 plot_dist.py [--particles output/particles.txt] \
                         [--type 1] [--vmax 20.0] [--all-snapshots]
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
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
})

# Mass of mist droplet (kg) — must match droplet.species column 3 (Molmass).
# Override on the CLI if you're plotting drop_1/drop_2/drop_3 instead.
M_MIST_DEFAULT = 2.796017461694916e-10


def read_particles(path, ptype, all_snapshots):
    """Yield (vx, vy, vz, type) arrays from one or all dump snapshots."""
    with open(path) as f:
        text = f.read()

    snaps = re.split(r"ITEM: TIMESTEP\s*\n(\d+)\s*\n", text)[1:]
    if not snaps:
        sys.exit("no ITEM: TIMESTEP blocks found in dump")
    pairs = list(zip(snaps[0::2], snaps[1::2]))   # (timestep_str, body)
    if not all_snapshots:
        pairs = [pairs[-1]]

    rows = []
    for ts_str, body in pairs:
        m = re.search(r"ITEM: ATOMS([^\n]*)\n(.*?)(?=\nITEM:|\Z)",
                       body, flags=re.DOTALL)
        if not m:
            continue
        cols = m.group(1).split()
        idx = {name: i for i, name in enumerate(cols)}
        for ln in m.group(2).strip().splitlines():
            tok = ln.split()
            if len(tok) < len(cols):
                continue
            t = int(float(tok[idx["type"]]))
            if t != ptype:
                continue
            vx = float(tok[idx["vx"]]); vy = float(tok[idx["vy"]]); vz = float(tok[idx["vz"]])
            rows.append((vx, vy, vz))
    if not rows:
        sys.exit(f"no particles of type={ptype} found in dump")
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def main(argv):
    ap = argparse.ArgumentParser(description="Plot f(E), f(theta) for droplets.")
    ap.add_argument("--particles", default="output/particles.txt",
                    help="Path to SPARTA particle dump (default output/particles.txt)")
    ap.add_argument("--type", type=int, default=1,
                    help="Particle type to filter on (default 1 = mist)")
    ap.add_argument("--mass", type=float, default=M_MIST_DEFAULT,
                    help="Per-particle mass in kg (default mist 2.796e-10)")
    ap.add_argument("--all-snapshots", action="store_true",
                    help="Merge every recorded snapshot in the dump (better stats).")
    ap.add_argument("--bins", type=int, default=40,
                    help="Histogram bin count (default 40)")
    ap.add_argument("--vmax", type=float, default=20.0,
                    help="Launch-cap speed in m/s; sets E_max = 0.5*m*vmax^2 "
                         "for E-axis normalization (default 20).")
    ap.add_argument("--out", default="dist.png", help="Output PNG path")
    args = ap.parse_args(argv)

    vx, vy, vz = read_particles(args.particles, args.type, args.all_snapshots)
    speed = np.sqrt(vx*vx + vy*vy + vz*vz)
    KE_J = 0.5 * args.mass * speed**2
    E_max = 0.5 * args.mass * args.vmax**2
    KE_norm = KE_J / E_max

    # Polar angle from +y axis (Cartesian convention here: +y = +Z = vertical).
    theta_deg = np.degrees(np.arccos(np.clip(vy / np.maximum(speed, 1e-30), -1.0, 1.0)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].hist(KE_norm, bins=args.bins,
                 color="C0", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel(r"$E / E_\mathrm{max}\quad(E_\mathrm{max}=\frac{1}{2}\,m\,v_\mathrm{max}^2)$")
    axes[0].set_ylabel(r"counts")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(theta_deg, bins=args.bins, range=(0.0, 180.0),
                 color="C3", edgecolor="black", alpha=0.85)
    axes[1].set_xlabel(r"polar angle $\theta$ (deg)")
    axes[1].set_ylabel(r"counts")
    axes[1].set_xlim(0, 180)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
