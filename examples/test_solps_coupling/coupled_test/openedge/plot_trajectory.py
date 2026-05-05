#!/usr/bin/env python3
"""Plot droplet trajectories in (R, Z) from a SPARTA particle dump.

Reads output/particles.txt across all recorded snapshots, groups records
by particle id, and draws each particle's path through R-Z space.  An
optional wall.surf overlay shows the divertor outline for context.

Usage:
    python3 plot_trajectory.py [--particles output/particles.txt] \
                               [--wall ../../input/wall.surf] \
                               [--type 1] [--max-tracks 200] \
                               [--xlim 2.5 5.0] [--ylim -4 -2.5] \
                               [--out trajectory.png]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

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
    "lines.linewidth": 1.2,
})


def read_particles_all(path, ptype):
    """Return dict id -> list of (timestep, x, y) for particles of type ptype."""
    with open(path) as f:
        text = f.read()
    snaps = re.split(r"ITEM: TIMESTEP\s*\n(\d+)\s*\n", text)[1:]
    pairs = list(zip(snaps[0::2], snaps[1::2]))
    if not pairs:
        sys.exit("no ITEM: TIMESTEP blocks found in dump")
    tracks = defaultdict(list)
    for ts_str, body in pairs:
        ts = int(ts_str)
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
            pid = int(float(tok[idx["id"]]))
            x = float(tok[idx["x"]])
            y = float(tok[idx["y"]])
            tracks[pid].append((ts, x, y))
    if not tracks:
        sys.exit(f"no particles of type={ptype} found in dump")
    return tracks


def parse_wall(path):
    """Return list of (x, y) line endpoint pairs from wall.surf."""
    text = Path(path).read_text().splitlines()
    pts, lines, section = {}, [], None
    for ln in text:
        s = ln.strip()
        if s == "Points": section = "P"; continue
        if s == "Lines":  section = "L"; continue
        if not s or s.startswith("#"): continue
        tk = s.split()
        if section == "P" and len(tk) == 3:
            pts[int(tk[0])] = (float(tk[1]), float(tk[2]))
        elif section == "L" and len(tk) == 3:
            lines.append((int(tk[1]), int(tk[2])))
    return pts, lines


def main(argv):
    ap = argparse.ArgumentParser(description="Plot droplet trajectories.")
    ap.add_argument("--particles", default="output/particles.txt",
                    help="SPARTA particle dump (default output/particles.txt)")
    ap.add_argument("--wall", default="../../input/wall.surf",
                    help="wall.surf overlay (default ../../input/wall.surf)")
    ap.add_argument("--type", type=int, default=1,
                    help="Particle type to plot (default 1 = mist).")
    ap.add_argument("--max-tracks", type=int, default=200,
                    help="Cap on number of trajectories drawn (default 200).")
    ap.add_argument("--min-points", type=int, default=1,
                    help="Skip particles with fewer than this many recorded points (default 1).")
    ap.add_argument("--xlim", nargs=2, type=float, default=[3.0, 4.0],
                    help="R-axis limits (m), default 3 4.")
    ap.add_argument("--ylim", nargs=2, type=float, default=[-4.0, 2.0],
                    help="Z-axis limits (m), default -4 2.")
    ap.add_argument("--out", default="trajectory.png", help="Output PNG path")
    args = ap.parse_args(argv)

    tracks = read_particles_all(args.particles, args.type)

    # Filter and (optionally) cap
    items = [(pid, sorted(pts)) for pid, pts in tracks.items()
             if len(pts) >= args.min_points]
    items.sort(key=lambda it: -len(it[1]))    # longest tracks first
    if args.max_tracks > 0 and len(items) > args.max_tracks:
        items = items[:args.max_tracks]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Wall overlay
    try:
        pts, lines = parse_wall(args.wall)
        for p1, p2 in lines:
            x1, y1 = pts[p1]; x2, y2 = pts[p2]
            ax.plot([x1, x2], [y1, y2], color="black", lw=1.2, zorder=1)
    except FileNotFoundError:
        print(f"warning: wall file {args.wall} not found, skipping overlay")

    # Trajectories: color by sequence of arrival, alpha for legibility
    cmap = plt.get_cmap("turbo")
    for k, (pid, pts) in enumerate(items):
        col = cmap((k % 64) / 63.0)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        if len(pts) >= 2:
            ax.plot(xs, ys, color=col, lw=1.0, alpha=0.7, zorder=2)
            ax.plot(xs[0], ys[0], 'o', color=col, ms=3, zorder=3)   # start
            ax.plot(xs[-1], ys[-1], 's', color=col, ms=3, zorder=3) # end
        else:
            ax.plot(xs[0], ys[0], 'o', color=col, ms=4, alpha=0.8, zorder=3)

    ax.set_xlabel(r"$R\ (\mathrm{m})$")
    ax.set_ylabel(r"$Z\ (\mathrm{m})$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if args.xlim: ax.set_xlim(*args.xlim)
    if args.ylim: ax.set_ylim(*args.ylim)
    ax.set_title(rf"droplet trajectories ($N={len(items)}$ shown, "
                 rf"type={args.type})")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
