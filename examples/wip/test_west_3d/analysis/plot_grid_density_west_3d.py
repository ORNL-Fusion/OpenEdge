#!/usr/bin/env python3
"""Plot a 3D impurity-density view from an OpenEdge grid dump.

Example:
  python3 analysis/plot_grid_density_west_3d.py \
    --dump output/tmp.grid.density \
    --out output/grid_density_3d.west.png \
    --log --threshold 1e10 --max-points 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_grid_dump(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    blocks = {}
    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue
        ts = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF CELLS":
            raise RuntimeError("Bad dump format: NUMBER OF CELLS")
        nc = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        bounds = []
        for j in range(3):
            lo, hi = map(float, lines[i + 1 + j].split()[:2])
            bounds.append((lo, hi))
        i += 4

        if not lines[i].startswith("ITEM: CELLS"):
            raise RuntimeError("Bad dump format: CELLS header")
        header = lines[i].split()[2:]
        i += 1

        cols = [[] for _ in header]
        for _ in range(nc):
            row = lines[i].split()
            i += 1
            for j in range(len(header)):
                cols[j].append(float(row[j]))
        blocks[ts] = (header, [np.array(c) for c in cols], bounds)

    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def choose_timestep(blocks, timestep_arg):
    keys = sorted(blocks)
    if timestep_arg == "last":
        return keys[-1]
    return int(timestep_arg)


def extract_density_block(blocks, timestep):
    header, cols, bounds = blocks[timestep]
    name_to_col = {name: arr for name, arr in zip(header, cols)}
    required = ["xc", "yc", "zc"]
    for key in required:
        if key not in name_to_col:
            raise RuntimeError(f"Missing required column '{key}'")

    value_names = [name for name in header if name not in {"id", "xc", "yc", "zc"}]
    if not value_names:
        raise RuntimeError("No density/value columns found in dump")

    values = np.zeros_like(name_to_col[value_names[0]])
    for name in value_names:
        values += name_to_col[name]

    return (
        name_to_col["xc"],
        name_to_col["yc"],
        name_to_col["zc"],
        values,
        bounds,
        value_names,
    )


def maybe_subsample(x, y, z, val, max_points):
    if max_points is None or len(val) <= max_points:
        return x, y, z, val
    rng = np.random.default_rng(42)
    idx = rng.choice(len(val), size=max_points, replace=False)
    return x[idx], y[idx], z[idx], val[idx]


def main():
    here = Path(__file__).resolve().parent.parent
    default_dump = here / "output" / "tmp.grid.density"
    default_out = here / "output" / "grid_density_3d.west.png"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(default_dump), help="grid dump file")
    ap.add_argument("--out", default=str(default_out), help="output PNG path")
    ap.add_argument("--timestep", default="last", help="last or explicit integer")
    ap.add_argument("--threshold", type=float, default=0.0, help="plot only values above this threshold")
    ap.add_argument("--max-points", type=int, default=50000, help="random cap on plotted points")
    ap.add_argument("--size", type=float, default=4.0, help="scatter marker size")
    ap.add_argument("--alpha", type=float, default=0.5, help="scatter alpha")
    ap.add_argument("--log", action="store_true", help="use log color scale")
    ap.add_argument("--elev", type=float, default=18.0, help="3D elevation angle")
    ap.add_argument("--azim", type=float, default=-62.0, help="3D azimuth angle")
    args = ap.parse_args()

    blocks = parse_grid_dump(Path(args.dump))
    ts = choose_timestep(blocks, args.timestep)
    x, y, z, val, bounds, value_names = extract_density_block(blocks, ts)

    mask = np.isfinite(val) & (val > args.threshold)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    val = val[mask]
    if len(val) == 0:
        raise RuntimeError("No cells remain after thresholding")

    x, y, z, val = maybe_subsample(x, y, z, val, args.max_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if args.log:
        positive = val[val > 0.0]
        norm = LogNorm(vmin=max(positive.min(), args.threshold if args.threshold > 0 else positive.min()),
                       vmax=positive.max())
    else:
        norm = None

    sc = ax.scatter(x, y, z, c=val, s=args.size, alpha=args.alpha, cmap="inferno", norm=norm)

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.04)
    cbar.set_label("Impurity density")

    title_scale = "log" if args.log else "linear"
    ax.set_title(f"WEST 3D impurity density, timestep {ts} ({title_scale})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Wrote {out}")
    print(f"  timestep: {ts}")
    print(f"  value columns summed: {value_names}")
    print(f"  points plotted: {len(val)}")
    print(f"  threshold: {args.threshold}")


if __name__ == "__main__":
    main()
