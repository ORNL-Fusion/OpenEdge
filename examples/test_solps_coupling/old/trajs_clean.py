#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from utils import surface


def parse_dump(filepath: Path):
    rows = {k: [] for k in ["timestep", "id", "x", "y", "z", "radius"]}
    with filepath.open() as f:
        lines = f.read().splitlines()

    i = 0
    timestep = None
    n_atoms = None
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue
        if line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            headers = lines[i].split()[2:]
            idx = {name: j for j, name in enumerate(headers)}
            if timestep is None or n_atoms is None:
                raise RuntimeError("Incomplete dump block before ATOMS")
            for k in range(n_atoms):
                vals = lines[i + 1 + k].split()
                rows["timestep"].append(timestep)
                rows["id"].append(int(vals[idx["id"]]))
                rows["x"].append(float(vals[idx["x"]]))
                rows["y"].append(float(vals[idx["y"]]))
                rows["z"].append(float(vals[idx["z"]]))
                radius_idx = idx.get("radius")
                radius_val = float(vals[radius_idx]) if radius_idx is not None else np.nan
                rows["radius"].append(radius_val)
            i += 1 + n_atoms
            continue
        i += 1

    for key in rows:
        rows[key] = np.asarray(rows[key])
    return rows


def radius_metric(radius, mode="percent_loss"):
    rad = np.asarray(radius, dtype=float)
    valid = np.isfinite(rad) & (rad > 0.0)
    if not np.any(valid):
        return np.full(rad.shape, np.nan)
    r0 = rad[np.where(valid)[0][0]]
    if mode == "percent_loss":
        return 100.0 * (1.0 - rad / r0)
    if mode == "ratio":
        return rad / r0
    if mode == "radius":
        return rad
    raise ValueError("mode must be radius, ratio, or percent_loss")


def add_traj_colored(ax, R, Z, values, cmap, vmin, vmax):
    if len(R) < 2:
        return
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    if np.count_nonzero(valid) < 2:
        return
    points = np.array([R, Z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    keep = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
    if not np.any(keep):
        return
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs[keep], cmap=cmap, norm=norm, linewidth=2.4, alpha=0.95)
    lc.set_array(vals[:-1][keep])
    ax.add_collection(lc)


def main():
    parser = argparse.ArgumentParser(description="Plot cleaned droplet trajectories.")
    parser.add_argument("--dump", default="case.drag.mode.1", help="Dump file (Lammps format).")
    parser.add_argument("--color-mode", choices=["percent_loss", "ratio", "radius"], default="percent_loss")
    parser.add_argument("--out", default="Figs/trajs_clean.png", help="Output PNG path.")
    parser.add_argument("--marker-size", type=float, default=2.5, help="Scatter marker size for trajectory points.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    dump_path = base / args.dump
    if not dump_path.exists():
        raise FileNotFoundError(f"Missing dump: {dump_path}")
    data = parse_dump(dump_path)

    wall = surface(str(base / "wall.surf"), "2D")
    core = surface(str(base / "core.surf"), "2D")
    Rwall, Zwall = wall.polygon.exterior.xy
    Rcore, Zcore = core.polygon.exterior.xy

    ids = np.unique(data["id"])
    metrics = []
    mode_used = args.color_mode
    for pid in ids:
        mask = data["id"] == pid
        metric = radius_metric(data["radius"][mask], mode=mode_used)
        finite = metric[np.isfinite(metric)]
        if finite.size:
            metrics.append(finite)
    if not metrics:
        # fallback to raw radius mode if requested mode has no finite data
        fallback_metric = np.asarray(data["radius"], dtype=float)
        fallback_finite = fallback_metric[np.isfinite(fallback_metric)]
        if fallback_finite.size == 0:
            raise RuntimeError("No valid radius data available for coloring.")
        metrics = [fallback_finite]
        mode_used = "radius"
        print("Warning: selected color_mode produced no finite data; falling back to 'radius'.")
    metrics = np.concatenate(metrics)
    vmin = float(np.nanmin(metrics))
    vmax = float(np.nanmax(metrics))
    if args.color_mode == "percent_loss":
        vmin = 0.0
        vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = "viridis"
    for pid in ids:
        mask = data["id"] == pid
        if not np.any(mask):
            continue
        metric = radius_metric(data["radius"][mask], mode=mode_used)
        add_traj_colored(
            ax,
            data["x"][mask],
            data["y"][mask],
            metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if args.marker_size > 0:
            pts = np.asarray([data["x"][mask], data["y"][mask]], dtype=float).T
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=args.marker_size**4,
                c="k",
                alpha=0.3,
                linewidths=0,
                rasterized=True,
                zorder=5,
            )

    ax.plot(Rwall, Zwall, color="k", lw=3.0, label="Wall")
    ax.plot(Rcore, Zcore, color="forestgreen", lw=3.0, label="Core")

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    label_map = {
        "percent_loss": "Radius loss (%)",
        "ratio": "r / r0",
        "radius": "Radius (m)",
    }
    cbar.set_label(label_map[mode_used])

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Droplet trajectories (cleaned)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
    exit()


#    data = parse_dump("case.drag.mode.1")

    wall = surface("wall.surf", "2D")
    core = surface("core.surf", "2D")
    Rwall, Zwall = wall.polygon.exterior.xy
    Rcore, Zcore = core.polygon.exterior.xy

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = "viridis"
    ax.plot(Rwall, Zwall, color="k", lw=3.0, label="Wall")
    ax.plot(Rcore, Zcore, color="forestgreen", lw=3.0, label="Core")

#    ax.plot(2.48363, -1.96148, 'ks', ms=10) # 0 -7.0318 -0.0429335 -1.28749 3.49502e-05 0 0
#    ax.plot(2.79735, 0.567552, 'o', ms=4) # 0 1.19157 12.2444 -1.75597 3.49502e-05 0 0
#    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#    sm.set_array([])
#    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
#    label_map = {
#        "percent_loss": "Radius loss (%)",
#        "ratio": "r / r0",
#        "radius": "Radius (m)",
#    }
#    cbar.set_label(label_map[mode_used])

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Droplet trajectories (cleaned)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

#    out_path = base / args.out
#    out_path.parent.mkdir(parents=True, exist_ok=True)
#    fig.savefig(out_path, dpi=300, bbox_inches="tight")
#    print(f"Saved {out_path}")
    plt.show()
    
