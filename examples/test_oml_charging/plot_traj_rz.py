"""Plot particle trajectories in R-Z from a SPARTA/OpenEdge dump file."""

import argparse
import os
from collections import defaultdict

import numpy as np


def parse_dump(filename: str):
    """Parse ITEM-style dump with dynamic ATOMS columns."""
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = 0
    ts = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            ts = int(lines[i + 1])
            i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1])
            i += 2
        elif s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            c = {name: idx for idx, name in enumerate(cols)}
            for key in ["id", "x", "y"]:
                if key not in c:
                    raise RuntimeError(f"Missing required column '{key}' in dump header: {s}")
            for _ in range(n):
                p = lines[i + 1].split()
                rows.append((ts, int(p[c["id"]]), float(p[c["x"]]), float(p[c["y"]])))
                i += 1
            i += 1
        else:
            i += 1

    if not rows:
        raise RuntimeError(f"No particle records found in {filename}")

    arr = np.asarray(rows, dtype=float)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2], arr[:, 3]


def maybe_overlay_surfaces(ax):
    if not os.path.exists("utils.py"):
        return
    try:
        from utils import surface
    except Exception:
        return

    for surf_file, color, lw, label in [
        ("wall.surf", "k", 2.0, "Wall"),
        ("core.surf", "#2f7d32", 2.0, "Core"),
    ]:
        if os.path.exists(surf_file):
            S = surface(surf_file, "2D")
            x, y = S.polygon.exterior.xy
            ax.plot(x, y, "-", color=color, lw=lw, label=label)


def main():
    p = argparse.ArgumentParser(description="Plot R-Z trajectories from dump")
    p.add_argument("--dump", default="case.out", help="Input dump file (default: case.out)")
    p.add_argument("--out", default="Figs/traj_rz.png", help="Output PNG")
    p.add_argument("--max-ids", type=int, default=20, help="Max number of particle IDs to plot")
    args = p.parse_args()

    ts, pid, R, Z = parse_dump(args.dump)

    by_id = defaultdict(list)
    for i in range(pid.size):
        by_id[int(pid[i])].append(i)

    ids = sorted(by_id.keys())[: max(1, args.max_ids)]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    maybe_overlay_surfaces(ax)

    for k, ip in enumerate(ids):
        idx = np.asarray(by_id[ip], dtype=int)
        order = np.argsort(ts[idx])
        ii = idx[order]
        ax.plot(R[ii], Z[ii], "-", lw=1.2, alpha=0.9, label=f"id={ip}" if len(ids) <= 10 else None)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"Trajectories in R-Z ({os.path.basename(args.dump)})")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    if len(ids) <= 10:
        ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
