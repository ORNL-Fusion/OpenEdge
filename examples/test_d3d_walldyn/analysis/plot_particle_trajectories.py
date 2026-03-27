#!/usr/bin/env python3
"""Plot a few particle trajectories directly from an OpenEdge particle dump."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_particle_dump(path):
    timesteps = []
    ids = []
    types = []
    x = []
    y = []
    z = []

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    natoms = 0
    timestep = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            natoms = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: ATOMS"):
            for _ in range(natoms):
                cols = lines[i + 1].split()
                ids.append(int(cols[0]))
                types.append(int(cols[1]))
                x.append(float(cols[2]))
                y.append(float(cols[3]))
                z.append(float(cols[4]))
                timesteps.append(timestep)
                i += 1
            i += 1
        else:
            i += 1

    return {
        "timestep": np.asarray(timesteps, dtype=int),
        "id": np.asarray(ids, dtype=int),
        "type": np.asarray(types, dtype=int),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
    }


def build_tracks(data, max_step=None, require_contiguous=False):
    order = np.lexsort((data["timestep"], data["id"]))
    ts = data["timestep"][order]
    pid = data["id"][order]
    ptype = data["type"][order]
    x = data["x"][order]
    y = data["y"][order]
    z = data["z"][order]

    tracks = []
    by_id = defaultdict(list)
    for i, p in enumerate(pid):
        by_id[int(p)].append(i)

    for particle_id, idxs in by_id.items():
        current = []
        prev_xyz = None
        prev_t = None
        for idx in idxs:
            xyz = np.array([x[idx], y[idx], z[idx]], dtype=float)
            split = False
            if prev_xyz is not None and max_step is not None:
                if np.linalg.norm(xyz - prev_xyz) > max_step:
                    split = True
            if prev_t is not None and require_contiguous and ts[idx] != prev_t + 1:
                split = True

            if split and len(current) >= 2:
                tracks.append(current)
                current = []
            elif split:
                current = []

            current.append(
                {
                    "id": particle_id,
                    "type": int(ptype[idx]),
                    "timestep": int(ts[idx]),
                    "xyz": xyz,
                }
            )
            prev_xyz = xyz
            prev_t = ts[idx]

        if len(current) >= 2:
            tracks.append(current)

    tracks.sort(key=len, reverse=True)
    return tracks


def plot_tracks(tracks, output_path, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    for i, track in enumerate(tracks):
        pts = np.array([node["xyz"] for node in track])
        pid = track[0]["id"]
        ptype = track[0]["type"]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            lw=2.0,
            color=cmap(i % 10),
            label=f"id={pid}, type={ptype}, n={len(track)}",
        )
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=cmap(i % 10), s=18)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def trajectory_metrics(track):
    pts = np.array([node["xyz"] for node in track], dtype=float)
    ts = np.array([node["timestep"] for node in track], dtype=int)

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    path_length = float(seg_len.sum())
    net_displacement = float(np.linalg.norm(pts[-1] - pts[0]))

    dt_steps = np.diff(ts)
    valid = dt_steps > 0
    if np.any(valid):
        speeds = seg_len[valid] / dt_steps[valid]
        mean_speed = float(np.mean(speeds))
    else:
        mean_speed = 0.0

    return path_length, net_displacement, mean_speed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statefile", nargs="?", default="output/particles")
    parser.add_argument("--output", default="output/particle_trajectories.png")
    parser.add_argument("--ntraj", type=int, default=3, help="Number of trajectories to plot")
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Split tracks when consecutive samples jump farther than this distance",
    )
    parser.add_argument(
        "--contiguous",
        action="store_true",
        help="Split tracks when timestep samples are not contiguous",
    )
    args = parser.parse_args()

    data = parse_particle_dump(args.statefile)
    tracks = build_tracks(data, max_step=args.max_step, require_contiguous=args.contiguous)
    if not tracks:
        raise RuntimeError("No trajectory segments with at least 2 points were found")

    selected = tracks[: args.ntraj]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_tracks(selected, out, f"Particle trajectories from {Path(args.statefile).name}")

    print(f"Wrote {out}")
    for i, track in enumerate(selected, start=1):
        path_length, net_disp, mean_speed_per_step = trajectory_metrics(track)
        print(
            f"  track {i}: id={track[0]['id']} type={track[0]['type']} "
            f"points={len(track)} timesteps={track[0]['timestep']}..{track[-1]['timestep']} "
            f"path={path_length:.6e} m net={net_disp:.6e} m "
            f"net/path={(net_disp / path_length if path_length > 0 else 0.0):.3f} "
            f"mean_step_speed={mean_speed_per_step:.6e} m/step"
        )


if __name__ == "__main__":
    main()
