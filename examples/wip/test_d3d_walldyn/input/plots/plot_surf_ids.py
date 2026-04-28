#!/usr/bin/env python3
"""Plot a SPARTA/OpenEdge 3D .surf file and overlay surface IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "surf_file",
        nargs="?",
        default="input/geometry/d3d_refined_wedge_capped.surf",
        help="Path to SPARTA/OpenEdge 3D surface file",
    )
    parser.add_argument("--alpha", type=float, default=0.18, help="Surface transparency")
    parser.add_argument(
        "--face-color",
        default="#8fb7d9",
        help="Triangle face color for the base mesh",
    )
    parser.add_argument(
        "--edge-color",
        default="black",
        help="Triangle edge color",
    )
    parser.add_argument(
        "--label-step",
        type=int,
        default=1,
        help="Label every Nth surface id (default: every surface)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=7.0,
        help="Surface id label font size",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=22.0,
        help="3D camera elevation",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D camera azimuth",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional PNG output path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="PNG resolution when --out is used",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save only; do not open an interactive window",
    )
    return parser.parse_args()


def read_surface(path: Path):
    lines = path.read_text().splitlines()
    try:
        points_idx = lines.index("Points")
        tris_idx = lines.index("Triangles")
    except ValueError as exc:
        raise ValueError(f"Could not find Points/Triangles sections in {path}") from exc

    n_points = None
    n_tris = None
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "points":
            n_points = int(parts[0])
        if len(parts) == 2 and parts[1] == "triangles":
            n_tris = int(parts[0])
    if n_points is None or n_tris is None:
        raise ValueError(f"Could not determine point/triangle counts in {path}")

    point_lines = [line for line in lines[points_idx + 1 : tris_idx] if line.strip()]
    tri_lines = [line for line in lines[tris_idx + 1 :] if line.strip()]
    point_lines = point_lines[:n_points]
    tri_lines = tri_lines[:n_tris]

    points = np.array([[float(tok) for tok in line.split()[1:4]] for line in point_lines], dtype=float)
    tri_table = np.array([[int(tok) for tok in line.split()] for line in tri_lines], dtype=int)
    surf_ids = tri_table[:, 0]
    triangles = tri_table[:, 1:4] - 1
    verts = points[triangles]
    centroids = verts.mean(axis=1)
    return surf_ids, verts, centroids


def main() -> int:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    surf_path = Path(args.surf_file)
    if not surf_path.is_absolute():
        surf_path = Path.cwd() / surf_path
    if not surf_path.exists():
        raise SystemExit(f"Surface file not found: {surf_path}")

    surf_ids, verts, centroids = read_surface(surf_path)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        verts,
        facecolors=args.face_color,
        edgecolors=args.edge_color,
        linewidths=0.35,
        alpha=args.alpha,
    )
    ax.add_collection3d(mesh)

    step = max(args.label_step, 1)
    for idx in range(0, len(surf_ids), step):
        cx, cy, cz = centroids[idx]
        ax.text(cx, cy, cz, str(surf_ids[idx]), fontsize=args.font_size, color="darkred")

    xyz = verts.reshape(-1, 3)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0e-12)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(spans)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{surf_path.name} with surface IDs")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
