#!/usr/bin/env python3
"""Refine a 2D SPARTA/OpenEdge surface polyline by subdividing segments.

This keeps the same geometry and only inserts extra vertices along each edge.

Examples:
  python3 analysis/refine_sparta_surface.py \
      --input input/wall.txt \
      --output input/wall_refined.txt \
      --max-segment-length 0.002

  python3 analysis/refine_sparta_surface.py \
      --input input/wall.txt \
      --output input/wall_x4.txt \
      --subdivide 4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


Point2 = tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input 2D SPARTA/OpenEdge surf file")
    parser.add_argument("--output", type=Path, required=True, help="Output refined surf file")
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=None,
        help="Target maximum segment length [m]. Longer edges are split automatically.",
    )
    parser.add_argument(
        "--subdivide",
        type=int,
        default=None,
        help="Uniformly split every original segment into this many pieces.",
    )
    return parser.parse_args()


def read_sparta_polyline(path: Path) -> list[Point2]:
    points: dict[int, Point2] = {}
    edges: dict[int, int] = {}
    section: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low == "points":
            section = "points"
            continue
        if low == "lines":
            section = "lines"
            continue
        if low.startswith("surface"):
            continue

        cols = line.split()
        if section == "points" and len(cols) >= 3 and cols[0].isdigit():
            pid = int(cols[0])
            points[pid] = (float(cols[1]), float(cols[2]))
        elif section == "lines" and len(cols) >= 3 and cols[0].isdigit():
            edges[int(cols[1])] = int(cols[2])

    if not points or not edges:
        raise RuntimeError(f"Could not parse Points/Lines from {path}")

    start = min(edges)
    ordered_ids = [start]
    current = start
    visited: set[int] = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        nxt = edges.get(current)
        if nxt is None:
            raise RuntimeError(f"Broken edge chain in {path}: missing successor for point {current}")
        if nxt == start:
            break
        ordered_ids.append(nxt)
        current = nxt

    if len(ordered_ids) < 3:
        raise RuntimeError(f"Need at least 3 ordered vertices in {path}, found {len(ordered_ids)}")

    return [points[idx] for idx in ordered_ids]


def subdivide_polyline(points: list[Point2], max_segment_length: float | None, subdivide: int | None) -> list[Point2]:
    refined: list[Point2] = []
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        seg_len = math.hypot(x2 - x1, y2 - y1)

        if subdivide is not None:
            nsub = max(1, subdivide)
        elif max_segment_length is not None:
            nsub = max(1, int(math.ceil(seg_len / max_segment_length)))
        else:
            nsub = 1

        for k in range(nsub):
            t = k / nsub
            refined.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

    return refined


def write_sparta_polyline(path: Path, points: list[Point2]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    with path.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i, (x, y) in enumerate(points, start=1):
            f.write(f"{i} {x:.12g} {y:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(1, n + 1):
            j = i + 1 if i < n else 1
            f.write(f"{i} {i} {j}\n")


def main() -> None:
    args = parse_args()
    if (args.max_segment_length is None) == (args.subdivide is None):
        raise SystemExit("Specify exactly one of --max-segment-length or --subdivide")
    if args.max_segment_length is not None and args.max_segment_length <= 0.0:
        raise SystemExit("--max-segment-length must be > 0")
    if args.subdivide is not None and args.subdivide < 1:
        raise SystemExit("--subdivide must be >= 1")

    poly = read_sparta_polyline(args.input)
    refined = subdivide_polyline(poly, args.max_segment_length, args.subdivide)
    write_sparta_polyline(args.output, refined)

    print(f"Read {args.input}: {len(poly)} vertices")
    print(f"Wrote {args.output}: {len(refined)} vertices")


if __name__ == "__main__":
    main()
