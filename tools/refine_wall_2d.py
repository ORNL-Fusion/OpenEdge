#!/usr/bin/env python3
"""Refine a 2D SPARTA wall.surf by bisecting long line segments.

Splits any line whose length exceeds --max-edge into N equal sub-segments
along the original straight segment. The wall shape is preserved exactly:
new vertices are inserted on the original line, so the geometry never bends.

Optionally restrict refinement to a region (segment midpoint inside box) or
to a range of original line IDs.

The output mirrors the SPARTA 2D surf format:

    surface geometry

    <Np> points
    <Nl> lines

    Points

    <id> <x> <y>
    ...

    Lines

    <id> <p1> <p2>
    ...

A JSON map of {old_line_id: [new_first, new_last]} is written next to
--map for convenience when updating SPARTA `group surf id` commands.

Example
-------

    # Bisect any segment longer than 1cm in the lower outer divertor strike box
    python tools/refine_wall_2d.py \\
        --input  examples/wip/test_diii_d_oedge_jm/input/wall.surf \\
        --output examples/wip/test_diii_d_oedge_jm/input/wall_refined.surf \\
        --max-edge 0.01 \\
        --region -1.30 -1.10  1.35 1.85 \\
        --map    examples/wip/test_diii_d_oedge_jm/input/wall_id_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_surf(path: Path):
    text = path.read_text().splitlines()
    npts = nlines = None
    section = None
    points: Dict[int, Tuple[float, float]] = {}
    lines: List[Tuple[int, int, int]] = []
    for raw in text:
        s = raw.strip()
        if not s:
            continue
        if section is None and s.lower().endswith("points"):
            tok = s.split()
            if len(tok) == 2 and tok[0].isdigit():
                npts = int(tok[0])
                continue
        if section is None and s.lower().endswith("lines"):
            tok = s.split()
            if len(tok) == 2 and tok[0].isdigit():
                nlines = int(tok[0])
                continue
        if s.lower() == "points":
            section = "points"
            continue
        if s.lower() == "lines":
            section = "lines"
            continue
        tok = s.split()
        if section == "points" and len(tok) >= 3:
            points[int(tok[0])] = (float(tok[1]), float(tok[2]))
        elif section == "lines" and len(tok) >= 3:
            lines.append((int(tok[0]), int(tok[1]), int(tok[2])))
        # else: ignore (header, type lines, etc.)

    if npts is None or nlines is None:
        sys.exit("could not parse <N> points / <N> lines header")
    if len(points) != npts or len(lines) != nlines:
        sys.exit(f"count mismatch: header {npts}/{nlines} vs read "
                 f"{len(points)}/{len(lines)}")
    return points, lines


def seglen(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def in_region(a, b, region) -> bool:
    if region is None:
        return True
    xmin, xmax, ymin, ymax = region
    mx = 0.5 * (a[0] + b[0])
    my = 0.5 * (a[1] + b[1])
    return xmin <= mx <= xmax and ymin <= my <= ymax


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-edge", required=True, type=float,
                    help="target max segment length (units of wall.surf, usually m)")
    ap.add_argument("--region", nargs=4, type=float,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="only refine segments whose midpoint is inside this box "
                         "(x = first wall coord, usually Z; y = second, usually R)")
    ap.add_argument("--ids", nargs=2, type=int, metavar=("LO", "HI"),
                    help="only refine original line IDs in [LO,HI] (inclusive)")
    ap.add_argument("--map", type=Path,
                    help="optional JSON: {old_line_id: [new_first, new_last]}")
    args = ap.parse_args()

    points, lines_in = parse_surf(args.input)

    # Copy original points first; their new IDs match insertion order.
    new_points: List[Tuple[float, float]] = []
    pid_old_to_new: Dict[int, int] = {}
    for old_id in sorted(points.keys()):
        new_points.append(points[old_id])
        pid_old_to_new[old_id] = len(new_points)

    new_lines: List[Tuple[int, int]] = []
    id_map: Dict[int, List[int]] = {}
    n_split = 0

    for old_id, p1, p2 in lines_in:
        a = points[p1]
        b = points[p2]
        L = seglen(a, b)

        select = True
        if args.ids is not None and not (args.ids[0] <= old_id <= args.ids[1]):
            select = False
        if select and not in_region(a, b, args.region):
            select = False

        if not select or L <= args.max_edge:
            new_lines.append((pid_old_to_new[p1], pid_old_to_new[p2]))
            id_map[old_id] = [len(new_lines), len(new_lines)]
            continue

        n = max(2, math.ceil(L / args.max_edge))
        first_new_line = len(new_lines) + 1
        prev_pid = pid_old_to_new[p1]
        for k in range(1, n):
            t = k / n
            new_points.append((a[0] + t * (b[0] - a[0]),
                               a[1] + t * (b[1] - a[1])))
            cur_pid = len(new_points)
            new_lines.append((prev_pid, cur_pid))
            prev_pid = cur_pid
        new_lines.append((prev_pid, pid_old_to_new[p2]))
        id_map[old_id] = [first_new_line, len(new_lines)]
        n_split += 1

    out: List[str] = []
    out.append("surface geometry")
    out.append("")
    out.append(f"{len(new_points)} points")
    out.append(f"{len(new_lines)} lines")
    out.append("")
    out.append("Points")
    out.append("")
    for i, (x, y) in enumerate(new_points, start=1):
        out.append(f"{i} {x:.10g} {y:.10g}")
    out.append("")
    out.append("Lines")
    out.append("")
    for i, (p1, p2) in enumerate(new_lines, start=1):
        out.append(f"{i} {p1} {p2}")
    out.append("")
    args.output.write_text("\n".join(out))

    if args.map is not None:
        args.map.write_text(json.dumps(id_map, indent=2))

    print(f"in:    {len(points)} points, {len(lines_in)} lines")
    print(f"out:   {len(new_points)} points, {len(new_lines)} lines")
    print(f"split: {n_split} segments refined "
          f"(max_edge = {args.max_edge}"
          f"{', region = ' + str(tuple(args.region)) if args.region else ''}"
          f"{', ids = ' + str(tuple(args.ids)) if args.ids else ''})")


if __name__ == "__main__":
    main()
