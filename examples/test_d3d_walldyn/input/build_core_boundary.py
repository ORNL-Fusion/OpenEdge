#!/usr/bin/env python3
"""Revolve a 2D flux-surface polygon (R,Z) into a closed 3D toroidal wedge.

Reads a SPARTA-format 2D surface file (Points + Lines) and produces a
watertight triangulated 3D closed surface for use as a core boundary
in OpenEdge/SPARTA.

The surface is closed with solid caps at phi_min and phi_max (not periodic).
All normals point outward by default; use --invert to flip them inward
(for use with SPARTA's `invert` keyword on read_surf).

Outputs:
- SPARTA 3D triangular surface (.surf)
- OBJ mesh for visualization
- Metadata JSON with triangle group ranges
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Path to 2D SPARTA surface file (R,Z polygon)")
    p.add_argument("--phi-min-deg", type=float, default=0.0,
                   help="Wedge start toroidal angle [deg]")
    p.add_argument("--phi-max-deg", type=float, default=30.0,
                   help="Wedge end toroidal angle [deg]")
    p.add_argument("--prefix", type=Path, default=Path("core_boundary"),
                   help="Output prefix path")
    p.add_argument("--min-edge", type=float, default=0.01,
                   help="Minimum edge length [m]; closer points are merged")
    return p.parse_args()


def read_sparta_2d(path: Path) -> List[Point2]:
    """Read a SPARTA 2D surface file and return the ordered R,Z points."""
    points = {}
    lines_section = []
    section = None

    with path.open("r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s == "Points":
                section = "points"
                continue
            elif s == "Lines":
                section = "lines"
                continue
            elif "points" in s and section is None:
                continue
            elif "lines" in s and section is None:
                continue
            elif "Geometry" in s or "surface" in s.lower():
                continue

            cols = s.split()
            if section == "points" and len(cols) >= 3:
                pid = int(cols[0])
                r, z = float(cols[1]), float(cols[2])
                points[pid] = (r, z)
            elif section == "lines" and len(cols) >= 3:
                v1, v2 = int(cols[1]), int(cols[2])
                lines_section.append((v1, v2))

    if not lines_section:
        return [points[i] for i in sorted(points.keys())]

    adj = {}
    for v1, v2 in lines_section:
        adj.setdefault(v1, []).append(v2)
        adj.setdefault(v2, []).append(v1)

    start = lines_section[0][0]
    ordered = [start]
    visited = {start}
    cur = start
    while True:
        found = False
        for nxt in adj.get(cur, []):
            if nxt not in visited:
                ordered.append(nxt)
                visited.add(nxt)
                cur = nxt
                found = True
                break
        if not found:
            break

    return [points[i] for i in ordered]


def decimate_polygon(rz: List[Point2], min_edge: float) -> List[Point2]:
    """Remove points that are closer than min_edge to their predecessor."""
    if len(rz) < 4:
        return rz
    result = [rz[0]]
    for i in range(1, len(rz)):
        dr = rz[i][0] - result[-1][0]
        dz = rz[i][1] - result[-1][1]
        if math.hypot(dr, dz) >= min_edge:
            result.append(rz[i])
    if len(result) > 2:
        dr = result[-1][0] - result[0][0]
        dz = result[-1][1] - result[0][1]
        if math.hypot(dr, dz) < min_edge:
            result.pop()
    return result


def signed_area_rz(poly: List[Point2]) -> float:
    n = len(poly)
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _cross2d(o: Point2, a: Point2, b: Point2) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool:
    d1 = _cross2d(p, a, b)
    d2 = _cross2d(p, b, c)
    d3 = _cross2d(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(poly: List[Point2]) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon (CCW winding)."""
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    while len(idx) > 3:
        ear_found = False
        m = len(idx)
        for i in range(m):
            prev_i = idx[(i - 1) % m]
            curr_i = idx[i]
            next_i = idx[(i + 1) % m]
            if _cross2d(poly[prev_i], poly[curr_i], poly[next_i]) <= 0:
                continue
            is_ear = True
            for j in range(m):
                if j == (i - 1) % m or j == i or j == (i + 1) % m:
                    continue
                if _point_in_triangle(poly[idx[j]],
                                      poly[prev_i], poly[curr_i], poly[next_i]):
                    is_ear = False
                    break
            if is_ear:
                tris.append((prev_i, curr_i, next_i))
                idx.pop(i)
                ear_found = True
                break
        if not ear_found:
            raise RuntimeError(
                f"Ear clipping failed with {len(idx)} vertices remaining")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def rz_to_xyz(rz: List[Point2], phi_rad: float) -> List[Point3]:
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    return [(r * c, r * s, z) for r, z in rz]


def build_watertight_wedge(n: int, rz: List[Point2], xyz: List[Point3],
                           phi0: float, phi1: float) -> List[Tri]:
    """Build a fully watertight closed wedge surface.

    Vertex layout:
      [0..n-1]   = phi_min ring
      [n..2n-1]  = phi_max ring

    Wall quads connect the two rings. Caps close each end.
    All normals point outward (away from the enclosed volume).

    For the wall, "outward" means away from the toroidal axis (radially out
    from the enclosed core volume). For the caps, outward means in the
    -phi direction for phi_min cap and +phi direction for phi_max cap.

    The key to watertight: wall boundary edges at phi_min are (i, (i+1)%n)
    going one direction. The cap must use the REVERSE edge ((i+1)%n, i) to
    match. Same logic at phi_max.
    """
    tris: List[Tri] = []

    # --- Wall triangles ---
    # Each quad [i, j, n+j, n+i] is split into 2 triangles.
    # Wall edges at phi_min boundary: edge (i -> j) where j=(i+1)%n
    # Wall edges at phi_max boundary: edge (n+j -> n+i)
    # So caps must use reverse edges to match.
    for i in range(n):
        j = (i + 1) % n
        # Two triangles per quad, normals pointing outward
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))

    # --- phi_min cap (vertices 0..n-1) ---
    # Wall uses edge (i -> j) along phi_min boundary.
    # Cap must use edge (j -> i) = reversed, so cap winding is CW when
    # viewed from outside (i.e., looking in the +phi direction).
    # Ear-clip gives CCW triangles in R-Z plane. When viewed from -phi
    # direction (outside the phi_min cap), CCW in R-Z corresponds to the
    # cap normal pointing in -phi direction (outward). This means the
    # ear-clip winding (a,b,c) gives outward normals for phi_min cap.
    # But we need the boundary edges to be reversed relative to wall.
    # Wall edge at phi_min: (i, i+1). Cap edge must be (i+1, i).
    # Ear-clip CCW gives edges going (a->b->c->a). The boundary edges
    # of the cap polygon go 0->1->2->...->n-1->0 (CCW).
    # This means cap boundary edge (i, i+1) MATCHES wall edge (i, i+1).
    # That's the SAME direction = DUPLICATE, not watertight!
    # Fix: reverse the cap winding to CW = flip each triangle.
    rz_tris = ear_clip_triangulate(rz)

    for a, b, c in rz_tris:
        # Flipped winding for phi_min cap (CW when viewed from R-Z plane)
        # This makes boundary edges go (i+1 -> i), matching wall's need
        tris.append((c, b, a))  # reversed

    # --- phi_max cap (vertices n..2n-1) ---
    # Wall uses edge (n+j -> n+i) along phi_max boundary, i.e., (n+i+1 -> n+i).
    # Cap must use edge (n+i -> n+i+1) = reversed.
    # Ear-clip CCW in R-Z plane: boundary goes 0->1->2...
    # With offset n, boundary edges go (n+0 -> n+1 -> ...).
    # That matches what we need: (n+i -> n+i+1).
    # So phi_max cap uses the ear-clip winding directly (not flipped).
    for a, b, c in rz_tris:
        tris.append((n + a, n + b, n + c))  # original winding

    return tris


def verify_watertight(tris: List[Tri]) -> bool:
    """Check that every directed edge (a,b) has exactly one reverse (b,a)."""
    edges = Counter()
    for a, b, c in tris:
        edges[(a, b)] += 1
        edges[(b, c)] += 1
        edges[(c, a)] += 1

    duplicates = sum(1 for v in edges.values() if v > 1)
    unmatched = sum(1 for (a, b) in edges if edges.get((b, a), 0) == 0)

    if duplicates > 0 or unmatched > 0:
        print(f"  WARNING: {duplicates} duplicate edges, {unmatched} unmatched edges")
        return False
    print(f"  Watertight check passed: {len(edges)} directed edges, all matched")
    return True


def write_sparta_3d(path: Path, xyz: List[Point3], tris: List[Tri]) -> None:
    with path.open("w") as f:
        f.write("surface geometry\n\n")
        f.write(f"{len(xyz)} points\n")
        f.write(f"{len(tris)} triangles\n\n")
        f.write("Points\n\n")
        for i, p in enumerate(xyz, start=1):
            f.write(f"{i} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        f.write("\nTriangles\n\n")
        for i, t in enumerate(tris, start=1):
            f.write(f"{i} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def write_obj(path: Path, xyz: List[Point3], tris: List[Tri]) -> None:
    with path.open("w") as f:
        f.write("# core boundary wedge mesh\n")
        for p in xyz:
            f.write(f"v {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def main() -> None:
    args = parse_args()

    rz = read_sparta_2d(args.input)
    print(f"Read {len(rz)} points from {args.input}")

    rz = decimate_polygon(rz, args.min_edge)
    n = len(rz)
    print(f"After decimation (min_edge={args.min_edge}): {n} points")

    # Ensure CCW winding in R-Z plane
    if signed_area_rz(rz) < 0.0:
        rz = list(reversed(rz))
        print("  Reversed winding to CCW")

    phi0 = math.radians(args.phi_min_deg)
    phi1 = math.radians(args.phi_max_deg)

    xyz0 = rz_to_xyz(rz, phi0)
    xyz1 = rz_to_xyz(rz, phi1)
    xyz = xyz0 + xyz1

    all_tris = build_watertight_wedge(n, rz, xyz, phi0, phi1)
    n_wall = 2 * n
    n_cap = n - 2  # ear-clip of n-gon = n-2 triangles

    print(f"Wedge: {args.phi_min_deg:.1f} -> {args.phi_max_deg:.1f} deg")
    print(f"Wall triangles:       {n_wall}")
    print(f"Cap triangles (each): {n_cap}")
    print(f"Total triangles:      {len(all_tris)}")

    # Verify watertight before writing
    verify_watertight(all_tris)

    prefix = args.prefix
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)

    out_surf = Path(f"{prefix}_wedge_capped.surf")
    out_obj = Path(f"{prefix}_wedge_capped.obj")
    out_meta = Path(f"{prefix}_meta.json")

    write_sparta_3d(out_surf, xyz, all_tris)
    write_obj(out_obj, xyz, all_tris)

    meta = {
        "input_file": str(args.input),
        "phi_min_deg": args.phi_min_deg,
        "phi_max_deg": args.phi_max_deg,
        "dphi_deg": args.phi_max_deg - args.phi_min_deg,
        "n_rz_vertices": n,
        "min_edge": args.min_edge,
        "n_wall_triangles": n_wall,
        "n_cap_triangles_each": n_cap,
        "n_total_triangles": len(all_tris),
        "triangle_groups": {
            "core_wall": {"first": 1, "last": n_wall},
            "cap_phi_min": {"first": n_wall + 1, "last": n_wall + n_cap},
            "cap_phi_max": {"first": n_wall + n_cap + 1, "last": n_wall + 2 * n_cap},
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print()
    print("Add to your SPARTA input script:")
    print(f"  read_surf          input/{out_surf.name} group core")
    print(f"  surf_collide       CORE specular")
    print(f"  surf_modify        core collide CORE react none")
    print()
    print(f"Wrote: {out_surf}")
    print(f"Wrote: {out_obj}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
