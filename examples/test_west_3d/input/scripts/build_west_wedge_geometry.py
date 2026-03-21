#!/usr/bin/env python3
"""Build a WEST toroidal wedge surface from 2D SPARTA wall file.

Reads a 2D SPARTA surf file (Points + Lines in R-Z plane) and produces
a 3D toroidal wedge with ear-clipped caps.

Outputs:
- 3D SPARTA/OpenEdge triangular surface for a toroidal wedge [phi_min, phi_max]
- OBJ mesh for quick visualization
- ASCII STL mesh
- JSON metadata with triangle group indices
- Watertightness check (edge-matching, inspired by SPARTA tools/stl2surf.py)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall", type=Path, required=True,
                    help="Path to 2D SPARTA wall file (Points + Lines)")
    p.add_argument("--phi-min-deg", type=float, default=0.0,
                    help="Wedge start toroidal angle [deg]")
    p.add_argument("--phi-max-deg", type=float, default=30.0,
                    help="Wedge end toroidal angle [deg]")
    p.add_argument("--simplify", type=float, default=0.002,
                    help="Douglas-Peucker simplification tolerance [m] (0 to disable)")
    p.add_argument("--prefix", type=Path, default=Path("west"),
                    help="Output prefix path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Read 2D SPARTA surf file
# ---------------------------------------------------------------------------

def read_sparta_2d_wall(path: Path) -> List[Point2]:
    """Read a SPARTA 2D surf file and return the closed polyline as (R, Z) list.

    Assumes the file has Points and Lines sections, and Lines form a single
    closed loop in order (line i connects point i to point i+1, last wraps).
    """
    lines_text = path.read_text(encoding="utf-8").splitlines()

    n_points = 0
    n_lines = 0
    points_raw = {}
    edges = []

    section = None
    for ln in lines_text:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("surface"):
            continue
        if "points" in s and not s.startswith("Points"):
            n_points = int(s.split()[0])
            continue
        if "lines" in s and not s.startswith("Lines"):
            n_lines = int(s.split()[0])
            continue
        if s == "Points":
            section = "points"
            continue
        if s == "Lines":
            section = "lines"
            continue

        cols = s.split()
        if section == "points" and len(cols) >= 3:
            idx = int(cols[0])
            r, z = float(cols[1]), float(cols[2])
            points_raw[idx] = (r, z)
        elif section == "lines" and len(cols) >= 3:
            i, j = int(cols[1]), int(cols[2])
            edges.append((i, j))

    if not points_raw or not edges:
        raise RuntimeError(f"Could not parse Points/Lines from {path}")

    # Reconstruct ordered loop by walking edges
    adj = {}
    for i, j in edges:
        adj[i] = j

    start = edges[0][0]
    loop = [start]
    cur = adj[start]
    while cur != start:
        loop.append(cur)
        cur = adj[cur]

    rz = [points_raw[i] for i in loop]
    return rz


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _perpendicular_dist(p: Point2, a: Point2, b: Point2) -> float:
    """Perpendicular distance from point p to line segment a-b."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dz)
    if length < 1e-30:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dz) / length


def douglas_peucker(pts: List[Point2], tol: float) -> List[Point2]:
    """Simplify a polyline using the Douglas-Peucker algorithm."""
    if len(pts) <= 2:
        return pts

    # Find the point farthest from the line between first and last
    d_max = 0.0
    idx = 0
    for i in range(1, len(pts) - 1):
        d = _perpendicular_dist(pts[i], pts[0], pts[-1])
        if d > d_max:
            d_max = d
            idx = i

    if d_max > tol:
        left = douglas_peucker(pts[:idx + 1], tol)
        right = douglas_peucker(pts[idx:], tol)
        return left[:-1] + right
    else:
        return [pts[0], pts[-1]]


def simplify_closed_polygon(pts: List[Point2], tol: float) -> List[Point2]:
    """Apply Douglas-Peucker to a closed polygon."""
    if tol <= 0 or len(pts) < 4:
        return pts
    # Open the polygon, simplify, close again
    extended = pts + [pts[0]]
    simplified = douglas_peucker(extended, tol)
    # Remove the closing duplicate
    if len(simplified) > 1:
        simplified = simplified[:-1]
    return simplified


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
    """Ear-clipping triangulation of a simple CCW polygon."""
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
                f"Ear clipping failed with {len(idx)} vertices remaining "
                f"(degenerate polygon?)")

    tris.append((idx[0], idx[1], idx[2]))
    return tris


# ---------------------------------------------------------------------------
# 3D wedge construction
# ---------------------------------------------------------------------------

def rz_to_xyz(rz: List[Point2], phi_rad: float) -> List[Point3]:
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    return [(r * c, r * s, z) for r, z in rz]


def build_wedge_wall_tris(n: int) -> List[Tri]:
    """Lateral wall: quads between phi_min [0..n-1] and phi_max [n..2n-1]."""
    tris: List[Tri] = []
    for i in range(n):
        j = (i + 1) % n
        a, b, c, d = i, j, n + j, n + i
        tris.append((a, b, c))
        tris.append((a, c, d))
    return tris


def build_cap_tris(
    rz: List[Point2],
    xyz: List[Point3],
    offset: int,
    phi_rad: float,
    outward_sign: float,
) -> List[Tri]:
    """Triangulate a phi-face cap with correct outward normals."""
    rz_tris = ear_clip_triangulate(rz)

    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    out_dir = (-s_phi * outward_sign, c_phi * outward_sign, 0.0)

    cap_tris: List[Tri] = []
    for a, b, c in rz_tris:
        va = xyz[offset + a]
        vb = xyz[offset + b]
        vc = xyz[offset + c]
        ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
        wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
        nx = uy * wz - uz * wy
        ny = uz * wx - ux * wz
        nz = ux * wy - uy * wx
        dot = nx * out_dir[0] + ny * out_dir[1] + nz * out_dir[2]
        if dot >= 0:
            cap_tris.append((offset + a, offset + b, offset + c))
        else:
            cap_tris.append((offset + a, offset + c, offset + b))

    return cap_tris


# ---------------------------------------------------------------------------
# Watertightness check (from SPARTA tools/stl2surf.py)
# ---------------------------------------------------------------------------

def check_watertight(tris: List[Tri]) -> bool:
    """Check if a triangle mesh is watertight via edge matching."""
    ehash = {}
    dup = 0
    for tri in tris:
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            if e in ehash:
                dup += 1
            else:
                ehash[e] = 1

    unmatch = 0
    for edge in ehash:
        if (edge[1], edge[0]) not in ehash:
            unmatch += 1

    if dup or unmatch:
        print(f"WARNING: surface is NOT watertight")
        if dup:
            print(f"  Duplicate edges: {dup}")
        if unmatch:
            print(f"  Unmatched edges: {unmatch}")
        return False
    else:
        print("Watertight check: PASSED")
        return True


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_sparta_3d(path: Path, xyz: List[Point3], tris: List[Tri]) -> None:
    with path.open("w", encoding="utf-8") as f:
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
    with path.open("w", encoding="utf-8") as f:
        f.write("# WEST wedge wall mesh\n")
        for p in xyz:
            f.write(f"v {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def write_ascii_stl(path: Path, xyz: List[Point3], tris: List[Tri],
                     name: str = "west_wedge") -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in tris:
            a, b, c = xyz[tri[0]], xyz[tri[1]], xyz[tri[2]]
            ux, uy, uz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
            vx, vy, vz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
            nx = uy*vz - uz*vy
            ny = uz*vx - ux*vz
            nz = ux*vy - uy*vx
            mag = math.sqrt(nx*nx + ny*ny + nz*nz)
            if mag > 0:
                nx, ny, nz = nx/mag, ny/mag, nz/mag
            f.write(f"  facet normal {nx:.12g} {ny:.12g} {nz:.12g}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.12g} {a[1]:.12g} {a[2]:.12g}\n")
            f.write(f"      vertex {b[0]:.12g} {b[1]:.12g} {b[2]:.12g}\n")
            f.write(f"      vertex {c[0]:.12g} {c[1]:.12g} {c[2]:.12g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.phi_max_deg <= args.phi_min_deg:
        raise ValueError("phi_max_deg must be greater than phi_min_deg")

    rz = read_sparta_2d_wall(args.wall)

    # Remove near-duplicate consecutive points (degenerate edges break ear-clipping)
    cleaned: List[Point2] = [rz[0]]
    for i in range(1, len(rz)):
        dr = rz[i][0] - cleaned[-1][0]
        dz = rz[i][1] - cleaned[-1][1]
        if math.hypot(dr, dz) > 1e-6:
            cleaned.append(rz[i])
    # Check closing edge too
    dr = cleaned[-1][0] - cleaned[0][0]
    dz = cleaned[-1][1] - cleaned[0][1]
    if math.hypot(dr, dz) < 1e-6:
        cleaned.pop()
    if len(cleaned) < len(rz):
        print(f"Removed {len(rz) - len(cleaned)} near-duplicate vertices")
    rz = cleaned

    # Simplify polygon to remove nearly-collinear points
    # (prevents degenerate thin triangles that crash SPARTA's cut3d)
    if args.simplify > 0:
        n_before = len(rz)
        rz = simplify_closed_polygon(rz, args.simplify)
        print(f"Simplified: {n_before} -> {len(rz)} vertices (tol={args.simplify} m)")

    # Ensure CCW winding for ear-clipping
    if signed_area_rz(rz) < 0.0:
        rz = list(reversed(rz))

    n = len(rz)
    phi0 = math.radians(args.phi_min_deg)
    phi1 = math.radians(args.phi_max_deg)

    xyz0 = rz_to_xyz(rz, phi0)
    xyz1 = rz_to_xyz(rz, phi1)
    xyz = xyz0 + xyz1

    wall_tris = build_wedge_wall_tris(n)
    cap_min_tris = build_cap_tris(rz, xyz, 0, phi0, outward_sign=+1.0)
    cap_max_tris = build_cap_tris(rz, xyz, n, phi1, outward_sign=-1.0)
    all_tris = wall_tris + cap_min_tris + cap_max_tris

    n_wall = len(wall_tris)
    n_cap_min = len(cap_min_tris)
    n_cap_max = len(cap_max_tris)

    # Watertightness check
    check_watertight(all_tris)

    prefix = args.prefix
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)

    out_surf = Path(f"{prefix}_wedge_capped.surf")
    out_obj = Path(f"{prefix}_wedge_capped.obj")
    out_stl = Path(f"{prefix}_wedge_capped.stl")
    out_meta = Path(f"{prefix}_meta.json")

    write_sparta_3d(out_surf, xyz, all_tris)
    write_obj(out_obj, xyz, all_tris)
    write_ascii_stl(out_stl, xyz, all_tris)

    meta = {
        "wall_file": str(args.wall),
        "phi_min_deg": args.phi_min_deg,
        "phi_max_deg": args.phi_max_deg,
        "dphi_deg": args.phi_max_deg - args.phi_min_deg,
        "n_rz_vertices": n,
        "n_wall_triangles": n_wall,
        "n_cap_min_triangles": n_cap_min,
        "n_cap_max_triangles": n_cap_max,
        "n_total_triangles": len(all_tris),
        "triangle_groups": {
            "wall": {"first": 1, "last": n_wall},
            "caps": {"first": n_wall + 1, "last": len(all_tris)},
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Read wall file: {args.wall}")
    print(f"Boundary loop vertices: {n}")
    print(f"Wedge angle: {args.phi_min_deg:.1f} -> {args.phi_max_deg:.1f} deg")
    print(f"Wall triangles:      {n_wall}")
    print(f"Cap triangles (min): {n_cap_min}")
    print(f"Cap triangles (max): {n_cap_max}")
    print(f"Total triangles:     {len(all_tris)}")
    print()
    print("Add to your SPARTA input script:")
    print(f"  read_surf          input/geometry/{out_surf.name} particle check")
    print(f"  group              wall surf id 1:{n_wall}")
    print(f"  group              caps surf id {n_wall+1}:{len(all_tris)}")
    print()
    print(f"Wrote: {out_surf}")
    print(f"Wrote: {out_obj}")
    print(f"Wrote: {out_stl}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
