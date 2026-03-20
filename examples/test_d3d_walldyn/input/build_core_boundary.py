#!/usr/bin/env python3
"""Revolve a 2D flux-surface polygon (R,Z) into a 3D toroidal wedge surface.

Reads a SPARTA-format 2D surface file (Points + Lines) and produces a
triangulated 3D wedge with phi-periodic caps, suitable for use as a core
boundary in OpenEdge/SPARTA.

Outputs:
- SPARTA 3D triangular surface (.surf)
- OBJ mesh for visualization
- Metadata JSON with triangle group ranges
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
    p.add_argument("--input", type=Path, required=True,
                   help="Path to 2D SPARTA surface file (R,Z polygon)")
    p.add_argument("--phi-min-deg", type=float, default=0.0,
                   help="Wedge start toroidal angle [deg]")
    p.add_argument("--phi-max-deg", type=float, default=30.0,
                   help="Wedge end toroidal angle [deg]")
    p.add_argument("--prefix", type=Path, default=Path("core_boundary"),
                   help="Output prefix path")
    p.add_argument("--flip-normals", action="store_true",
                   help="Flip wall normals to point inward (toward core)")
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
                lid = int(cols[0])
                v1, v2 = int(cols[1]), int(cols[2])
                lines_section.append((v1, v2))

    if not lines_section:
        # Fall back: just return points in order
        return [points[i] for i in sorted(points.keys())]

    # Follow the line connectivity to get ordered polygon
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


def build_wedge_wall_tris(n: int, flip: bool = False) -> List[Tri]:
    """Build wall triangles between phi_min (0..n-1) and phi_max (n..2n-1).

    Default normals point outward (away from axis). With flip=True, normals
    point inward (toward axis), which is what you want for a core boundary
    that should reflect/absorb particles coming from outside.
    """
    tris: List[Tri] = []
    for i in range(n):
        j = (i + 1) % n
        a, b, c, d = i, j, n + j, n + i
        if flip:
            tris.append((a, c, b))
            tris.append((a, d, c))
        else:
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
    """Triangulate a phi-face cap with outward-pointing normals."""
    rz_tris = ear_clip_triangulate(rz)
    cap_tris: List[Tri] = []
    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    out_dir = (-s_phi * outward_sign, c_phi * outward_sign, 0.0)

    for a, b, c in rz_tris:
        va, vb, vc = xyz[offset + a], xyz[offset + b], xyz[offset + c]
        ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
        wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
        nx = uy * wz - uz * wy
        ny = uz * wx - ux * wz
        nz_ = ux * wy - uy * wx
        dot = nx * out_dir[0] + ny * out_dir[1] + nz_ * out_dir[2]
        if dot >= 0:
            cap_tris.append((offset + a, offset + b, offset + c))
        else:
            cap_tris.append((offset + a, offset + c, offset + b))
    return cap_tris


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
    n = len(rz)
    print(f"Read {n} points from {args.input}")

    # Ensure CCW winding
    if signed_area_rz(rz) < 0.0:
        rz = list(reversed(rz))
        print("  Reversed winding to CCW")

    phi0 = math.radians(args.phi_min_deg)
    phi1 = math.radians(args.phi_max_deg)

    xyz0 = rz_to_xyz(rz, phi0)
    xyz1 = rz_to_xyz(rz, phi1)
    xyz = xyz0 + xyz1

    # Wall tris: flip normals so they point inward (core boundary blocks
    # particles coming from outside, i.e. from larger psi)
    wall_tris = build_wedge_wall_tris(n, flip=args.flip_normals)

    # Cap tris for periodic boundaries
    cap_min_tris = build_cap_tris(rz, xyz, 0, phi0, outward_sign=+1.0)
    cap_max_tris = build_cap_tris(rz, xyz, n, phi1, outward_sign=-1.0)

    all_tris = wall_tris + cap_min_tris + cap_max_tris
    n_wall = len(wall_tris)
    n_cap_min = len(cap_min_tris)
    n_cap_max = len(cap_max_tris)

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
        "flip_normals": args.flip_normals,
        "n_wall_triangles": n_wall,
        "n_cap_min_triangles": n_cap_min,
        "n_cap_max_triangles": n_cap_max,
        "n_total_triangles": len(all_tris),
        "triangle_groups": {
            "core_wall": {"first": 1, "last": n_wall},
            "cap_phi_min": {"first": n_wall + 1, "last": n_wall + n_cap_min},
            "cap_phi_max": {"first": n_wall + n_cap_min + 1,
                            "last": n_wall + n_cap_min + n_cap_max},
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"Wedge: {args.phi_min_deg:.1f} -> {args.phi_max_deg:.1f} deg")
    print(f"Wall triangles:       {n_wall}")
    print(f"Cap triangles (min):  {n_cap_min}")
    print(f"Cap triangles (max):  {n_cap_max}")
    print(f"Total triangles:      {len(all_tris)}")
    print()
    print("Add to your SPARTA input script:")
    print(f"  read_surf          input/{out_surf.name} group core_all")
    print(f"  group              core_wall surf id 1:{n_wall}")
    print(f"  group              core_caps surf id {n_wall+1}:{len(all_tris)}")
    print(f"  surf_collide       CORE_WALL specular")
    print(f"  surf_collide       CORE_PERIODIC toroidal {args.phi_max_deg - args.phi_min_deg:.1f}")
    print(f"  surf_modify        core_wall collide CORE_WALL react none")
    print(f"  surf_modify        core_caps collide CORE_PERIODIC react none")
    print()
    print(f"Wrote: {out_surf}")
    print(f"Wrote: {out_obj}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
