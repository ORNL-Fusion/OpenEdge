#!/usr/bin/env python3
"""Build a DIII-D toroidal wedge surface from SOLPS mesh.extra.

Outputs:
- 2D SPARTA/OpenEdge wall file in (R,Z)
- 3D SPARTA/OpenEdge triangular surface for a toroidal wedge [phi_min, phi_max]
- OBJ mesh for quick visualization
- STL mesh for quick visualization
- CSV pairing of periodic phi-side vertices
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-extra", type=Path, required=True, help="Path to SOLPS mesh.extra")
    p.add_argument("--phi-min-deg", type=float, default=0.0, help="Wedge start toroidal angle [deg]")
    p.add_argument("--phi-max-deg", type=float, default=20.0, help="Wedge end toroidal angle [deg]")
    p.add_argument("--tol", type=float, default=1.0e-8, help="Point merge tolerance [m]")
    p.add_argument("--prefix", type=Path, default=Path("d3d_wedge"), help="Output prefix path")
    return p.parse_args()


def _qkey(p: Sequence[float], tol: float) -> Tuple[int, int]:
    return (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))


def read_mesh_segments(mesh_extra: Path, tol: float) -> Tuple[List[Point2], List[Tuple[int, int]]]:
    if not mesh_extra.exists():
        raise FileNotFoundError(f"mesh.extra not found: {mesh_extra}")

    id_of: Dict[Tuple[int, int], int] = {}
    points: List[Point2] = []
    edges: List[Tuple[int, int]] = []

    def vid(p: Sequence[float]) -> int:
        k = _qkey(p, tol)
        if k not in id_of:
            id_of[k] = len(points)
            points.append((float(p[0]), float(p[1])))
        return id_of[k]

    with mesh_extra.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            cols = s.split()
            if len(cols) < 4:
                continue
            r1, z1, r2, z2 = map(float, cols[:4])
            i = vid((r1, z1))
            j = vid((r2, z2))
            if i != j:
                edges.append((i, j))

    if not edges:
        raise RuntimeError("No valid segments parsed from mesh.extra")

    return points, edges


def extract_outer_loop(points: List[Point2], edges: List[Tuple[int, int]]) -> List[int]:
    # Interior edges in mesh.extra usually appear twice; keep boundary-only edges.
    edge_count = Counter(tuple(sorted(e)) for e in edges)
    boundary_edges = [e for e in edge_count if edge_count[e] == 1]
    if not boundary_edges:
        raise RuntimeError("No boundary edges found in mesh.extra")

    adj: Dict[int, List[int]] = defaultdict(list)
    edge_set = set()
    for i, j in boundary_edges:
        adj[i].append(j)
        adj[j].append(i)
        edge_set.add(tuple(sorted((i, j))))

    unused = set(edge_set)
    loops: List[List[int]] = []

    while unused:
        e0 = next(iter(unused))
        start, cur = e0
        loop = [start, cur]
        unused.remove(e0)

        prev = start
        while True:
            nbrs = adj[cur]
            nxt = None
            for cand in nbrs:
                if cand == prev:
                    continue
                ek = tuple(sorted((cur, cand)))
                if ek in unused:
                    nxt = cand
                    break
            if nxt is None:
                if cur == start:
                    break
                if start in nbrs:
                    cur = start
                    break
                break

            loop.append(nxt)
            unused.remove(tuple(sorted((cur, nxt))))
            prev, cur = cur, nxt
            if cur == start:
                break

        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    if not loops:
        raise RuntimeError("Could not reconstruct a closed boundary loop from mesh.extra")

    def perimeter(loop: List[int]) -> float:
        n = len(loop)
        p = 0.0
        for i in range(n):
            a = points[loop[i]]
            b = points[loop[(i + 1) % n]]
            p += math.hypot(b[0] - a[0], b[1] - a[1])
        return p

    outer = max(loops, key=perimeter)
    return outer


def signed_area_rz(poly: List[Point2]) -> float:
    n = len(poly)
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _cross2d(o: Point2, a: Point2, b: Point2) -> float:
    """Signed area of triangle OAB (positive if CCW)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool:
    d1 = _cross2d(p, a, b)
    d2 = _cross2d(p, b, c)
    d3 = _cross2d(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(poly: List[Point2]) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon.

    Returns triangle index triples (into the original poly list).
    Polygon must be in CCW winding order.
    """
    n = len(poly)
    if n < 3:
        return []

    # work with index list that shrinks as ears are clipped
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    while len(idx) > 3:
        ear_found = False
        m = len(idx)
        for i in range(m):
            prev_i = idx[(i - 1) % m]
            curr_i = idx[i]
            next_i = idx[(i + 1) % m]

            # ear candidate must be a convex vertex (positive cross product for CCW)
            if _cross2d(poly[prev_i], poly[curr_i], poly[next_i]) <= 0:
                continue

            # check no other vertex lies inside this triangle
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
                f"(degenerate polygon?)"
            )

    # last 3 vertices form the final triangle
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def build_cap_tris(
    rz: List[Point2],
    xyz: List[Point3],
    offset: int,
    phi_rad: float,
    outward_sign: float,
) -> List[Tri]:
    """Triangulate a phi-face cap using ear-clipping on the R-Z polygon.

    offset: vertex index offset in the global vertex list (0 for phi_min, n for phi_max)
    outward_sign: +1.0 if cap normal should point in -phi direction (phi_min cap),
                  -1.0 if cap normal should point in +phi direction (phi_max cap).
    We flip winding to get outward normals.
    """
    # Ear-clip in R-Z plane (polygon is CCW after signed_area check)
    rz_tris = ear_clip_triangulate(rz)

    cap_tris: List[Tri] = []
    # For each triangle, compute its normal and check it points outward
    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    # Outward normal for phi_min cap: points in -phi direction = (-sin(phi), cos(phi), 0)
    # scaled by outward_sign
    out_dir = (-s_phi * outward_sign, c_phi * outward_sign, 0.0)

    for a, b, c in rz_tris:
        va = xyz[offset + a]
        vb = xyz[offset + b]
        vc = xyz[offset + c]
        # compute triangle normal
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


def write_sparta_2d(path: Path, rz: List[Point2]) -> None:
    n = len(rz)
    with path.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i, (r, z) in enumerate(rz, start=1):
            f.write(f"{i} {r:.12g} {z:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            a = i + 1
            b = (i + 1) % n + 1
            f.write(f"{i+1} {a} {b}\n")


def rz_to_xyz(rz: List[Point2], phi_rad: float) -> List[Point3]:
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    xyz: List[Point3] = []
    for r, z in rz:
        xyz.append((r * c, r * s, z))
    return xyz


def build_wedge_wall_tris(n: int) -> List[Tri]:
    # side 0 indices: [0..n-1], side 1 indices: [n..2n-1]
    tris: List[Tri] = []
    for i in range(n):
        j = (i + 1) % n
        a = i
        b = j
        c = n + j
        d = n + i
        tris.append((a, b, c))
        tris.append((a, c, d))
    return tris


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
        f.write("# d3d wedge wall mesh\n")
        for p in xyz:
            f.write(f"v {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def _vec_sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(u: Point3, v: Point3) -> Point3:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _norm(v: Point3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def write_ascii_stl(path: Path, xyz: List[Point3], tris: List[Tri], name: str = "d3d_wedge_wall") -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in tris:
            a = xyz[tri[0]]
            b = xyz[tri[1]]
            c = xyz[tri[2]]
            u = _vec_sub(b, a)
            v = _vec_sub(c, a)
            n = _cross(u, v)
            mag = _norm(n)
            if mag > 0.0:
                n = (n[0] / mag, n[1] / mag, n[2] / mag)
            else:
                n = (0.0, 0.0, 0.0)

            f.write(f"  facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.12g} {a[1]:.12g} {a[2]:.12g}\n")
            f.write(f"      vertex {b[0]:.12g} {b[1]:.12g} {b[2]:.12g}\n")
            f.write(f"      vertex {c[0]:.12g} {c[1]:.12g} {c[2]:.12g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def write_periodic_pair_csv(path: Path, n: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vertex_id_phi_min", "vertex_id_phi_max"])
        for i in range(n):
            w.writerow([i + 1, n + i + 1])


def main() -> None:
    args = parse_args()

    if args.phi_max_deg <= args.phi_min_deg:
        raise ValueError("phi_max_deg must be greater than phi_min_deg")

    points, edges = read_mesh_segments(args.mesh_extra, args.tol)
    loop_ids = extract_outer_loop(points, edges)
    rz = [points[i] for i in loop_ids]

    # Ensure a consistent winding for deterministic output.
    if signed_area_rz(rz) < 0.0:
        rz = list(reversed(rz))

    n = len(rz)
    phi0 = math.radians(args.phi_min_deg)
    phi1 = math.radians(args.phi_max_deg)

    xyz0 = rz_to_xyz(rz, phi0)
    xyz1 = rz_to_xyz(rz, phi1)
    xyz = xyz0 + xyz1
    wall_tris = build_wedge_wall_tris(n)

    # Build phi-face cap triangulations via ear-clipping
    # phi_min cap: vertices [0..n-1], outward normal points in -phi direction
    cap_min_tris = build_cap_tris(rz, xyz, 0, phi0, outward_sign=+1.0)
    # phi_max cap: vertices [n..2n-1], outward normal points in +phi direction
    cap_max_tris = build_cap_tris(rz, xyz, n, phi1, outward_sign=-1.0)

    all_tris = wall_tris + cap_min_tris + cap_max_tris

    n_wall = len(wall_tris)
    n_cap_min = len(cap_min_tris)
    n_cap_max = len(cap_max_tris)

    prefix = args.prefix
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)

    out_wall_2d = Path(f"{prefix}_wall_rz.surf")
    out_capped_3d = Path(f"{prefix}_wedge_capped.surf")
    out_wall_3d = Path(f"{prefix}_wedge_wall.surf")
    out_obj = Path(f"{prefix}_wedge_capped.obj")
    out_stl = Path(f"{prefix}_wedge_capped.stl")
    out_pair = Path(f"{prefix}_phi_periodic_pairs.csv")
    out_meta = Path(f"{prefix}_meta.json")

    write_sparta_2d(out_wall_2d, rz)
    write_sparta_3d(out_capped_3d, xyz, all_tris)
    write_sparta_3d(out_wall_3d, xyz, wall_tris)
    write_obj(out_obj, xyz, all_tris)
    write_ascii_stl(out_stl, xyz, all_tris)
    write_periodic_pair_csv(out_pair, n)

    meta = {
        "mesh_extra": str(args.mesh_extra),
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
            "cap_phi_min": {"first": n_wall + 1, "last": n_wall + n_cap_min},
            "cap_phi_max": {"first": n_wall + n_cap_min + 1,
                            "last": n_wall + n_cap_min + n_cap_max},
        },
        "outputs": {
            "wall_2d": str(out_wall_2d),
            "capped_3d": str(out_capped_3d),
            "wall_3d": str(out_wall_3d),
            "obj": str(out_obj),
            "stl": str(out_stl),
            "periodic_pairs": str(out_pair),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Read mesh.extra: {args.mesh_extra}")
    print(f"Boundary loop vertices: {n}")
    print(f"Wedge angle: {args.phi_min_deg:.3f} -> {args.phi_max_deg:.3f} deg")
    print(f"Wall triangles: {n_wall}")
    print(f"Cap triangles (phi_min): {n_cap_min}")
    print(f"Cap triangles (phi_max): {n_cap_max}")
    print(f"Total triangles: {len(all_tris)}")
    print(f"Surface groups for input script:")
    print(f"  group wall surf id 1:{n_wall}")
    print(f"  group caps surf id {n_wall+1}:{len(all_tris)}")
    print(f"Wrote: {out_wall_2d}")
    print(f"Wrote: {out_capped_3d}")
    print(f"Wrote: {out_wall_3d}")
    print(f"Wrote: {out_obj}")
    print(f"Wrote: {out_stl}")
    print(f"Wrote: {out_pair}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
