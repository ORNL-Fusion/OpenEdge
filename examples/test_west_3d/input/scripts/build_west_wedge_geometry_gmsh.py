#!/usr/bin/env python3
"""Build a WEST toroidal wedge surface with Gmsh from a 2D SPARTA wall file.

This is a Gmsh-backed alternative to ``build_west_wedge_geometry.py``.
It reads a 2D SPARTA wall file (Points + Lines in the R-Z plane), creates a
planar cross-section in the X-Z plane, revolves it about the Z axis, meshes the
resulting wedge surfaces with Gmsh, and writes:

- a 3D SPARTA/OpenEdge triangular ``.surf`` file
- an OBJ mesh
- an ASCII STL mesh
- a JSON metadata file with triangle group ranges

The generated surfaces are ordered as:
- wall triangles first
- cap triangles after

Usage example:
    python3 build_west_wedge_geometry_gmsh.py \
        --wall input/data/wall.txt \
        --phi-min-deg 0 --phi-max-deg 30 \
        --mesh-size 0.01 \
        --prefix input/geometry/west
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wall",
        type=Path,
        required=True,
        help="Path to 2D SPARTA wall file (Points + Lines)",
    )
    p.add_argument(
        "--phi-min-deg",
        type=float,
        default=0.0,
        help="Wedge start toroidal angle [deg]",
    )
    p.add_argument(
        "--phi-max-deg",
        type=float,
        default=30.0,
        help="Wedge end toroidal angle [deg]",
    )
    p.add_argument(
        "--simplify",
        type=float,
        default=0.002,
        help="Douglas-Peucker simplification tolerance [m] (0 to disable)",
    )
    p.add_argument(
        "--mesh-size",
        type=float,
        default=0.01,
        help="Target Gmsh surface mesh size [m]",
    )
    p.add_argument(
        "--mesh-size-min",
        type=float,
        default=None,
        help="Optional lower bound for Gmsh mesh size [m]",
    )
    p.add_argument(
        "--mesh-size-max",
        type=float,
        default=None,
        help="Optional upper bound for Gmsh mesh size [m]",
    )
    p.add_argument(
        "--algo",
        type=int,
        default=6,
        help="Gmsh 2D meshing algorithm (default: 6, Frontal-Delaunay)",
    )
    p.add_argument(
        "--prefix",
        type=Path,
        default=Path("west"),
        help="Output prefix path",
    )
    p.add_argument(
        "--write-msh",
        action="store_true",
        help="Also write a Gmsh .msh file",
    )
    return p.parse_args()


def require_gmsh():
    try:
        import gmsh  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The Python gmsh module is required for this script.\n"
            "Install it in your active environment, for example:\n"
            "  pip install gmsh\n"
            f"Original import error: {exc}"
        )
    return gmsh


def read_sparta_2d_wall(path: Path) -> List[Point2]:
    """Read a SPARTA 2D surf file and return the closed polyline as (R, Z)."""
    lines_text = path.read_text(encoding="utf-8").splitlines()

    points_raw = {}
    edges: List[Tuple[int, int]] = []
    section = None

    for ln in lines_text:
        s = ln.strip()
        if not s or s.startswith("surface"):
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

    adj = {i: j for i, j in edges}
    start = edges[0][0]
    loop = [start]
    cur = adj[start]
    while cur != start:
        loop.append(cur)
        cur = adj[cur]

    return [points_raw[i] for i in loop]


def _perpendicular_dist(p: Point2, a: Point2, b: Point2) -> float:
    dx, dz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dz)
    if length < 1e-30:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dz) / length


def douglas_peucker(pts: List[Point2], tol: float) -> List[Point2]:
    if len(pts) <= 2:
        return pts

    d_max = 0.0
    idx = 0
    for i in range(1, len(pts) - 1):
        d = _perpendicular_dist(pts[i], pts[0], pts[-1])
        if d > d_max:
            d_max = d
            idx = i

    if d_max > tol:
        left = douglas_peucker(pts[: idx + 1], tol)
        right = douglas_peucker(pts[idx:], tol)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def simplify_closed_polygon(pts: List[Point2], tol: float) -> List[Point2]:
    if tol <= 0 or len(pts) < 4:
        return pts
    extended = pts + [pts[0]]
    simplified = douglas_peucker(extended, tol)
    if len(simplified) > 1:
        simplified = simplified[:-1]
    return simplified


def signed_area_rz(poly: Sequence[Point2]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def clean_polygon(rz: List[Point2], tol: float) -> List[Point2]:
    cleaned: List[Point2] = [rz[0]]
    for i in range(1, len(rz)):
        dr = rz[i][0] - cleaned[-1][0]
        dz = rz[i][1] - cleaned[-1][1]
        if math.hypot(dr, dz) > 1e-6:
            cleaned.append(rz[i])
    dr = cleaned[-1][0] - cleaned[0][0]
    dz = cleaned[-1][1] - cleaned[0][1]
    if math.hypot(dr, dz) < 1e-6:
        cleaned.pop()

    if tol > 0:
        cleaned = simplify_closed_polygon(cleaned, tol)

    if signed_area_rz(cleaned) < 0.0:
        cleaned = list(reversed(cleaned))
    return cleaned


def triangle_normal(a: Point3, b: Point3, c: Point3) -> Point3:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    return (
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx,
    )


def dot(a: Point3, b: Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def mean_point(points: Iterable[Point3]) -> Point3:
    pts = list(points)
    n = len(pts)
    return (
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
        sum(p[2] for p in pts) / n,
    )


def normalize_phi(phi: float) -> float:
    twopi = 2.0 * math.pi
    while phi < 0.0:
        phi += twopi
    while phi >= twopi:
        phi -= twopi
    return phi


def angle_diff(a: float, b: float) -> float:
    d = normalize_phi(a - b)
    if d > math.pi:
        d -= 2.0 * math.pi
    return abs(d)


def classify_triangle(
    tri_pts: Sequence[Point3],
    phi_min: float,
    phi_max: float,
    tol: float = 1e-4,
) -> str:
    phis = [normalize_phi(math.atan2(p[1], p[0])) for p in tri_pts]
    mean_phi = math.atan2(
        sum(math.sin(phi) for phi in phis),
        sum(math.cos(phi) for phi in phis),
    )
    mean_phi = normalize_phi(mean_phi)
    spread = max(angle_diff(phi, mean_phi) for phi in phis)

    if spread < tol:
        if angle_diff(mean_phi, normalize_phi(phi_min)) < 5 * tol:
            return "cap_min"
        if angle_diff(mean_phi, normalize_phi(phi_max)) < 5 * tol:
            return "cap_max"
    return "wall"


def extract_surface_triangles(gmsh, phi_min: float, phi_max: float) -> Tuple[List[Point3], List[Tri], List[Tri], List[Tri]]:
    node_map: dict[int, int] = {}
    xyz: List[Point3] = []
    wall_tris: List[Tri] = []
    cap_min_tris: List[Tri] = []
    cap_max_tris: List[Tri] = []

    surfaces = gmsh.model.getEntities(2)
    for dim, tag in surfaces:
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim, tag)
        for elem_type, node_tags in zip(elem_types, elem_node_tags):
            props = gmsh.model.mesh.getElementProperties(elem_type)
            if len(props) >= 6:
                name = props[0]
                num_nodes = props[3]
            else:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Unexpected getElementProperties() return for element type {elem_type}: {props}"
                )
            if not str(name).startswith("Triangle") or num_nodes != 3:
                continue

            for i in range(0, len(node_tags), 3):
                tri_nodes = [int(node_tags[i]), int(node_tags[i + 1]), int(node_tags[i + 2])]
                tri_ids: List[int] = []
                tri_pts: List[Point3] = []
                for node_tag in tri_nodes:
                    if node_tag not in node_map:
                        node_info = gmsh.model.mesh.getNode(node_tag)
                        coords = node_info[0]
                        node_map[node_tag] = len(xyz)
                        xyz.append((coords[0], coords[1], coords[2]))
                    tri_id = node_map[node_tag]
                    tri_ids.append(tri_id)
                    tri_pts.append(xyz[tri_id])

                group = classify_triangle(tri_pts, phi_min, phi_max)
                if group.startswith("cap"):
                    center = mean_point(tri_pts)
                    out_sign = +1.0 if group == "cap_min" else -1.0
                    out_dir = (
                        -math.sin(phi_min if group == "cap_min" else phi_max) * out_sign,
                        math.cos(phi_min if group == "cap_min" else phi_max) * out_sign,
                        0.0,
                    )
                    n = triangle_normal(*tri_pts)
                    tri = tuple(tri_ids)  # type: ignore[assignment]
                    if dot(n, out_dir) < 0:
                        tri = (tri_ids[0], tri_ids[2], tri_ids[1])
                    if group == "cap_min":
                        cap_min_tris.append(tri)
                    else:
                        cap_max_tris.append(tri)
                else:
                    wall_tris.append(tuple(tri_ids))  # type: ignore[arg-type]

    return xyz, wall_tris, cap_min_tris, cap_max_tris


def check_watertight(tris: Sequence[Tri]) -> bool:
    ehash = {}
    dup = 0
    for tri in tris:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if edge in ehash:
                dup += 1
            else:
                ehash[edge] = 1

    unmatch = 0
    for edge in ehash:
        if (edge[1], edge[0]) not in ehash:
            unmatch += 1

    if dup or unmatch:
        print("WARNING: surface is NOT watertight")
        if dup:
            print(f"  Duplicate edges: {dup}")
        if unmatch:
            print(f"  Unmatched edges: {unmatch}")
        return False

    print("Watertight check: PASSED")
    return True


def write_sparta_3d(path: Path, xyz: Sequence[Point3], tris: Sequence[Tri]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{len(xyz)} points\n")
        f.write(f"{len(tris)} triangles\n\n")
        f.write("Points\n\n")
        for i, p in enumerate(xyz, start=1):
            f.write(f"{i} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        f.write("\nTriangles\n\n")
        for i, tri in enumerate(tris, start=1):
            f.write(f"{i} {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_obj(path: Path, xyz: Sequence[Point3], tris: Sequence[Tri]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# WEST wedge wall mesh (Gmsh)\n")
        for p in xyz:
            f.write(f"v {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        for tri in tris:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_ascii_stl(path: Path, xyz: Sequence[Point3], tris: Sequence[Tri], name: str = "west_wedge") -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in tris:
            a, b, c = xyz[tri[0]], xyz[tri[1]], xyz[tri[2]]
            nx, ny, nz = triangle_normal(a, b, c)
            mag = math.sqrt(nx * nx + ny * ny + nz * nz)
            if mag > 0:
                nx /= mag
                ny /= mag
                nz /= mag
            f.write(f"  facet normal {nx:.12g} {ny:.12g} {nz:.12g}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.12g} {a[1]:.12g} {a[2]:.12g}\n")
            f.write(f"      vertex {b[0]:.12g} {b[1]:.12g} {b[2]:.12g}\n")
            f.write(f"      vertex {c[0]:.12g} {c[1]:.12g} {c[2]:.12g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def build_gmsh_wedge(gmsh, rz: Sequence[Point2], phi_min: float, phi_max: float, mesh_size: float, algo: int) -> None:
    phi0 = phi_min
    dphi = phi_max - phi_min
    c0 = math.cos(phi0)
    s0 = math.sin(phi0)

    gmsh.model.add("west_wedge")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", algo)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)

    point_tags = []
    for r, z in rz:
        point_tags.append(gmsh.model.geo.addPoint(r * c0, r * s0, z, mesh_size))

    line_tags = []
    n = len(point_tags)
    for i in range(n):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n]))

    loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
    base_surface = gmsh.model.geo.addPlaneSurface([loop_tag])

    out = gmsh.model.geo.revolve([(2, base_surface)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dphi)
    gmsh.model.geo.synchronize()

    volume_tags = [tag for dim, tag in out if dim == 3]
    if volume_tags:
        boundary = gmsh.model.getBoundary([(3, volume_tags[0])], oriented=False, recursive=False)
        gmsh.model.addPhysicalGroup(2, [tag for dim, tag in boundary if dim == 2], name="wedge_surfaces")


def main() -> None:
    args = parse_args()
    if args.phi_max_deg <= args.phi_min_deg:
        raise ValueError("phi_max_deg must be greater than phi_min_deg")

    gmsh = require_gmsh()

    rz = read_sparta_2d_wall(args.wall)
    n_before = len(rz)
    rz = clean_polygon(rz, args.simplify)
    removed = n_before - len(rz)
    if removed > 0:
        print(f"Simplified/cleaned: {n_before} -> {len(rz)} vertices")

    phi_min = math.radians(args.phi_min_deg)
    phi_max = math.radians(args.phi_max_deg)

    gmsh.initialize()
    try:
        if args.mesh_size_min is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", args.mesh_size_min)
        if args.mesh_size_max is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size_max)

        build_gmsh_wedge(gmsh, rz, phi_min, phi_max, args.mesh_size, args.algo)
        gmsh.model.mesh.generate(2)

        xyz, wall_tris, cap_min_tris, cap_max_tris = extract_surface_triangles(gmsh, phi_min, phi_max)
        all_tris = wall_tris + cap_min_tris + cap_max_tris

        check_watertight(all_tris)

        prefix = args.prefix
        if prefix.parent != Path(""):
            prefix.parent.mkdir(parents=True, exist_ok=True)

        out_surf = Path(f"{prefix}_wedge_capped.surf")
        out_obj = Path(f"{prefix}_wedge_capped.obj")
        out_stl = Path(f"{prefix}_wedge_capped.stl")
        out_meta = Path(f"{prefix}_meta.json")
        out_msh = Path(f"{prefix}_wedge_capped.msh")

        write_sparta_3d(out_surf, xyz, all_tris)
        write_obj(out_obj, xyz, all_tris)
        write_ascii_stl(out_stl, xyz, all_tris)

        if args.write_msh:
            gmsh.write(str(out_msh))

        meta = {
            "wall_file": str(args.wall),
            "phi_min_deg": args.phi_min_deg,
            "phi_max_deg": args.phi_max_deg,
            "dphi_deg": args.phi_max_deg - args.phi_min_deg,
            "n_rz_vertices": len(rz),
            "mesh_size": args.mesh_size,
            "mesh_size_min": args.mesh_size_min,
            "mesh_size_max": args.mesh_size_max,
            "gmsh_algorithm": args.algo,
            "n_wall_triangles": len(wall_tris),
            "n_cap_min_triangles": len(cap_min_tris),
            "n_cap_max_triangles": len(cap_max_tris),
            "n_total_triangles": len(all_tris),
            "triangle_groups": {
                "wall": {"first": 1, "last": len(wall_tris)},
                "caps": {"first": len(wall_tris) + 1, "last": len(all_tris)},
            },
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Read wall file: {args.wall}")
        print(f"Boundary loop vertices: {len(rz)}")
        print(f"Wedge angle: {args.phi_min_deg:.1f} -> {args.phi_max_deg:.1f} deg")
        print(f"Mesh size: {args.mesh_size:.4g} m")
        print(f"Wall triangles:      {len(wall_tris)}")
        print(f"Cap triangles (min): {len(cap_min_tris)}")
        print(f"Cap triangles (max): {len(cap_max_tris)}")
        print(f"Total triangles:     {len(all_tris)}")
        print()
        print("Add to your SPARTA input script:")
        print(f"  read_surf          input/geometry/{out_surf.name} particle check")
        print(f"  group              wall surf id 1:{len(wall_tris)}")
        print(f"  group              caps surf id {len(wall_tris) + 1}:{len(all_tris)}")
        print()
        print(f"Wrote: {out_surf}")
        print(f"Wrote: {out_obj}")
        print(f"Wrote: {out_stl}")
        print(f"Wrote: {out_meta}")
        if args.write_msh:
            print(f"Wrote: {out_msh}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
