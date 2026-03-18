#!/usr/bin/env python3
"""Build a refined DIII-D toroidal wedge surface from SOLPS mesh.extra.

Compared to build_d3d_wedge_geometry.py, this script:
- Subdivides wall segments adaptively (more near divertor)
- Adds toroidal subdivisions for better wall resolution
- Keeps cap triangulation coarse (ear-clipping, flat faces)
- Prints surface ID ranges for SPARTA group commands

Usage:
    python refine_wedge_geometry.py --mesh-extra mesh.extra \
        --phi-min-deg 0 --phi-max-deg 30 \
        --wall-refine 10 --divertor-refine 20 \
        --toroidal-segments 4 \
        --divertor-zmin -1.5 --divertor-zmax -0.5 \
        --divertor-rmin 1.0 --divertor-rmax 1.8 \
        --prefix d3d_refined
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Reuse mesh.extra parsing and ear-clipping from the original builder
# ---------------------------------------------------------------------------

def _qkey(p, tol):
    return (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))


def read_mesh_segments(mesh_extra: Path, tol: float = 1e-8):
    from collections import Counter, defaultdict
    id_of: Dict[Tuple[int, int], int] = {}
    points: List[Point2] = []
    edges: List[Tuple[int, int]] = []

    def vid(p):
        k = _qkey(p, tol)
        if k not in id_of:
            id_of[k] = len(points)
            points.append((float(p[0]), float(p[1])))
        return id_of[k]

    with mesh_extra.open("r") as f:
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

    # extract outer boundary loop
    edge_count = Counter(tuple(sorted(e)) for e in edges)
    boundary_edges = [e for e in edge_count if edge_count[e] == 1]
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
                    break
                break
            loop.append(nxt)
            unused.remove(tuple(sorted((cur, nxt))))
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    def perimeter(loop):
        n = len(loop)
        return sum(math.hypot(points[loop[(i+1)%n]][0] - points[loop[i]][0],
                              points[loop[(i+1)%n]][1] - points[loop[i]][1])
                   for i in range(n))

    outer = max(loops, key=perimeter)
    rz = [points[i] for i in outer]

    # ensure CCW winding
    area = sum(rz[i][0]*rz[(i+1)%len(rz)][1] - rz[(i+1)%len(rz)][0]*rz[i][1]
               for i in range(len(rz))) * 0.5
    if area < 0:
        rz = list(reversed(rz))

    return rz


def is_in_divertor_region(r, z, rmin, rmax, zmin, zmax):
    return rmin <= r <= rmax and zmin <= z <= zmax


def subdivide_polyline(rz: List[Point2], n_default: int, n_divertor: int,
                       rmin: float, rmax: float, zmin: float, zmax: float) -> List[Point2]:
    """Subdivide the R-Z polyline, inserting more points near the divertor."""
    refined = []
    n = len(rz)
    for i in range(n):
        r1, z1 = rz[i]
        r2, z2 = rz[(i + 1) % n]
        mid_r = 0.5 * (r1 + r2)
        mid_z = 0.5 * (z1 + z2)

        if is_in_divertor_region(mid_r, mid_z, rmin, rmax, zmin, zmax):
            nsub = n_divertor
        else:
            nsub = n_default

        for k in range(nsub):
            t = k / nsub
            refined.append((r1 + t * (r2 - r1), z1 + t * (z2 - z1)))

    return refined


# ---------------------------------------------------------------------------
# Ear-clipping for cap triangulation (copied from original builder)
# ---------------------------------------------------------------------------

def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p, a, b, c):
    d1 = _cross2d(p, a, b)
    d2 = _cross2d(p, b, c)
    d3 = _cross2d(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def ear_clip_triangulate(poly):
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []
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
                if j in ((i - 1) % m, i, (i + 1) % m):
                    continue
                if _point_in_triangle(poly[idx[j]], poly[prev_i], poly[curr_i], poly[next_i]):
                    is_ear = False
                    break
            if is_ear:
                tris.append((prev_i, curr_i, next_i))
                idx.pop(i)
                ear_found = True
                break
        if not ear_found:
            raise RuntimeError(f"Ear clipping failed with {len(idx)} vertices remaining")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


# ---------------------------------------------------------------------------
# 3D mesh construction
# ---------------------------------------------------------------------------

def rz_to_xyz(rz, phi_rad):
    c, s = math.cos(phi_rad), math.sin(phi_rad)
    return [(r * c, r * s, z) for r, z in rz]


def build_wall_tris(n_rz: int, n_tor: int) -> List[Tri]:
    """Build wall triangles for n_rz poloidal vertices x (n_tor+1) toroidal rings.

    Vertex layout: ring k has indices [k*n_rz .. (k+1)*n_rz - 1]
    """
    tris = []
    for k in range(n_tor):
        for i in range(n_rz):
            j = (i + 1) % n_rz
            a = k * n_rz + i
            b = k * n_rz + j
            c = (k + 1) * n_rz + j
            d = (k + 1) * n_rz + i
            tris.append((a, b, c))
            tris.append((a, c, d))
    return tris


def _tri_area_3d(va, vb, vc):
    ux, uy, uz = vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]
    wx, wy, wz = vc[0]-va[0], vc[1]-va[1], vc[2]-va[2]
    cx = uy*wz - uz*wy
    cy = uz*wx - ux*wz
    cz = ux*wy - uy*wx
    return 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)


def delaunay_triangulate_polygon(rz):
    """Constrained Delaunay triangulation of a simple polygon in R-Z plane.

    Uses scipy Delaunay on the polygon vertices, then filters to keep only
    triangles whose centroid lies inside the polygon.
    """
    import numpy as np
    from scipy.spatial import Delaunay
    from matplotlib.path import Path as MplPath

    pts = np.array(rz)
    tri = Delaunay(pts)

    # Filter: keep triangles whose centroid is inside the polygon
    poly_path = MplPath(pts)
    result = []
    for simplex in tri.simplices:
        centroid = pts[simplex].mean(axis=0)
        if poly_path.contains_point(centroid):
            result.append(tuple(simplex))

    return result


def build_cap_tris(rz, xyz, offset, phi_rad, outward_sign, wall_tris,
                   min_area=1e-10):
    """Triangulate a phi-face cap using Delaunay triangulation.

    Uses scipy Delaunay on R-Z polygon vertices (much better triangle quality
    than ear-clipping for refined polygons). Filters degenerate triangles and
    fixes winding for watertightness.
    """
    try:
        rz_tris = delaunay_triangulate_polygon(rz)
    except Exception:
        print("  WARNING: Delaunay failed, falling back to ear-clipping")
        rz_tris = ear_clip_triangulate(rz)

    c_phi = math.cos(phi_rad)
    s_phi = math.sin(phi_rad)
    out_dir = (-s_phi * outward_sign, c_phi * outward_sign, 0.0)

    cap_tris = []
    skipped = 0
    for a, b, c in rz_tris:
        va, vb, vc = xyz[offset + a], xyz[offset + b], xyz[offset + c]

        # Skip degenerate triangles
        if _tri_area_3d(va, vb, vc) < min_area:
            skipped += 1
            continue

        ux, uy, uz = vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]
        wx, wy, wz = vc[0]-va[0], vc[1]-va[1], vc[2]-va[2]
        nx = uy*wz - uz*wy
        ny = uz*wx - ux*wz
        nz = ux*wy - uy*wx
        dot = nx*out_dir[0] + ny*out_dir[1] + nz*out_dir[2]
        if dot >= 0:
            cap_tris.append((offset+a, offset+b, offset+c))
        else:
            cap_tris.append((offset+a, offset+c, offset+b))

    if skipped:
        print(f"  Skipped {skipped} degenerate cap triangles (area < {min_area})")

    # Build set of directed wall edges for consistency check
    wall_edges = set()
    for a, b, c in wall_tris:
        wall_edges.add((a, b))
        wall_edges.add((b, c))
        wall_edges.add((c, a))

    # Fix any cap triangle whose shared edge has the same direction as wall
    fixed = 0
    for i, (a, b, c) in enumerate(cap_tris):
        edges = [(a, b), (b, c), (c, a)]
        for e in edges:
            if e in wall_edges:
                cap_tris[i] = (a, c, b)
                fixed += 1
                break
    if fixed:
        print(f"  Fixed {fixed} cap triangle windings for watertightness")

    return cap_tris


def write_sparta_3d(path, xyz, tris):
    with path.open("w") as f:
        f.write("surface geometry\n\n")
        f.write(f"{len(xyz)} points\n")
        f.write(f"{len(tris)} triangles\n\n")
        f.write("Points\n\n")
        for i, p in enumerate(xyz, 1):
            f.write(f"{i} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
        f.write("\nTriangles\n\n")
        for i, t in enumerate(tris, 1):
            f.write(f"{i} {t[0]+1} {t[1]+1} {t[2]+1}\n")


def write_ascii_stl(path, xyz, tris, name="d3d_wedge"):
    with path.open("w") as f:
        f.write(f"solid {name}\n")
        for tri in tris:
            a, b, c = xyz[tri[0]], xyz[tri[1]], xyz[tri[2]]
            u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
            n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            mag = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
            if mag > 0:
                n = (n[0]/mag, n[1]/mag, n[2]/mag)
            else:
                n = (0, 0, 0)
            f.write(f"  facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n")
            f.write("    outer loop\n")
            for p in (a, b, c):
                f.write(f"      vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-extra", type=Path, required=True)
    p.add_argument("--phi-min-deg", type=float, default=0.0)
    p.add_argument("--phi-max-deg", type=float, default=30.0)
    p.add_argument("--wall-refine", type=int, default=5,
                   help="Subdivisions per original segment (main chamber)")
    p.add_argument("--divertor-refine", type=int, default=15,
                   help="Subdivisions per original segment (divertor region)")
    p.add_argument("--toroidal-segments", type=int, default=4,
                   help="Number of toroidal strips for wall mesh")
    p.add_argument("--divertor-rmin", type=float, default=1.0)
    p.add_argument("--divertor-rmax", type=float, default=1.8)
    p.add_argument("--divertor-zmin", type=float, default=-1.5)
    p.add_argument("--divertor-zmax", type=float, default=-0.5)
    p.add_argument("--prefix", type=Path, default=Path("d3d_refined"))
    # Emit mask options (requires h5py and numpy)
    p.add_argument("--emit-plasma-h5", type=Path, default=None,
                   help="Plasma HDF5 for emit mask (enables emit_wall generation)")
    p.add_argument("--emit-bfield-h5", type=Path, default=None,
                   help="B-field HDF5 for emit mask")
    p.add_argument("--emit-threshold", type=float, default=0.01,
                   help="Relative flux threshold for emit_wall (default 0.01)")
    p.add_argument("--emit-mass-amu", type=float, default=2.0,
                   help="Background ion mass for Bohm flux (default 2.0 D+)")
    return p.parse_args()


def main():
    args = parse_args()
    phi0 = math.radians(args.phi_min_deg)
    phi1 = math.radians(args.phi_max_deg)
    n_tor = args.toroidal_segments

    # Read and extract boundary loop from mesh.extra
    rz_raw = read_mesh_segments(args.mesh_extra)
    print(f"Original boundary vertices: {len(rz_raw)}")

    # Subdivide poloidally
    rz = subdivide_polyline(rz_raw, args.wall_refine, args.divertor_refine,
                            args.divertor_rmin, args.divertor_rmax,
                            args.divertor_zmin, args.divertor_zmax)
    n_rz = len(rz)
    print(f"Refined boundary vertices: {n_rz}")

    # Build toroidal rings: (n_tor + 1) rings from phi0 to phi1
    xyz = []
    for k in range(n_tor + 1):
        phi = phi0 + k * (phi1 - phi0) / n_tor
        xyz.extend(rz_to_xyz(rz, phi))

    # Wall triangles
    wall_tris = build_wall_tris(n_rz, n_tor)
    n_wall = len(wall_tris)

    # Cap triangles using Delaunay on the refined polygon
    # (much better quality than ear-clipping for dense point sets)
    cap_min_tris = build_cap_tris(rz, xyz, 0, phi0,
                                   outward_sign=+1.0, wall_tris=wall_tris)
    n_cap_min = len(cap_min_tris)

    cap_max_offset = n_tor * n_rz
    cap_max_tris = build_cap_tris(rz, xyz, cap_max_offset, phi1,
                                   outward_sign=-1.0, wall_tris=wall_tris)
    n_cap_max = len(cap_max_tris)

    all_tris = wall_tris + cap_min_tris + cap_max_tris
    n_total = len(all_tris)

    # Surface ID ranges (1-based, for SPARTA group commands)
    wall_first, wall_last = 1, n_wall
    cap_first = n_wall + 1
    cap_last = n_total

    # Output files
    prefix = args.prefix
    out_surf = Path(f"{prefix}_wedge_capped.surf")
    out_stl = Path(f"{prefix}_wedge_capped.stl")
    out_meta = Path(f"{prefix}_meta.json")

    write_sparta_3d(out_surf, xyz, all_tris)
    write_ascii_stl(out_stl, xyz, all_tris)

    meta = {
        "mesh_extra": str(args.mesh_extra),
        "phi_min_deg": args.phi_min_deg,
        "phi_max_deg": args.phi_max_deg,
        "n_rz_original": len(rz_raw),
        "n_rz_refined": n_rz,
        "n_toroidal_segments": n_tor,
        "wall_refine": args.wall_refine,
        "divertor_refine": args.divertor_refine,
        "divertor_region": {
            "rmin": args.divertor_rmin, "rmax": args.divertor_rmax,
            "zmin": args.divertor_zmin, "zmax": args.divertor_zmax,
        },
        "n_vertices": len(xyz),
        "n_wall_triangles": n_wall,
        "n_cap_min_triangles": n_cap_min,
        "n_cap_max_triangles": n_cap_max,
        "n_total_triangles": n_total,
        "triangle_groups": {
            "wall": {"first": wall_first, "last": wall_last},
            "caps": {"first": cap_first, "last": cap_last},
        },
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))

    print(f"\nWedge: {args.phi_min_deg:.1f} -> {args.phi_max_deg:.1f} deg, "
          f"{n_tor} toroidal segments")
    print(f"Wall triangles:     {n_wall:>6d}  (surf id {wall_first} - {wall_last})")
    print(f"Cap triangles:      {n_cap_min + n_cap_max:>6d}  (surf id {cap_first} - {cap_last})")
    print(f"Total triangles:    {n_total:>6d}")
    print(f"Min cap area:       check metadata for triangle quality")
    print(f"\nSPARTA input commands:")
    print(f"  read_surf {out_surf.name} group wall_and_caps")
    print(f"  group wall surf id <> {wall_first} {wall_last}")
    print(f"  group caps surf id <> {cap_first} {cap_last}")
    print(f"\nOutput: {out_surf}, {out_stl}, {out_meta}")

    # Optional: generate emit_wall subgroup from plasma/B-field data
    if args.emit_plasma_h5 is not None and args.emit_bfield_h5 is not None:
        from generate_emit_groups import (compute_segment_flux,
                                          active_segments_to_triangle_ids,
                                          write_sparta_include)

        print(f"\n--- Emit mask generation ---")
        flux = compute_segment_flux(rz, args.emit_plasma_h5,
                                    args.emit_bfield_h5, args.emit_mass_amu)

        peak = flux.max()
        if peak > 0:
            cutoff = args.emit_threshold * peak
            active = [i for i in range(n_rz) if flux[i] >= cutoff]
            ranges = active_segments_to_triangle_ids(active, n_rz, n_tor)
            n_emit = sum(hi - lo + 1 for lo, hi in ranges)

            print(f"Peak Bohm flux: {peak:.3e} m^-2 s^-1")
            print(f"Threshold: {args.emit_threshold:.1%} of peak = {cutoff:.3e}")
            print(f"Active poloidal segments: {len(active)} / {n_rz} "
                  f"({100*len(active)/n_rz:.1f}%)")
            print(f"Emit wall triangles: {n_emit} / {n_wall} "
                  f"({100*n_emit/n_wall:.1f}%)")

            out_emit = Path(f"{prefix}_emit_wall_groups.sparta")
            write_sparta_include(out_emit, ranges, wall_first, wall_last)

            meta["triangle_groups"]["emit_wall"] = {
                "first_ranges": [[lo, hi] for lo, hi in ranges],
                "n_triangles": n_emit,
                "threshold": args.emit_threshold,
                "n_active_poloidal": len(active),
            }
            Path(out_meta).write_text(json.dumps(meta, indent=2))
            print(f"Updated {out_meta}")
        else:
            print("WARNING: No positive flux found — skipping emit mask")


if __name__ == "__main__":
    main()
