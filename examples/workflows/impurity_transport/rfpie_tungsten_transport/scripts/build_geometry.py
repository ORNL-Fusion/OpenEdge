#!/usr/bin/env python3
"""Convert the original RFPIE binary STL geometry into SPARTA surfaces.

The original CAD uses millimetres.  The generated OpenEdge surfaces use SI
metres.  The plasma-facing tungsten disk is separated from the rest of the
target so it can be its own sputtering/sheath group.  Every target-face tile
carries a custom three-column array named ``sheath``:

    sheath[1] = Vdc [V]
    sheath[2] = Vrf peak amplitude [V]
    sheath[3] = RF phase offset [rad]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CASE_DIR.parents[3]


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    ntri = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * ntri:
        raise ValueError(f"{path} is not the expected binary STL layout")
    normals = np.empty((ntri, 3), dtype=float)
    triangles = np.empty((ntri, 3, 3), dtype=float)
    offset = 84
    for i in range(ntri):
        record = struct.unpack_from("<12fH", raw, offset)
        normals[i] = record[:3]
        triangles[i] = np.asarray(record[3:12]).reshape(3, 3)
        offset += 50
    return normals, triangles


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def remesh_planar_face(triangles: np.ndarray, nrings: int) -> np.ndarray:
    """Refine a convex planar face while preserving its boundary vertices."""
    if nrings < 1:
        raise ValueError("target_face_radial_rings must be >= 1")
    coords: dict[tuple[float, float, float], np.ndarray] = {}
    edge_count: dict[tuple[tuple[float, float, float],
                           tuple[float, float, float]], int] = {}
    for tri in triangles:
        keys = [tuple(round(float(v), 12) for v in xyz) for xyz in tri]
        for key, xyz in zip(keys, tri):
            coords[key] = np.asarray(xyz, dtype=float)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((keys[i], keys[j])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    adjacency: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("plasma-facing boundary is not one closed polygon")

    start = min(adjacency)
    ordered = [start]
    previous = None
    current = start
    while True:
        candidates = [v for v in adjacency[current] if v != previous]
        nxt = candidates[0]
        if nxt == start:
            break
        ordered.append(nxt)
        previous, current = current, nxt
        if len(ordered) > len(adjacency):
            raise RuntimeError("failed to traverse plasma-facing boundary")
    if len(ordered) != len(adjacency):
        raise RuntimeError("plasma-facing boundary contains multiple loops")
    polygon = np.asarray([coords[key] for key in ordered])
    signed_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if signed_area < 0.0:
        polygon = polygon[::-1]

    # Use fewer azimuthal points on inner rings so element size stays roughly
    # isotropic. Keeping all boundary vertices makes the mesh conform exactly
    # to the original side wall. Delaunay is only a triangulator here; no
    # geometry is smoothed or moved.
    from scipy.spatial import Delaunay

    center = polygon.mean(axis=0)
    nboundary = len(polygon)
    rel = polygon[:, :2] - center[:2]
    theta_boundary = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    theta0 = theta_boundary[0]
    theta_mod = (theta_boundary - theta0) % (2.0 * np.pi)
    order = np.argsort(theta_mod)
    theta_axis = theta_mod[order]
    radius_axis = np.linalg.norm(rel[order], axis=1)
    theta_axis = np.r_[theta_axis, 2.0 * np.pi]
    radius_axis = np.r_[radius_axis, radius_axis[0]]

    points = [center]
    for iring in range(1, nrings):
        frac = iring / nrings
        ntheta = max(6, int(round(nboundary * frac)))
        theta_rel = 2.0 * np.pi * np.arange(ntheta) / ntheta
        radius = frac * np.interp(theta_rel, theta_axis, radius_axis)
        theta = theta0 + theta_rel
        zface = np.full(ntheta, center[2])
        points.extend(np.column_stack([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta), zface]))
    points.extend(polygon)
    points = np.asarray(points)
    connectivity = Delaunay(points[:, :2]).simplices
    refined = points[connectivity]
    cross_z = np.cross(refined[:, 1] - refined[:, 0],
                       refined[:, 2] - refined[:, 0])[:, 2]
    refined[cross_z < 0.0] = refined[cross_z < 0.0][:, [0, 2, 1]]
    return refined


def write_surf(path: Path, triangles: np.ndarray, title: str,
               custom: np.ndarray | None = None) -> None:
    """Write a deduplicated 3-D SPARTA surface file."""
    vertex_ids: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    connectivity: list[tuple[int, int, int]] = []
    for tri in triangles:
        ids = []
        for xyz in tri:
            key = tuple(round(float(v), 12) for v in xyz)
            if key not in vertex_ids:
                vertex_ids[key] = len(vertices) + 1
                vertices.append(key)
            ids.append(vertex_ids[key])
        connectivity.append(tuple(ids))

    if custom is not None and custom.shape != (len(connectivity), 3):
        raise ValueError("custom sheath values must have shape (ntri, 3)")

    with path.open("w") as handle:
        handle.write(f"# {title}\n")
        if custom is not None:
            handle.write("# triangle custom columns: sheath[Vdc,Vrf,phase_rad]\n")
        handle.write(f"\n{len(vertices)} points\n{len(connectivity)} triangles\n\n")
        handle.write("Points\n\n")
        for i, xyz in enumerate(vertices, 1):
            handle.write(f"{i} {xyz[0]:.12e} {xyz[1]:.12e} {xyz[2]:.12e}\n")
        handle.write("\nTriangles\n\n")
        for i, ids in enumerate(connectivity, 1):
            suffix = ""
            if custom is not None:
                suffix = " " + " ".join(f"{v:.12e}" for v in custom[i - 1])
            handle.write(f"{i} {ids[0]} {ids[1]} {ids[2]}{suffix}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CASE_DIR / "input/config.json")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "input")
    parser.add_argument("--vdc", type=float, help="override target DC voltage [V]")
    parser.add_argument("--vrf", type=float, help="override target RF peak voltage [V]")
    parser.add_argument("--phase", type=float, help="override target RF phase [rad]")
    parser.add_argument("--rings", type=int, help="override target-face radial rings")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    geom = cfg["geometry"]
    sheath = dict(cfg["target_sheath"])
    if args.vdc is not None:
        sheath["vdc_v"] = args.vdc
    if args.vrf is not None:
        sheath["vrf_v"] = args.vrf
    if args.phase is not None:
        sheath["phase_rad"] = args.phase
    if args.rings is not None:
        geom["target_face_radial_rings"] = args.rings
    scale = float(geom["stl_length_to_m"])
    domain_path = REPO_DIR / geom["source_domain_stl"]
    target_path = REPO_DIR / geom["source_target_stl"]

    domain_normals, domain_tri = read_binary_stl(domain_path)
    target_normals, target_tri = read_binary_stl(target_path)
    domain_tri *= scale
    target_tri *= scale

    # The exposed face is the planar +z disk at the maximum target z.
    zmax = float(target_tri[:, :, 2].max())
    face_mask = (np.all(np.isclose(target_tri[:, :, 2], zmax, atol=1.0e-10), axis=1)
                 & (target_normals[:, 2] > 0.9))
    if not np.any(face_mask):
        raise RuntimeError("no +z plasma-facing target triangles were identified")

    target_face_raw = target_tri[face_mask]
    target_face = remesh_planar_face(
        target_face_raw, int(geom["target_face_radial_rings"]))
    target_body = target_tri[~face_mask]
    tile_values = np.column_stack([
        np.full(len(target_face), float(sheath["vdc_v"])),
        np.full(len(target_face), float(sheath["vrf_v"])),
        np.full(len(target_face), float(sheath["phase_rad"])),
    ])
    # Keep a single watertight target for read_surf/check.  Body triangles
    # come first, followed by the biased face ID range.  Body waveform values
    # are zero and never enter the target-only sputter/sheath group.
    target_all = np.concatenate([target_body, target_face], axis=0)
    all_values = np.vstack([np.zeros((len(target_body), 3)), tile_values])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_surf(args.output_dir / "domain.surf", domain_tri,
               "RFPIE simulation-domain shell; source cad/sim_domain.stl; SI metres")
    write_surf(args.output_dir / "target_body.surf", target_body,
               "RFPIE tungsten target sides/back; source exposed_tungsten_sample.stl")
    write_surf(args.output_dir / "target_face.surf", target_face,
               "RFPIE plasma-facing tungsten target tiles", tile_values)
    write_surf(args.output_dir / "target.surf", target_all,
               "RFPIE watertight W target; body IDs then plasma-facing IDs",
               all_values)
    (args.output_dir / "geometry_groups.inc").write_text(
        "# generated by scripts/build_geometry.py\n"
        f"group target_body surf id 1:{len(target_body)}\n"
        f"group target surf id {len(target_body)+1}:{len(target_all)}\n")

    summary = {
        "source_units": "mm",
        "output_units": "m",
        "domain_triangles": int(len(domain_tri)),
        "target_face_triangles": int(len(target_face)),
        "target_face_source_triangles": int(len(target_face_raw)),
        "target_face_radial_rings": int(geom["target_face_radial_rings"]),
        "target_body_triangles": int(len(target_body)),
        "target_body_id_range": [1, int(len(target_body))],
        "target_face_id_range": [int(len(target_body) + 1), int(len(target_all))],
        "target_face_area_m2": float(triangle_areas(target_face).sum()),
        "target_z_m": zmax,
        "domain_bounds_m": [domain_tri.reshape(-1, 3).min(axis=0).tolist(),
                            domain_tri.reshape(-1, 3).max(axis=0).tolist()],
        "target_bounds_m": [target_tri.reshape(-1, 3).min(axis=0).tolist(),
                            target_tri.reshape(-1, 3).max(axis=0).tolist()],
        "sheath_columns": ["Vdc_V", "Vrf_peak_V", "phase_rad"],
        "sheath_values": [float(sheath["vdc_v"]), float(sheath["vrf_v"]),
                           float(sheath["phase_rad"])],
    }
    (args.output_dir / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
