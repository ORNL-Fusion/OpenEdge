#!/usr/bin/env python3
"""Populate /mesh/wall_surf_cell in an existing plasma.h5 from a wall.surf.

For each wall segment (one per line in wall.surf), find the SOLPS mesh
triangle whose barycenter is closest to the segment centroid, then look
up that triangle's /mesh/cell_index. Write the per-segment SOLPS cell
index back to the same plasma.h5 as /mesh/wall_surf_cell.

OpenEdge's compute surface/physical/sputter prefers this map over its
nearest-triangle fallback because it identifies the SOLPS cell
ADJACENT TO THE WALL (matching what SOLPS calls the target volume cell)
rather than the cell containing the wall.surf segment centroid (which
may be one cell upstream when wall.surf sits a few mm outside the
SOLPS mesh boundary).

Usage:  python3 add_wall_surf_cell.py <plasma.h5> <wall.surf>
"""

from pathlib import Path
import sys
import numpy as np
import h5py
from scipy.spatial import cKDTree


def parse_wall(path):
    text = Path(path).read_text().splitlines()
    pts, lns, sec = {}, [], None
    for line in text:
        s = line.strip()
        if s == "Points": sec = "p"; continue
        if s == "Lines":  sec = "l"; continue
        if not s or not s[0].isdigit(): continue
        tok = s.split()
        if sec == "p":
            pts[int(tok[0])] = (float(tok[1]), float(tok[2]))   # (Z, R)
        elif sec == "l":
            lns.append((int(tok[0]), int(tok[1]), int(tok[2])))
    return pts, lns


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: add_wall_surf_cell.py <plasma.h5> <wall.surf>")
    plasma_h5 = Path(sys.argv[1])
    wall_surf = Path(sys.argv[2])
    if not plasma_h5.exists() or not wall_surf.exists():
        sys.exit(f"missing input(s)")

    with h5py.File(plasma_h5, 'r') as f:
        vr = f['mesh/vtx_r'][:]
        vz = f['mesh/vtx_z'][:]
        tri = f['mesh/triangles'][:]
        cell_index = f['mesh/cell_index'][:]
    bary = np.column_stack([
        (vr[tri[:, 0]] + vr[tri[:, 1]] + vr[tri[:, 2]]) / 3.0,
        (vz[tri[:, 0]] + vz[tri[:, 1]] + vz[tri[:, 2]]) / 3.0])
    valid = cell_index >= 0
    tree = cKDTree(bary[valid])
    valid_idx = np.where(valid)[0]

    pts, lns = parse_wall(wall_surf)
    n = len(lns)
    # wall.surf segment IDs are 1-based; we store an array indexed by
    # (id - 1). If IDs aren't contiguous 1..N we still fill the array of
    # size max(id) for safe direct indexing.
    max_id = max(sid for sid, _, _ in lns)
    out = np.full(max_id, -1, dtype=np.int32)
    for sid, p1, p2 in lns:
        z = 0.5 * (pts[p1][0] + pts[p2][0])
        r = 0.5 * (pts[p1][1] + pts[p2][1])
        # tree is over (R, Z); query at (R, Z) of segment centroid.
        _, k = tree.query([r, z])
        tri_global = int(valid_idx[k])
        out[sid - 1] = int(cell_index[tri_global])

    print(f"populated wall_surf_cell: {n} segments, "
          f"valid={(out >= 0).sum()}/{out.size}, "
          f"cell range [{out[out >= 0].min()}, {out[out >= 0].max()}]")

    with h5py.File(plasma_h5, 'r+') as f:
        if 'mesh/wall_surf_cell' in f:
            del f['mesh/wall_surf_cell']
        f.create_dataset('mesh/wall_surf_cell', data=out)
    print(f"wrote /mesh/wall_surf_cell to {plasma_h5}")


if __name__ == "__main__":
    main()
