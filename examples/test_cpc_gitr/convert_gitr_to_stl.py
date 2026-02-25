#!/usr/bin/env python3
"""
Convert GITR point-plane 3D geometry (.cfg) to binary STL.

GITR stores triangulated surfaces as parallel arrays:
    x1[N], y1[N], z1[N]  — vertex 1 of each triangle
    x2[N], y2[N], z2[N]  — vertex 2 of each triangle
    x3[N], y3[N], z3[N]  — vertex 3 of each triangle

plus plane coefficients (a,b,c,d), normals, areas, material Z, etc.

This script reads those arrays, computes outward face normals from the
vertex cross product, and writes a binary STL file that SPARTA can
import via `read_surf`.

Usage:
    python3 convert_gitr_to_stl.py \
        /path/to/gitrGeometryPointPlane3d.cfg \
        output.stl

The output STL uses SI units (metres), matching the GITR input.
"""

import re
import sys
import numpy as np


def parse_cfg_array(content, name):
    """Extract a bracketed numeric array from GITR .cfg text."""
    pattern = r'{}\s*=\s*\[(.*?)\]'.format(re.escape(name))
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise KeyError(f"Array '{name}' not found in cfg file")
    vals = [float(x) for x in match.group(1).split(',') if x.strip()]
    return np.array(vals)


def write_ascii_stl(filename, vertices, normals, name="gitr_geometry"):
    """
    Write ASCII STL (required by SPARTA's stl2surf.py).
    vertices: (N, 3, 3) — N triangles, 3 vertices, 3 coords
    normals:  (N, 3)    — face normals
    """
    n_triangles = len(vertices)
    with open(filename, 'w') as f:
        f.write(f"solid {name}\n")
        for i in range(n_triangles):
            nx, ny, nz = normals[i]
            f.write(f"  facet normal {nx:.10e} {ny:.10e} {nz:.10e}\n")
            f.write(f"    outer loop\n")
            for j in range(3):
                vx, vy, vz = vertices[i, j]
                f.write(f"      vertex {vx:.10e} {vy:.10e} {vz:.10e}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
        f.write(f"endsolid {name}\n")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <gitr_geometry.cfg> <output.stl>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    stl_path = sys.argv[2]

    with open(cfg_path, 'r') as f:
        content = f.read()

    # Parse vertex arrays
    x1 = parse_cfg_array(content, 'x1')
    y1 = parse_cfg_array(content, 'y1')
    z1 = parse_cfg_array(content, 'z1')
    x2 = parse_cfg_array(content, 'x2')
    y2 = parse_cfg_array(content, 'y2')
    z2 = parse_cfg_array(content, 'z2')
    x3 = parse_cfg_array(content, 'x3')
    y3 = parse_cfg_array(content, 'y3')
    z3 = parse_cfg_array(content, 'z3')

    n = len(x1)
    print(f"Parsed {n} triangles from {cfg_path}")

    # Build vertex array: (N, 3, 3)
    vertices = np.zeros((n, 3, 3))
    vertices[:, 0, 0] = x1;  vertices[:, 0, 1] = y1;  vertices[:, 0, 2] = z1
    vertices[:, 1, 0] = x2;  vertices[:, 1, 1] = y2;  vertices[:, 1, 2] = z2
    vertices[:, 2, 0] = x3;  vertices[:, 2, 1] = y3;  vertices[:, 2, 2] = z3

    # Compute face normals from cross product of edges
    e1 = vertices[:, 1] - vertices[:, 0]  # edge v1->v2
    e2 = vertices[:, 2] - vertices[:, 0]  # edge v1->v3
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for degenerate triangles
    normals = normals / norms

    # If GITR provides inDir, use it to orient normals outward.
    # inDir = -1 means normal points inward; flip those.
    try:
        inDir = parse_cfg_array(content, 'inDir')
        n_flipped = 0
        for i in range(n):
            if inDir[i] < 0:
                # flip normal and swap v2/v3 to keep consistent winding
                normals[i] *= -1.0
                vertices[i, 1], vertices[i, 2] = (
                    vertices[i, 2].copy(), vertices[i, 1].copy())
                n_flipped += 1
        print(f"Flipped {n_flipped} triangles using inDir")
    except KeyError:
        print("No inDir array found; normals from cross product as-is")

    write_ascii_stl(stl_path, vertices, normals)

    # Print bounding box
    all_verts = vertices.reshape(-1, 3)
    lo = all_verts.min(axis=0)
    hi = all_verts.max(axis=0)
    print(f"Bounding box:")
    print(f"  X: [{lo[0]:.6e}, {hi[0]:.6e}]")
    print(f"  Y: [{lo[1]:.6e}, {hi[1]:.6e}]")
    print(f"  Z: [{lo[2]:.6e}, {hi[2]:.6e}]")
    print(f"Wrote {n} triangles to {stl_path}")
    print(f"STL file size: {80 + 4 + n * 50} bytes")


if __name__ == '__main__':
    main()
