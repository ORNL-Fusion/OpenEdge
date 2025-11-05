#!/usr/bin/env python3
"""
Plot a SPARTA-style surface file with 'Points' and 'Triangles' sections.

Usage:
    python plot_surface3d.py -i flatSurface.txt [-o out.png]

This script reads the surface file, parses the points and triangles,
and renders a 3D triangulated mesh using matplotlib.

Author: ChatGPT
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_sparta_surface(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    try:
        pts_idx = lines.index("Points") + 1
        tri_idx = lines.index("Triangles") + 1
    except ValueError as e:
        raise RuntimeError("Could not find 'Points' or 'Triangles' section") from e

    pts = []
    i = pts_idx
    while i < tri_idx - 1:
        parts = lines[i].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                x, y, z = map(float, parts[1:])
                pts.append((x, y, z))
            except ValueError:
                pass
        i += 1

    tris = []
    for j in range(tri_idx, len(lines)):
        parts = lines[j].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                i1, i2, i3 = map(int, parts[1:])
                tris.append((i1 - 1, i2 - 1, i3 - 1))
            except ValueError:
                pass

    if not pts or not tris:
        raise RuntimeError("Parsed zero points or zero triangles; check file format.")
    return np.array(pts, dtype=float), np.array(tris, dtype=int)



if __name__ == "__main__":
    pts, tris = parse_sparta_surface("surfaces/flatSurface.txt")
#    plot_surface(pts, tris)
    
#    def plot_surface(pts, tris, save=None, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize=(8, 6)))
    ax = fig.add_subplot(111, projection='3d')

    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    ranges = xyz_max - xyz_min
    max_range = max(ranges)
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface mesh (SPARTA)")
    plt.show()
    
#    main()

