
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

def parse_sparta_surface(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    try:
        pts_idx = lines.index("Points") + 1
        tri_idx = lines.index("Triangles") + 1
    except ValueError as e:
        raise RuntimeError("Could not find 'Points' or 'Triangles' section") from e

    pts = []
    pt_ids = []
    i = pts_idx
    while i < tri_idx - 1:
        parts = lines[i].split()
        if len(parts) == 4:
            _id = int(parts[0])
            x, y, z = map(float, parts[1:])
            pt_ids.append(_id)
            pts.append((x, y, z))
        i += 1

    tris = []
    tri_ids = []
    for j in range(tri_idx, len(lines)):
        parts = lines[j].split()
        if len(parts) == 4:
            _id = int(parts[0])
            i1, i2, i3 = map(int, parts[1:])
            # SPARTA points are 1-based → convert to 0-based
            tris.append((i1 - 1, i2 - 1, i3 - 1))
            tri_ids.append(_id)

    if not pts or not tris:
        raise RuntimeError("Parsed zero points or zero triangles; check file format.")

    return (np.array(pts, dtype=float),
            np.array(tris, dtype=int),
            np.array(pt_ids, dtype=int),
            np.array(tri_ids, dtype=int))


def plot_surface(pts, tris, save=None, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
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
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def triangle_normals_and_centroids(pts, tris):
    # pts: (N,3), tris: (M,3)
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    # unnormalized normals
    n = np.cross(v1 - v0, v2 - v0)
    # normalize
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_norm[n_norm == 0] = 1.0
    n = n / n_norm
    # centroids
    c = (v0 + v1 + v2) / 3.0
    return n, c


if __name__ == "__main__":

    pts_t, tris_t, pt_ids_t, tri_ids_t = parse_sparta_surface("target_0deg.txt")
    normals_t, centroids_t = triangle_normals_and_centroids(pts_t, tris_t)

    # direction from target toward skimmer: along -z
    d = np.array([0.0, 0.0, -1.0])

    # 1) normals roughly aligned with -z
    cos_thresh = 0.9   # ~25° cone
    face_mask = (normals_t @ d) > cos_thresh

    # 2) restrict to the front plane in z (avoid edges/back)
    z_c = centroids_t[:, 2]
    z_front = np.min(z_c)            # or np.max, depending on which side is facing skimmer
    tol = 1e-4
    plane_mask = np.abs(z_c - z_front) < tol

    mask = face_mask & plane_mask

    facing_tri_ids = tri_ids_t[mask]
    print("Triangles facing skimmer:", len(np.array(facing_tri_ids)))

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection='3d')

    # all target tris (faint)
    polys_all = [pts_t[t] for t in tris_t]
    poly_all = Poly3DCollection(polys_all, alpha=0.2, linewidths=0.2)
    ax.add_collection3d(poly_all)

    # facing-skimmer tris (bright)
    polys_face = [pts_t[t] for t in tris_t[mask]]
    poly_face = Poly3DCollection(polys_face, alpha=0.9, linewidths=0.5)
    poly_face.set_edgecolor("k")
    poly_face.set_facecolor("orange")
    ax.add_collection3d(poly_face)

    plt.show()
    
    # 321 1260

