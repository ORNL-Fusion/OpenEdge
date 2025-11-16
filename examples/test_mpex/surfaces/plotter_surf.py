
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

def Te_profile(x, y):
    """
    Analytic Te(r) from the MPEX challenge:
        Te(r) = (1 eV) + (4 eV) * exp(-(r / 2 cm)^12)
    r in meters, 2 cm = 0.02 m
    """
    r = np.sqrt(x**2 + y**2)
    R0 = 0.02  # 2 cm
    Te = 1.0 + 4.0 * np.exp(- (r / R0)**12)
    return Te  # in eV


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

def write_surface_with_Te(in_path, out_path):
    """
    Read a SPARTA surface file (Points/Triangles format),
    compute Te at each triangle centroid, and write a new file
    where each Triangles line has a 5th column = Te [eV].

    Output 'Points' section is unchanged; only 'Triangles' lines are extended.
    """
    # --- read raw lines (to preserve headers, comments, etc.) ---
    with open(in_path, "r") as f:
        raw_lines = f.readlines()

    # --- parse points & triangles, get IDs ---
    pts, tris, pt_ids, tri_ids = parse_sparta_surface(in_path)

    # triangle centroids and Te for each tri (no masking)
    normals, centroids = triangle_normals_and_centroids(pts, tris)
    Te_all = Te_profile(centroids[:, 0], centroids[:, 1])  # shape (ntris,)

    # find where 'Triangles' section starts
    lines_stripped = [ln.strip() for ln in raw_lines]
    try:
        tri_header_idx = lines_stripped.index("Triangles")  # index of the word 'Triangles'
    except ValueError as e:
        raise RuntimeError("Could not find 'Triangles' section in file") from e

    tri_start = tri_header_idx + 1  # first triangle line in raw_lines

    # --- write out new file ---
    with open(out_path, "w") as g:
        # write everything up to and including 'Triangles' unchanged
        for i in range(tri_start):
            g.write(raw_lines[i])

        # now rewrite each triangle line with Te appended
        for j in range(len(tris)):
            # original triangle line
            parts = raw_lines[tri_start + j].split()
            if len(parts) < 4:
                # skip weird lines (shouldn't happen in a clean file)
                continue

            tri_id = int(parts[0])
            i1, i2, i3 = parts[1], parts[2], parts[3]

            Te = Te_all[j]
            # new line: id   i1 i2 i3   Te
            g.write(f"{tri_id} {i1} {i2} {i3} {Te:.6g}\n")

        # if there was anything after the Triangles block, write it as-is
        for i in range(tri_start + len(tris), len(raw_lines)):
            g.write(raw_lines[i])



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
    angle = (0, 45, 85, ...)
    for angle in [0,45,85]:
    
        tol_angle = 2.0  # ± window in degrees


        in_surf  = f"target_{int(angle)}deg.txt"
        out_surf = f"target_{int(angle)}deg_Te.txt"
            
            
        pts_t, tris_t, pt_ids_t, tri_ids_t = parse_sparta_surface(in_surf)
        normals_t, centroids_t = triangle_normals_and_centroids(pts_t, tris_t)

        # Te at each triangle centroid
        Te_all = Te_profile(centroids_t[:, 0], centroids_t[:, 1])

        # direction from target toward skimmer: along -z
        d = np.array([0.0, 0.0, -1.0])
        d = d / np.linalg.norm(d)

        # angle between triangle normal and -z, in degrees
        cosang = normals_t @ d           # n · d
        cosang = np.clip(cosang, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cosang))

        # front side = normals that point at least somewhat toward skimmer
        front_mask = cosang > 0.0

        # triangles whose normal makes ~angle degrees with -z
        mask = front_mask & (np.abs(angles_deg - angle) < tol_angle)

        print(f"N({angle} deg) =", np.count_nonzero(mask))
        facing_tri_ids = tri_ids_t[mask]
        print("Triangles total and facing used:", len(tri_ids_t), len(facing_tri_ids))
        print(np.min(facing_tri_ids), np.max(facing_tri_ids))
        
        Te_face = Te_all[mask]

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111, projection='3d')

        # all tris (faint)
        polys_all = [pts_t[t] for t in tris_t]
        poly_all = Poly3DCollection(polys_all, alpha=0.1, linewidths=0.2)
        poly_all.set_facecolor("lightgrey")
        ax.add_collection3d(poly_all)

        # selected tris, colored by Te
        polys_face = [pts_t[t] for t in tris_t[mask]]
        norm = plt.Normalize(vmin=Te_face.min(), vmax=Te_face.max())
        cmap = plt.cm.inferno
        colors = cmap(norm(Te_face))

        poly_face = Poly3DCollection(polys_face, linewidths=0.5)
        poly_face.set_edgecolor("k")
        poly_face.set_facecolor(colors)
        ax.add_collection3d(poly_face)

        xyz_min = pts_t.min(axis=0)
        xyz_max = pts_t.max(axis=0)
        ranges = xyz_max - xyz_min
        max_range = ranges.max()
        mid = 0.5 * (xyz_min + xyz_max)
        ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
        ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
        ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Target triangles at ~{angle}° to -z, colored by Te")

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(Te_face)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.05)
        cbar.set_label("T_e [eV]")

    #    plt.tight_layout()
    #    plt.show()

    #    exit()
        

        # (Optional) plotting stuff you already have...
        pts_t, tris_t, pt_ids_t, tri_ids_t = parse_sparta_surface(in_surf)
        normals_t, centroids_t = triangle_normals_and_centroids(pts_t, tris_t)
        # ... your Te_all, masks, plotting, etc ...

        # finally write a new surf file with Te appended after each triangle
        write_surface_with_Te(in_surf, out_surf)
        print(f"Wrote surface with Te to {out_surf}")



        # 321 1260

