#!/usr/bin/env python3
"""
Diagnostic: plot plasma field values mapped onto the 3D wedge surface.

Reads the surface geometry (.surf file) and plasma.h5, computes R=sqrt(x²+y²)
and Z=z for each triangle centroid, interpolates ne/Te/flux from the 2D grid,
and plots to verify the mapping is correct.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator


def read_sparta_surf(path):
    """Read SPARTA triangle surface file. Returns points (Npts,3) and tris (Ntri,3)."""
    with open(path) as f:
        lines = f.readlines()

    npoints = ntris = 0
    reading = None
    points = []
    tris = []

    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if "points" in tok and len(tok) == 2:
            npoints = int(tok[0])
            continue
        if "triangles" in tok and len(tok) == 2:
            ntris = int(tok[0])
            continue
        if tok[0] == "Points":
            reading = "points"
            continue
        if tok[0] == "Triangles":
            reading = "tris"
            continue

        if reading == "points" and len(tok) >= 4:
            points.append([float(tok[1]), float(tok[2]), float(tok[3])])
        elif reading == "tris" and len(tok) >= 4:
            tris.append([int(tok[1]) - 1, int(tok[2]) - 1, int(tok[3]) - 1])

    return np.array(points), np.array(tris)


def main():
    surf_file = "input/geometry/d3d_refined_wedge_capped.surf"
    plasma_file = "input/data/plasma.h5"

    pts, tris = read_sparta_surf(surf_file)
    print(f"Surface: {len(pts)} points, {len(tris)} triangles")

    # Triangle centroids in Cartesian
    centroids = np.array([pts[tri].mean(axis=0) for tri in tris])
    xc, yc, zc = centroids[:, 0], centroids[:, 1], centroids[:, 2]

    # Map to cylindrical
    R = np.sqrt(xc**2 + yc**2)
    Z = zc

    print(f"Centroid R range: [{R.min():.4f}, {R.max():.4f}]")
    print(f"Centroid Z range: [{Z.min():.4f}, {Z.max():.4f}]")

    # Load plasma data
    with h5py.File(plasma_file, "r") as f:
        r = f["r"][...]
        z = f["z"][...]
        ne = f["dens_e"][...]
        te = f["temp_e"][...]
        ni = f["dens_i"][...]
        vpar = f["parr_flow"][...]

    print(f"Plasma r range: [{r[0]:.4f}, {r[-1]:.4f}]")
    print(f"Plasma z range: [{z[0]:.4f}, {z[-1]:.4f}]")

    # Interpolate with bounds_error=False, fill_value=0 (same as C++ fix)
    ne_interp = RegularGridInterpolator((z, r), ne, method="linear",
                                         bounds_error=False, fill_value=0.0)
    te_interp = RegularGridInterpolator((z, r), te, method="linear",
                                         bounds_error=False, fill_value=0.0)
    ni_interp = RegularGridInterpolator((z, r), ni, method="linear",
                                         bounds_error=False, fill_value=0.0)
    vpar_interp = RegularGridInterpolator((z, r), vpar, method="linear",
                                           bounds_error=False, fill_value=0.0)

    query = np.column_stack((Z, R))
    ne_surf = ne_interp(query)
    te_surf = te_interp(query)
    ni_surf = ni_interp(query)
    vpar_surf = vpar_interp(query)
    flux_surf = np.abs(ni_surf * vpar_surf)

    # Count in/out of bounds
    in_r = (R >= r[0]) & (R <= r[-1])
    in_z = (Z >= z[0]) & (Z <= z[-1])
    in_bounds = in_r & in_z
    print(f"\nTriangles in plasma grid:  {in_bounds.sum()} / {len(tris)}")
    print(f"Triangles out of bounds:   {(~in_bounds).sum()} / {len(tris)}")
    print(f"  R out of range: {(~in_r).sum()}")
    print(f"  Z out of range: {(~in_z).sum()}")

    # Separate wall (id 1-136) and caps (137-268)
    wall_mask = np.arange(len(tris)) < 136
    cap_mask = ~wall_mask

    print(f"\nWall triangles with non-zero flux: {(flux_surf[wall_mask] > 0).sum()} / {wall_mask.sum()}")
    print(f"Cap triangles with non-zero flux:  {(flux_surf[cap_mask] > 0).sum()} / {cap_mask.sum()}")

    # ---- Plot 1: R-Z scatter of triangle centroids colored by log10(ne) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for ax, (vals, title, cmap) in zip(axes.flat, [
        (np.where(ne_surf > 0, np.log10(ne_surf), np.nan), "log10(ne) on surface", "inferno"),
        (te_surf, "Te [eV] on surface", "magma"),
        (np.where(flux_surf > 0, np.log10(flux_surf), np.nan), "log10(|ni*vpar|) on surface", "hot"),
        (in_bounds.astype(float), "In plasma grid (1=yes, 0=no)", "RdYlGn"),
    ]):
        sc = ax.scatter(R, Z, c=vals, s=8, cmap=cmap, edgecolors="none")
        # Draw plasma grid boundary
        ax.axvline(r[0], color="cyan", ls="--", lw=0.8, label="plasma grid")
        ax.axvline(r[-1], color="cyan", ls="--", lw=0.8)
        ax.axhline(z[0], color="cyan", ls="--", lw=0.8)
        ax.axhline(z[-1], color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)

    fig.savefig("output/surf_flux_diagnostic.png", dpi=180)
    plt.close(fig)
    print(f"\nWrote output/surf_flux_diagnostic.png")

    # ---- Plot 2: 2D plasma field with surface overlay ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    extent = [r[0], r[-1], z[0], z[-1]]

    ax = axes[0]
    log_ne = np.where(ne > 0, np.log10(ne), np.nan)
    im = ax.imshow(log_ne, origin="lower", extent=extent, aspect="auto", cmap="inferno")
    ax.scatter(R[wall_mask], Z[wall_mask], c="lime", s=3, label="wall tris", zorder=5)
    ax.scatter(R[cap_mask], Z[cap_mask], c="red", s=3, label="cap tris", zorder=5)
    ax.legend(fontsize=8)
    ax.set_title("log10(ne) + surface centroids")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    log_flux = np.where(np.abs(ni * vpar) > 0, np.log10(np.abs(ni * vpar)), np.nan)
    im = ax.imshow(log_flux, origin="lower", extent=extent, aspect="auto", cmap="hot")
    ax.scatter(R[wall_mask], Z[wall_mask], c="lime", s=3, label="wall tris", zorder=5)
    ax.set_title("log10(|ni*vpar|) + wall centroids")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.savefig("output/plasma_vs_surface.png", dpi=180)
    plt.close(fig)
    print(f"Wrote output/plasma_vs_surface.png")


if __name__ == "__main__":
    main()
