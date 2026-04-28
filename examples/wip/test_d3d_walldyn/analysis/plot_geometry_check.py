#!/usr/bin/env python3
"""
Geometry diagnostic: overlay SOLPS 2D wall, OpenEdge 3D surface (projected to R,Z),
and plasma fields to check alignment.

This answers:
1. Does the 3D wedge wall match the SOLPS wall boundary in R,Z?
2. Where do the surface normals point relative to the plasma?
3. Which surface elements see positive v_normal (incident flux)?
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def read_sparta_2d_wall(path):
    """Read 2D SPARTA wall file (R,Z lines)."""
    pts = []
    reading = False
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "Points":
                reading = True
                continue
            if tok[0] == "Lines":
                break
            if reading and len(tok) >= 3:
                pts.append([float(tok[1]), float(tok[2])])
    pts = np.array(pts)
    # Close the polygon
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    return pts


def read_sparta_3d_surf(path):
    """Read 3D SPARTA triangle surface. Returns points (N,3) and tris (M,3)."""
    points, tris = [], []
    reading = None
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "Points":
                reading = "points"; continue
            if tok[0] == "Triangles":
                reading = "tris"; continue
            if reading == "points" and len(tok) >= 4:
                points.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif reading == "tris" and len(tok) >= 4:
                tris.append([int(tok[1])-1, int(tok[2])-1, int(tok[3])-1])
    return np.array(points), np.array(tris)


def triangle_normals(pts, tris):
    """Compute outward normals for each triangle."""
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-30] = 1.0
    return normals / norms


def main():
    wall_2d_file = "input/data/wall.txt"
    surf_3d_file = "input/geometry/d3d_refined_wedge_capped.surf"
    plasma_file = "input/data/plasma.h5"

    # Read geometries
    wall_rz = read_sparta_2d_wall(wall_2d_file)
    pts_3d, tris_3d = read_sparta_3d_surf(surf_3d_file)
    normals_3d = triangle_normals(pts_3d, tris_3d)

    # Centroids in 3D
    centroids = np.array([pts_3d[t].mean(axis=0) for t in tris_3d])
    xc, yc, zc = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    R = np.sqrt(xc**2 + yc**2)
    Z = zc
    phi = np.degrees(np.arctan2(yc, xc))

    # Project normals to cylindrical (nr, nz)
    rr = np.maximum(R, 1e-12)
    cos_phi = xc / rr
    sin_phi = yc / rr
    nr_cyl = normals_3d[:, 0] * cos_phi + normals_3d[:, 1] * sin_phi  # R component
    nz_cyl = normals_3d[:, 2]  # Z component
    nt_cyl = -normals_3d[:, 0] * sin_phi + normals_3d[:, 1] * cos_phi  # toroidal

    # Separate wall (id 1-136) and caps (137-268)
    wall_mask = np.arange(len(tris_3d)) < 136
    cap_mask = ~wall_mask

    # Load plasma
    with h5py.File(plasma_file, "r") as f:
        r = f["r"][...]
        z = f["z"][...]
        ne = f["dens_e"][...]
        ni = f["dens_i"][...]
        vpar = f["parr_flow"][...]
        vpar_r = f["parr_flow_r"][...]
        vpar_z = f["parr_flow_z"][...]

    ni_interp = RegularGridInterpolator((z, r), ni, bounds_error=False, fill_value=0.0)
    vr_interp = RegularGridInterpolator((z, r), vpar_r, bounds_error=False, fill_value=0.0)
    vz_interp = RegularGridInterpolator((z, r), vpar_z, bounds_error=False, fill_value=0.0)

    query = np.column_stack((Z, R))
    ni_s = ni_interp(query)
    vr_s = vr_interp(query)
    vz_s = vz_interp(query)

    # v_normal = v . n_surf (positive = into wall)
    v_normal = -(vr_s * nr_cyl + vz_s * nz_cyl)
    incident_flux = np.where(v_normal > 0, ni_s * v_normal, 0.0)

    # ---- PLOTS ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    extent = [r[0], r[-1], z[0], z[-1]]

    # Panel 1: Wall alignment check
    ax = axes[0, 0]
    log_ne = np.where(ne > 0, np.log10(ne), np.nan)
    ax.imshow(log_ne, origin="lower", extent=extent, aspect="auto", cmap="inferno", alpha=0.7)
    ax.plot(wall_rz[:, 0], wall_rz[:, 1], "c-", lw=2.5, label="SOLPS wall (2D)")
    ax.scatter(R[wall_mask], Z[wall_mask], c="lime", s=6, zorder=5, label="3D wall tris")
    ax.scatter(R[cap_mask], Z[cap_mask], c="red", s=4, zorder=5, alpha=0.5, label="3D cap tris")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Wall alignment: SOLPS 2D vs 3D wedge")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    # Panel 2: Surface normals (R,Z components) on wall triangles
    ax = axes[0, 1]
    ax.plot(wall_rz[:, 0], wall_rz[:, 1], "k-", lw=1.5)
    q = ax.quiver(R[wall_mask], Z[wall_mask], nr_cyl[wall_mask], nz_cyl[wall_mask],
                  scale=20, color="blue", width=0.003, headwidth=4)
    ax.set_title("Surface normals (R,Z) — wall triangles")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")

    # Panel 3: v_normal on wall (positive = into wall, negative = away)
    ax = axes[0, 2]
    ax.plot(wall_rz[:, 0], wall_rz[:, 1], "k-", lw=1.5)
    vmax = np.nanmax(np.abs(v_normal[wall_mask]))
    if vmax == 0: vmax = 1
    sc = ax.scatter(R[wall_mask], Z[wall_mask], c=v_normal[wall_mask], s=15,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, edgecolors="none")
    ax.set_title("v_normal on wall (red=into wall, blue=away)")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="v_normal [m/s]")

    # Panel 4: Incident flux (only positive v_normal)
    ax = axes[1, 0]
    ax.plot(wall_rz[:, 0], wall_rz[:, 1], "k-", lw=1.5)
    log_flux = np.full(wall_mask.sum(), np.nan)
    m = incident_flux[wall_mask] > 0
    log_flux[m] = np.log10(incident_flux[wall_mask][m])
    sc = ax.scatter(R[wall_mask], Z[wall_mask], c=log_flux, s=15, cmap="hot", edgecolors="none")
    ax.set_title("log10(incident flux) — wall only")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="log10(ni*v_normal)")

    # Panel 5: Toroidal angle distribution
    ax = axes[1, 1]
    ax.scatter(phi[wall_mask], Z[wall_mask], c=incident_flux[wall_mask] > 0,
               cmap="RdYlGn", s=8)
    ax.set_title("Wall tris: phi vs Z (green = has flux)")
    ax.set_xlabel("phi [deg]"); ax.set_ylabel("Z [m]")

    # Panel 6: Cap normal check (should be purely toroidal)
    ax = axes[1, 2]
    ax.bar(["nr", "nz", "nt"], [
        np.mean(np.abs(nr_cyl[cap_mask])),
        np.mean(np.abs(nz_cyl[cap_mask])),
        np.mean(np.abs(nt_cyl[cap_mask])),
    ])
    ax.set_title("Cap normals (should be mostly toroidal)")
    ax.set_ylabel("mean |component|")

    # Stats
    print(f"Wall triangles: {wall_mask.sum()}")
    print(f"  with v_normal > 0 (incident): {(v_normal[wall_mask] > 0).sum()}")
    print(f"  with incident flux > 0:       {(incident_flux[wall_mask] > 0).sum()}")
    print(f"  max incident flux:            {incident_flux[wall_mask].max():.3e}")
    print(f"Cap triangles: {cap_mask.sum()}")
    print(f"  with incident flux > 0:       {(incident_flux[cap_mask] > 0).sum()}")

    fig.savefig("output/geometry_check.png", dpi=180)
    plt.close(fig)
    print(f"\nWrote output/geometry_check.png")


if __name__ == "__main__":
    main()
