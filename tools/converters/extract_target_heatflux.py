#!/usr/bin/env python3
"""
Extract target heat flux and particle flux from SOLPS balance.nc.

Computes the heat and particle flux deposited on divertor target plates
by summing the face-centered fluxes at the target boundary cells.
This is equivalent to the SOLPS `wlld` b2plot command.

The target cells are at:
  - Outer target: ix = 0..nx-1, iy = 0  (poloidal face in y-direction)
  - Inner target: ix = 0..nx-1, iy = ny  (poloidal face in y-direction)

Output: HDF5 file with target heat flux profiles (s [m], q [W/m²], Gamma_D+ [m⁻²s⁻¹])
mapped along the target arc length.

Usage:
    python extract_target_heatflux.py /path/to/solps_run --out target_heatflux.h5
    python extract_target_heatflux.py /path/to/solps_run --out target_heatflux.h5 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def extract_target_profiles(nc_path: str | Path):
    """Extract target heat flux and particle flux from balance.nc.

    Returns dict with keys:
      outer/inner -> {s, R, Z, q_total, q_electron, q_ion, gamma_D}
    where s is arc length along the target [m].
    """
    from netCDF4 import Dataset

    ds = Dataset(str(nc_path), "r")

    nx = len(ds.dimensions["nx_plus2"]) - 2
    ny = len(ds.dimensions["ny_plus2"]) - 2

    # Cell corner coordinates: crx, cry shape (5, ny+2, nx+2)
    crx = np.array(ds.variables["crx"])
    cry = np.array(ds.variables["cry"])

    # Face areas: hy shape (ny+2, nx+2) — radial face area (target-normal)
    hy = np.array(ds.variables["hy"])

    # --- Heat fluxes at faces ---
    # fhe: electron heat flux through faces (ny+2, nx+2, 2)
    #   [:,:,0] = poloidal (x) face, [:,:,1] = radial (y) face
    # fhi: ion heat flux through faces
    fhe = np.array(ds.variables["fhe"])
    fhi = np.array(ds.variables["fhi"])

    # Reshape if needed (some versions store flat)
    if fhe.ndim == 2:
        fhe = fhe.reshape(ny + 2, nx + 2, -1)
    if fhi.ndim == 2:
        fhi = fhi.reshape(ny + 2, nx + 2, -1)

    # Total heat flux through radial (y) faces
    fht_y_e = fhe[:, :, 1]  # electron, y-face [W]
    fht_y_i = fhi[:, :, 1]  # ion, y-face [W]

    # --- Particle flux at faces ---
    # fna: particle flux through faces (ns, ny+2, nx+2, 2)
    fna = np.array(ds.variables["fna"])
    if fna.ndim == 3:
        fna = fna.reshape(-1, ny + 2, nx + 2, 2)
    # D+ is species index 1
    fna_D_y = fna[1, :, :, 1]  # D+ flux through y-faces [particles/s]

    # Radial face areas at targets
    # hy[iy, ix] is the face area for the radial face at (ix, iy)
    s_interior = np.s_[1:-1]  # strip guard cells in x

    results = {}

    for target, iy_face in [("outer", 1), ("inner", ny)]:
        # Target face: iy_face = 1 for outer (bottom boundary),
        #              iy_face = ny for inner (top boundary)
        # Heat flux density = face flux [W] / face area [m²]
        area = hy[iy_face, s_interior]
        area = np.where(area > 1e-20, area, np.nan)

        q_e = np.abs(fht_y_e[iy_face, s_interior]) / area  # W/m²
        q_i = np.abs(fht_y_i[iy_face, s_interior]) / area
        q_total = q_e + q_i

        # Particle flux density
        gamma_D = np.abs(fna_D_y[iy_face, s_interior]) / area  # m⁻²s⁻¹

        # Target cell midpoints (R, Z) from corners
        # Use corners 0,1 (bottom) for outer target, 2,3 (top) for inner
        if target == "outer":
            R_pts = 0.5 * (crx[0, iy_face, s_interior] + crx[1, iy_face, s_interior])
            Z_pts = 0.5 * (cry[0, iy_face, s_interior] + cry[1, iy_face, s_interior])
        else:
            R_pts = 0.5 * (crx[2, iy_face, s_interior] + crx[3, iy_face, s_interior])
            Z_pts = 0.5 * (cry[2, iy_face, s_interior] + cry[3, iy_face, s_interior])

        # Arc length along target
        dR = np.diff(R_pts)
        dZ = np.diff(Z_pts)
        ds = np.sqrt(dR**2 + dZ**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])

        # Clean NaNs
        mask = np.isfinite(q_total) & np.isfinite(gamma_D)
        results[target] = {
            "s": s[mask],
            "R": R_pts[mask],
            "Z": Z_pts[mask],
            "q_total": q_total[mask],
            "q_electron": q_e[mask],
            "q_ion": q_i[mask],
            "gamma_D": gamma_D[mask],
            "area": area[mask] if np.all(np.isfinite(area[mask])) else area[mask],
        }

    ds.close()
    return results


def write_target_heatflux_h5(results: dict, out_path: str | Path):
    """Write target heat flux profiles to HDF5."""
    with h5py.File(str(out_path), "w") as f:
        for target in ("outer", "inner"):
            if target not in results:
                continue
            d = results[target]
            g = f.create_group(target)
            g.create_dataset("s", data=d["s"])          # arc length [m]
            g.create_dataset("R", data=d["R"])          # R coordinates [m]
            g.create_dataset("Z", data=d["Z"])          # Z coordinates [m]
            g.create_dataset("q_total", data=d["q_total"])      # W/m²
            g.create_dataset("q_electron", data=d["q_electron"])  # W/m²
            g.create_dataset("q_ion", data=d["q_ion"])          # W/m²
            g.create_dataset("gamma_D", data=d["gamma_D"])      # m⁻²s⁻¹


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_path", type=Path, help="SOLPS run directory (containing balance.nc)")
    p.add_argument("--out", type=Path, default=Path("target_heatflux.h5"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    nc_path = args.run_path / "balance.nc"
    if not nc_path.exists():
        print(f"ERROR: {nc_path} not found")
        return

    print(f"Reading {nc_path}")
    results = extract_target_profiles(nc_path)

    for target in ("outer", "inner"):
        if target in results:
            d = results[target]
            print(f"\n  {target.upper()} target: {len(d['s'])} cells")
            print(f"    s range: [{d['s'].min():.4f}, {d['s'].max():.4f}] m")
            print(f"    q_total: [{d['q_total'].min():.2e}, {d['q_total'].max():.2e}] W/m²")
            print(f"    gamma_D: [{d['gamma_D'].min():.2e}, {d['gamma_D'].max():.2e}] m⁻²s⁻¹")

    write_target_heatflux_h5(results, args.out)
    print(f"\nWrote: {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("SOLPS Target Profiles (from balance.nc)")

        for j, target in enumerate(("outer", "inner")):
            if target not in results:
                continue
            d = results[target]

            ax = axes[0, j]
            ax.plot(d["s"] * 100, d["q_total"] / 1e6, "r-", lw=1.5, label="Total")
            ax.plot(d["s"] * 100, d["q_electron"] / 1e6, "b--", lw=1, label="Electron")
            ax.plot(d["s"] * 100, d["q_ion"] / 1e6, "g--", lw=1, label="Ion")
            ax.set_ylabel("Heat flux [MW/m²]")
            ax.set_title(f"{target.capitalize()} target")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[1, j]
            ax.plot(d["s"] * 100, d["gamma_D"], "k-", lw=1.5)
            ax.set_ylabel("D+ flux [m⁻²s⁻¹]")
            ax.set_xlabel("Arc length [cm]")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("target_profiles.png", dpi=150)
        print("Saved: target_profiles.png")
        plt.show()


if __name__ == "__main__":
    main()
