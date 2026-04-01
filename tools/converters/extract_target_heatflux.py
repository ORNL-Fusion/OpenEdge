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
    print(f"  Grid: nx={nx}, ny={ny}")

    # Cell corner coordinates: crx, cry
    # Shape can be (4, ny+2, nx+2) or (5, ny+2, nx+2) depending on version
    crx = np.array(ds.variables["crx"])
    cry = np.array(ds.variables["cry"])

    # Face areas: hy shape (ny+2, nx+2) — radial face area
    hy = np.array(ds.variables["hy"])

    # --- Heat fluxes at faces ---
    # Layout: (2, ny+2, nx+2) where dim 0 = face direction (0=x/poloidal, 1=y/radial)
    # Or: (ny+2, nx+2, 2) in some versions
    have_split = "fhe" in ds.variables and "fhi" in ds.variables
    have_total = "fht" in ds.variables

    def _read_hf(varname):
        a = np.array(ds.variables[varname])
        # Ensure shape is (2, ny+2, nx+2): dim 0 = face direction
        if a.shape == (ny + 2, nx + 2, 2):
            a = a.transpose(2, 0, 1)
        return a

    def _sum_components(prefix):
        """Sum all heat flux components matching prefix (e.g. 'fhe_')."""
        acc = None
        for vn in sorted(ds.variables):
            if vn.startswith(prefix):
                a = _read_hf(vn)
                acc = a if acc is None else acc + a
        return acc

    if have_split:
        fhe = _read_hf("fhe")
        fhi = _read_hf("fhi")
        fht_y_e = fhe[1]  # radial (y) face, electron [W]
        fht_y_i = fhi[1]  # radial (y) face, ion [W]
    elif have_total:
        fht = _read_hf("fht")
        fht_y_e = 0.5 * fht[1]
        fht_y_i = 0.5 * fht[1]
        print("  Using fht (total) — no electron/ion split")
    else:
        fhe_sum = _sum_components("fhe_")
        fhi_sum = _sum_components("fhi_")
        if fhe_sum is not None and fhi_sum is not None:
            fht_y_e = fhe_sum[1]  # y-face
            fht_y_i = fhi_sum[1]
            n_e = sum(1 for v in ds.variables if v.startswith("fhe_"))
            n_i = sum(1 for v in ds.variables if v.startswith("fhi_"))
            print(f"  Summed {n_e} fhe_* + {n_i} fhi_* components")
        else:
            hf_vars = [v for v in ds.variables if "fh" in v]
            ds.close()
            raise KeyError(f"No heat flux variables found. fh* vars: {hf_vars}")

    # --- Particle flux at faces ---
    # Try fna_tot first (total), then fna, then sum fna_* components
    # Shape: (ns, 2, ny+2, nx+2) where dim 1 = face direction
    fna_D_y = None
    if "fna_tot" in ds.variables:
        fna = np.array(ds.variables["fna_tot"])
        # D+ is species index 1, y-face is dim 1 index 1
        fna_D_y = fna[1, 1]  # (ny+2, nx+2)
        print("  Using fna_tot for D+ particle flux")
    elif "fna" in ds.variables:
        fna = np.array(ds.variables["fna"])
        if fna.ndim == 4:
            fna_D_y = fna[1, 1]
        elif fna.ndim == 3:
            fna_D_y = fna[1, :, :]  # assume (ns, ny+2, nx+2) single face
        print("  Using fna for D+ particle flux")
    else:
        # No particle flux — set to zero, adatom model won't work
        fna_D_y = np.zeros((ny + 2, nx + 2))
        print("  WARNING: no fna/fna_tot — D+ flux set to zero")

    # --- Extract target profiles ---
    s_interior = np.s_[1:-1]  # strip guard cells in x

    results = {}

    for target, iy_face in [("outer", 1), ("inner", ny)]:
        # Heat flux density = face flux [W] / face area [m²]
        area = hy[iy_face, s_interior]
        area = np.where(area > 1e-20, area, np.nan)

        q_e = np.abs(fht_y_e[iy_face, s_interior]) / area
        q_i = np.abs(fht_y_i[iy_face, s_interior]) / area
        q_total = q_e + q_i

        gamma_D = np.abs(fna_D_y[iy_face, s_interior]) / area

        # Target cell midpoints from corners
        # corners 0,1 = bottom face; corners 2,3 = top face
        nc = crx.shape[0]  # 4 or 5
        if target == "outer":
            R_pts = 0.5 * (crx[0, iy_face, s_interior] + crx[1, iy_face, s_interior])
            Z_pts = 0.5 * (cry[0, iy_face, s_interior] + cry[1, iy_face, s_interior])
        else:
            R_pts = 0.5 * (crx[2, iy_face, s_interior] + crx[min(3, nc-1), iy_face, s_interior])
            Z_pts = 0.5 * (cry[2, iy_face, s_interior] + cry[min(3, nc-1), iy_face, s_interior])

        # Arc length along target
        dR = np.diff(R_pts)
        dZ = np.diff(Z_pts)
        ds_arc = np.sqrt(dR**2 + dZ**2)
        s = np.concatenate([[0.0], np.cumsum(ds_arc)])

        mask = np.isfinite(q_total) & np.isfinite(gamma_D)
        results[target] = {
            "s": s[mask], "R": R_pts[mask], "Z": Z_pts[mask],
            "q_total": q_total[mask], "q_electron": q_e[mask],
            "q_ion": q_i[mask], "gamma_D": gamma_D[mask],
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
