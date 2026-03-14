#!/usr/bin/env python3
"""Generate a sequence of B-field HDF5 snapshots from a reference equilibrium
(.equ file) by applying rigid radial shifts to the psi grid.

This implements strike point sweeping for WEST: shifting psi(R,Z) by ΔR
translates the poloidal flux map horizontally, moving strike points on the
divertor targets without rerunning the equilibrium solver.

B-field components are recomputed from the shifted psi:
    B_R =  -(1/R) ∂ψ/∂Z
    B_Z =  (1/R)  ∂ψ/∂R
    B_t =  btf * rtf / R      (vacuum toroidal field)

Usage:
    python gen_bfield_sweep.py \
        --equ input/equilibrium.equ \
        --delta-r 0.02 0.04 0.02 0.0 -0.02 -0.04 -0.02 0.0 \
        --times   0.0  0.5  1.0  1.5  2.0   2.5   3.0  3.5 \
        --outdir input/bfield_sweep/

Output:
    bfield_sweep/bfield_t0.h5 .. bfield_tN.h5
    bfield_sweep/bfield_times.txt   (manifest for compute bfield/timedep)
"""

import argparse
import os
import sys

import h5py
import numpy as np


def parse_equ_file(path):
    """Parse a SOLPS-style equilibrium (.equ) file.

    Returns dict with keys: jm, km, btf, rtf, psib, r, z, psi
    where psi has shape (km, jm) and includes the psib offset.
    """
    equ = {"jm": 0, "km": 0, "btf": 0.0, "rtf": 0.0, "psib": 0.0}

    with open(path) as f:
        lines = f.readlines()

    def find_scalar(name, lines):
        for line in lines:
            if name in line and "=" in line and ":=" not in line:
                idx = line.index("=")
                return float(line[idx + 1 :].split()[0])
        return 0.0

    def find_int(name, lines):
        for line in lines:
            if name in line and "=" in line and ":=" not in line:
                # For km, make sure "km" comes before "=" to avoid matching
                # other variables
                if name == "km":
                    pos_name = line.find(name)
                    pos_eq = line.find("=")
                    if pos_name > pos_eq:
                        continue
                idx = line.index("=")
                return int(line[idx + 1 :].split()[0])
        return 0

    equ["jm"] = find_int("jm", lines)
    equ["km"] = find_int("km", lines)
    equ["btf"] = find_scalar("btf", lines)
    equ["rtf"] = find_scalar("rtf", lines)
    equ["psib"] = find_scalar("psib", lines)

    if equ["jm"] <= 0 or equ["km"] <= 0:
        raise ValueError(f"Failed to parse jm/km from {path}")

    # Find and read r(1:jm)
    idx_r = None
    for i, line in enumerate(lines):
        if "r(1:jm)" in line:
            idx_r = i + 1
            break
    if idx_r is None:
        raise ValueError("Cannot find r(1:jm) in equilibrium file")

    r_vals = []
    i = idx_r
    while len(r_vals) < equ["jm"] and i < len(lines):
        r_vals.extend(float(v) for v in lines[i].split())
        i += 1
    equ["r"] = np.array(r_vals[: equ["jm"]])

    # Find and read z(1:km)
    idx_z = None
    for i, line in enumerate(lines):
        if "z(1:km)" in line:
            idx_z = i + 1
            break
    if idx_z is None:
        raise ValueError("Cannot find z(1:km) in equilibrium file")

    z_vals = []
    i = idx_z
    while len(z_vals) < equ["km"] and i < len(lines):
        z_vals.extend(float(v) for v in lines[i].split())
        i += 1
    equ["z"] = np.array(z_vals[: equ["km"]])

    # Find and read psi(j,k)
    idx_psi = None
    for i, line in enumerate(lines):
        if "psi(j,k)" in line:
            idx_psi = i + 1
            break
    if idx_psi is None:
        raise ValueError("Cannot find psi(j,k) in equilibrium file")

    total = equ["jm"] * equ["km"]
    psi_flat = []
    i = idx_psi
    while len(psi_flat) < total and i < len(lines):
        psi_flat.extend(float(v) for v in lines[i].split())
        i += 1
    psi_flat = np.array(psi_flat[:total])

    # psi stored as psi(j,k) → reshape to (km, jm), add psib
    equ["psi"] = psi_flat.reshape(equ["km"], equ["jm"]) + equ["psib"]

    return equ


def compute_bfield_from_psi(r, z, psi, btf, rtf):
    """Compute B-field components from psi(Z,R) on an (R,Z) grid.

    psi has shape (nz, nr) where axis 0 = Z, axis 1 = R.

    Returns (br, bt, bz) each of shape (nz, nr).
        B_R =  -(1/R) ∂ψ/∂Z
        B_Z =  (1/R)  ∂ψ/∂R
        B_t =  btf * rtf / R
    """
    nr = len(r)
    nz = len(z)
    assert psi.shape == (nz, nr)

    # Compute partial derivatives using central differences
    dpsi_dr = np.zeros_like(psi)
    dpsi_dz = np.zeros_like(psi)

    # ∂ψ/∂R (along axis 1)
    for j in range(nr):
        if j == 0:
            dpsi_dr[:, j] = (psi[:, j + 1] - psi[:, j]) / (r[j + 1] - r[j])
        elif j == nr - 1:
            dpsi_dr[:, j] = (psi[:, j] - psi[:, j - 1]) / (r[j] - r[j - 1])
        else:
            dpsi_dr[:, j] = (psi[:, j + 1] - psi[:, j - 1]) / (r[j + 1] - r[j - 1])

    # ∂ψ/∂Z (along axis 0)
    for k in range(nz):
        if k == 0:
            dpsi_dz[k, :] = (psi[k + 1, :] - psi[k, :]) / (z[k + 1] - z[k])
        elif k == nz - 1:
            dpsi_dz[k, :] = (psi[k, :] - psi[k - 1, :]) / (z[k] - z[k - 1])
        else:
            dpsi_dz[k, :] = (psi[k + 1, :] - psi[k - 1, :]) / (z[k + 1] - z[k - 1])

    # Build 2D R array for division
    R2d = np.tile(r, (nz, 1))  # shape (nz, nr)
    R2d = np.maximum(R2d, 1.0e-10)  # avoid division by zero

    br = -dpsi_dz / R2d
    bz = dpsi_dr / R2d
    bt = btf * rtf / R2d

    return br, bt, bz


def apply_rigid_shift(equ, delta_r):
    """Apply rigid radial shift ΔR to the equilibrium.

    This shifts the R-coordinate grid by delta_r, effectively translating
    the poloidal flux map horizontally. The psi values stay the same on
    the shifted grid.
    """
    shifted = dict(equ)
    shifted["r"] = equ["r"] + delta_r
    # psi, z, and toroidal parameters stay the same
    return shifted


def write_bfield_h5(path, r, z, br, bt, bz):
    """Write B-field components to HDF5 file in OpenEdge format.

    Datasets: r(nr), z(nz), br(nz,nr), bt(nz,nr), bz(nz,nr)
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        f.create_dataset("br", data=br)
        f.create_dataset("bt", data=bt)
        f.create_dataset("bz", data=bz)


def main():
    parser = argparse.ArgumentParser(
        description="Generate B-field sweep snapshots from equilibrium"
    )
    parser.add_argument(
        "--equ", required=True, help="Path to reference equilibrium (.equ) file"
    )
    parser.add_argument(
        "--delta-r",
        nargs="+",
        type=float,
        required=True,
        help="Radial shifts in meters (one per snapshot)",
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        required=True,
        help="Simulation times in seconds (one per snapshot)",
    )
    parser.add_argument(
        "--outdir", default="bfield_sweep", help="Output directory"
    )
    parser.add_argument(
        "--prefix", default="bfield_t", help="Output filename prefix"
    )
    args = parser.parse_args()

    if len(args.delta_r) != len(args.times):
        print(
            f"Error: --delta-r ({len(args.delta_r)} values) and "
            f"--times ({len(args.times)} values) must have same length",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading equilibrium from: {args.equ}")
    equ = parse_equ_file(args.equ)
    print(
        f"  jm={equ['jm']} km={equ['km']} btf={equ['btf']:.3f} rtf={equ['rtf']:.3f}"
    )
    print(
        f"  R range: [{equ['r'][0]:.4f}, {equ['r'][-1]:.4f}] m"
    )
    print(
        f"  Z range: [{equ['z'][0]:.4f}, {equ['z'][-1]:.4f}] m"
    )

    manifest_lines = []
    for i, (dr, t) in enumerate(zip(args.delta_r, args.times)):
        shifted = apply_rigid_shift(equ, dr)
        br, bt, bz = compute_bfield_from_psi(
            shifted["r"], shifted["z"], shifted["psi"],
            shifted["btf"], shifted["rtf"],
        )

        fname = f"{args.prefix}{i}.h5"
        fpath = os.path.join(args.outdir, fname)
        write_bfield_h5(fpath, shifted["r"], shifted["z"], br, bt, bz)
        manifest_lines.append(f"{t}  {fname}")

        print(
            f"  [{i}] ΔR={dr:+.4f} m  t={t:.3f} s  → {fpath}"
            f"  |B| range: [{np.sqrt(br**2+bt**2+bz**2).min():.3f},"
            f" {np.sqrt(br**2+bt**2+bz**2).max():.3f}] T"
        )

    manifest_path = os.path.join(args.outdir, "bfield_times.txt")
    with open(manifest_path, "w") as f:
        f.write("# time_seconds  filename\n")
        for line in manifest_lines:
            f.write(line + "\n")
    print(f"\nManifest written to: {manifest_path}")
    print(f"Generated {len(args.times)} B-field snapshots")


if __name__ == "__main__":
    main()
