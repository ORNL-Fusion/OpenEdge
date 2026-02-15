#!/usr/bin/env python3
# Copyright 2025, OpenEdge contributors
# Authors: Abdourahmane Diaw, OpenEdge contributors
"""
3D gravity validation: compare SPARTA/OpenEdge particle trajectory against
constant-acceleration theory for one particle.

Theory:
    vz(t) = vz0 + gz * (t - t0)
    z(t)  = z0 + vz0*(t - t0) + 0.5*gz*(t - t0)^2

Usage (from OpenEdge top-level):
    python examples/test_gravity_3d/run_gravity_3d.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_dump_particle(filename: Path):
    """
    Parse SPARTA/OpenEdge dump file of the form:
      ITEM: TIMESTEP
      ITEM: NUMBER OF ATOMS
      ITEM: ATOMS id type x y z vx vy vz
    """
    timesteps, ids = [], []
    x, y, z = [], [], []
    vx, vy, vz = [], [], []
    zlo = None
    zhi = None
    z_periodic = False

    lines = filename.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            ts = int(lines[i + 1].strip())
            i += 2
        elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1].strip())
            i += 2
        elif lines[i].strip().startswith("ITEM: BOX BOUNDS"):
            words = lines[i].split()
            if len(words) >= 6:
                z_periodic = words[5].lower().startswith("p")
            zb = lines[i + 3].split()
            if len(zb) >= 2:
                zlo = float(zb[0])
                zhi = float(zb[1])
            i += 4
        elif lines[i].strip().startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            colmap = {name: idx for idx, name in enumerate(cols)}
            need = ["id", "x", "y", "z", "vx", "vy", "vz"]
            for name in need:
                if name not in colmap:
                    raise RuntimeError(f"Missing dump column '{name}' in {filename}")
            for j in range(n):
                row = lines[i + 1 + j].split()
                timesteps.append(ts)
                ids.append(int(row[colmap["id"]]))
                x.append(float(row[colmap["x"]]))
                y.append(float(row[colmap["y"]]))
                z.append(float(row[colmap["z"]]))
                vx.append(float(row[colmap["vx"]]))
                vy.append(float(row[colmap["vy"]]))
                vz.append(float(row[colmap["vz"]]))
            i += n + 1
        else:
            i += 1

    return (
        np.asarray(timesteps, dtype=float),
        np.asarray(ids, dtype=int),
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(vx, dtype=float),
        np.asarray(vy, dtype=float),
        np.asarray(vz, dtype=float),
        zlo,
        zhi,
        z_periodic,
    )


def setup_axes_style(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)


def unwrap_periodic(z: np.ndarray, zlo: float, zhi: float) -> np.ndarray:
    lz = zhi - zlo
    if lz <= 0.0:
        return z.copy()
    zu = z.copy()
    shift = 0.0
    for i in range(1, len(zu)):
        dz = zu[i] - zu[i - 1]
        if dz > 0.5 * lz:
            shift -= lz
        elif dz < -0.5 * lz:
            shift += lz
        zu[i] += shift
    return zu


def main():
    parser = argparse.ArgumentParser(description="Run and validate 3D gravity fix example")
    parser.add_argument("--sparta", default="../../src/spa_mpi", help="Path to SPARTA/OpenEdge executable")
    parser.add_argument("--mpi", type=int, default=1, help="MPI ranks (1 runs without mpirun)")
    parser.add_argument("--input", default="in.gravity3d", help="Input script")
    parser.add_argument("--dt", type=float, default=1.0e-3, help="Timestep [s]")
    parser.add_argument("--gz", type=float, default=-9.81, help="Gravity z-component [m/s^2]")
    parser.add_argument("--png", default="gravity3d_validation.png", help="Output PNG")
    parser.add_argument("--csv", default="gravity3d_validation.csv", help="Output CSV")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    sparta = (here / args.sparta).resolve()
    input_file = here / args.input
    dump_file = here / "dump.gravity3d"

    if dump_file.exists():
        dump_file.unlink()

    cmd = [str(sparta), "-in", str(input_file)]
    if args.mpi and args.mpi > 1:
        cmd = ["mpirun", "-np", str(args.mpi)] + cmd
    subprocess.run(cmd, cwd=here, check=True)

    tstep, pid, _x, _y, z, _vx, _vy, vz, zlo, zhi, z_periodic = parse_dump_particle(dump_file)

    # Use first particle ID only
    unique_ids = np.unique(pid)
    if len(unique_ids) == 0:
        raise RuntimeError("No particle records found in dump.gravity3d")
    sel = pid == unique_ids[0]

    t = tstep[sel] * args.dt
    z = z[sel]
    vz = vz[sel]
    if z_periodic and zlo is not None and zhi is not None:
        z = unwrap_periodic(z, zlo, zhi)

    t0 = t[0]
    z0 = z[0]
    vz0 = vz[0]

    vz_ref = vz0 + args.gz * (t - t0)
    z_ref = z0 + vz0 * (t - t0) + 0.5 * args.gz * (t - t0) ** 2

    vz_err = np.abs(vz - vz_ref)
    z_err = np.abs(z - z_ref)

    # Save data
    with (here / args.csv).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "z", "z_ref", "vz", "vz_ref", "z_err", "vz_err"])
        for i in range(len(t)):
            w.writerow([t[i], z[i], z_ref[i], vz[i], vz_ref[i], z_err[i], vz_err[i]])

    # Plot style similar to polarization test
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300, constrained_layout=True)

    # (a) vz(t)
    ax = axs[0, 0]
    ax.plot(t, vz, "o", ms=4, mfc="none", label="simulation $v_z$")
    ax.plot(t, vz_ref, "-", lw=1.8, label="model $v_z$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$v_z$ [m/s]")
    ax.set_title("$v_z(t)$: simulation vs model")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    # (b) z(t)
    ax = axs[0, 1]
    ax.plot(t, z, "s", ms=3.5, mfc="none", label="simulation $z$")
    ax.plot(t, z_ref, "-", lw=1.8, label="model $z$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$z$ [m]")
    ax.set_title("$z(t)$: simulation vs model")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    # (c) abs error in vz
    ax = axs[1, 0]
    ax.semilogy(t, np.maximum(vz_err, 1.0e-30), "o-", ms=3, lw=1.0,
                label=r"$|v_z-v_{z,ref}|$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$|v_z-v_{z,ref}|$ [m/s]")
    ax.set_title("Absolute velocity error")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    # (d) abs error in z
    ax = axs[1, 1]
    ax.semilogy(t, np.maximum(z_err, 1.0e-30), "o-", ms=3, lw=1.0,
                label=r"$|z-z_{ref}|$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$|z-z_{ref}|$ [m]")
    ax.set_title("Absolute position error")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    out_png = here / args.png
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_png}")
    print(f"Saved data: {here / args.csv}")
    print(f"max |vz-vz_ref| = {vz_err.max():.6e}")
    print(f"max |z-z_ref|   = {z_err.max():.6e}")


if __name__ == "__main__":
    main()
