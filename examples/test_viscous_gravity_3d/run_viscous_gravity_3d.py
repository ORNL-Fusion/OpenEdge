#!/usr/bin/env python3
# Copyright 2025, OpenEdge contributors
"""
3D viscous+gravity validation for one droplet particle.

Model used in this test:
    dvz/dt = -nuE * vz + gz
    dz/dt  = vz

Default test is the `nuE=0` limit (pure gravity through fix viscous).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dump_particle(filename: Path):
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
            for j in range(n):
                row = lines[i + 1 + j].split()
                timesteps.append(ts)
                ids.append(int(row[0]))
                x.append(float(row[2]))
                y.append(float(row[3]))
                z.append(float(row[4]))
                vx.append(float(row[5]))
                vy.append(float(row[6]))
                vz.append(float(row[7]))
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
    parser = argparse.ArgumentParser(description="Run and validate 3D viscous+gravity example")
    parser.add_argument("--sparta", default="../../src/spa_serial", help="Path to spa_serial")
    parser.add_argument("--input", default="in.viscous_gravity3d", help="Input script")
    parser.add_argument("--dt", type=float, default=1.0e-3, help="Timestep [s]")
    parser.add_argument("--gz", type=float, default=-9.81, help="Gravity z-component [m/s^2]")
    parser.add_argument("--nu", type=float, default=0.0, help="Constant drag rate nuE [1/s]")
    parser.add_argument("--png", default="viscous_gravity3d_validation.png", help="Output PNG")
    parser.add_argument("--csv", default="viscous_gravity3d_validation.csv", help="Output CSV")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    sparta = (here / args.sparta).resolve()
    input_file = here / args.input
    dump_file = here / "dump.viscous_gravity3d"

    if dump_file.exists():
        dump_file.unlink()

    subprocess.run([str(sparta), "-in", str(input_file)], cwd=here, check=True)

    tstep, pid, _x, _y, z, _vx, _vy, vz, zlo, zhi, z_periodic = parse_dump_particle(dump_file)
    unique_ids = np.unique(pid)
    if len(unique_ids) == 0:
        raise RuntimeError("No particle records found in dump.viscous_gravity3d")
    sel = pid == unique_ids[0]

    t = tstep[sel] * args.dt
    z = z[sel]
    vz = vz[sel]
    if z_periodic and zlo is not None and zhi is not None:
        z = unwrap_periodic(z, zlo, zhi)
    nu = args.nu

    tau = t - t[0]
    vz0 = vz[0]
    z0 = z[0]
    if nu > 0.0:
        v_inf = args.gz / nu
        vz_ref = (vz0 - v_inf) * np.exp(-nu * tau) + v_inf
        z_ref = z0 + (vz0 - v_inf) * (1.0 - np.exp(-nu * tau)) / nu + v_inf * tau
    else:
        v_inf = float("nan")
        vz_ref = vz0 + args.gz * tau
        z_ref = z0 + vz0 * tau + 0.5 * args.gz * tau * tau

    vz_err = np.abs(vz - vz_ref)
    z_err = np.abs(z - z_ref)
    if nu == 0.0 and len(t) > 1:
        slope = (vz[-1] - vz[0]) / (t[-1] - t[0])
        if abs(slope - args.gz) > 0.1 * max(1.0, abs(args.gz)):
            print("WARNING: measured dvz/dt does not match gz; check that spa_serial was rebuilt with the latest fix_viscous.cpp")

    with (here / args.csv).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "z", "z_ref", "vz", "vz_ref", "z_err", "vz_err", "nuE", "v_inf"])
        for i in range(len(t)):
            w.writerow([t[i], z[i], z_ref[i], vz[i], vz_ref[i], z_err[i], vz_err[i], nu, v_inf])

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300, constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(t, vz, "o", ms=4, mfc="none", label="simulation $v_z$")
    ax.plot(t, vz_ref, "-", lw=1.8, label="model $v_z$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$v_z$ [m/s]")
    ax.set_title("$v_z(t)$: viscous/gravity")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    ax = axs[0, 1]
    ax.plot(t, z, "s", ms=3.5, mfc="none", label="simulation $z$")
    ax.plot(t, z_ref, "-", lw=1.8, label="model $z$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$z$ [m]")
    ax.set_title("$z(t)$: viscous/gravity")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    ax = axs[1, 0]
    ax.semilogy(t, np.maximum(vz_err, 1.0e-30), "o-", ms=3, lw=1.0, label=r"$|v_z-v_{z,ref}|$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$|v_z-v_{z,ref}|$ [m/s]")
    ax.set_title("Absolute velocity error")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    ax = axs[1, 1]
    ax.semilogy(t, np.maximum(z_err, 1.0e-30), "o-", ms=3, lw=1.0, label=r"$|z-z_{ref}|$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$|z-z_{ref}|$ [m]")
    ax.set_title(f"Absolute position error (nuE={nu:.3e} 1/s)")
    setup_axes_style(ax)
    ax.legend(frameon=False)

    out_png = here / args.png
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_png}")
    print(f"Saved data: {here / args.csv}")
    print(f"nuE [1/s]       = {nu:.6e}")
    print(f"terminal vz [m/s]= {v_inf:.6e}")
    print(f"max |vz-vz_ref| = {vz_err.max():.6e}")
    print(f"max |z-z_ref|   = {z_err.max():.6e}")


if __name__ == "__main__":
    main()
