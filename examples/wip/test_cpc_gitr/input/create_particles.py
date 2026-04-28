#!/usr/bin/env python3
"""
Generate initial W particle source for CPC benchmark / GITR-CPC alignment.

Default behavior places particles at a point inferred from the surface geometry
center, then applies optional x/y/z push offsets so the source is moved inward.

SPARTA read_particles format:
  id  species_index  x  y  z  vx  vy  vz

Examples:
  python3 create_particles.py
  python3 create_particles.py 10000 particles.dat --surf input/gitr_geometry.surf
  python3 create_particles.py --push-z -5e-4 --push-x 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

# Physical constants
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19
KB_JK = 1.380649e-23

# Default impurity mass (W)
DEFAULT_MASS_AMU = 183.84


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create SPARTA particle source file")
    ap.add_argument("nP", nargs="?", type=int, default=10000,
                    help="number of particles (default: 10000)")
    ap.add_argument("outfile", nargs="?", default="particles.dat",
                    help="output particle file (default: particles.dat)")
    ap.add_argument("--surf", default="input/gitr_geometry.surf",
                    help="surface geometry file for domain bounds/center")
    ap.add_argument("--push-x", type=float, default=0.0,
                    help="offset source x position [m]")
    ap.add_argument("--push-y", type=float, default=0.0,
                    help="offset source y position [m]")
    ap.add_argument("--push-z", type=float, default=0.0,
                    help="offset source z position [m]")
    ap.add_argument("--margin", type=float, default=1.0e-4,
                    help="keep source at least this far from AABB faces [m]")
    ap.add_argument("--x0", type=float, default=0.0,
                    help="override source x coordinate [m]")
    ap.add_argument("--y0", type=float, default=0.0,
                    help="override source y coordinate [m]")
    ap.add_argument("--z0", type=float, default=1.0e-5,
                    help="override source z coordinate [m]")
    ap.add_argument("--mass-amu", type=float, default=DEFAULT_MASS_AMU,
                    help="impurity mass [amu] (default: 183.84 for W)")
    ap.add_argument("--energy-ev", type=float, default=9.0,
                    help="directed kinetic energy [eV] (default: 9.0)")
    ap.add_argument("--phi-deg", type=float, default=0.0,
                    help="GITR polar angle phi [deg] (from +z)")
    ap.add_argument("--theta-deg", type=float, default=0.0,
                    help="GITR azimuth angle theta [deg] in x-y plane")
    ap.add_argument("--temp-ev", type=float, default=1.0,
                    help="thermal temperature for velocity spread [eV] (default: 1.0)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="RNG seed for reproducible thermal velocity sampling")
    return ap.parse_args()


def parse_surf_points(path: Path) -> np.ndarray:
    pts = []
    in_points = False

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Points"):
                in_points = True
                continue
            if line.startswith("Triangles") or line.startswith("Lines"):
                break
            if in_points:
                cols = line.split()
                if len(cols) >= 4:
                    pts.append([float(cols[1]), float(cols[2]), float(cols[3])])

    if not pts:
        raise RuntimeError(f"No points parsed from {path}")
    return np.array(pts, dtype=float)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def choose_source_point(args: argparse.Namespace) -> tuple[float, float, float, tuple[float, float, float], tuple[float, float, float]]:
    points = parse_surf_points(Path(args.surf))
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ctr = 0.5 * (mins + maxs)

    # Start from explicit GITR-like source coordinates.
    x0 = args.x0
    y0 = args.y0
    z0 = args.z0

    # Apply user-directed "push" to move source inward.
    x0 += args.push_x
    y0 += args.push_y
    z0 += args.push_z

    # Keep seed point strictly inside the geometry AABB by margin.
    m = max(args.margin, 0.0)
    x0 = clamp(x0, mins[0] + m, maxs[0] - m)
    y0 = clamp(y0, mins[1] + m, maxs[1] - m)
    z0 = clamp(z0, mins[2] + m, maxs[2] - m)

    return x0, y0, z0, (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def main() -> None:
    args = parse_args()
    nP = args.nP
    outfile = args.outfile

    x0, y0, z0, bmin, bmax = choose_source_point(args)

    mass_kg = args.mass_amu * AMU_KG
    speed = np.sqrt(2.0 * max(args.energy_ev, 0.0) * EV_J / mass_kg)
    # Match GITR sampler in gitr.cpp:
    # vx = v*sin(phi)*cos(theta), vy = v*sin(phi)*sin(theta), vz = v*cos(phi)
    phi = np.deg2rad(args.phi_deg)
    theta = np.deg2rad(args.theta_deg)
    vx_drift = speed * np.sin(phi) * np.cos(theta)
    vy_drift = speed * np.sin(phi) * np.sin(theta)
    vz_drift = speed * np.cos(phi)
    if np.isclose(phi, 0.0):
        vx_drift = 0.0
        vy_drift = 0.0
        vz_drift = speed

    # Maxwellian thermal spread around the drift velocity.
    # 1D component sigma = sqrt(k_B T / m), with T from eV -> K.
    temp_ev = max(args.temp_ev, 0.0)
    temp_k = temp_ev * EV_J / KB_JK
    sigma_v = np.sqrt(KB_JK * temp_k / mass_kg) if temp_k > 0.0 else 0.0
    rng = np.random.default_rng(args.seed)
    dv = rng.normal(0.0, sigma_v, size=(nP, 3))
    vxyz = dv + np.array([vx_drift, vy_drift, vz_drift])

    drift_E_total = 0.5 * mass_kg * (vx_drift**2 + vy_drift**2 + vz_drift**2) / EV_J
    print(f"Creating {nP} W particles at ({x0:.6e},{y0:.6e},{z0:.6e})")
    print(f"Geometry AABB min={bmin}, max={bmax}")
    print(f"Mass = {args.mass_amu:.5f} amu")
    print(f"Angles: phi={args.phi_deg:.3f} deg, theta={args.theta_deg:.3f} deg")
    print(f"Directed energy = {args.energy_ev:.3f} eV, E_drift_total = {drift_E_total:.3f} eV")
    print(f"Thermal spread: T = {temp_ev:.3f} eV, sigma_v = {sigma_v:.6f} m/s")
    speed2 = np.sum(vxyz * vxyz, axis=1)
    ke_ev = 0.5 * mass_kg * speed2 / EV_J
    print(f"Kinetic energy stats [eV]: mean={ke_ev.mean():.4f}, min={ke_ev.min():.4f}, max={ke_ev.max():.4f}")

    with open(outfile, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("0\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{nP}\n")
        f.write("ITEM: BOX BOUNDS rr rr rr\n")
        f.write(f"{bmin[0]} {bmax[0]}\n")
        f.write(f"{bmin[1]} {bmax[1]}\n")
        f.write(f"{bmin[2]} {bmax[2]}\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        for i in range(nP):
            pid = i + 1
            ispecies = 1
            f.write(
                f"{pid} {ispecies} {x0} {y0} {z0} "
                f"{vxyz[i,0]} {vxyz[i,1]} {vxyz[i,2]}\n"
            )

    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
