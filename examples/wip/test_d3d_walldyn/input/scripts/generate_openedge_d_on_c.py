#!/usr/bin/env python3
"""
Generate OpenEdge surface interaction data for projectile -> carbon using RustBCA.

Output HDF5 datasets:
  - E      (nE,) incident energies in eV
  - A      (nA,) incident angles in degrees
  - spyld  (nE, nA) sputtering yield
  - rfyld  (nE, nA) particle reflection yield

Supported projectile presets:
  - D  -> carbon target, default output 1_on_6.h5
  - C  -> carbon target, default output 6_on_6.h5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from libRustBCA import reflection_coefficient, sputtering_yield


# RustBCA material dictionaries compatible with the Python bindings.
deuterium = {
    "symbol": "D",
    "name": "deuterium",
    "Z": 1.0,
    "m": 2.0,
    "Ec": 0.1,
    "Es": 1.5,
}

carbon = {
    "symbol": "C",
    "name": "carbon",
    "Z": 6.0,
    "m": 12.011,
    "n": 1.1331e29,
    "Es": 7.37,
    "Ec": 1.0,
    "Eb": 0.0,
    "Q": 1.70,
    "W": 1.84,
    "s": 2.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenEdge-format projectile-on-carbon sputtering/reflection yield tables."
    )
    parser.add_argument(
        "--projectile",
        choices=("D", "C"),
        default="D",
        help="Incident projectile preset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file path. Defaults to atomic-number naming based on --projectile.",
    )
    parser.add_argument("--e-min", type=float, default=1.0, help="Minimum incident energy [eV].")
    parser.add_argument("--e-max", type=float, default=1000.0, help="Maximum incident energy [eV].")
    parser.add_argument("--n-energy", type=int, default=40, help="Number of energy points.")
    parser.add_argument("--a-min", type=float, default=0.0, help="Minimum incident angle [deg].")
    parser.add_argument("--a-max", type=float, default=89.9, help="Maximum incident angle [deg].")
    parser.add_argument("--n-angle", type=int, default=40, help="Number of angle points.")
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Monte Carlo ions per (E, A) point.",
    )
    parser.add_argument(
        "--angle-floor",
        type=float,
        default=1.0e-4,
        help="Angles below this are evaluated at this floor to avoid exact-normal edge cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_energy < 2 or args.n_angle < 2:
        raise ValueError("n-energy and n-angle must both be >= 2.")
    if args.samples < 1:
        raise ValueError("samples must be >= 1.")

    projectile = deuterium if args.projectile == "D" else carbon
    default_name = "1_on_6.h5" if args.projectile == "D" else "6_on_6.h5"
    output = args.output if args.output is not None else Path(default_name)

    energies = np.linspace(args.e_min, args.e_max, args.n_energy, dtype=np.float64)
    angles = np.linspace(args.a_min, args.a_max, args.n_angle, dtype=np.float64)

    spyld = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)
    rfyld = np.zeros((args.n_energy, args.n_angle), dtype=np.float64)

    total = args.n_energy * args.n_angle
    start = time.time()
    idx = 0

    for ie, energy in enumerate(energies):
        for ia, angle in enumerate(angles):
            angle_eval = max(float(angle), args.angle_floor)
            spyld[ie, ia] = sputtering_yield(
                projectile, carbon, float(energy), angle_eval, args.samples
            )
            rfyld[ie, ia], _ = reflection_coefficient(
                projectile, carbon, float(energy), angle_eval, args.samples
            )
            idx += 1

        elapsed = time.time() - start
        done_frac = idx / total
        eta = elapsed / done_frac - elapsed if done_frac > 0 else 0.0
        print(
            f"[{ie + 1:4d}/{args.n_energy}] E={energy:8.2f} eV | "
            f"progress={100.0 * done_frac:6.2f}% | elapsed={elapsed:8.1f}s | eta={eta:8.1f}s",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5f:
        h5f.create_dataset("E", data=energies)
        h5f.create_dataset("A", data=angles)
        h5f.create_dataset("spyld", data=spyld)
        h5f.create_dataset("rfyld", data=rfyld)

    total_time = time.time() - start
    print(f"\nProjectile: {args.projectile}  Target: C")
    print(f"Wrote {output}")
    print(f"Total points: {total}, samples/point: {args.samples}, walltime: {total_time:.1f}s")
    print(f"spyld range: [{spyld.min():.6g}, {spyld.max():.6g}]")
    print(f"rfyld range: [{rfyld.min():.6g}, {rfyld.max():.6g}]")


if __name__ == "__main__":
    main()
