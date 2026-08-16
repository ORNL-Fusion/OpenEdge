#!/usr/bin/env python3
"""Generate the OpenEdge He-on-W sputter-yield grid with RustBCA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
RUSTBCA_SCRIPTS = Path("/Users/42d/Projects/code/RustBCA/scripts")
sys.path.insert(0, str(RUSTBCA_SCRIPTS))

import generate_openedge_pairs as generator  # noqa: E402
from materials import helium, tungsten  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "input")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--min-counts", type=int, default=20)
    parser.add_argument("--n-energy", type=int, default=50)
    parser.add_argument("--n-angle", type=int, default=45)
    parser.add_argument("--e-min", type=float, default=5.0)
    parser.add_argument("--e-max", type=float, default=10000.0)
    parser.add_argument("--a-min", type=float, default=0.0)
    parser.add_argument("--a-max", type=float, default=89.0)
    parser.add_argument("--angle-floor", type=float, default=0.1)
    args = parser.parse_args()

    generator.PAIRS["he_on_w"] = (helium, tungsten)
    args.pair = "he_on_w"
    energies = np.geomspace(args.e_min, args.e_max, args.n_energy)
    angles = np.linspace(args.a_min, args.a_max, args.n_angle)
    generator.run_pure(args, energies, angles)


if __name__ == "__main__":
    main()
