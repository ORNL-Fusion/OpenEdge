#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a surf_react pmi-compatible HDF5 table for WEST."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root.parents[1] / "database" / "surface" / "74_on_74.h5",
        help="Input OpenEdge surface database file with A/E/rfyld/spyld.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "input" / "74_on_74_pmi.h5",
        help="Output surf_react pmi-compatible HDF5 file.",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=0.8,
        help="Constant reflected-energy fraction used to populate RE.",
    )
    parser.add_argument(
        "--ebind",
        type=float,
        default=11.75,
        help="Binding energy in eV written to E_bind.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as src:
        energy = np.asarray(src["E"][:], dtype=np.float64)
        angle = np.asarray(src["A"][:], dtype=np.float64)
        rn = np.asarray(src["rfyld"][:], dtype=np.float64)
        spyld = np.asarray(src["spyld"][:], dtype=np.float64)

    if rn.shape != spyld.shape:
        raise SystemExit(f"shape mismatch: rfyld {rn.shape} vs spyld {spyld.shape}")

    re = np.full(rn.shape, float(args.re), dtype=np.float64)

    with h5py.File(args.output, "w") as dst:
        dst.create_dataset("E", data=energy)
        dst.create_dataset("A", data=angle)
        dst.create_dataset("RN", data=rn)
        dst.create_dataset("RE", data=re)
        dst.create_dataset("spyld", data=spyld)
        dst.create_dataset("E_bind", data=np.asarray(float(args.ebind), dtype=np.float64))

    print(f"Wrote {args.output}")
    print(
        "Using RN <- rfyld from source, RE <- constant "
        f"{args.re:g}, spyld <- source spyld, E_bind <- {args.ebind:g} eV"
    )


if __name__ == "__main__":
    main()
