#!/usr/bin/env python3
"""Install a generated He-on-W yield table into an existing processes.h5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processes", type=Path,
                        help="processes.h5 to update (make a backup first)")
    parser.add_argument("--source", type=Path,
                        default=CASE_DIR / "input/he_on_w.h5")
    args = parser.parse_args()
    if not args.processes.exists():
        raise SystemExit(f"missing {args.processes}")
    if not args.source.exists():
        raise SystemExit(f"missing {args.source}; run build_he_on_w.py first")

    with h5py.File(args.source, "r") as src:
        energy = src["E"][:]
        theta = src["A"][:]
        yield_grid = np.clip(src["spyld"][:], 0.0, None)
        source_note = src.attrs.get("source", "RustBCA He-on-W")
    with h5py.File(args.processes, "r+") as dst:
        path = "surface/sputter/he_on_w"
        if path in dst:
            del dst[path]
        group = dst.create_group(path)
        group.create_dataset("E", data=energy)
        group.create_dataset("theta", data=theta)
        group.create_dataset("Y", data=yield_grid)
        group.attrs["source"] = source_note
        group.attrs["units"] = "E: eV; theta: deg from surface normal; Y: atoms/ion"
        group.attrs["Es_eV"] = 8.68
    print(f"installed {path} into {args.processes}")


if __name__ == "__main__":
    main()
