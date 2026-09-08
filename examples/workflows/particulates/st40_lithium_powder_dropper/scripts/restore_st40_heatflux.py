#!/usr/bin/env python3
"""Restore SOLPS heat flux after an equilibrium-only HDF5 correction.

The corrected ST40 file retained the native SOLPS mesh but an earlier
regeneration omitted the optional q datasets.  This utility copies q only
after proving that every array defining the per-cell mapping is identical.
It must not be used between different SOLPS cases or meshes.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


MAPPING_DATASETS = (
    "b2/r_center",
    "b2/z_center",
    "b2/cell_polygons",
    "mesh/cell_index",
    "mesh/triangles",
    "mesh/vtx_r",
    "mesh/vtx_z",
)
HEAT_DATASETS = ("b2/q_par", "b2/q_perp", "mesh/q_par", "mesh/q_perp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(source: Path, target: Path) -> None:
    source_hash = sha256(source)
    with h5py.File(source, "r") as src, h5py.File(target, "r+") as dst:
        for name in MAPPING_DATASETS:
            if name not in src or name not in dst:
                raise RuntimeError(f"mapping gate missing {name}")
            if not np.array_equal(src[name][...], dst[name][...], equal_nan=True):
                raise RuntimeError(
                    f"mapping gate failed for {name}; refusing cross-case q copy"
                )

        for name in HEAT_DATASETS:
            if name not in src:
                raise RuntimeError(f"source heat-flux dataset missing: {name}")
            values = src[name][...]
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"source heat-flux dataset is non-finite: {name}")
            if name in dst:
                del dst[name]
            dataset = dst.create_dataset(name, data=values)
            dataset.attrs["units"] = "W m^-2"
            dataset.attrs["provenance"] = "SOLPS-13589 run_dir"

        dst.attrs["heatflux_source"] = "SOLPS-13589 run_dir"
        dst.attrs["heatflux_source_sha256"] = source_hash
        dst.attrs["heatflux_transfer_geometry_gate"] = "PASS"
        dst.attrs["heatflux_transfer_note"] = (
            "q restored after equilibrium-orientation correction; native "
            "SOLPS cell mapping is byte-identical"
        )

    print(f"PASS: restored physical q_par/q_perp in {target}")
    print(f"  source sha256: {source_hash}")
    print(f"  target sha256: {sha256(target)}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="converted SOLPS-13589 run_dir HDF5 carrying q")
    parser.add_argument(
        "--target", type=Path, required=True,
        help=("orientation-corrected intermediate HDF5 with the identical "
              "native mesh; pass this file to build_core_transit_background.py"),
    )
    args = parser.parse_args()
    restore(args.source, args.target)


if __name__ == "__main__":
    main()
