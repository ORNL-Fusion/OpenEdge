#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def iter_frames(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(f"{path}: malformed NUMBER OF ATOMS block at step {step}")
            natoms = int(handle.readline().strip())

            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"{path}: malformed BOX BOUNDS block at step {step}")
            handle.readline()
            handle.readline()
            handle.readline()

            header = handle.readline().strip().split()[2:]
            col = {name: idx for idx, name in enumerate(header)}
            needed = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]
            missing = [name for name in needed if name not in col]
            if missing:
                raise RuntimeError(f"{path}: missing columns {missing} at step {step}")

            for _ in range(natoms):
                parts = handle.readline().split()
                yield {
                    "timestep": step,
                    "id": int(parts[col["id"]]),
                    "type": int(parts[col["type"]]),
                    "x": float(parts[col["x"]]),
                    "y": float(parts[col["y"]]),
                    "z": float(parts[col["z"]]),
                    "vx": float(parts[col["vx"]]),
                    "vy": float(parts[col["vy"]]),
                    "vz": float(parts[col["vz"]]),
                }


def main():
    parser = argparse.ArgumentParser(description="Extract one particle trajectory from a state dump.")
    parser.add_argument("--input", required=True, help="Path to state dump")
    parser.add_argument("--particle-id", type=int, default=None, help="Explicit particle id to extract")
    parser.add_argument(
        "--select",
        choices=["first", "longest"],
        default="longest",
        help="Fallback trajectory choice when --particle-id is not provided",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    by_id = defaultdict(list)
    for row in iter_frames(Path(args.input)):
        by_id[row["id"]].append(row)

    if not by_id:
        raise SystemExit(f"No particle records found in {args.input}")

    for rows in by_id.values():
        rows.sort(key=lambda row: row["timestep"])

    if args.particle_id is not None:
        if args.particle_id not in by_id:
            raise SystemExit(f"Particle id {args.particle_id} not found in {args.input}")
        pid = args.particle_id
    elif args.select == "first":
        pid = sorted(by_id)[0]
    else:
        pid = max(by_id, key=lambda key: len(by_id[key]))

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestep", "id", "type", "x", "y", "z", "vx", "vy", "vz"],
        )
        writer.writeheader()
        writer.writerows(by_id[pid])

    types_seen = sorted({row["type"] for row in by_id[pid]})
    print(f"Wrote {outpath} for particle {pid}; types seen along trajectory: {types_seen}")


if __name__ == "__main__":
    main()
