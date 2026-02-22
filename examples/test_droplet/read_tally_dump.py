#!/usr/bin/env python3
"""Read a SPARTA dump-tally file and export CSV data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_tally_dump(path: Path):
    rows = []
    step_counts = []

    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line != "ITEM: TIMESTEP":
            i += 1
            continue

        if i + 1 >= len(lines):
            break
        timestep = int(lines[i + 1].strip())
        i += 2

        if i >= len(lines) or lines[i].strip() != "ITEM: NUMBER OF TALLIES":
            raise ValueError(f"Malformed file near timestep {timestep}: missing NUMBER OF TALLIES")
        if i + 1 >= len(lines):
            raise ValueError(f"Malformed file near timestep {timestep}: missing tally count")
        ntally = int(lines[i + 1].strip())
        step_counts.append((timestep, ntally))
        i += 2

        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Malformed file near timestep {timestep}: missing BOX BOUNDS")
        i += 4  # header + 3 bounds lines

        if i >= len(lines) or not lines[i].startswith("ITEM: TALLIES"):
            raise ValueError(f"Malformed file near timestep {timestep}: missing TALLIES header")
        col_names = lines[i].split()[2:]
        i += 1

        for _ in range(ntally):
            if i >= len(lines):
                raise ValueError(f"Malformed file near timestep {timestep}: missing tally row")
            parts = lines[i].split()
            if len(parts) != len(col_names):
                raise ValueError(
                    f"Malformed row at timestep {timestep}: expected {len(col_names)} fields, got {len(parts)}"
                )
            rows.append([timestep] + parts)
            i += 1

    return col_names if rows or step_counts else [], rows, step_counts


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input SPARTA tally dump file")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix path (default: <input basename>)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_prefix = args.out_prefix if args.out_prefix else input_path.with_suffix("")

    col_names, rows, step_counts = parse_tally_dump(input_path)
    if not step_counts:
        raise SystemExit(f"No tally blocks found in: {input_path}")

    records_path = Path(f"{out_prefix}.records.csv")
    summary_path = Path(f"{out_prefix}.steps.csv")

    write_csv(summary_path, ["timestep", "number_of_tallies"], step_counts)
    write_csv(records_path, ["timestep"] + col_names, rows)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {records_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
#python3 build_transfer_matrix.py \
#    --in-I tmp.collision.wall.I.mode1.nemit200 \
#    --in-O tmp.collision.wall.O.mode1.nemit200 \
#    --launched-I 200 \
#    --launched-O 0
