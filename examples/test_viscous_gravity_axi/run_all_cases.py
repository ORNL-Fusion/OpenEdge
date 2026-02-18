#!/usr/bin/env python3
"""
Run all viscous+gravity validation cases and post-process with validation.py.

Examples:
  python3 examples/test_viscous_gravity/run_all_cases.py
  python3 examples/test_viscous_gravity/run_all_cases.py --spa src/spa_serial
  python3 examples/test_viscous_gravity/run_all_cases.py --mpi 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_case(example_dir: Path, spa_path: Path, input_file: str, mpi: int) -> None:
    cmd = [str(spa_path)]
    if mpi > 1:
        cmd = ["mpirun", "-np", str(mpi)] + cmd

    in_path = example_dir / input_file
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    print(f"\n==> Running {input_file}")
    with in_path.open("r", encoding="utf-8") as fin:
        proc = subprocess.run(cmd, stdin=fin, cwd=example_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"Case failed: {input_file} (exit {proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all viscous-gravity validation cases.")
    parser.add_argument(
        "--spa",
        type=Path,
        default=Path("/Users/42d/work/openedge-restart/sparta-upstream/src/spa_mpi"),
        help="Path to SPARTA/OpenEdge executable.",
    )
    parser.add_argument("--mpi", type=int, default=1, help="MPI ranks (1 = serial).")
    parser.add_argument("--dt", type=float, default=1.0e-5, help="Timestep [s] for validation.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Figs/viscous_gravity_validation.png"),
        help="Validation output PNG path (relative to example dir unless absolute).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    example_dir = Path(__file__).resolve().parent

    spa_path = args.spa
    if not spa_path.is_absolute():
        spa_path = (repo_root / spa_path).resolve()
    if not spa_path.exists():
        raise FileNotFoundError(f"SPARTA executable not found: {spa_path}")

    cases = ["in.case_1", "in.case_2", "in.case_3", "in.case_4"]
    for case in cases:
        run_case(example_dir, spa_path, case, args.mpi)

    out_path = args.out if args.out.is_absolute() else (example_dir / args.out)
    validation_script = example_dir / "validation.py"

    print("\n==> Running validation")
    proc = subprocess.run(
        [
            sys.executable,
            str(validation_script),
            "--dt",
            f"{args.dt:.12g}",
            "--out",
            str(out_path),
        ],
        cwd=example_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Validation failed (exit {proc.returncode})")

    print(f"\nDone. Figure: {out_path}")


if __name__ == "__main__":
    main()
