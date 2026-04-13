#!/usr/bin/env python3
from __future__ import annotations

import math
import argparse
from pathlib import Path


E_CHARGE = 1.602176634e-19
MASS_H = 1.67262192369e-27
Q_OVER_M = E_CHARGE / MASS_H
DT = 2.0e-10
BT = 1.0
V0_STORE = (1.0e5, 2.0e5, 0.0)  # (vR, vZ, vtoroidal)


def boris_push(v: tuple[float, float, float], e: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    qm = Q_OVER_M
    vminus = [v[i] + qm * e[i] * 0.5 * DT for i in range(3)]
    t = [qm * b[i] * 0.5 * DT for i in range(3)]
    t2 = sum(x * x for x in t)
    s = [2.0 * t[i] / (1.0 + t2) for i in range(3)]

    def cross(a: list[float], c: list[float]) -> list[float]:
        return [
            a[1] * c[2] - a[2] * c[1],
            a[2] * c[0] - a[0] * c[2],
            a[0] * c[1] - a[1] * c[0],
        ]

    vprime = cross(vminus, t)
    vprime = [vprime[i] + vminus[i] for i in range(3)]
    vplus = cross(vprime, s)
    vplus = [vplus[i] + vminus[i] for i in range(3)]
    return tuple(vplus[i] + qm * e[i] * 0.5 * DT for i in range(3))


def parse_last_velocity(path: Path) -> tuple[float, float, float]:
    lines = path.read_text().strip().splitlines()
    atom_line = lines[-1].split()
    return float(atom_line[5]), float(atom_line[6]), float(atom_line[7])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="output/boris_basis.dump")
    args = parser.parse_args()

    dump_path = Path(args.dump)
    actual = parse_last_velocity(dump_path)

    # Correct basis: store (R,Z,phi), rotate in RHS basis (R,phi,Z), map back.
    v_rhs = (V0_STORE[0], V0_STORE[2], V0_STORE[1])
    b_rhs = (0.0, BT, 0.0)
    e_rhs = (0.0, 0.0, 0.0)
    correct_rhs = boris_push(v_rhs, e_rhs, b_rhs)
    correct = (correct_rhs[0], correct_rhs[2], correct_rhs[1])

    # Old buggy basis: feed stored order directly to Cartesian Boris.
    wrong = boris_push(V0_STORE, (0.0, 0.0, 0.0), (0.0, 0.0, BT))

    err_correct = math.sqrt(sum((actual[i] - correct[i]) ** 2 for i in range(3)))
    err_wrong = math.sqrt(sum((actual[i] - wrong[i]) ** 2 for i in range(3)))

    print("Actual velocity     :", actual)
    print("Correct-basis model :", correct)
    print("Wrong-basis model   :", wrong)
    print("Error vs correct    :", err_correct)
    print("Error vs wrong      :", err_wrong)

    if err_correct < 1.0 and err_wrong > 1.0e3:
        print("PASS: runtime matches corrected 2D Boris basis mapping.")
        return 0

    print("FAIL: runtime does not match corrected 2D Boris basis mapping.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
