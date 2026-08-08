#!/usr/bin/env python3
"""Ballistic validation: z(t) and vz(t) under constant gravity vs analytic.

Exits 0 on PASS, 1 on FAIL (for regression use).
"""

import sys
from pathlib import Path
import numpy as np

G = -9.81
DT = 1.0e-3
VZ_TOL = 1e-9    # m/s, absolute
Z_TOL = 1e-5     # m, absolute


def parse_dump(path):
    t, z, vz = [], [], []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            step = int(lines[i + 1]); i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1]); i += 2
        elif s.startswith("ITEM: ATOMS"):
            col = {k: j for j, k in enumerate(s.split()[2:])}
            v = lines[i + 1].split()
            t.append(step * DT)
            z.append(float(v[col["z"]]))
            vz.append(float(v[col["vz"]]))
            i += 1 + n
        else:
            i += 1
    return np.asarray(t), np.asarray(z), np.asarray(vz)


def main():
    base = Path(__file__).resolve().parent
    t, z, vz = parse_dump(base / "output" / "dump.gravity3d")

    err_vz = np.max(np.abs(vz - G * t))
    err_z = np.max(np.abs(z - 0.5 * G * t * t))

    checks = [("max |vz - g t|", err_vz, VZ_TOL),
              ("max |z - g t^2/2|", err_z, Z_TOL)]
    failed = False
    for name, val, tol in checks:
        ok = val < tol
        failed |= not ok
        print(f"  {name}: {val:.3e} (tol {tol}) {'ok' if ok else 'FAIL'}")
    print(f"  final: vz = {vz[-1]:.6f} (analytic {G*t[-1]:.6f}), "
          f"z = {z[-1]:.6f} (analytic {0.5*G*t[-1]**2:.6f})")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
