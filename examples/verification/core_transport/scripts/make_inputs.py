#!/usr/bin/env python3
"""Synthetic 1-D core-transport cases shared by the Aurora (Fortran) reference and the C++ port.

Units are Aurora's (cm, s): rvol grid 0..50 cm, five-state chain with bell-shaped
ionization/recombination profiles, D and v varying in radius, an edge neutral source,
parallel loss outside 40 cm.  Written with full double precision so both drivers see
identical numbers.
"""
import os, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "input")
os.makedirs(out, exist_ok=True)

ir, nion, a = 61, 5, 50.0
rr = np.linspace(0.0, a, ir)
rr[1:] += 0.3 * np.sin(np.pi * rr[1:] / a) ** 2 * (a / ir)      # mildly non-uniform
x = rr / a
diff = np.array([1.0e4 * (0.5 + x**2) * (1 + 0.1 * z) for z in range(nion)])
conv = np.array([-2.0e2 * x * (1 + 0.2 * z) for z in range(nion)])  # inward pinch
s_rates = np.array([2.0e3 * (1 + z) * np.exp(-((x - 0.2 * z) / 0.5) ** 2) for z in range(nion)])
s_rates[-1] = 0.0                                                    # no ionization out of the last state
r_rates = np.array([1.0e3 * (1 + 0.5 * z) * (0.2 + x**2) for z in range(nion)])  # index z: z+1 -> z
r_rates[-1] = 0.0
n0 = 1.0e8 * np.exp(-((rr - 46.0) / 2.5) ** 2)                        # neutral density / source profile
pl = np.where(rr > 40.0, 5.0e2 * (rr - 40.0) / 10.0, 0.0)
src = np.array([3.0e11 * np.exp(-((rr - 44.0) / 3.0) ** 2) * (z == 2) for z in range(nion)])
n_init = np.array([1.0e9 * (1 - x**2) * (z + 1) / nion for z in range(nion)])


def write(name, dt, nsteps, dlen, evolve, use_src, use_n0, use_pl, r0=None):
    r = r_rates.copy()
    if r0 is not None:
        r[0] = r0
    with open(os.path.join(out, f"case_{name}.dat"), "w") as f:
        f.write(f"{ir} {nion} {nsteps}\n")
        f.write(f"{dt!r} {dlen!r} {int(evolve)} {int(use_src)} {int(use_n0)} {int(use_pl)}\n")
        for arr in (rr, n0, pl, diff, conv, s_rates, r, src, n_init):
            for row in np.atleast_2d(arr):
                f.write(" ".join(repr(float(v)) for v in row) + "\n")
    print("wrote", name)


# Aurora-comparable (no per-state src): fixed neutral density; transported neutrals with a volume
# source (Aurora run with a padded fixed state, see ref_aurora/aurora_ref.f90); closed boundary
write("neutral", 1.0e-4, 300, 1.0, evolve=False, use_src=False, use_n0=True, use_pl=True)
write("evolve", 1.0e-4, 300, 1.0, evolve=True, use_src=False, use_n0=True, use_pl=True)
write("closed", 1.0e-4, 300, 0.0, evolve=True, use_src=False, use_n0=True, use_pl=False)
# C++-only: per-state volume source (the fix's injection mode) with full recombination, conservation checked
write("source", 1.0e-4, 300, 1.0, evolve=True, use_src=True, use_n0=False, use_pl=True)
write("source_fixed_n0", 1.0e-4, 300, 1.0, evolve=False, use_src=True, use_n0=True, use_pl=True, r0=0.0)
