#!/usr/bin/env python3
"""Single 1 um Li grain at rest, axi slots (x=Z, y=R). 1 um (not 50 um)
puts the orbit-Coulomb drag term in play: lambda_D ~ 7.5 um > R_d."""

import os

RD = 1.0e-6
RHO_LI = 534.0
MASS = 4.0 / 3.0 * 3.141592653589793 * RD**3 * RHO_LI

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
os.makedirs(base, exist_ok=True)

with open(os.path.join(base, "grain.species"), "w") as f:
    f.write("# 1 um Li grain macroparticle\n")
    f.write("# ID Molwt(amu) Molmass(kg) RotDof RotRel VibDof VibRel "
            "VibTemp(K) specwt charge radius(m) temp(K)\n")
    f.write(f"grain {MASS/1.66053906660e-27:.9e} {MASS:.9e} "
            f"0 0 0 0 0 1 0 {RD:.3e} 300.0\n")


def write_one(name, v):
    with open(os.path.join(base, name), "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n")
        f.write("ITEM: BOX BOUNDS oo ao pp\n-1.0 2.0\n0.0 10.0\n-0.5 0.5\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        f.write(f"1 1 0.5 5.13 0.0 {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")


write_one("grain.rest", (0.0, 0.0, 0.0))          # drag / efield runs
write_one("grain.vphi", (0.0, 0.0, 1000.0))       # free toroidal kinematics
write_one("grain.v100", (100.0, 0.0, 0.0))        # neutral-drag decay
print(f"wrote grain.species (m={MASS:.4e} kg), grain.rest, grain.vphi")
