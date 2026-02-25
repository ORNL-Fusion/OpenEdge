#!/usr/bin/env python3
"""
Generate initial W particle source for CPC benchmark.

Point source at origin (0, 0, 1e-5 m — just above the angled surface),
10 eV kinetic energy directed in +z.

SPARTA read_particles format:
  id  species_index  icell  x  y  z  vx  vy  vz

species_index: 1 = W (neutral) in our species list

Usage:
    python3 create_particles.py <nP> <output_file>
"""

import sys
import numpy as np

# Physical constants
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19

# W parameters
mass_amu = 184.0
mass_kg = mass_amu * AMU_KG

# Source parameters from CPC case
energy_eV = 10.0
x0, y0, z0 = 0.0, 0.0, 1.0e-5  # just above surface at origin

# Velocity from 10 eV in z-direction
vz = np.sqrt(2.0 * energy_eV * EV_J / mass_kg)

nP = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
outfile = sys.argv[2] if len(sys.argv) > 2 else "particles.dat"

print(f"Creating {nP} W particles at ({x0},{y0},{z0})")
print(f"Energy = {energy_eV} eV, vz = {vz:.4f} m/s")

with open(outfile, "w") as f:
    f.write("ITEM: TIMESTEP\n")
    f.write("0\n")
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(f"{nP}\n")
    f.write("ITEM: BOX BOUNDS rr rr oo\n")
    f.write("-0.016 0.016\n")
    f.write("-0.016 0.016\n")
    f.write("-0.01 0.031\n")
    f.write("ITEM: ATOMS id type x y z vx vy vz\n")
    for i in range(nP):
        pid = i + 1
        ispecies = 1  # W (first in species list, 1-indexed)
        f.write(f"{pid} {ispecies} {x0} {y0} {z0} 0.0 0.0 {vz}\n")

print(f"Wrote {outfile}")
