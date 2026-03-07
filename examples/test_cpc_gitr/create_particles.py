#!/usr/bin/env python3
"""
Generate initial W particle source for CPC benchmark (GITR convention).

Point source at (0, 0, 1e-5 m) — just above the angled surface.
6 eV kinetic energy in +x and 6 eV in +z (12 eV total, 45 deg).

SPARTA read_particles format:
  id  species_index  x  y  z  vx  vy  vz

species_index: 1 = W (neutral) in our species list

Usage:
    python3 create_particles.py [nP] [output_file]
"""

import sys
import numpy as np

# Physical constants
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19

# W parameters
mass_amu = 184.0
mass_kg = mass_amu * AMU_KG

# Source parameters (GITR convention): 6 eV in x + 6 eV in z
energy_per_component_eV = 6.0
x0, y0, z0 = 0.0, 0.0, 1.0e-5  # just above surface at origin

vx = np.sqrt(2.0 * energy_per_component_eV * EV_J / mass_kg)
vz = vx  # symmetric 45 deg launch

nP = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
outfile = sys.argv[2] if len(sys.argv) > 2 else "particles.dat"

E_total = 0.5 * mass_kg * (vx**2 + vz**2) / EV_J
print(f"Creating {nP} W particles at ({x0},{y0},{z0})")
print(f"Energy = {energy_per_component_eV} eV/component, E_total = {E_total:.2f} eV")
print(f"vx = vz = {vx:.6f} m/s")

with open(outfile, "w") as f:
    f.write("ITEM: TIMESTEP\n")
    f.write("0\n")
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(f"{nP}\n")
    f.write("ITEM: BOX BOUNDS rr rr rr\n")
    f.write("-0.016 0.016\n")
    f.write("-0.016 0.016\n")
    f.write("-0.01 0.031\n")
    f.write("ITEM: ATOMS id type x y z vx vy vz\n")
    for i in range(nP):
        pid = i + 1
        ispecies = 1  # W (first in species list, 1-indexed)
        f.write(f"{pid} {ispecies} {x0} {y0} {z0} {vx} 0.0 {vz}\n")

print(f"Wrote {outfile}")
