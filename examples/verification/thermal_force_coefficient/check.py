#!/usr/bin/env python3
from pathlib import Path
import math

AMU = 1.66053906660e-27
QE = 1.60217646e-19  # Update::echarge used by fix force/thermal
DT = 1.0e-6
GRAD_TI = 1.0
MASS_BG = 2.0 * AMU

species = {
    1: (10.81 * AMU, 1),
    2: (10.81 * AMU, 2),
    3: (10.81 * AMU, 3),
    4: (10.81 * AMU, 4),
    5: (12.011 * AMU, 1),
    6: (12.011 * AMU, 2),
    7: (12.011 * AMU, 3),
    8: (12.011 * AMU, 4),
}


def beta_i(mass_impurity, charge):
    mu = mass_impurity / (mass_impurity + MASS_BG)
    return -3.0 * (
        1.0
        - mu
        - 5.0
        * math.sqrt(2.0)
        * (1.1 * mu**2.5 - 0.35 * mu**1.5)
        * charge**2
    ) / (2.6 - 2.0 * mu + 5.4 * mu**2)


lines = Path("output/particles.1").read_text().splitlines()
start = next(
    index for index, line in enumerate(lines) if line.strip() == "ITEM: ATOMS id type vx vy vz"
) + 1
rows = [line.split() for line in lines[start:]]
assert len(rows) == len(species), f"expected 8 particles, found {len(rows)}"

maximum_relative_error = 0.0
for row in rows:
    particle_id, species_id = map(int, row[:2])
    vx, vy, vz = map(float, row[2:])
    mass, charge = species[species_id]
    expected = beta_i(mass, charge) * QE * GRAD_TI * DT / mass
    relative_error = abs(vz - expected) / max(abs(expected), 1.0e-300)
    maximum_relative_error = max(maximum_relative_error, relative_error)
    assert abs(vx) < 1.0e-14 and abs(vy) < 1.0e-14
    assert relative_error < 2.0e-12, (
        f"particle {particle_id}: vz={vz:.17g}, expected={expected:.17g}, "
        f"relative error={relative_error:.3e}"
    )

print(f"PASS: DIVIMP ion thermal coefficient, max relative error {maximum_relative_error:.3e}")
