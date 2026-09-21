#!/usr/bin/env python3
"""Build the deterministic ST40 grain-size response scan.

This is deliberately a response scan, not a claimed experimental particle-size
distribution.  Every diameter receives the same 30 launch conditions so that
diameter is the only changed input.  A measured number- or mass-weighted PSD can
be applied later without rerunning the individual trajectories.

OpenEdge initializes grain mass, radius, and temperature from the particle
species.  The read-particles file therefore assigns one species type to each
diameter while prescribing only ID, position, and velocity.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from build_core_transit_cohort import build as build_launch_grid


ROOT = Path(__file__).resolve().parents[1]
PARTICLES = ROOT / "input" / "core_transit_size_scan.particles"
MANIFEST = ROOT / "input" / "core_transit_size_scan.csv"
SPECIES = ROOT / "input" / "grains_size_scan.species"
RECYCLE = ROOT / "input" / "li_size_scan.recycle"

# Wide enough to locate the survival transition around the 100-um baseline.
# These are diagnostic bins, not a measured ST40 powder distribution.
DIAMETERS_UM = (40, 60, 80, 100, 125, 150, 200, 250, 300)
LI_DENSITY_KG_M3 = 534.0
LI_ATOM_MASS_AMU = 6.94
ATOMIC_MASS_KG = 1.66053906660e-27
INITIAL_TEMPERATURE_K = 293.15


def species_id(diameter_um: int) -> str:
    return f"grain_d{diameter_um:03d}um"


def grain_mass(radius_m: float) -> float:
    return (4.0 / 3.0) * math.pi * LI_DENSITY_KG_M3 * radius_m**3


def build_rows() -> list[dict[str, float | int | str]]:
    launches = build_launch_grid()
    rows: list[dict[str, float | int | str]] = []
    particle_id = 1
    for type_index, diameter_um in enumerate(DIAMETERS_UM, start=1):
        radius_m = 0.5e-6 * diameter_um
        mass_kg = grain_mass(radius_m)
        for launch in launches:
            rows.append(
                {
                    "id": particle_id,
                    "species_type": type_index,
                    "species_id": species_id(diameter_um),
                    "diameter_um": diameter_um,
                    "radius_um": 0.5 * diameter_um,
                    "mass_kg": mass_kg,
                    "atoms_per_grain": mass_kg
                    / (LI_ATOM_MASS_AMU * ATOMIC_MASS_KG),
                    "diagnostic_number_weight": 1.0,
                    **{key: value for key, value in launch.items() if key != "id"},
                }
            )
            particle_id += 1
    return rows


def write_species() -> None:
    with SPECIES.open("w") as stream:
        stream.write(
            "# Deterministic Li grain-size response scan.\n"
            "# Diagnostic bins only: this is not a measured ST40 PSD.\n"
            "# ID Molwt(amu) Molmass(kg) RotDof RotRel VibDof VibRel "
            "VibTemp(K) specwt charge radius(m) temp(K)\n"
        )
        for diameter_um in DIAMETERS_UM:
            radius_m = 0.5e-6 * diameter_um
            mass_kg = grain_mass(radius_m)
            mass_amu = mass_kg / ATOMIC_MASS_KG
            stream.write(
                f"{species_id(diameter_um)} {mass_amu:.16g} {mass_kg:.16g} "
                f"0 0 0 0 0 1 0 {radius_m:.16g} {INITIAL_TEMPERATURE_K:.8g}\n"
            )


def write_recycle() -> None:
    species = ["Li", "Li+", "Li2+", "Li3+"] + [
        species_id(diameter_um) for diameter_um in DIAMETERS_UM
    ]
    with RECYCLE.open("w") as stream:
        stream.write(
            "# Li on the ST40 wall. All species stick in this Stage-1 scan.\n\n"
        )
        stream.write("\n\n".join(f"{atom} --> Li\nA 0.0" for atom in species))
        stream.write("\n")


def write_particles_and_manifest(rows: list[dict[str, float | int | str]]) -> None:
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    with PARTICLES.open("w") as stream:
        stream.write(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n"
            f"{len(rows)}\n"
            "ITEM: BOX BOUNDS oo ao pp\n"
            "-1.0 1.0\n0.0 1.05\n-0.05 0.05\n"
            "ITEM: ATOMS id type x y z vx vy vz\n"
        )
        for row in rows:
            # Axisymmetric OpenEdge slots are x=Z, y=R, z=phi.
            stream.write(
                f"{row['id']} {row['species_type']} "
                f"{row['z0_m']:.14g} {row['r0_m']:.14g} 0 "
                f"{row['vz0_m_s']:.14g} {row['vr0_m_s']:.14g} 0\n"
            )


def main() -> None:
    rows = build_rows()
    write_species()
    write_recycle()
    write_particles_and_manifest(rows)
    launches_per_size = len(rows) // len(DIAMETERS_UM)
    print(f"wrote {PARTICLES.relative_to(ROOT)}: {len(rows)} deterministic grains")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"wrote {SPECIES.relative_to(ROOT)} and {RECYCLE.relative_to(ROOT)}")
    print(
        f"diagnostic diameters [um]: {', '.join(map(str, DIAMETERS_UM))}; "
        f"{launches_per_size} identical launch conditions per diameter"
    )


if __name__ == "__main__":
    main()
