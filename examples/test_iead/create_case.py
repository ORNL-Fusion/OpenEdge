#!/usr/bin/env python3
"""
Generate OpenEdge IEAD test cases for comparison with sheath_tracker.f90.

Creates:
  - wall.surf               flat wall surface at z = Lz
  - particles_alpha{N}.dat  initial particle files for each tilt angle
  - in.iead_alpha{N}        input files for each tilt angle

Physics setup matches sheath_tracker.f90:
  Te = Ti = 5 eV,  ni = 1e19 m^-3,  B = 0.5 T
  Ta ions: Ta2+ (8%), Ta3+ (62%), Ta4+ (30%), mass = 180.948 amu
  Tilt angles: 0, 45, 85 deg from wall normal
  Sheath: Borodkina model (default) with self-consistent floating potential

Note: The Fortran code uses a fixed MPS scale length (rho_D) for all
angles, while OpenEdge's Borodkina model scales lmps = rho_D/cos(alpha).
This produces physically meaningful differences at oblique angles.

Usage:
    python3 create_case.py [--np 10000] [--seed 12345]
"""

import argparse
import math
import os

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Physical constants
# ═══════════════════════════════════════════════════════════════
QE   = 1.602176634e-19
AMU  = 1.660539067e-27
EPS0 = 8.854187817e-12
ME   = 9.1093837015e-31
PI   = math.pi

# ═══════════════════════════════════════════════════════════════
#  Plasma / sheath parameters (matching sheath_tracker.f90)
# ═══════════════════════════════════════════════════════════════
Te_eV  = 5.0
Ti_eV  = 5.0
ni0    = 1.0e19
Bmag   = 0.5
Ta_amu = 180.948
mD_amu = 2.0       # background D+ sets sheath scale

# Species: (name, charge Z, fraction)
SPECIES = [
    ("Ta2+", 2, 0.08),
    ("Ta3+", 3, 0.62),
    ("Ta4+", 4, 0.30),
]

# Tilt angles (degrees from wall normal)
ANGLES = [0, 45, 85]

# ═══════════════════════════════════════════════════════════════
#  Derived scales
# ═══════════════════════════════════════════════════════════════
Te   = Te_eV * QE
Ti   = Ti_eV * QE
m_Ta = Ta_amu * AMU
mD   = mD_amu * AMU

lD = math.sqrt(EPS0 * Te / (ni0 * QE**2))

cs_D  = math.sqrt(Te / mD)
wci_D = QE * Bmag / mD
rho_D = cs_D / wci_D

Lz = 25.0 * rho_D + 40.0 * lD

# Per-species sound speed and thermal velocity
sp_cs  = [math.sqrt(Z * Te / m_Ta) for _, Z, _ in SPECIES]
sp_vth = [math.sqrt(Ti / m_Ta)] * len(SPECIES)  # same mass, same Ti
sp_wci = [Z * QE * Bmag / m_Ta for _, Z, _ in SPECIES]
sp_rho = [cs / wci for cs, wci in zip(sp_cs, sp_wci)]

# Time step: match Fortran dt = 0.1 * min(2*pi/wci_max, lD/cs_min)
wci_max = max(sp_wci)
cs_min  = min(sp_cs)
dt_fortran = 0.1 * min(2.0 * PI / wci_max, lD / cs_min)

# OpenEdge Boris subcycling
# Use timestep * boris_subcycles to give dt_sub ~ dt_fortran
TIMESTEP = 1.0e-8
BORIS_SUBCYCLES = int(round(TIMESTEP / dt_fortran))
BORIS_SUBCYCLES = max(BORIS_SUBCYCLES, 10)
# BORIS_SUBCYCLES *= 10  # extra resolution for thin Debye sheath
dt_sub = TIMESTEP / BORIS_SUBCYCLES

# Run steps: need enough time for slowest case (85 deg tilt)
# Worst case: vz ~ cs_min * cos(85) ~ 200 m/s, transit ~ 80 us
# Use 1 ms total (matches Fortran MAX_STEPS * dt)
RUN_STEPS = int(1.0e-3 / TIMESTEP)

# Domain: centered at R=1m to avoid cylindrical singularity
# (the plasma/fields compute stores B in cylindrical and converts
#  to Cartesian using phi angle; at R=1 the conversion is identity)
X_LO, X_HI = 0.99, 1.01
Y_LO, Y_HI = -0.01, 0.01
Z_LO = -0.001
Z_HI = Lz + 0.002  # small margin above wall

NX, NY, NZ = 4, 4, 20

# Floating potential (matching Fortran sheath formula)
phi_float_eV = 0.5 * math.log(mD / (2.0 * PI * ME) / (1.0 + Ti_eV / Te_eV)) * Te_eV


def print_banner():
    print("=" * 60)
    print("  IEAD test case generator")
    print("=" * 60)
    print(f"  Te = Ti  = {Te_eV} eV")
    print(f"  ni       = {ni0:.0e} m^-3")
    print(f"  |B|      = {Bmag} T")
    print(f"  lam_D    = {lD*1e6:.2f} um")
    print(f"  rho_D+   = {rho_D*1e3:.3f} mm")
    print(f"  Lz       = {Lz*1e3:.3f} mm")
    print(f"  phi_float= {phi_float_eV:.2f} eV")
    print(f"  dt_fort  = {dt_fortran:.3e} s")
    print(f"  dt_sub   = {dt_sub:.3e} s  (subcycles={BORIS_SUBCYCLES})")
    print(f"  timestep = {TIMESTEP:.1e} s")
    print(f"  run      = {RUN_STEPS} steps ({RUN_STEPS*TIMESTEP*1e3:.2f} ms)")
    print("-" * 60)
    for i, (name, Z, frac) in enumerate(SPECIES):
        print(f"  {name}  frac={frac:.2f}  cs={sp_cs[i]:.0f} m/s  "
              f"rho={sp_rho[i]*1e3:.2f} mm  wci={sp_wci[i]:.2e} rad/s")
    print("=" * 60)


def create_wall_surf(fname="wall.surf"):
    """Flat wall at z=Lz, two triangles covering the x-y domain.
    Normal points -z (toward incoming particles)."""
    pts = [
        (X_LO, Y_LO, Lz),
        (X_HI, Y_LO, Lz),
        (X_HI, Y_HI, Lz),
        (X_LO, Y_HI, Lz),
    ]
    with open(fname, "w") as f:
        f.write("# Flat wall surface for IEAD test\n\n")
        f.write("4 points\n")
        f.write("2 triangles\n\n")
        f.write("Points\n\n")
        for i, (x, y, z) in enumerate(pts):
            f.write(f"{i+1} {x:.15e} {y:.15e} {z:.15e}\n")
        f.write("\nTriangles\n\n")
        # Winding order gives normal in -z (toward particles)
        f.write("1 1 3 2\n")
        f.write("2 1 4 3\n")
    print(f"  Created {fname}")


def create_particles(alpha_deg, Np_total, seed, fname):
    """Generate initial particle file matching Fortran initialization."""
    rng = np.random.default_rng(seed)
    alpha_rad = math.radians(alpha_deg)
    bhat = np.array([math.sin(alpha_rad), 0.0, math.cos(alpha_rad)])

    with open(fname, "w") as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{Np_total}\n")
        bnd = "pp pp oo"
        f.write(f"ITEM: BOX BOUNDS {bnd}\n")
        f.write(f"{X_LO} {X_HI}\n")
        f.write(f"{Y_LO} {Y_HI}\n")
        f.write(f"{Z_LO} {Z_HI}\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")

        pid = 0
        for isp, (name, Z, frac) in enumerate(SPECIES):
            Np_sp = max(100, int(round(Np_total * frac)))
            cs  = sp_cs[isp]
            vth = sp_vth[isp]
            v_drift = cs * bhat

            for _ in range(Np_sp):
                pid += 1
                # Position: random in x,y, at z=epsilon
                x = rng.uniform(X_LO + 1e-6, X_HI - 1e-6)
                y = rng.uniform(Y_LO + 1e-6, Y_HI - 1e-6)
                z = 1.0e-5

                # Velocity: drift + thermal, reject-resample until vz > 0
                vx = v_drift[0] + vth * rng.standard_normal()
                vy = v_drift[1] + vth * rng.standard_normal()
                while True:
                    vz = v_drift[2] + vth * rng.standard_normal()
                    if vz > 0.0:
                        break

                itype = isp + 1  # 1-indexed species
                f.write(f"{pid} {itype} {x:.10e} {y:.10e} {z:.10e} "
                        f"{vx:.10e} {vy:.10e} {vz:.10e}\n")

    print(f"  Created {fname}  ({pid} particles)")


def create_input(alpha_deg, Np_total, fname):
    """Generate OpenEdge input file for one tilt angle."""
    alpha_rad = math.radians(alpha_deg)
    Bx = Bmag * math.sin(alpha_rad)
    By = 0.0
    Bz = Bmag * math.cos(alpha_rad)

    pfname = f"particles_alpha{alpha_deg}.dat"

    content = f"""\
################################################################################
# IEAD test: Ta mixture Chodura sheath, alpha = {alpha_deg} deg
#
# Compare against sheath_tracker.f90 sheath model.
# Physics: Te=Ti={Te_eV} eV, ni={ni0:.0e}, B={Bmag} T, Ta2+/Ta3+/Ta4+
# Wall at z = {Lz:.6e} m, sheath entrance at z = 0
################################################################################

# ---- Simulation setup ----
seed                    12345
dimension               3
units                   si

boundary                p p o
global                  gridcut 0.0 comm/sort yes
global                  fnum 1.0

create_box              {X_LO} {X_HI} {Y_LO} {Y_HI} {Z_LO} {Z_HI:.15e}
create_grid             {NX} {NY} {NZ}
balance_grid            rcb cell

# ---- Species ----
species                 plasma.species Ta2+ Ta3+ Ta4+
mixture                 allTa Ta2+ Ta3+ Ta4+
mixture                 allTa Ta2+ frac 0.08
mixture                 allTa Ta3+ frac 0.62
mixture                 allTa Ta4+ frac 0.30

# ---- Flat wall surface at z = Lz ----
read_surf               wall.surf

# ---- Surface collision: vanish with per-particle CSV log ----
surf_collide            absorb vanish log yes file output/vanish_alpha{alpha_deg}
surf_modify             all collide absorb

# ---- Plasma fields (constant, uniform) ----
compute                 cplasma plasma/fields all constant &
                        magnetic_field {Bx:.15e} {By:.15e} {Bz:.15e} &
                        electric_field 0 0 0 &
                        temp_e {Te_eV} temp_i {Ti_eV} &
                        dens_e {ni0:.6e} dens_i {ni0:.6e} &
                        parrflow 0.0 &
                        bx by bz ex ey ez temp_e temp_i dens_i dens_e parrflow

# ---- Magnetic field for Boris pusher ----
fix                     bfield bfield/grid c_cplasma[1] c_cplasma[2] c_cplasma[3]
global                  bfield grid bfield 0

# ---- Electric field (background = 0, sheath adds per-particle) ----
fix                     efield efield/grid c_cplasma[4] c_cplasma[5] c_cplasma[6]
global                  efield grid efield 0

# ---- Charged-particle pusher: Boris with sheath kick ----
compute                 cgeom nearest_surf/grid all all dist nx ny nz surfidx
global                  pusher mode boris plasma cplasma subcycles {BORIS_SUBCYCLES} &
                        sheath kick geom cgeom mD_amu {mD_amu}

# ---- Load particles ----
read_particles          {pfname} 0

# ---- Diagnostics ----
shell                   mkdir -p output

# Particle count stats
stats                   1000
stats_style             step cpu np

# ---- Time integration ----
timestep                {TIMESTEP:.6e}
run                     {RUN_STEPS}
"""
    with open(fname, "w") as f:
        f.write(content)
    print(f"  Created {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate IEAD test cases")
    parser.add_argument("--np", type=int, default=10000,
                        help="Total particle count (default: 10000)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed for particle generation")
    args = parser.parse_args()

    print_banner()

    os.makedirs("output", exist_ok=True)
    create_wall_surf("wall.surf")

    for alpha in ANGLES:
        pfname = f"particles_alpha{alpha}.dat"
        infname = f"in.iead_alpha{alpha}"
        create_particles(alpha, args.np, args.seed + alpha, pfname)
        create_input(alpha, args.np, infname)

    print("\nTo run (one angle at a time):")
    print(f"  mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.iead_alpha0")
    print(f"  mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.iead_alpha45")
    print(f"  mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.iead_alpha85")
    print("\nThen compare:")
    print(f"  python3 compare_iead.py")


if __name__ == "__main__":
    main()
