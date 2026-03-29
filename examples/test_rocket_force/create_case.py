#!/usr/bin/env python3
"""
Create test case for rocket force validation.

Setup: single droplet launched with the same initial particle state used in
`test_droplet`, but evolved across three rocket-force values eta = 0.0, 0.5,
1.0. Unlike the earlier synthetic version of this case, the inputs here reuse
the real `plasma.h5` heat-flux data and equilibrium-backed magnetic field from
`examples/test_evaporation/input`, so the rocket-force test follows the same
plasma/background path as `test_droplet`.

Usage:
    python3 create_case.py
    # then run each case:
    mpirun -np 1 spa_mpi -in in.eta_0p0
    mpirun -np 1 spa_mpi -in in.eta_0p5
    mpirun -np 1 spa_mpi -in in.eta_1p0
    # then compare:
    python3 compare_rocket.py
"""

from pathlib import Path

import numpy as np

# ======================================================================
# Parameters
# ======================================================================
# Domain: R in [2, 6], Z in [-4, 3.8] (matching CAT geometry scale)
r_min, r_max = 2.0, 6.0
z_min, z_max = -4.0, 3.8
base_dir = Path(__file__).resolve().parent
real_plasma_file = Path("../test_evaporation/input/plasma.h5")
real_equilibrium_file = Path("../test_evaporation/input/g000001.00001_symm.X4.equ")
wall_surface_file = Path("wall.surf")
core_surface_file = Path("core.surf")
heatflux_scale = 4.0

# Droplet initial conditions (aligned with test_droplet scale)
rd0 = 2.5e-3     # m
rho_li = 534.0   # kg/m^3
md0 = rho_li * (4.0/3.0) * np.pi * rd0**3
Td0 = 773.15     # K
amu_kg = 1.66053906660e-27
molwt0 = md0 / amu_kg

# Droplet start position/velocity patterned after test_droplet/source.
x0 = 3.41714   # R
y0 = -3.53336  # Z
vx0 = -3.39584
vy0 = 21.0707
vphi0 = 0.0

if not (base_dir / real_plasma_file).exists():
    raise FileNotFoundError(f"Missing plasma input: {base_dir / real_plasma_file}")
if not (base_dir / real_equilibrium_file).exists():
    raise FileNotFoundError(f"Missing equilibrium input: {base_dir / real_equilibrium_file}")
if not (base_dir / wall_surface_file).exists():
    raise FileNotFoundError(f"Missing wall surface: {base_dir / wall_surface_file}")

print(f"Using plasma file: {real_plasma_file}")
print(f"Using equilibrium file: {real_equilibrium_file}")
print(f"Using wall surface: {wall_surface_file}")
print(f"Using heatflux/scale = {heatflux_scale}")

# ======================================================================
# Write species file
# ======================================================================
species_file = "droplet.species"
with open(species_file, 'w') as f:
    f.write("# Species data\n")
    f.write("# ID     Molwt (amu)           Molmass (kg)           Rotational dof    RotRel    Vibrational dof    VibRel    VibTemp (K)    species wt    charge\n")
    f.write(f"drop_1     {molwt0:.15e}     {md0:.15e}      0      0.0    0      0.0    0.0        1.0        0.0\n")
print(f"Written {species_file}")

# ======================================================================
# Write particle source file (single droplet)
# ======================================================================
source_file = "source.1"
with open(source_file, 'w') as f:
    f.write("ITEM: TIMESTEP\n")
    f.write("0\n")
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write("1\n")
    f.write("ITEM: BOX BOUNDS rr rr pp\n")
    f.write(f"{r_min} {r_max}\n")
    f.write(f"{z_min} {z_max}\n")
    f.write("-0.05 0.05\n")
    f.write("ITEM: ATOMS id type x y z vx vy vz\n")
    z_mid = 0.0  # 2D, z doesn't matter
    f.write(f"1 1 {x0} {y0} {z_mid} {vx0} {vy0} {vphi0}\n")
print(f"Written {source_file}")

# ======================================================================
# Write input scripts for eta = 0, 0.5, 1.0
# ======================================================================
dt = 1e-5
t_end = 10.5
nsteps = int(t_end / dt)
dump_every = 10

eta_values = [0.0, 0.5, 1.0]

for eta in eta_values:
    eta_str = f"{eta:.1f}".replace(".", "p")
    fname = f"in.eta_{eta_str}"
    dump_file = f"traj_eta_{eta_str}"

    with open(fname, 'w') as f:
        f.write(f"# Rocket force test: eta = {eta}\n")
        f.write("seed                12345\n")
        f.write("dimension           2\n")
        f.write("boundary            r r p\n")
        f.write("global              gridcut 0.0 comm/sort no\n")
        f.write(f"create_box          {r_min} {r_max} {z_min} {z_max} -0.05 0.05\n\n")
        f.write("create_grid         100 100 1\n")
        f.write("balance_grid        rcb cell\n\n")
        f.write(f"species {species_file} drop_1\n")
        f.write("mixture DropletSource drop_1 frac 1.0 nrho 1\n\n")
        f.write("# Real wall surface: remove particle when it leaves the vessel\n")
        f.write(f"read_surf {wall_surface_file.as_posix()} group surfID particle check\n")
        f.write(f"surf_collide wallVanish vanish log yes file collide_vanish_eta_{eta_str}.csv\n")
        f.write("surf_modify surfID collide wallVanish react none\n\n")
        
        f.write(f"read_surf {core_surface_file.as_posix()} invert group coreID particle check\n")
        f.write(f"surf_collide coreVanish vanish \n")
        f.write("surf_modify coreID collide coreVanish react none\n\n")


        f.write("read_particles source.1 0\n\n")
        f.write(f"variable dt equal {dt}\n")
        f.write(f"variable N equal {nsteps}\n\n")
        f.write("variable pmass particle mass\n\n")
        f.write("# Plasma fields from the real CAT/SOLPS-derived plasma file\n")
        f.write(f"compute cp plasma/fields all file {real_plasma_file.as_posix()} &\n")
        f.write(f"    equilibrium {real_equilibrium_file.as_posix()} &\n")
        f.write("    bx by bz temp_e dens_e temp_i dens_i parrflow er et ez\n\n")
        f.write(f"# Evaporation with rocket force eta={eta}\n")
        f.write(f"fix fevap evaporation 1 DropletSource &\n")
        f.write(f"    mass {md0:.6e} radius {rd0} temp {Td0} &\n")
        f.write(f"    heatflux/compute cp heatflux/scale {heatflux_scale} &\n")
        f.write(f"    rocket_eta {eta}\n\n")
        f.write("# Drag driven by the same real plasma and equilibrium B-field\n")
        f.write("fix fdrag drag 1 2.0 1 &\n")
        f.write("    plasma c_cp[4] c_cp[6] c_cp[7] c_cp[8] &\n")
        f.write("    bfield c_cp[1] c_cp[3] c_cp[2] &\n")
        f.write(f"    gravity 0 -9.8 0 model epstein mass {md0:.6e} radius {rd0} temp {Td0} &\n")
        f.write("    cylindrical yes\n\n")
        f.write(f"dump 10 particle all {dump_every} {dump_file} &\n")
        f.write("    id type x y z vx vy vz v_pmass temp radius\n\n")
        f.write("timestep ${dt}\n")
        f.write(f"stats {dump_every}\n")
        f.write("stats_style step cpu np\n\n")
        f.write("variable npart equal np\n")
        f.write("fix fhalt halt 1 v_npart <= 0\n\n")
        f.write("run ${N}\n")

    print(f"Written {fname} (eta={eta})")

print("\nDone. Run with:")
print("  mpirun -np 1 spa_mpi -in in.eta_0p0")
print("  mpirun -np 1 spa_mpi -in in.eta_0p5")
print("  mpirun -np 1 spa_mpi -in in.eta_1p0")
print("Then: python3 compare_rocket.py")
