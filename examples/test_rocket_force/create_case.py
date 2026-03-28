#!/usr/bin/env python3
"""
Create test case for rocket force validation.

Setup: single droplet in a synthetic plasma with a known temperature gradient.
The case is intentionally self-contained, but it mirrors the single-droplet
transport style used in test_droplet: nonzero initial launch velocity plus
Epstein drag and gravity. Three cases are run with eta = 0.0, 0.5, 1.0.

The temperature gradient is purely in the +R direction, so the rocket force
should deflect the droplet in the -R direction while it falls under gravity.

Usage:
    python3 create_case.py
    # then run each case:
    mpirun -np 1 spa_mpi -in in.eta_0p0
    mpirun -np 1 spa_mpi -in in.eta_0p5
    mpirun -np 1 spa_mpi -in in.eta_1p0
    # then compare:
    python3 compare_rocket.py
"""

import numpy as np
import h5py
import os

# ======================================================================
# Parameters
# ======================================================================
# Domain: R in [2, 6], Z in [-4, 3.8] (matching CAT geometry scale)
r_min, r_max = 2.0, 6.0
z_min, z_max = -4.0, 3.8
nr, nz = 200, 400

r_arr = np.linspace(r_min, r_max, nr)
z_arr = np.linspace(z_min, z_max, nz)

# Constant background plasma used by fix drag (synthetic, no SOLPS input)
ne0 = 1.5746e20  # m^-3
te0 = 10.0       # eV
ti0 = 10.0       # eV
vp0 = 0.0        # m/s
br0 = 0.0
bt0 = 0.0
bz0 = 0.0

# Constant heat flux. Keep this lower than the original bare rocket-force test
# so the droplet does not enter the near-zero-mass regime too quickly.
q_mag_val = 5e6  # W/m^2

# Temperature gradient written in the same units as plasma.h5: eV/m.
# The rocket-force implementation currently uses grad_Te for direction only.
grad_te_r_val = 100.0
grad_te_z_val = 0.0

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

# ======================================================================
# Write heatflux HDF5 with grad_Te
# ======================================================================
hf_file = "heatflux_rocket.h5"
print(f"Writing {hf_file}...")
with h5py.File(hf_file, 'w') as f:
    # Write both root-level and grouped coordinate datasets so the file is
    # accepted by the different heat-flux readers used across OpenEdge trees.
    f.create_dataset('R', data=r_arr)
    f.create_dataset('Z', data=z_arr)
    f.create_dataset('r', data=r_arr)
    f.create_dataset('z', data=z_arr)
    grid = f.create_group('grid')
    grid.create_dataset('Rc', data=r_arr)
    grid.create_dataset('Zc', data=z_arr)
    rr, zz = np.meshgrid(r_arr, z_arr)
    grid.create_dataset('R', data=rr)
    grid.create_dataset('Z', data=zz)

    # Uniform heat flux
    q_mag = np.full((nz, nr), q_mag_val)
    f.create_dataset('q_mag', data=q_mag)

    # Uniform temperature gradient (dTe/dR > 0, dTe/dZ = 0)
    grad_r = np.full((nz, nr), grad_te_r_val)
    grad_z = np.full((nz, nr), grad_te_z_val)
    f.create_dataset('grad_te_r', data=grad_r)
    f.create_dataset('grad_te_z', data=grad_z)

print(f"  q_mag = {q_mag_val:.1e} W/m^2")
print(f"  grad_te_r = {grad_te_r_val:.3e}")
print(f"  grad_te_z = {grad_te_z_val}")

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
t_end = 0.05
nsteps = int(t_end / dt)
dump_every = 50

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
        f.write("read_particles source.1 0\n\n")
        f.write(f"variable dt equal {dt}\n")
        f.write(f"variable N equal {nsteps}\n\n")
        f.write("variable pmass particle mass\n\n")
        f.write(f"variable te grid \"{te0}\"\n")
        f.write(f"variable ti grid \"{ti0}\"\n")
        f.write(f"variable ni grid \"{ne0}\"\n")
        f.write(f"variable vp grid \"{vp0}\"\n")
        f.write(f"variable Br grid \"{br0}\"\n")
        f.write(f"variable Bt grid \"{bt0}\"\n")
        f.write(f"variable Bz grid \"{bz0}\"\n\n")
        f.write(f"# Evaporation with rocket force eta={eta}\n")
        f.write(f"fix fevap evaporation 1 DropletSource &\n")
        f.write(f"    mass {md0:.6e} radius {rd0} temp {Td0} &\n")
        f.write(f"    heatflux/file {hf_file} &\n")
        f.write(f"    rocket_eta {eta}\n\n")
        f.write("# test_droplet-like transport with synthetic constant plasma inputs\n")
        f.write(f"fix fdrag drag 1 2.0 1 plasma te ti ni vp bfield Br Bt Bz &\n")
        f.write(f"    gravity 0 -9.81 0 model epstein mass {md0:.6e} radius {rd0} temp {Td0} &\n")
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
