#!/usr/bin/env python3
"""
Create test case for rocket force validation.

Setup: single droplet in a uniform plasma with a known temperature gradient.
The rocket force should push the droplet in the -grad_Te direction (away from
hot plasma).  Three cases are run with eta = 0.0, 0.5, 1.0.

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

# Uniform plasma conditions
ne0 = 1e19       # m^-3
te0 = 50.0       # eV
ti0 = 50.0       # eV

# Constant heat flux (MW/m^2) — strong enough to evaporate
q_mag_val = 50e6  # W/m^2

# Temperature gradient: dTe/dR = 100 eV/m, dTe/dZ = 0
# This gives a clear directional rocket force in -R
grad_te_r_val = 100.0 * 1.602e-19  # convert eV/m to J/m
grad_te_z_val = 0.0

# Droplet initial conditions
rd0 = 2.5e-3     # m
rho_li = 512.0   # kg/m^3
md0 = rho_li * (4.0/3.0) * np.pi * rd0**3
Td0 = 773.15     # K

# Droplet start position: center of domain
x0 = 4.0   # R
y0 = 0.0   # Z
vx0 = 0.0  # initially at rest
vy0 = 0.0

# ======================================================================
# Write heatflux HDF5 with grad_Te
# ======================================================================
hf_file = "heatflux_rocket.h5"
print(f"Writing {hf_file}...")
with h5py.File(hf_file, 'w') as f:
    f.create_dataset('r', data=r_arr)
    f.create_dataset('z', data=z_arr)

    # Uniform heat flux
    q_mag = np.full((nz, nr), q_mag_val)
    f.create_dataset('q_mag', data=q_mag)

    # Uniform temperature gradient (dTe/dR > 0, dTe/dZ = 0)
    grad_r = np.full((nz, nr), grad_te_r_val)
    grad_z = np.full((nz, nr), grad_te_z_val)
    f.create_dataset('grad_te_r', data=grad_r)
    f.create_dataset('grad_te_z', data=grad_z)

print(f"  q_mag = {q_mag_val:.1e} W/m^2")
print(f"  grad_te_r = {grad_te_r_val:.3e} J/m (={100.0} eV/m)")
print(f"  grad_te_z = {grad_te_z_val}")

# ======================================================================
# Write species file
# ======================================================================
species_file = "droplet.species"
with open(species_file, 'w') as f:
    f.write("# Species file for rocket force test\n")
    f.write("# species-ID  molwt(AMU)  charge  rotDOF  vibDOF  rotREL  vibREL  vibTEMP\n")
    f.write("drop_1  6.941  0  0  0  0  0  0\n")
print(f"Written {species_file}")

# ======================================================================
# Write particle source file (single droplet)
# ======================================================================
source_file = "source.1"
with open(source_file, 'w') as f:
    f.write("SPARTA\n\n")
    f.write("1 particles\n")
    f.write("2 custom\n")
    f.write("hit_flag int 0\n")
    f.write("hit_surf_id int 0\n\n")
    f.write("Particles\n\n")
    # id species x y z vx vy vz (then customs)
    z_mid = 0.0  # 2D, z doesn't matter
    f.write(f"1 drop_1 {x0} {y0} {z_mid} {vx0} {vy0} 0.0 0 0\n")
print(f"Written {source_file}")

# ======================================================================
# Write input scripts for eta = 0, 0.5, 1.0
# ======================================================================
dt = 1e-5
t_end = 1.0  # 1 second
nsteps = int(t_end / dt)
dump_every = 100

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
        f.write(f"# Evaporation with rocket force eta={eta}\n")
        f.write(f"fix fevap evaporation 1 DropletSource &\n")
        f.write(f"    mass {md0:.6e} radius {rd0} temp {Td0} &\n")
        f.write(f"    heatflux/file {hf_file} &\n")
        f.write(f"    rocket_eta {eta}\n\n")
        f.write("# Gravity (downward in Z)\n")
        f.write("global gravity -9.81 0 1 0\n\n")
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
