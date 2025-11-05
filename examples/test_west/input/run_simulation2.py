# Copyright 2024, OpenEdge contributors
# Authors: Abdou Diaw
# License: GPL-2.0 license
"""
This test file is part of OpenEdge, a Particle transport code.

This test simulates the motion of a charge particle in a uniform and external magnetic field.

"""


# ------
# Imports
# ------
import numpy as np
import random

import numpy as np
#from scipy.constants import m_e, e, k, epsilon_0, m_p
# imports
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess as sp
#from geometry import get_magnetic_field
#from source import createSource

# Number of MPI cores
MPI_cores = '8'

import numpy as np
import matplotlib.pyplot as plt
import random
import random
#from geometry import get_magnetic_field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import compute_line_normal

import matplotlib.cm as cm
#from closest_surfaces import load_points, load_triangles, find_closest_surface, calculate_velocity


#geqdsk_file = f'../g000001.00001_symm'

# read magnetic field
#rs, zs, r2D, z2D, flux2D, Br2D,Bz2D ,Bphi2D = get_magnetic_field(geqdsk_file)

def write_lammps_format(filename, x, y, z, vx, vy, vz, particle_type):
    n_atoms = 1
    print(f"Writing {n_atoms} atoms to {filename}")
    with open(filename, 'w') as file:
        # Write the headers
        min_x, max_x = -6, 6 #  -6 6 -6 6. -4 4
        min_y, max_y = -6, 6
        min_z, max_z = -4, 4
        
        file.write("ITEM: TIMESTEP\n")
        file.write("0\n")
        file.write("ITEM: NUMBER OF ATOMS\n")
        file.write(f"{n_atoms}\n")
        file.write("ITEM: BOX BOUNDS pp pp pp\n")
        file.write(f"{min_x} {max_x}\n")
        file.write(f"{min_y} {max_y}\n")
        file.write(f"{min_z} {max_z}\n")
        file.write("ITEM: ATOMS id type x y z vx vy vz\n")
        
        file.write(f"2 {particle_type} {x} {y} {z} {vx} {vy} {vz}\n")
  

def parse_file(filename):
    timesteps = []
    x_coords = []
    y_coords = []
    z_coords = []
    vx_coords = []
    vy_coords = []
    vz_coords = []
    mass=[]

    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "ITEM: TIMESTEP":
                timestep = int(lines[i + 1].strip())
                i += 2  # Move to next line after timestep

            elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
                num_atoms = int(lines[i + 1].strip())
                i += 2  # Move to the line after the number of atoms

            elif lines[i].strip() == "ITEM: ATOMS id type x y z vx vy vz v_pmass":
                if num_atoms > 0:
                    # Only add timestep if atoms are present
                    timesteps.append(timestep)
                    # Loop to parse all atoms for the current timestep
                    for _ in range(num_atoms):
                        atom_data = lines[i + 1].strip().split()
                        x_coords.append(float(atom_data[2]))
                        y_coords.append(float(atom_data[3]))
                        z_coords.append(float(atom_data[4]))
                        vx_coords.append(float(atom_data[5]))
                        vy_coords.append(float(atom_data[6]))
                        vz_coords.append(float(atom_data[7]))
                        mass.append(float(atom_data[8]))
                        i += 1  # Move to the next atom data line
                i += 1  # Move to the next line after the "ITEM: ATOMS" section

            else:
                i += 1  # Move to next line if no match

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass


import os
from utils import surface #, parser
import matplotlib

if __name__ == "__main__":

    path = '../'
    run_path = '../'
#    wall = surface("input/reduce_surf.txt", "2D")
    wall = surface("wall.txt", "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy
    filename = f"state/state_yes"
    filename1 = f"state/state_no"
    import numpy as np
    import matplotlib.pyplot as plt

#    # Load your data
#    timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass = parse_file(filename)
#    _timesteps, _x_coords, _y_coords, _z_coords, _vx_coords, _vy_coords, _vz_coords, _mass = parse_file(filename1)
#
#    # Ensure numpy arrays
#    t  = np.asarray(timesteps, dtype=float)
#    x  = np.asarray(x_coords, dtype=float)
#    y  = np.asarray(y_coords, dtype=float)
#    z  = np.asarray(z_coords, dtype=float)
#    vx = np.asarray(vx_coords, dtype=float)
#    vy = np.asarray(vy_coords, dtype=float)
#    vz = np.asarray(vz_coords, dtype=float)
#    mass = np.asarray(mass, dtype=float)
#
#    # Ensure numpy arrays
#    _t  = np.asarray(_timesteps, dtype=float)
#    _x  = np.asarray(_x_coords, dtype=float)
#    _y  = np.asarray(_y_coords, dtype=float)
#    _z  = np.asarray(_z_coords, dtype=float)
#    _vx = np.asarray(_vx_coords, dtype=float)
#    _vy = np.asarray(_vy_coords, dtype=float)
#    _vz = np.asarray(_vz_coords, dtype=float)
#    _mass = np.asarray(_mass, dtype=float)
#
#    # Cylindrical radius
#    R = np.sqrt(x**2 + z**2)
#    _R = np.sqrt(_x**2 + _z**2)
#
#    # Speeds and kinetic energy
#    speed = np.sqrt(vx**2 + vy**2 + vz**2)
#
#    K = 0.5 * mass * speed**2
#    K0 = K[0] if K[0] != 0 else 1.0
#    rel_drift = (K - K0) / K0

#    print(f"[Conservation] max |ΔK/K0| = {np.max(np.abs(rel_drift)):.3e}")

    # -------- R–Z trajectory --------
    fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=300)
#    ax1.plot(R, y, marker='o', linewidth=1, label='coll')
#    ax1.plot(_R, _y, marker='^', linewidth=1, label='no col')
    ax1.plot(Rwall, Zwall, 'g', linewidth=3.)
#    ax1.plot(_rwall, _zwall, 'g', linewidth=3.)

#    ax1.contour(r2D, z2D, flux2D, levels=100, colors='teal', linewidths=0.8)
#    ax1.plot(3.2, 0., 'ko', markersize=5)
    ax1.set_xlabel('R [m]', weight='semibold', fontsize=10)
    ax1.set_ylabel('Z [m]', weight='semibold', fontsize=10)
#    ax1.set_xlim(3.44231,3.4425495)
#    ax1.set_ylim(-3.35930,-3.35895)
    ax1.legend(loc='best', fontsize=6)
#    fig1.suptitle('Li trajectory (R–Z)', fontsize=12, weight='semibold')
    plt.tight_layout()
    plt.show()




