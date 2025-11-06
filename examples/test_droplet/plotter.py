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
#from geometry/ import get_magnetic_field
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
    temp =[]
    radius  = []
#        timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius

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

            elif lines[i].strip() == "ITEM: ATOMS id type x y z vx vy vz v_pmass temp radius":
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
                        temp.append(float(atom_data[9]))
                        radius.append(float(atom_data[10]))
                        i += 1  # Move to the next atom data line
                i += 1  # Move to the next line after the "ITEM: ATOMS" section

            else:
                i += 1  # Move to next line if no match

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius


import os
from utils import surface #, parser
import matplotlib

if __name__ == "__main__":

#    path = '../'
#    run_path = '../'
##    wall = surface("input/reduce_surf.txt", "2D")
#    wall = surface("../surfaces/wall.txt", "2D")
#    fname = os.path.join(run_path, "mesh.extra")
#    d = np.loadtxt(fname)
#    _rwall = np.vstack((d[:,0], d[:,2]))
#    _zwall = np.vstack((d[:,1], d[:,3]))
#    domain = wall.polygon
#    Rwall, Zwall = domain.exterior.xy
#    filename = f"state/state_yes"
#    filename1 = f"state/state_no"
    import numpy as np
    import matplotlib.pyplot as plt

#    # Load your data
    timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius = parse_file("LigamentSource")

#    # Ensure numpy arrays
    dt = 1e-3
    times  = np.asarray(timesteps*np.asarray(dt), dtype=float)
    radii = np.asarray(radius, dtype=float)
    temp = np.asarray(temp-np.asarray(273.15), dtype=float)


    #fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax1.tick_params(axis='both', direction='in')
#    # Slice data to show every 10th point
#    times_sampled = t
#    radii_sampled = radius

    # Plot radius with log-scaled charge color using 'cividis' colormap
#    sc = ax1.scatter(times_sampled, radii_sampled, c=norm_charges_sampled, cmap='cividis', s=80, edgecolor='k') #, label="Radius (m)")
    ax1.plot(times, radii, color="navy", lw=2)  # Keep the full line plot for context
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Radius (m)", color="navy")
    ax1.tick_params(axis="y", labelcolor="navy")
    ax1.grid(True, linestyle="--", alpha=0.5)
#    ax1.set_ylim(0, max(radii_sampled)+max(radii_sampled)*1e-1)
    # Second y-axis for temperature using a contrasting color
    ax2 = ax1.twinx()
    ax2.plot(times, temp, color="darkorange", lw=2)
    ax2.set_ylabel("Temperature (°C)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Lower and resize the colorbar
#    cbar_ax = fig.add_axes([0.25, 0.3, 0.5, 0.02])  # [left, bottom, width, height]
#    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label="Charge ($10^7 e$)")
#    cbar.ax.tick_params(labelsize=8)  # Reduce tick label size if needed


            
    plt.tight_layout(pad=2.0)  # Increase padding
#    Q_g = 1e-6*Qs
#    ax1.set_title(f"Droplet lifetime: ($Q_g$ = {Q_g} $MW/m^2$, $T_0$ = {T0}°C)", fontsize=12, weight='bold')

    ax1.grid(True, linestyle='--', alpha=0.7)

    # Save the figure with improved DPI, tight bounding box, and specified background color
#    plt.savefig(f"Figs/case_{case}.png",
#            dpi=300, bbox_inches="tight", facecolor='white')



    plt.show()


