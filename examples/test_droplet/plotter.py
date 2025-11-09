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
    wall = surface("wall.surf", "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy
    core = surface('core.surf', "2D")
    domain = core.polygon
    rcore, zcore = domain.exterior.xy
    import numpy as np
    import matplotlib.pyplot as plt

    # Load your data
    timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius = parse_file("LigamentSource_w_g")

    print(radius)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # overlay walls, tidy axes
    ax.plot(rcore, zcore, "k", lw=2.5)
    ax.plot(Rwall, Zwall ,"k", lw=2.5)

    ax.plot(x_coords,y_coords, 'ko')
    

    ax.set_aspect("equal")

    ax.set_xlabel("R (m)", weight="semibold", fontsize=10)
    ax.set_ylabel("Z (m)", weight="semibold", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(2.38, 3.75);
    ax.set_ylim(-3.75, -2.25)
    plt.show()
    
#    exit()
#

    PlotterMassEvol=True
    
    if PlotterMassEvol:
    #    # Ensure numpy arrays
        dt = 1e-5*1000
        time=8.5
        N = int(time/dt)
        N1 = int(N/100)
#        dt = 1e-07*N1


#        rd  =  50e-6
#        temp0 = 773.15
#        times  = np.asarray(timesteps*np.asarray(dt), dtype=float)
#        radii = np.asarray(radius/np.array(rd), dtype=float)

        rd     = 50e-6           # meters
        temp0K = 773.15          # Kelvin
        dt     = float(dt)
     
        # build arrays
        times  = np.asarray(timesteps, dtype=float) * dt
        
        radii  = np.asarray(radius,   dtype=float)      # normalized so t0 should be 1.0
        tempC  = np.asarray(temp,     dtype=float)

#        times = np.insert(times, 0, 0.0)
        radii[0] = rd
#        tempC = np.insert(tempC, 0, temp0K)            # initial temp in °C
        tempC[0] = temp0K
        print(np.unique(tempC))
        
        #fig, ax1 = plt.subplots(figsize=(10, 6))
        import matplotlib.ticker as mticker
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        ax1.tick_params(axis='both', direction='in')
        ax1.plot(times, radii/rd, color="navy", lw=2)  # Keep the full line plot for context
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("$R_d(t)/R_d(0)$", color="navy")
        ax1.tick_params(axis="y", labelcolor="navy")
        ax1.grid(True, linestyle="--", alpha=0.5)

        fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_scientific(False)
        ax1.yaxis.set_major_formatter(fmt)

        ax2 = ax1.twinx()
        ax2.plot(times, tempC-273, color="darkorange", lw=2)
        ax2.set_ylabel("Temperature (°C)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

                
        plt.tight_layout(pad=2.0)  # Increase padding
    #    Q_g = 1e-6*Qs
    #    ax1.set_title(f"Droplet lifetime: ($Q_g$ = {Q_g} $MW/m^2$, $T_0$ = {T0}°C)", fontsize=12, weight='bold')

        ax1.grid(True, linestyle='--', alpha=0.7)

        # Save the figure with improved DPI, tight bounding box, and specified background color
    #    plt.savefig(f"Figs/case_{case}.png",
    #            dpi=300, bbox_inches="tight", facecolor='white')



        plt.show()


