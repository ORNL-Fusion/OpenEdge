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
    ids = []
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
                        ids.append(int(atom_data[1]))
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

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius, ids


import os
from utils import surface #, parser
import matplotlib

if __name__ == "__main__":


    import numpy as np
    import matplotlib.pyplot as plt

    def load_one_id(fname):
        ts,x,y,z,vr,vz_c,vphi,mass,temp,r,ids = parse_file(fname)
        ids = np.asarray(ids); pick = np.unique(ids)[0]
        m = (ids == pick)
        tstep = np.asarray(ts,float)[m]
        vr    = np.asarray(vr,float)[m]
        vz    = np.asarray(vz_c,float)[m]   # this is v_z (cyl)
        vphi  = np.asarray(vphi,float)[m]
        zpos  = np.asarray(z,float)[m]
        m_d   = float(np.asarray(mass,float)[m][0])
        return tstep, vr, vz, vphi, zpos, m_d

    runs = [
        {"file":"LigamentSource_with_g"   ,"dt":1e-3,"label":"g=-9.8","g":-9.8},
    ]
    label ='test_1'
    path = '../'
    #    run_path = '../'
    wall = surface("wall.surf", "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy
    core = surface('core.surf', "2D")
    domain = core.polygon
    rcore, zcore = domain.exterior.xy
    import numpy as np
    import matplotlib.pyplot as plt
    #        import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # runs to compare
    runs = [
        {"file": "case.pure", "dt": 1e-3, "label": "pure"},
        {"file": "case.g", "dt": 1e-3, "label": "g 20"},
        {"file": "case.g.9", "dt": 1e-3, "label": "g-9"},
        {"file": "case.g.90", "dt": 1e-3, "label": "g-90"},

#        {"file": "case.6.noG", "dt": 1e-3, "label": "instability no G"},
#        {"file": "case.6.withG", "dt": 1e-3, "label": "instability with G"},
#        {"file": "case.4", "dt": 1e-3, "label": "no G"},
#        {"file": "case.5", "dt": 1e-3, "label": "with G"}
    ]

    # --- helper: plot R-Z trajectories for one run ---
    def plot_trajs_RZ(ax, file, dt, label, color=None, lw=1.5, alpha=0.9):
        # parse_file is assumed to return flat arrays for all recorded points
        #   timesteps, x, y, z, vx, vy, vz, mass, temp, radius, ids
        tstep, x, y, z, vx, vy, vz, mass, temp, radius, ids = parse_file(file)
        t = np.asarray(tstep, dtype=float) * float(dt)

        # group by particle id and draw path in R-Z (x ~ R, y ~ Z)
#        uniq = np.unique(ids)
#        for k, pid in enumerate(uniq):
#            m = (ids == pid)
            # draw one line per particle
        ax.plot(x,y, '-', lw=lw, alpha=alpha, color=color)

        # add a single legend handle per run
        ax.plot([], [], '-', lw=lw, color=color, label=label)

    # --- main figure: R-Z trajectories for all runs ---
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for run in runs:
        plot_trajs_RZ(ax, run["file"], run["dt"], run["label"], color=next(colors))

    # overlays (tokamak geometry etc.) if you already have them
    # Rwall, Zwall, rcore, zcore assumed defined
    ax.plot(Rwall, Zwall, 'k-', lw=2.5)
    ax.plot(rcore, zcore, 'g-', lw=2.5)

##group            divertor_outer surf id 40 #<> 37 50
#group            divertor_inner surf id <> 10 17

    # R-Z plot cosmetics
    ax.set_xlim(3.20, 3.45)
    ax.set_ylim(-3.8, -3.0)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=10, frameon=True)
    plt.tight_layout()
    plt.show()

