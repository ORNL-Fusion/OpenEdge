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
    import os
    import glob
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # --- discover runs automatically from current folder ---
    dt_default = 1e-3

    case_files = sorted(glob.glob("case.*"))

    def parse_params(fname):
        """
        Extract site, vmag, angle from a filename like:
        case.outer.mist.site.37.vmag.0.1.angle.2.0
        """
        base = os.path.basename(fname)

        m_site  = re.search(r"site\.(\d+)", base)
        # vmag: between 'vmag.' and '.angle'
        m_vmag  = re.search(r"vmag\.([0-9.]+)\.angle", base)
        # angle: after 'angle.' to end of string
        m_angle = re.search(r"angle\.([0-9.]+)$", base)

        site  = int(m_site.group(1)) if m_site else None
        vmag  = float(m_vmag.group(1)) if m_vmag else None
        angle = float(m_angle.group(1)) if m_angle else None

        return site, vmag, angle


    runs = []
    for f in case_files:
        site, vmag, angle = parse_params(f)
        label = f"site {site}" if site is not None else os.path.basename(f)
        runs.append({
            "file": f,
            "dt": dt_default,
            "site": site,
            "vmag": vmag,
            "angle": angle,
            "label": label,
        })

    print("Found runs:")
    for r in runs:
        print(f"  {r['file']}  ->  site={r['site']}  vmag={r['vmag']}  angle={r['angle']}")

    # --- build color map per (vmag, angle) group ---
    group_to_color = {}
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for r in runs:
        key = (r["vmag"], r["angle"])
        if key not in group_to_color:
            group_to_color[key] = next(colors)
    # --- helper: plot R-Z trajectories for one run ---
    def plot_trajs_RZ(ax, file, dt, label,
                      color=None, lw=1.2, alpha=0.5,
                      add_label=False, pid=None):
        tstep, x, y, z, vx, vy, vz, mass, temp, radius, ids = parse_file(file)

        # make sure these are numpy arrays
        t     = np.asarray(tstep, dtype=float) * float(dt)  # unused for now
        x     = np.asarray(x, dtype=float)
        y     = np.asarray(y, dtype=float)
        ids   = np.asarray(ids)

        # choose which particle id to keep
        if pid is None:
            uniq = np.unique(ids)
            pid = uniq[0]

        m = (ids == pid)          # boolean mask

        # boolean indexing now works because x, y are numpy arrays
        ax.plot(x[m], y[m], "o", alpha=alpha, color=color, markersize=2)

        if add_label:
            ax.plot([], [], "o", color=color, label=f"{label}, id={pid}")



    # --- main figure: R-Z trajectories for all runs ---
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))

    # 1) plot all trajectories, colored by (vmag, angle)
    for r in runs:
        key = (r["vmag"], r["angle"])
        color = group_to_color[key]
        # either force a specific id:
        # plot_trajs_RZ(ax, r["file"], r["dt"], r["label"], color=color,
        #               add_label=False, pid=123456)

        # or let it pick the first id present in that file:
        plot_trajs_RZ(ax, r["file"], r["dt"], r["label"], color=color,
                      add_label=False)


    # overlays (tokamak geometry etc.) if you already have them
    ax.plot(Rwall, Zwall, "k-", lw=2.5, label="_nolegend_")
    ax.plot(rcore, zcore, "g-", lw=2.5, label="_nolegend_")

    # 2) legend: one entry per (vmag, angle) group
    handles = []
    labels  = []
    for (vmag, angle), color in group_to_color.items():
        h = ax.plot([], [], "o", color=color)[0]
        handles.append(h)
        labels.append(f"vmag={vmag}, angle={angle}")

    leg1 = ax.legend(handles, labels, title="Groups", ncol=2, fontsize=9, frameon=True)
    ax.add_artist(leg1)

    # Optional: second legend per site (if you really want it)
    # This can get crowded if many sites, so comment out if noisy.
    site_handles = []
    site_labels  = []
    for r in runs:
        key = (r["vmag"], r["angle"])
        color = group_to_color[key]
        h = ax.plot([], [], "o", color=color)[0]
        site_handles.append(h)
        site_labels.append(r["label"])
    # ax.legend(site_handles, site_labels, title="Sites", ncol=3, fontsize=8, frameon=True, loc="lower left")

    # R-Z plot cosmetics
    ax.set_xlim(3.1,3.52)
    ax.set_ylim(-3.8,-3.1)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

