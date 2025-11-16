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
        {"file":"case.2"   ,"dt":1e-3,"label":"g=-9.8","g":-9.8},
    ]
    label ='test_1'
    nuE = 0.868098  # 1/s (from your print)

    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.tick_params(axis='both', direction='in')

    for run in runs:
        tstep, vr, vz, vphi, zpos, m_d = load_one_id(run["file"])
        t  = tstep * run["dt"]
        g = run["g"]
        nuE   = 0.868098
        uparz = 0.0
        g     = -9.8
        vz_inf = uparz + g/nuE
        vz_analytic = vz_inf + (vz[0] - vz_inf) * np.exp(-nuE*(t - t[0]))
        ax.plot(t, vz, 'o', ms=3, label=f"simulation")
        ax.plot(t, vz_analytic, '-', lw=2, label=f"analytic")
        ax.set_xlabel('time (s)'); ax.set_ylabel(r'$v_z$ (m/s)'); ax.grid(alpha=0.3); ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)


        fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        plt.tight_layout(pad=2.0)  # Increase padding
        ax.grid(True, linestyle='--', alpha=0.7)
    
#        Save the figure with improved DPI, tight bounding box, and specified background color
        plt.savefig(f"Figs/numerical_{label}.png",dpi=300, bbox_inches="tight", facecolor='white')
        plt.tight_layout(); plt.show()
#    
#
#    exit()
#
##    path = '../'
##    run_path = '../'
#    wall = surface("wall.surf", "2D")
#    domain = wall.polygon
#    Rwall, Zwall = domain.exterior.xy
#    core = surface('core.surf', "2D")
#    domain = core.polygon
#    rcore, zcore = domain.exterior.xy
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    # Load your data
#    timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius, ids = parse_file("LigamentSource_w_g")
#
##        import numpy as np
#    import matplotlib.pyplot as plt
#
#    # ---- inputs ----
#    runs = [
#        {"file": "LigamentSource_without_g", "dt": 1e-3, "label": "no G"},{"file": "LigamentSource_with_g", "dt": 1e-3, "label": "with G=9.8"}]
#    nuE = 0.868098     # [1/s]
#    g   = -9.8      # [m/s^2]  (negative = downward)
#    rd = 50.e-6
#    def load_one_id(fname):
#        # parse_file should return columns in cylindrical order: (v_r, v_z, v_phi)
#        ts, x, y, z, vx, vy, vz, mass, temp, radius, pid = parse_file(fname)
#        pid = np.asarray(pid)
#        pid_sel = np.unique(pid)[0]
#        m = (pid == pid_sel)
#        tstep = np.asarray(ts, float)[m]
#        vr    = np.asarray(vx, float)[m]      # v_r
#        vz_c  = np.asarray(vy, float)[m]      # v_z (cyl)
#        vphi  = np.asarray(vz, float)[m]      # v_phi
#        zpos  = np.asarray(z,  float)[m]
#        m_d   = float(np.asarray(mass, float)[m][0])
#        return tstep, vr, vz_c, vphi, zpos, m_d
#
#    fig, (axV, axE) = plt.subplots(1, 2, figsize=(15, 4.2))
#
#    for run in runs:
#        tstep, vr, vz_c, vphi, zpos, m_d = load_one_id(run["file"])
#
#        t  = tstep * run["dt"]
#        v  = np.sqrt(vr*vr + vz_c*vz_c + vphi*vphi)
#
#        # analytic component along z with constant g and drag:
#        vz_inf = g/nuE
#        t0, vz0 = t[0], vz_c[0]
#        fac = np.exp(-nuE*(t - t0))
#        vz_analytic = vz_inf + (vz0 - vz_inf)*fac
#        v_analytic  = np.sqrt(vr*0 + vphi*0 + vz_analytic*vz_analytic)  # here vr=vphi≈0
#
#        # speed panel (linear y)
#        axV.plot(t, v, 'o', ms=3, label=f"data ({run['label']})")
#        axV.plot(t, v_analytic, '-', lw=2, label=f"analytic, $\\nu_E={nuE:.3g}\\,\\mathrm{{s^{{-1}}}}$")
#
#        # energies
#        K = 0.5*m_d*v*v
#        U = m_d*g*zpos
##        axE.plot(t, K, '-', lw=1.4, label=f"K ({run['label']})")
##        axE.plot(t, U, '--', lw=1.0, label=f"U ({run['label']})")
#        radius[0] = rd
#        axE.plot(t, radius, '-', lw=2.0, label=f"radius")
#
#        # residual check (should be a straight exponential decay)
#        res = np.abs(v - v_analytic)
#        # avoid log(0) issues:
#        eps = max(1e-12, 1e-6*np.max(v_analytic))
##        axR.semilogy(t, np.maximum(res, eps), 'o', ms=3, label=run["label"])
#
#    axV.set_xlabel("time [s]"); axV.set_ylabel("speed $v$ [m/s]"); axV.grid(alpha=0.3); axV.legend()
#    axE.set_xlabel("time [s]"); axE.set_ylabel("radius [m]");      axE.grid(alpha=0.3); axE.legend()
##    axR.set_xlabel("time [s]"); axR.set_ylabel(r"$|v - v_\mathrm{analytic}|$ [m/s]"); axR.grid(alpha=0.3); axR.legend()
#
#    plt.tight_layout(); plt.show()
#
#
#
#    exit()
#    times  = np.asarray(timesteps, dtype=float)
#    plt.plot(times, np.log(v), 'o')
#    plt.show()
#
#    exit()
##            times  = np.asarray(timesteps, dtype=float) * dt
##    print(radius)
#    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#    # overlay walls, tidy axes
#    ax.plot(rcore, zcore, "k", lw=2.5)
#    ax.plot(Rwall, Zwall ,"k", lw=2.5)
#
#    ax.plot(x_coords,y_coords, 'ko')
#    
#
#    ax.set_aspect("equal")
#
#    ax.set_xlabel("R (m)", weight="semibold", fontsize=10)
#    ax.set_ylabel("Z (m)", weight="semibold", fontsize=10)
#    ax.grid(alpha=0.3, linestyle="--")
#    ax.set_xlim(2.38, 3.75);
#    ax.set_ylim(-3.75, -2.25)
#    plt.show()
#    
##    exit()
##
#
#    PlotterMassEvol=True
#    
#    if PlotterMassEvol:
#    #    # Ensure numpy arrays
#        dt = 1e-5*1000
#        time=8.5
#        N = int(time/dt)
#        N1 = int(N/100)
##        dt = 1e-07*N1
#
#
##        rd  =  50e-6
##        temp0 = 773.15
##        times  = np.asarray(timesteps*np.asarray(dt), dtype=float)
##        radii = np.asarray(radius/np.array(rd), dtype=float)
#
#        rd     = 50e-6           # meters
#        temp0K = 773.15          # Kelvin
#        dt     = float(dt)
#     
#        # build arrays
#        times  = np.asarray(timesteps, dtype=float) * dt
#        
#        radii  = np.asarray(radius,   dtype=float)      # normalized so t0 should be 1.0
#        tempC  = np.asarray(temp,     dtype=float)
#
##        times = np.insert(times, 0, 0.0)
#        radii[0] = rd
##        tempC = np.insert(tempC, 0, temp0K)            # initial temp in °C
#        tempC[0] = temp0K
#        print(np.unique(tempC))
#        
#        #fig, ax1 = plt.subplots(figsize=(10, 6))
#        import matplotlib.ticker as mticker
#        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
#        ax1.tick_params(axis='both', direction='in')
#        ax1.plot(times, radii/rd, color="navy", lw=2)  # Keep the full line plot for context
#        
#
#
#        ax2 = ax1.twinx()
#        ax2.plot(times, tempC-273, color="darkorange", lw=2)
#        ax2.set_ylabel("Temperature (°C)", color="darkorange")
#        ax2.tick_params(axis="y", labelcolor="darkorange")
#
#        ax1.set_xlabel("Time (s)")
#        ax1.set_ylabel("$R_d(t)/R_d(0)$", color="navy")
#        ax1.tick_params(axis="y", labelcolor="navy")
#        ax1.grid(True, linestyle="--", alpha=0.5)
#
#        fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
#        fmt.set_scientific(False)
#        ax1.yaxis.set_major_formatter(fmt)
#        plt.tight_layout(pad=2.0)  # Increase padding
#    #    Q_g = 1e-6*Qs
#    #    ax1.set_title(f"Droplet lifetime: ($Q_g$ = {Q_g} $MW/m^2$, $T_0$ = {T0}°C)", fontsize=12, weight='bold')
#
#        ax1.grid(True, linestyle='--', alpha=0.7)
#
#        # Save the figure with improved DPI, tight bounding box, and specified background color
#    #    plt.savefig(f"Figs/case_{case}.png",
#    #            dpi=300, bbox_inches="tight", facecolor='white')
#
#
#
#        plt.show()
#
#
