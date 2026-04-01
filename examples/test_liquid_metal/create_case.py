#!/usr/bin/env python3
"""
Create a minimal 2D test case for fix liquid_metal.

Generates:
  - wall.surf: angled divertor wall (line segments)
  - Li.species: lithium species file
  - in.liquid_metal: SPARTA input script

The wall is a 20 cm line at 43 degrees (typical divertor angle).
Heat flux is constant (no plasma.h5 needed).
The fix outputs Tsurf, evap_flux, adatom_flux, h_film per surface element.
"""

import numpy as np
import os

# --- Wall geometry ---
# Angled line from (R0, Z0) to (R1, Z1), ~20 cm at 43 degrees
# In SPARTA 2D: x = R, y = Z
alpha_deg = 43.0
L = 0.20  # 20 cm divertor length
nseg = 50  # number of line segments

R0, Z0 = 1.3, -1.0
dR = L * np.cos(np.radians(alpha_deg))
dZ = L * np.sin(np.radians(alpha_deg))
R1, Z1 = R0 + dR, Z0 + dZ

# Generate points along the wall
Rs = np.linspace(R0, R1, nseg + 1)
Zs = np.linspace(Z0, Z1, nseg + 1)

# Write surf file
with open("wall.surf", "w") as f:
    npts = len(Rs)
    nlines = nseg
    f.write(f"# divertor wall surface: {nseg} segments at {alpha_deg} deg\n")
    f.write(f"\n{npts} points\n{nlines} lines\n\n")
    f.write("Points\n\n")
    for i in range(npts):
        f.write(f"{i+1} {Rs[i]:.8f} {Zs[i]:.8f}\n")
    f.write("\nLines\n\n")
    for i in range(nlines):
        f.write(f"{i+1} 1 {i+1} {i+2}\n")

print(f"wall.surf: {nseg} segments, R=[{R0:.3f},{R1:.3f}], Z=[{Z0:.3f},{Z1:.3f}]")

# --- Species file ---
with open("Li.species", "w") as f:
    f.write("# Li species for liquid metal test\n")
    f.write("# species mass(amu) charge diam(m) rotDOF vibDOF rotTemp(K) vibTemp(K)\n")
    f.write("Li  6.941  0  2.7e-10  0  0  0  0\n")

print("Li.species written")

# --- Domain bounds (with margin around the wall) ---
margin = 0.05
xlo, xhi = R0 - margin, R1 + margin
ylo, yhi = Z0 - margin, Z1 + margin

# Heat flux: constant 5 MW/m² (typical outer divertor)
q_hf = 5.0e6  # W/m²

# Write OpenEdge input
with open("in.liquid_metal", "w") as f:
    f.write(f"""# Minimal test case for fix liquid_metal
# Tests MHD film solver + Antoine+HK evaporation + ad-atom model
#
# Run: mpirun -np 1 $OE_BIN -in in.liquid_metal

seed               12345
dimension           2
boundary            oo oo p

# --- Domain and grid ---
create_box          {xlo:.6f} {xhi:.6f} {ylo:.6f} {yhi:.6f} -0.5 0.5
create_grid         20 20 1

# --- Species ---
species             Li.species Li
mixture             all Li

# --- Surface ---
read_surf           wall.surf group wall

surf_collide        absorb vanish
surf_modify         wall collide absorb

# --- Liquid metal film model ---
# Constant heat flux, evaporation enabled
# Outputs: Tsurf [C], evap_flux [atoms/m²/s], adatom_flux [atoms/m²/s], h_film [m]

fix flm liquid_metal wall 1 {q_hf:.1e} \\
    h0 0.005 U0 8.0 Bs 5.0 alpha {alpha_deg:.1f} width 1.67 Tin 350.0 \\
    qss 1.0e6 Nx 201 Ny 51 evap yes \\
    dp_flux {1.0e22:.1e} Yad 1e-3 E_eff 0.9

# --- Time stepping ---
# Just 1 step — the strip solver runs to steady state internally
timestep            1.0e-6
run                 1

# --- Dump per-surface outputs ---
# Columns: surf-ID, Tsurf, evap_flux, adatom_flux, h_film
dump dsurf surf wall 1 dump.surf.* id s_Tsurf_lm s_evap_lm s_adatom_lm s_h_lm
run                 1
""")

print(f"in.liquid_metal written (q_hf={q_hf:.0e} W/m², alpha={alpha_deg}°)")
print(f"\nTo run:")
print(f"  python3 create_case.py")
print(f"  mpirun -np 1 /path/to/spa_mpi -in in.liquid_metal")
print(f"\nTo plot:")
print(f"  python3 plot_results.py")
