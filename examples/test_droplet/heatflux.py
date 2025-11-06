#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write_heatflux_oe.py
--------------------
Create an OpenEdge-friendly HDF5 file with total heat-flux density on the SOLPS cell-center (R,Z) grid.

What it does
============
1) Loads a SOLPS run via quixote (expects b2fplasmf present).
2) Reads total heat flux on faces (fht) and face areas (sx, sy).
3) Converts to flux density [W/m^2] on faces, centers to cells, and computes |q|.
4) Writes HDF5:
   - grid/R, grid/Z (float32, meters)
   - fields/q_x, fields/q_y, fields/q_mag (float32, W/m^2)
   with units + provenance attrs.

Usage
=====
python write_heatflux_oe.py 

Notes
=====
- Cell-center coordinates are taken from shot.cx/shot.cy. If you prefer crx/cry, pass --use-crx.
- If fht is not available, script tries to sum components (fhe,fhi,fhj,fhm,fhp[,fnt]).
"""


from pathlib import Path
import argparse
import numpy as np
import h5py
from datetime import datetime
import quixote as qx
import matplotlib.pyplot as plt
from quixote import GridDataPlot, VesselPlot

    
def write_heatflux_oe(Rc_out,Zc_out,Q_out):
    with h5py.File("heatflux.h5", "w") as h5:
        ggrid   = h5.create_group("grid")
        gfields = h5.create_group("fields")

        dR = ggrid.create_dataset("Rc", data=Rc_out, compression="gzip", shuffle=True, chunks=True)
        dZ = ggrid.create_dataset("Zc", data=Zc_out, compression="gzip", shuffle=True, chunks=True)
        dQ = gfields.create_dataset("q_mag", data=Q_out, compression="gzip", shuffle=True, chunks=True)

  
    print("Wrote heatflux.h5 with datasets: grid/R, grid/Z, fields/q_mag")

run = Path("/Users/42d/ORNL Dropbox/Abdou DIaw/addLi/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up")

shot = qx.SolpsData(str(run))   # <-- just the run path


fht = shot.fht.astype(float)          # (nx, ny, 2)
sx  = shot.sx.astype(float)           # (nx, ny)  left/x faces
sy  = shot.sy.astype(float)           # (nx, ny)  bottom/y faces

# flux density [W/m^2] on faces; avoid /0 at the padded boundaries
qtx = np.divide(fht[...,0], sx, out=np.full_like(sx, np.nan), where=sx > 0)  # poloidal faces
qty = np.divide(fht[...,1], sy, out=np.full_like(sy, np.nan), where=sy > 0)  # radial  faces


# average adjacent faces onto cell centers
qx_c = 0.5*(qtx + np.roll(qtx, -1, axis=0)); qx_c[-1,:] = np.nan
qy_c = 0.5*(qty + np.roll(qty, -1, axis=1)); qy_c[:, -1] = np.nan
q_mag = np.sqrt(qx_c*qx_c + qy_c*qy_c)       # (nx, ny)
q_mag_plot = np.ma.array(q_mag, mask=np.isnan(q_mag))

cmap = plt.get_cmap('inferno').copy()
cmap.set_bad('0.92')  # light grey for masked areas

Rc, Zc = shot.crx[:, :, -1], shot.cry[:, :, -1]                         # guaranteed (nx,


# Ensure finite arrays for HDF5
Rc_out = Rc.astype(np.float32)
Zc_out = Zc.astype(np.float32)
Q_out  = np.where(np.isfinite(q_mag), q_mag, np.nan).astype(np.float32)

#write_heatflux_oe(Rc_out,Zc_out,Q_out)

from scipy.interpolate import griddata
points = np.vstack((Rc_out.flatten(), Zc_out.flatten())).T
def interpolate_field(field_data):
    interp_linear = griddata(points, field_data.flatten(), grid_points, method='linear')
    interp_nearest = griddata(points, field_data.flatten(), grid_points, method='nearest')
    return np.where(np.isnan(interp_linear), interp_nearest, interp_linear).reshape(nZ, nR)
    
nR, nZ = 500, 500
grid_r = np.linspace(2.3, 5.7, nR)
grid_z = np.linspace(-3.93, 4., nZ)
grid_rr, grid_zz = np.meshgrid(grid_r, grid_z)
grid_points = np.vstack((grid_rr.flatten(), grid_zz.flatten())).T

Q_out_int = interpolate_field(Q_out)

write_heatflux_oe(grid_r,grid_z,Q_out_int)



exit()
# Plotting
Plotter = True
if Plotter:

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot cell-centered field on the (R,Z) grid
    GridDataPlot(shot, q_mag_plot, canvas=ax, cmap=cmap, norm='log',  xlim=[2.0, 6], ylim=[-4, 3.8])

    # Overlay vessel / walls
    VesselPlot(shot, canvas=ax, color='k', lw=0.8)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Total heat-flux density |q| [W m$^{-2}$]')

    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
