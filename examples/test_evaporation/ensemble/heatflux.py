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

write_heatflux_oe(Rc_out,Zc_out,Q_out)

exit()
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

#write_heatflux_oe(grid_r,grid_z,Q_out_int)
#
#
#
#exit()
# Plotting
Plotter = True
import matplotlib.patheffects as pe
from utils import surface
#if Plotter:

import matplotlib.patheffects as pe

def site_label(kind:str, i:int) -> str:
    # kind = "I" or "O"
    return rf"$S^{{\mathrm{{{kind}}}}}_{{{i}}}$"

def label_endpoints(ax, R, Z, kind="I", which="both", dx=0.006, dz=0.006):
    """Label only first/last point as S^{kind}_{i} with a white halo."""
    halo = [pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()]
    N = len(R)
    pick = []
    if which in ("both", "first"):
        pick.append(0)
    if which in ("both", "last"):
        pick.append(N-1)
    for j in pick:
        txt = site_label(kind, j+1)      # j is 0-based, labels are 1-based
        ax.text(R[j]+dx, Z[j]+dz, txt, fontsize=10, weight='bold',
                color='k', zorder=50, clip_on=False, path_effects=halo)

##    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig, ax = plt.subplots(figsize=(4,3), dpi=400)
# Plot cell-centered field on the (R,Z) grid
GridDataPlot(shot, q_mag_plot, canvas=ax, cmap=cmap, norm='log',  xlim=[2.41, 4], ylim=[-3.8, -2.4])

wall = surface("wall.surf", "2D")
domain = wall.polygon
core = surface("core.surf", "2D")
rcore, zcore = core.polygon.exterior.xy
Rwall, Zwall = domain.exterior.xy
ax.plot(Rwall, Zwall, 'k-', lw=2.5)
ax.plot(rcore, zcore, 'g-', lw=2.5)
innersite=np.loadtxt("innersite.txt", unpack=True, skiprows=1)
outersite=np.loadtxt("outersite.txt", unpack=True, skiprows=1)
#ax.plot(innersite[1], innersite[2], 'ro', ms=2.5)
#ax.plot(outersite[1], outersite[2], 'ko', ms=2.5)
ax.tick_params(axis="both", labelsize=10, direction="in")
ax.set_ylabel(r"Z(m)",fontsize=14)
ax.set_ylabel(r"R(m)",fontsize=14)
ax.grid(ls="--", alpha=0.5)

halo = [pe.Stroke(linewidth=1.2, foreground='white'), pe.Normal()]

halo = [pe.Stroke(linewidth=1.0, foreground='white'), pe.Normal()]  # thinner halo
#ax.text(..., fontsize=9, weight='normal', path_effects=halo)


inner = np.loadtxt("innersite.txt", unpack=True, skiprows=1)  # cols: ?, R, Z ...
outer = np.loadtxt("outersite.txt", unpack=True, skiprows=1)

Ri, Zi = inner[1], inner[2]
Ro, Zo = outer[1], outer[2]

# halo effect so points are readable on any background


# use:
label_endpoints(ax, Ri, Zi, kind="I", which="both")
label_endpoints(ax, Ro, Zo, kind="O", which="both")


si = ax.scatter(Ri, Zi, s=10, marker='o', c='#d62728', edgecolor='k',
                linewidths=0.5, zorder=20, path_effects=halo, label='Inner sites')
so = ax.scatter(Ro, Zo, s=10, marker='^', c='k', edgecolor='white',
                linewidths=0.6, zorder=21, path_effects=halo, label='Outer sites')
ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title('heat-flux density |q| (W.m$^{-2}$)')

ax.grid(True, ls='--', alpha=0.3)
plt.tight_layout()
fig.savefig('Figs/launching_sites.png', dpi=400, bbox_inches="tight", facecolor="white")
#fig.savefig('launching_sites.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close
#plt.show()
#
#import numpy as np
#import matplotlib.patheffects as pe
#
#fig, ax = plt.subplots(figsize=(5.2, 5.6), dpi=400)
#
## field
#GridDataPlot(shot, q_mag_plot, canvas=ax, cmap=cmap, norm='log',
#             xlim=[2.41, 4.0], ylim=[-3.8, -2.4])
#
## boundaries
#wall = surface("wall.surf", "2D")
#core = surface("core.surf", "2D")
#Rwall, Zwall = wall.polygon.exterior.xy
#Rcore, Zcore = core.polygon.exterior.xy
#ax.plot(Rwall, Zwall, color='k', lw=2.2, zorder=5)
#ax.plot(Rcore, Zcore, color='g', lw=1.8, zorder=5)
#
## sites
#inner = np.loadtxt("innersite.txt", unpack=True, skiprows=1)  # cols: ?, R, Z ...
#outer = np.loadtxt("outersite.txt", unpack=True, skiprows=1)
#
#Ri, Zi = inner[1], inner[2]
#Ro, Zo = outer[1], outer[2]
#
## halo effect so points are readable on any background
#halo = [pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()]
#
## use scatter for better control
#si = ax.scatter(Ri, Zi, s=28, marker='o', c='#d62728', edgecolor='k',
#                linewidths=0.5, zorder=20, path_effects=halo, label='Inner sites')
#so = ax.scatter(Ro, Zo, s=28, marker='^', c='k', edgecolor='white',
#                linewidths=0.6, zorder=21, path_effects=halo, label='Outer sites')

# optional: enumerate sites with tiny offset to avoid overlap
#import matplotlib.patheffects as pe
#
#def label_endpoints(ax, R, Z, prefix, which="both", dx=0.006, dz=0.006):
#    """Label only the first/last point of a polyline."""
#    halo = [pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()]
#    N = len(R)
#    idxs = []
#    if which in ("both", "first"):
#        idxs.append(0)
#    if which in ("both", "last"):
#        idxs.append(N-1)
#    for j in idxs:
#        ax.text(R[j] + dx, Z[j] + dz, f"{prefix}{j+1}",
#                fontsize=9, weight='bold', color='k',
#                zorder=50, clip_on=False, path_effects=halo)
#
## after plotting points:
## inner sites
#label_endpoints(ax, Ri, Zi, prefix="I", which="both")   # or "first"
## outer sites
#label_endpoints(ax, Ro, Zo, prefix="O", which="both")   # or "first"


## cosmetics
#ax.set_aspect('equal', adjustable='box')
#ax.tick_params(axis="both", labelsize=11, direction="in")
#ax.grid(ls="--", alpha=0.35)
#ax.set_xlabel(r"R [m]", fontsize=14)      # <- was ylabel twice
#ax.set_ylabel(r"Z [m]", fontsize=14)
#ax.set_title(r"Total heat-flux density $|q|$ [W m$^{-2}$]", fontsize=16, pad=6)
#
## tidy legend
#leg = ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
#for t in leg.get_texts():
#    t.set_fontsize(10)
#
#fig.savefig('launching_sites.png', dpi=300, bbox_inches='tight', facecolor='white')
#plt.tight_layout()
#plt.show()
