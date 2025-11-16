#from freeqdsk import geqdsk
from matplotlib import pyplot as plt, Path
from scipy.constants import k, e, m_p
import numpy as np
import scipy.io
from scipy.ndimage import maximum_filter, gaussian_filter

import h5py
import numpy as np


data = h5py.File("bfield.h5", "r")

r = data["r"][:]
z = data["z"][:]
rs,zs = np.meshgrid(r,z)
br = data["br"][:] #.T
bz = data["bz"][:] #.T
bt = data["bt"][:] #.T
#
## Target coordinates
#target_r = 3.4277
#target_z = -3.46093
## Find the indices closest to the target coordinates
#r_idx = (np.abs(r - target_r)).argmin()
#z_idx = (np.abs(z - target_z)).argmin()
#
## Extract field values at the closest indices
##br_value = br[z_idx, r_idx]
##bz_value = bz[z_idx, r_idx]
##bt_value = bt[z_idx, r_idx]
#
#br_value = br[r_idx, z_idx]
#bz_value = bz[r_idx, z_idx]
#bt_value = bt[r_idx, z_idx]
#
##B: -0.295127 8.750185 -0.031703
## Output the results
#print(f"Field data at (r = {target_r}, z = {target_z}):")
#print(f"  Br: {br_value}")
#print(f"  Bz: {bz_value}")
#print(f"  Bt: {bt_value}")
##exit()
#plt.pcolormesh(rs,zs, data["br"][:].T)
fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

# Adjust space between subplots
#fig1.subplots_adjust(wspace=0.4, hspace=0.4)
#d = np.loadtxt("mesh.extra") #, unpack=True)

# Extract the R and Z coordinates
#ltx_wall_r= np.array([d[:, 0], d[:, 2]])
#ltx_wall_z = np.array([d[:, 1], d[:, 3]])
pcolormesh_3 = ax1.pcolormesh(rs,zs, br)
#ax1.contour(r2D, z2D, bt.T, levels, colors='black')
#ax1.plot(ltx_wall_r, ltx_wall_z, 'b', lw=2.5, label='wall')
ax1.set_xlabel('R (m)', fontsize=14, fontname='Times New Roman')
ax1.set_ylabel('Z (m)', fontsize=14, fontname='Times New Roman')
ax1.set_title("Bt2D [T]", fontsize=14, fontname='Times New Roman')
fig1.colorbar(pcolormesh_3, ax=ax1)

plt.show()
data.close


