# Copyright 2024, OpenEdge contributors
# Authors: Abdou Diaw
# License: GPL-2.0 license
"""
This script is part of OpenEdge, a particle transport code.

The script `interpolate_and_save_plasma_field` reads plasma and magnetic field data from SOLPS,
interpolates it to a specified grid, and converts it into an HDF5 format compatible with OpenEdge. 

The data includes electron and ion temperatures and densities, magnetic field components, 
and parallel gradients along magnetic field lines, among other plasma parameters.

You need the quixote software to process solps data. Quixote was developed by Iván Paradela Pérez (ORNL)
https://gitlab.com/ipar/quixote

Parameters:
- solps_folder: Directory containing input SOLPS data files.
- openedge_plasma_file: Filename to store interpolated plasma data in OpenEdge format.
- openedge_bfield_file: Filename to store interpolated magnetic field data in OpenEdge format.
- nR, nZ: Number of grid points in radial (R) and axial (Z) directions for interpolation.

The function will create two HDF5 files (openedge_plasma_file and openedge_bfield_file) with 
interpolated plasma and magnetic field data that OpenEdge can use for simulations.
"""

import os
import sys
import subprocess

#sys.path.insert(0, "/Users/42d/quixote-master")
# Import other necessary modules
import h5py
from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path
import quixote


import numpy as np
import h5py
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def interpolate_and_save_plasma_field(solps_folder, openedge_plasma_file, openedge_bfield_file, Plotter=False):
    plasma_data = quixote.SolpsData(solps_folder, name='d3d')
    
    # find inner and outer flux
    # Assuming plasma_data.crx and plasma_data.cry are arrays of r and z coordinates for flux surfaces


    n_r = 100
    n_z = 120

    # radial (0–15 cm) and axial (-1–20 cm) grids
    r = np.linspace(0.0, 4, n_r)
    z = np.linspace(-0.01, 0.20, n_z)

    R, Z = np.meshgrid(r, z, indexing="xy")

    # super-Gaussian Te(r)
    Te = 1.0 + 4.0 * np.exp(-(R / 0.02)**12)
    ne = 1e19 * np.exp(-(R / 0.02)**12)
    ni =ne
    Ti = Te
    
    
##     Interpolate all fields
#    parr_flow_grid = interpolate_field(parr_flow)
    te_grid = interpolate_field(te)
    ne_grid = interpolate_field(ne)
    ti_grid = interpolate_field(ti)
    ni_grid = interpolate_field(ni)
    ua_z_grid = interpolate_field(ua_z)
    ua_r_grid = interpolate_field(ua_r)
    ua_t_grid = interpolate_field(ua_t)
    grad_te_z_grid = interpolate_field(grad_te_z)
    grad_te_r_grid = interpolate_field(grad_te_r)
    grad_te_t_grid = interpolate_field(grad_te_t)
    grad_ti_z_grid = interpolate_field(grad_ti_z)
    grad_ti_r_grid = interpolate_field(grad_ti_r)
    grad_ti_t_grid = interpolate_field(grad_ti_t)
    b_r_grid = interpolate_field(Br)
    b_z_grid = interpolate_field(Bz)
    b_phi_grid = interpolate_field(Bt)
  

    print("b_phi_grid outside h5", np.unique(b_phi_grid))
    print("b_r_grid outside h5", np.unique(b_r_grid))
    print("b_phi_grid outside h5", np.unique(b_phi_grid))
    
#    # Save interpolated plasma data to HDF5 file
    with h5py.File(openedge_plasma_file, 'w') as file:
        file.create_dataset('r', data=grid_r)
        file.create_dataset('z', data=grid_z)
        file.create_dataset('temp_e', data=te_grid)
        file.create_dataset('temp_i', data=ti_grid)
        file.create_dataset('dens_e', data=ne_grid)
        file.create_dataset('dens_i', data=ni_grid)
        file.create_dataset('parr_flow', data=parr_flow_grid)
        file.create_dataset('parr_flow_r', data=ua_r_grid)
        file.create_dataset('parr_flow_z', data=ua_z_grid)
        file.create_dataset('parr_flow_t', data=ua_t_grid)
        file.create_dataset('grad_te_r', data=grad_te_r_grid)
        file.create_dataset('grad_te_z', data=grad_te_z_grid)
        file.create_dataset('grad_te_t', data=grad_te_t_grid)
        file.create_dataset('grad_ti_r', data=grad_ti_r_grid)
        file.create_dataset('grad_ti_z', data=grad_ti_z_grid)
        file.create_dataset('grad_ti_t', data=grad_ti_t_grid)

    # Save interpolated magnetic field data to HDF5 file
    with h5py.File(openedge_bfield_file, 'w') as file:
        print("b_phi_grid inside h5", np.unique(b_phi_grid))
        print("b_r_grid inside h5", np.unique(b_r_grid))
        print("b_phi_grid inside h5", np.unique(b_phi_grid))
        file.create_dataset('r', data=grid_r)
        file.create_dataset('z', data=grid_z)
        file.create_dataset('br', data=b_r_grid)
        file.create_dataset('bz', data=b_z_grid)
        file.create_dataset('bt', data=b_phi_grid)

import numpy as np
import h5py
import matplotlib.pyplot as plt

def interpolate_and_save_plasma_field_const(solps_folder,  # unused, kept for API compat
                                      openedge_plasma_file,
                                      openedge_bfield_file,
                                      Plotter=False):
    """
    Build analytic MPEX challenge plasma + B field and save to two HDF5 files
    in the format OpenEdge expects.

    Plasma:
        ne(r) = 1e19 * exp(-(r/0.02)^12)
        Te(r) = Ti(r) = 1 eV + 4 eV * exp(-(r/0.02)^12)
        no parallel flow, no gradients

    B field:
        Bz = 0.5 T, Br = Bt = 0
    """

    # --- 1D grids (r,z) ---
    n_r = 100
    n_z = 120

    # radial (0–15 cm) and axial (-1–20 cm) grids
    grid_r = np.linspace(0.0, 0.15, n_r)      # [m]
    grid_z = np.linspace(-0.01, 0.20, n_z)    # [m]

    # 2D mesh for fields
    R, Z = np.meshgrid(grid_r, grid_z, indexing="xy")   # shape (n_z, n_r)

    # --- analytic profiles (all in eV and m^-3) ---
    r0 = 0.02  # 2 cm
    ne0 = 1e19

    ne = ne0 * np.exp(-(R / r0) ** 12)
    Te = 1.0 + 4.0 * np.exp(-(R / r0) ** 12)
    Ti = Te.copy()
    ni = ne.copy()

    # zeros with same 2D shape
    zeros = np.zeros_like(Te)

    # --- save plasma file ---
    with h5py.File(openedge_plasma_file, 'w') as f:
        # 1D geometry
        f.create_dataset('r', data=grid_r)
        f.create_dataset('z', data=grid_z)

        # core thermodynamic fields (2D, z×r)
        f.create_dataset('temp_e', data=Te)
        f.create_dataset('temp_i', data=Ti)
        f.create_dataset('dens_e', data=ne)
        f.create_dataset('dens_i', data=ni)

        # parallel flow and its components (all zero)
        f.create_dataset('parr_flow',   data=zeros)
        f.create_dataset('parr_flow_r', data=zeros)
        f.create_dataset('parr_flow_z', data=zeros)
        f.create_dataset('parr_flow_t', data=zeros)

        # temperature gradients (all zero)
        f.create_dataset('grad_te_r', data=zeros)
        f.create_dataset('grad_te_z', data=zeros)
        f.create_dataset('grad_te_t', data=zeros)

        f.create_dataset('grad_ti_r', data=zeros)
        f.create_dataset('grad_ti_z', data=zeros)
        f.create_dataset('grad_ti_t', data=zeros)

    # --- save B-field file ---
    B0 = 0.5  # Tesla, axial field

    with h5py.File(openedge_bfield_file, 'w') as f:
        f.create_dataset('r', data=grid_r)
        f.create_dataset('z', data=grid_z)

        Br = zeros          # 0 everywhere
        Bz = B0 * np.ones_like(Te)
        Bt = zeros

        f.create_dataset('br', data=Br)
        f.create_dataset('bz', data=Bz)
        f.create_dataset('bt', data=Bt)

    # --- optional quick plots ---
    if Plotter:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        ax0, ax1, ax2 = axes

        im0 = ax0.pcolormesh(grid_r, grid_z, ne,
                             shading="auto")
        ax0.set_title(r"$n_e(r,z)$")
        ax0.set_xlabel("r (m)")
        ax0.set_ylabel("z (m)")
        fig.colorbar(im0, ax=ax0)

        im1 = ax1.pcolormesh(grid_r, grid_z, Te,
                             shading="auto")
        ax1.set_title(r"$T_e(r,z)$ [eV]")
        ax1.set_xlabel("r (m)")
        ax1.set_ylabel("z (m)")
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.pcolormesh(grid_r, grid_z, Bz,
                             shading="auto")
        ax2.set_title(r"$B_z(r,z)$ [T]")
        ax2.set_xlabel("r (m)")
        ax2.set_ylabel("z (m)")
        fig.colorbar(im2, ax=ax2)

        for ax in axes:
            ax.grid(True, ls="--", alpha=0.3)

        fig.tight_layout()
        plt.show()



if __name__ == "__main__":
    from pathlib import Path

    solps_folder = Path("/Users/42d/ORNL Dropbox/Abdou DIaw/addLi/"
                        "fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up")

    openedge_plasma_file = "input/plasma.h5"
    openedge_bfield_file = "input/bfield.h5"

    print("Exists?", solps_folder.exists())  # should be True
#    interpolate_and_save_plasma_field(str(solps_folder), openedge_plasma_file, openedge_bfield_file, Plotter=True)

    interpolate_and_save_plasma_field_const(solps_folder,  # unused, kept for API compat
                                          openedge_plasma_file,
                                          openedge_bfield_file,
                                          Plotter=False)
