# Copyright 2024, SPARTA-PMI contributors
# Authors: Abdou Diaw
# License: GPL-2.0 license
"""
This test file is part of SPARTA-PMI, a Particle transport code.

This test simulates the motion of an initial source at the core of a plasma, and examines how thermal gradient forces affect the spatial distribution of various oxygen species densities over time. Specifically, the simulation is run for different values of the perpendicular diffusion coefficient (D_perp), and the results are plotted to illustrate the impact on species distribution.

The test performs the following steps:
1. Initializes an initial particle source at the core of the plasma.
2. Simulates the cross-field diffusion for different values of the perpendicular diffusion coefficient (D_perp).
3. Reads the resulting species densities from the output files.
4. Plots the spatial distribution of each species density at a specific axial position (Z=0) for each D_perp value.
5. Provides visual comparisons to analyze the effect of varying D_perp on the distribution of oxygen species.

This script is designed to aid in the understanding of how variations in cross-field diffusion parameters influence particle transport within a WEST plasma.
"""


# ------
# Imports
# ------
import os
#os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
import matplotlib.colors as mcolors


import numpy as np
from scipy.constants import m_e, e, k, epsilon_0, m_p
# imports
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
import time
import subprocess as sp
import os

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
#from sample_soledge_data import interpolate_and_sample_oxygen, interpolate_and_sample_oxygen_charge

#from o2_w1 import main2

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
#from soledge_oxygen_wall_dist import interpolate_soledge_at_wall
#from sample_soledge_data import interpolate_and_sample_oxygen_hybrid
import h5py

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
import matplotlib.colors as mcolors


from utilities import surface, parser_density
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import h5py


# Number of MPI cores
MPI_cores = '6'



# Number of MPI cores
MPI_cores = '4'

def run_sparta_pmi_simulation(sparta_pmi_lmp, input_file):
    """
    Run a simulation with a particle going through a uniform magnetic field using SPARTA-PMI.

    Parameters:
    ----------
    sparta_pmi_lmp: str
        Path to the SPARTA-PMI executable.
    input_file: str
        Path to the input file for SPARTA-PMI.

    """
    sparta_pmi_script = open(input_file)
    args = ['mpirun', '-np', MPI_cores, sparta_pmi_lmp]
    sp.Popen(args, stdin=sparta_pmi_script).wait()
    sparta_pmi_script.close()
    time.sleep(2)



def plotter(open_edge_dir, geofile):
    files = sorted(os.listdir(open_edge_dir))
    labels = ["Li", "Li+", "Li2+", "Li3+"]

    d = np.loadtxt(geofile)

    wall_r= np.array([d[:, 0], d[:, 2]])
    wall_z = np.array([d[:, 1], d[:, 3]])

    wall = surface('input/reduce_surf.txt', "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy

    for fname in files:
        data = parser_density(os.path.join(open_edge_dir, fname), 2, 2+4)  # x,y + 4 species
        if not data:
            continue

        t = max(data.keys())
        xcs, ycs = data[t][0], data[t][1]
     
  
        species_arrays = data[t][3:3+4]     # four species, in order

        x = np.unique(xcs)
        y = np.unique(ycs)
        print(f"x {x},  y {y}")

        if x.size < 2 or y.size < 2:
            print(f"[skip] {fname}: not enough grid points.")
            continue

        # indices of each point in the (x,y) grid
        ix = np.searchsorted(x, xcs)
        iy = np.searchsorted(y, ycs)

        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

        for s, dens in enumerate(species_arrays):
            grid = np.zeros((y.size, x.size), dtype=float)
            # accumulate in case of duplicates
            np.add.at(grid, (iy, ix), dens)

            ax = axs[s]
            rs, zs = np.meshgrid(x, y)
            print(rs, zs)

            density_grid_smoothed = grid
            
            print(f"species {s} {np.unique(grid)}")

            if np.any(density_grid_smoothed):  # Check if the smoothed grid has data
                norm = matplotlib.colors.LogNorm(vmin=np.min(density_grid_smoothed[density_grid_smoothed > 0]),vmax=np.max(density_grid_smoothed))
                density_mesh = ax.pcolormesh(rs, zs, density_grid_smoothed, shading='auto', cmap='plasma', norm=norm)


                cbar = fig.colorbar(density_mesh, ax=ax, pad=0.01, fraction=0.04, shrink=0.8)
                cbar.set_label('Density [m$^{-3}$]', rotation=270, labelpad=15)
                
                ax.grid(alpha=0.3, linestyle='--')
              
            ax.plot(wall_r, wall_z, 'g', lw=2.5)
#            ax.plot(Rwall, Zwall, 'k', lw=2.5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(2.4,3.9)
            ax.set_ylim(-3.8,-2.6)
            ax.set_title(labels[s])

            ax.set_xlabel("R (m)", weight='semibold', fontsize=10)
            ax.set_ylabel("Z (m)", weight='semibold', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            
        plt.show()
        # plt.close(fig)  # uncomment if looping over many files


from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

PLOT_MODE = "imshow"     # "imshow" | "contourf" | "pcolormesh"
SMOOTH_SIGMA = 0.0       # e.g. 0.8 for light smoothing; 0.0 disables
VMIN_Q = 1               # percentile for vmin (robust to outliers)
VMAX_Q = 99              # percentile for vmax

# colormap with transparent "bad" (NaN) cells
_cmap = plt.get_cmap("plasma").copy()
_cmap.set_bad(alpha=0.0)

def plotter(open_edge_dir, geofile):
    files = sorted(os.listdir(open_edge_dir))
    labels = ["Li", "Li+", "Li2+", "Li3+"]

#    d = np.loadtxt(geofile)

#    wall_r= np.array([d[:, 0], d[:, 2]])
#    wall_z = np.array([d[:, 1], d[:, 3]])

    wall = surface('input/wall.txt', "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy

    core = surface('input/core.txt', "2D")
    domain = core.polygon
    rcore, zcore = domain.exterior.xy

    for fname in files:
        data = parser_density(os.path.join(open_edge_dir, fname), 2, 2+25)  # x,y + 4 species
        if not data:
            continue

        t = max(data.keys())
        xcs, ycs = data[t][0], data[t][1]
     
  
        species_arrays = data[t][3:3+4]     # four species, in order

        x = np.unique(xcs)
        y = np.unique(ycs)

        # indices of each point in the (x,y) grid
        ix = np.searchsorted(x, xcs)
        iy = np.searchsorted(y, ycs)



        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

#        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

        for s, dens in enumerate(species_arrays):
            sum_grid   = np.zeros((y.size, x.size), dtype=float)
            count_grid = np.zeros((y.size, x.size), dtype=float)
            np.add.at(sum_grid,   (iy, ix), dens)
            np.add.at(count_grid, (iy, ix), 1.0)
        
            # sum & count to average duplicates, start with zeros to allow np.add.at
            sum_grid   = np.zeros((y.size, x.size), dtype=float)
            count_grid = np.zeros((y.size, x.size), dtype=float)
            np.add.at(sum_grid,   (iy, ix), dens)
            np.add.at(count_grid, (iy, ix), 1.0)

            with np.errstate(invalid="ignore", divide="ignore"):
                grid = sum_grid / count_grid
                
            grid[count_grid == 0] = np.nan  # empty cells become NaN (transparent)

            # optional light smoothing (skips NaNs by filling, then re-mask)
            if SMOOTH_SIGMA > 0:
                fill = np.nanmedian(grid) if np.isfinite(grid).any() else 0.0
                tmp = np.where(np.isfinite(grid), grid, fill)
                tmp = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)
                grid = np.where(np.isfinite(grid), tmp, np.nan)

            # positive, finite data for LogNorm limits
            pos = grid[np.isfinite(grid) & (grid > 0)]
            ax = axs[s]

            if pos.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
#                ax.plot(wall_r, wall_z, "g", lw=2.5)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(2.4, 3.9); ax.set_ylim(-3.8, -2.6)
                ax.set_title(labels[s])
                ax.set_xlabel("R (m)"); ax.set_ylabel("Z (m)")
                ax.grid(alpha=0.3, linestyle="--")
                continue

            vmin = np.percentile(pos, VMIN_Q)
            vmax = np.percentile(pos, VMAX_Q)
            if vmin <= 0 or vmin >= vmax:  # last resort fallback
                vmin = pos.min()
                vmax = pos.max()

            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

            if PLOT_MODE == "imshow":
                extent = [x.min(), x.max(), y.min(), y.max()]
                im = ax.imshow(
                    grid, extent=extent, origin="lower", aspect="auto",
                    cmap=_cmap, norm=norm, interpolation="nearest"
                )
                mappable = im

            elif PLOT_MODE == "contourf":
                rs, zs = np.meshgrid(x, y)
                cf = ax.contourf(
                    rs, zs, grid, levels=80, cmap=_cmap, norm=norm
                )
                mappable = cf

            else:  # "pcolormesh"
                rs, zs = np.meshgrid(x, y)
                pm = ax.pcolormesh(
                    rs, zs, grid, shading="nearest", cmap=_cmap, norm=norm
                )
                mappable = pm

            cbar = fig.colorbar(mappable, ax=ax, pad=0.01, fraction=0.04, shrink=0.8)
            cbar.set_label('Density [m$^{-3}$]', rotation=270, labelpad=15)

            # overlay walls, tidy axes
            ax.plot(rcore, zcore, "b", lw=2.5)
            ax.plot(Rwall, Zwall ,"k", lw=2.5)
            ax.set_aspect("equal", adjustable="box")
#            ax.set_xlim(2.42, 4.07); ax.set_ylim(-3.8, -2.1)
            ax.set_title(labels[s])
            ax.set_xlabel("R (m)", weight="semibold", fontsize=10)
            ax.set_ylabel("Z (m)", weight="semibold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
                    
        plt.show()
        

def parser(filename, start_species, end_species):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i + 1].strip())
                number_of_cells = int(lines[i + 3].strip())
                xcs = []
                ycs = []
                density_values = [[] for _ in range(start_species, end_species + 1)]
                for j in range(number_of_cells):
                    cell_data_index = i + 5 + j
                    if cell_data_index >= len(lines):  # Check to avoid going out of bounds
                        break
                    cell_data_line = lines[cell_data_index]
                    cell_data = cell_data_line.split()
                    if len(cell_data) < end_species + 1:  # Ensure we have enough data in the line
                        continue
                    try:
                        xc = float(cell_data[1])
                        yc = float(cell_data[2])
                        densities = [float(cell_data[k]) for k in range(start_species, end_species + 1)]
                        xcs.append(xc)
                        ycs.append(yc)
                        for idx, density in enumerate(densities):
                            density_values[idx].append(density)
                    except ValueError:
                        # Skip if not a valid float
                        continue
                results[timestep] = [xcs, ycs] + density_values
    return results
    
def plotter(open_edge_dir, geofile):
    files = sorted(os.listdir(open_edge_dir))
    labels = ["Li", "Li+", "Li2+", "Li3+"]

#    d = np.loadtxt(geofile)

#    wall_r= np.array([d[:, 0], d[:, 2]])
#    wall_z = np.array([d[:, 1], d[:, 3]])

    wall = surface('input/wall.txt', "2D")
    domain = wall.polygon
    Rwall, Zwall = domain.exterior.xy

    core = surface('input/core.txt', "2D")
    domain = core.polygon
    rcore, zcore = domain.exterior.xy

    for fname in files:
        data = parser_density(os.path.join(open_edge_dir, fname), 2, 2+25)  # x,y + 4 species
        if not data:
            continue

        t = max(data.keys())
        xcs, ycs = data[t][0], data[t][1]
     
  
        species_arrays = data[t][3:3+4]     # four species, in order

        x = np.unique(xcs)
        y = np.unique(ycs)

        # indices of each point in the (x,y) grid
        ix = np.searchsorted(x, xcs)
        iy = np.searchsorted(y, ycs)



        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

#        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

        for s, dens in enumerate(species_arrays):
            sum_grid   = np.zeros((y.size, x.size), dtype=float)
            count_grid = np.zeros((y.size, x.size), dtype=float)
            np.add.at(sum_grid,   (iy, ix), dens)
            np.add.at(count_grid, (iy, ix), 1.0)
        
            # sum & count to average duplicates, start with zeros to allow np.add.at
            sum_grid   = np.zeros((y.size, x.size), dtype=float)
            count_grid = np.zeros((y.size, x.size), dtype=float)
            np.add.at(sum_grid,   (iy, ix), dens)
            np.add.at(count_grid, (iy, ix), 1.0)

            with np.errstate(invalid="ignore", divide="ignore"):
                grid = sum_grid / count_grid
                
            grid[count_grid == 0] = np.nan  # empty cells become NaN (transparent)

            # optional light smoothing (skips NaNs by filling, then re-mask)
            if SMOOTH_SIGMA > 0:
                fill = np.nanmedian(grid) if np.isfinite(grid).any() else 0.0
                tmp = np.where(np.isfinite(grid), grid, fill)
                tmp = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)
                grid = np.where(np.isfinite(grid), tmp, np.nan)

            # positive, finite data for LogNorm limits
            pos = grid[np.isfinite(grid) & (grid > 0)]
            ax = axs[s]

            if pos.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
#                ax.plot(wall_r, wall_z, "g", lw=2.5)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(2.4, 3.9); ax.set_ylim(-3.8, -2.6)
                ax.set_title(labels[s])
                ax.set_xlabel("R (m)"); ax.set_ylabel("Z (m)")
                ax.grid(alpha=0.3, linestyle="--")
                continue

            vmin = np.percentile(pos, VMIN_Q)
            vmax = np.percentile(pos, VMAX_Q)
            if vmin <= 0 or vmin >= vmax:  # last resort fallback
                vmin = pos.min()
                vmax = pos.max()

            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

            if PLOT_MODE == "imshow":
                extent = [x.min(), x.max(), y.min(), y.max()]
                im = ax.imshow(
                    grid, extent=extent, origin="lower", aspect="auto",
                    cmap=_cmap, norm=norm, interpolation="nearest"
                )
                mappable = im

            elif PLOT_MODE == "contourf":
                rs, zs = np.meshgrid(x, y)
                cf = ax.contourf(
                    rs, zs, grid, levels=80, cmap=_cmap, norm=norm
                )
                mappable = cf

            else:  # "pcolormesh"
                rs, zs = np.meshgrid(x, y)
                pm = ax.pcolormesh(
                    rs, zs, grid, shading="nearest", cmap=_cmap, norm=norm
                )
                mappable = pm
                
            if np.any(grid):  # Check if the grid has data
                # Plot using a logarithmic scale if requested
                norm = matplotlib.colors.LogNorm(vmin=np.min(grid[grid > 0]), vmax=np.max(grid))
                pcm = ax.pcolormesh(x, y, grid, shading='auto', cmap='plasma', norm=norm)
                fig.colorbar(pcm, ax=ax, label='Density [m$^{-3}$]')
                

            cbar = fig.colorbar(mappable, ax=ax, pad=0.01, fraction=0.04, shrink=0.8)
            cbar.set_label('Density [m$^{-3}$]', rotation=270, labelpad=15)

            # overlay walls, tidy axes
            ax.plot(rcore, zcore, "b", lw=2.5)
            ax.plot(Rwall, Zwall ,"k", lw=2.5)
            ax.set_aspect("equal", adjustable="box")
#            ax.set_xlim(2.42, 4.07); ax.set_ylim(-3.8, -2.1)
            ax.set_title(labels[s])
            ax.set_xlabel("R (m)", weight="semibold", fontsize=10)
            ax.set_ylabel("Z (m)", weight="semibold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
                    
        plt.show()
  
def main():
    path = '../'

    data = parser(f"tmp.grid", 2, 2+1)

    last_timestep = max(data.keys())
    species_labels = ["Ez"]


    logplot= True
    xcs, ycs = data[last_timestep][:2]
    x = np.unique(xcs)
    y = np.unique(ycs)
    Xmin, Xmax = np.inf, -np.inf
 

    fig, axes = plt.subplots(1,1, figsize=(8, 8))  # 2 rows, 5 columns for O to W+

#        axes = axes.flatten()
    tungsten_density_grid = np.zeros((len(y), len(x)))
    for species_index in [3]:  # Adjust to plot O to O7+
        id = species_index
        densities = data[last_timestep][id]
#        print(densities)
  

        density_grid = np.zeros((len(y), len(x)))
        for i, (xc, yc, density) in enumerate(zip(xcs, ycs, densities)):
            j = np.where(x == xc)[0]
            k = np.where(y == yc)[0]
            density_grid[k, j] = density
            
        print(density_grid)
        exit()
        ax = axes
        norm = matplotlib.colors.Normalize(vmin=min(np.unique(density)), vmax=max(np.unique(density)))
        pcm = ax.pcolormesh(x, y, density_grid, shading="auto", cmap='plasma', norm=norm)

        fig.colorbar(pcm, ax=ax, label='E [V/M$]')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
#        ax.set_title(f'Species {species_labels[id-2]}')
        ax.axis('scaled')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.axis('scaled')

#        plt.savefig(f'Figs/case.{tag}.png', dpi=300, bbox_inches='tight')
    plt.show()
        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def centers_to_edges(c):
    """Convert monotonically increasing centers → edges."""
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]     = c[0] - 0.5 * (c[1] - c[0])
    e[-1]    = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def plot_field_from_tmpgrid(data, field_index=3, title="E-field", logplot=False, cmap="plasma"):
    last = max(data.keys())
    xcs, ycs = data[last][:2]
    vals     = data[last][field_index]   # 1D array of your field

    # Unique coordinates + inverse indices for vectorized placement
    x, x_idx = np.unique(xcs, return_inverse=True)
    y, y_idx = np.unique(ycs, return_inverse=True)

    # Assemble grid
    G = np.full((y.size, x.size), np.nan)
    G[y_idx, x_idx] = vals

    # Convert centers to edges for pcolormesh
    xe = centers_to_edges(x)
    ye = centers_to_edges(y)

    # Choose normalization
    finite = np.isfinite(G)
    if logplot:
        # guard against nonpositive entries for LogNorm
        positive = finite & (G > 0)
        if not np.any(positive):
            raise ValueError("No positive values for LogNorm.")
        vmin = np.nanpercentile(G[positive], 1)
        vmax = np.nanpercentile(G[positive], 99)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = np.nanpercentile(G[finite], 1)
        vmax = np.nanpercentile(G[finite], 99)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(xe, ye, G, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("E [V/m]")  # change units label if needed

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

# ---- use it ----
def main():
    path = "../"
    data = parser("tmp.grid", 2, 2+1)   # your existing loader

    # Pick the right field index—3 in your snippet
    plot_field_from_tmpgrid(data, field_index=3, title="Sheath E-field", logplot=False)

if __name__ == "__main__":
    main()
