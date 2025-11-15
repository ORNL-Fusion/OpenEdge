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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.patches as patches
from utilities import surface, parser_density
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path


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


# geometry constants (from your gmsh scripts)
Z_SKIMMER_FRONT = 0.029       # skimmer upstream face
SKIMMER_THK     = 0.002
Z_SKIMMER_BACK  = Z_SKIMMER_FRONT + SKIMMER_THK
Z_SKIMMER_C     = 0.5 * (Z_SKIMMER_FRONT + Z_SKIMMER_BACK)

Z_TARGET_C      = 0.15        # target center (z_target)
TARGET_THK      = 0.002
Z_TARGET_FRONT  = Z_TARGET_C - 0.5 * TARGET_THK
Z_TARGET_BACK   = Z_TARGET_C + 0.5 * TARGET_THK

R_SKIMMER_IN    = 0.025       # inner radius of skimmer ring
TARGET_SIZE     = 0.08        # 8 cm x 8 cm plate

def main():
    # 3 coords + Nfields (here: 11 f_fixID[*])
    data = parser("tmp.grid.density", 3, 3 + 1 + 10)

    last_timestep = max(data.keys())
    xcs, ycs, zcs = data[last_timestep][:3]

    # convert to numpy arrays
    xcs = np.array(xcs, dtype=float)
    ycs = np.array(ycs, dtype=float)
    zcs = np.array(zcs, dtype=float)

    # ---- collect all species densities f_fixID[1..N] ----
    n_fields = len(data[last_timestep]) - 3
    densities_list = [
        np.array(data[last_timestep][3 + i], dtype=float)
        for i in range(n_fields)
    ]

    # total density (sum over charge states)
    densities_tot = np.sum(densities_list, axis=0)

    # ----------------- 2D XY map (sum over z) -----------------
    x, x_idx = np.unique(xcs, return_inverse=True)
    y, y_idx = np.unique(ycs, return_inverse=True)

    density_grid = np.zeros((len(y), len(x)))
    for i, d in enumerate(densities_tot):
        j = x_idx[i]   # column in x
        k = y_idx[i]   # row in y
        density_grid[k, j] += d

    pos = density_grid > 0
    norm = matplotlib.colors.LogNorm(
        vmin=density_grid[pos].min(),
        vmax=density_grid[pos].max()
    )

    # ----------------- 1D n(z): radial average per species -----------------
    # beam center & radius
    xc0 = x.mean()
    yc0 = y.mean()
    R0  = 0.02   # 2 cm beam radius

    r = np.sqrt((xcs - xc0)**2 + (ycs - yc0)**2)
    mask_beam = r <= R0

    z_bins, z_idx = np.unique(zcs, return_inverse=True)

    count_z = np.bincount(
        z_idx,
        weights=mask_beam.astype(float),
        minlength=len(z_bins),
    )
    denom = np.maximum(count_z, 1.0)

    n_z_species = []
    for d_s in densities_list:
        dens_in_beam_s = np.where(mask_beam, d_s, 0.0)
        sum_z_s = np.bincount(
            z_idx, weights=dens_in_beam_s, minlength=len(z_bins)
        )
        n_z_species.append(sum_z_s / denom)

    # ----------------- plotting: map + multi-species profile -----------------
    fig, (ax_map, ax_prof) = plt.subplots(
        1, 2, figsize=(9, 4), dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    ax_map.tick_params(axis='both', labelsize=16, direction='in')
    ax_prof.tick_params(axis='both', labelsize=16, direction='in')

    # ----- 2D XY map (z-integrated, total Ta density) -----
    pcm = ax_map.pcolormesh(
        x, y, density_grid, shading="auto",
        cmap="plasma", norm=norm
    )
    ax_map.set_xlabel('X (m)', weight="semibold", fontsize=12)
    ax_map.set_ylabel('Y (m)', weight="semibold", fontsize=12)
    ax_map.grid(True, linestyle="--", alpha=0.5)
    ax_map.set_title('Total Tantalum Density (m$^{-3}$)',
                     weight="semibold", fontsize=12)
    fig.colorbar(pcm, ax=ax_map) #, label='Density (m$^{-3}$)')

#    # skimmer aperture footprint (circle)
#    circ = patches.Circle(
#        (xc0, yc0), radius=R_SKIMMER_IN,
#        fill=False, linestyle='-', linewidth=1.2
#    )
#    ax_map.add_patch(circ)
#    ax_map.text(
#        xc0, yc0 + R_SKIMMER_IN + 0.003,
#        "Skimmer", ha='center', va='bottom', fontsize=9
#    )

    # target footprint (square plate)
#    half = TARGET_SIZE / 2.0
#    rect = patches.Rectangle(
#        (xc0 - half, yc0 - half),
#        TARGET_SIZE, TARGET_SIZE,
#        fill=False, linestyle='--', linewidth=1.2
#    )
#    ax_map.add_patch(rect)
#    ax_map.text(
#        xc0 + half + 0.003, yc0,
#        "Target", ha='left', va='center', fontsize=9
#    )

    # ----- 1D radial-averaged n(z) for each charge state -----
    species_labels = [
        "Ta", "Ta$^+$", "Ta$^{2+}$", "Ta$^{3+}$", "Ta$^{4+}$",
        "Ta$^{5+}$", "Ta$^{6+}$", "Ta$^{7+}$", "Ta$^{8+}$",
        "Ta$^{9+}$", "Ta$^{10+}$"
    ][:n_fields]

    for n_z, label in zip(n_z_species, species_labels):
        ax_prof.semilogy(
            z_bins, n_z, marker='o', ms=3, lw=1.0, label=label
        )

    # highlight skimmer and target locations along z
    ax_prof.axvspan(
        Z_SKIMMER_FRONT, Z_SKIMMER_BACK,
        alpha=0.1, hatch='///', edgecolor='k', facecolor='blue'
    )
    ax_prof.text(
        Z_SKIMMER_C, ax_prof.get_ylim()[0]*1.2,
        "Skimmer", rotation=90,
        ha='center', va='bottom', fontsize=8
    )

    ax_prof.axvspan(
        Z_TARGET_FRONT, Z_TARGET_BACK,
        alpha=0.5, hatch='\\\\\\', edgecolor='k', facecolor='red'
    )
    ax_prof.text(
        Z_TARGET_C, ax_prof.get_ylim()[0]*1.2,
        "Target", rotation=90,alpha=0.5,
        ha='center', va='bottom', fontsize=8
    )

    ax_prof.set_xlabel('Z (m)', weight="semibold", fontsize=12)
    ax_prof.set_ylabel('Density (m$^{-3}$)', weight="semibold", fontsize=12)
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.legend(fontsize=9, loc="best")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.3)

#    fig.savefig("Figs/out_png, dpi=400, bbox_inches="tight", facecolor="white")
        
    plt.show()

if __name__ == "__main__":
    main()


