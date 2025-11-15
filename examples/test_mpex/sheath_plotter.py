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
 
def parser(filename, start_species, end_species):
    """
    start_species/end_species are f_Fdist indices (e.g., 11..13 for EX,EY,EZ).
    """
    results = {}
    with open(filename, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("ITEM: TIMESTEP"):
            # read timestep
            if i+1 >= n: break
            timestep = int(lines[i+1].strip())

            # read number of cells
            # Expect: i+2 -> "ITEM: NUMBER OF CELLS", i+3 -> integer
            j = i + 2
            if j+1 >= n or not lines[j].startswith("ITEM: NUMBER OF CELLS"):
                # scan forward defensively
                j = i+1
                while j < n and not lines[j].startswith("ITEM: NUMBER OF CELLS"):
                    j += 1
                if j+1 >= n: break
            number_of_cells = int(lines[j+1].strip())

            # find the CELLS header line
            k = j + 2
            while k < n and not lines[k].startswith("ITEM: CELLS"):
                k += 1
            if k >= n: break

            # parse header tokens after "ITEM: CELLS"
            header_tokens = lines[k].split()[2:]  # drop "ITEM:" "CELLS"
            col_index = {name: idx for idx, name in enumerate(header_tokens)}

            # convenience: where are xc,yc?
            # Header is: id xc yc zc f_Fdist[1] ...
            # We'll take xc,yc by name; if missing, fall back to fixed positions.
            xc_idx = col_index.get('xc', 1)
            yc_idx = col_index.get('yc', 2)

            # map requested f_Fdist indices to header positions
            wanted_cols = []
            for s in range(start_species, end_species + 1):
                key = f"f_Fdist[{s}]"
                if key not in col_index:
                    # tolerate missing columns by skipping
                    continue
                wanted_cols.append(col_index[key])

            xcs, ycs = [], []
            density_values = [[] for _ in wanted_cols]

            # first data row is k+1, then number_of_cells rows
            first_row = k + 1
            last_row = min(first_row + number_of_cells, n)
            for r in range(first_row, last_row):
                parts = lines[r].split()
                # align with header length (header excludes the leading 'id' field in file row)
                # File row tokens are: id xc yc zc f_Fdist[1] ...
                # We skip the first token (cell id) to align with header_tokens.
                row = parts[1:]
                if len(row) < max([xc_idx, yc_idx, *(wanted_cols or [0])]) + 1:
                    continue
                try:
                    xcs.append(float(row[xc_idx]))
                    ycs.append(float(row[yc_idx]))
                    for jcol, col in enumerate(wanted_cols):
                        density_values[jcol].append(float(row[col]))
                except ValueError:
                    continue

            results[timestep] = [xcs, ycs] + density_values

            # advance i to after this block to continue scanning (if multi-timestep files)
            i = last_row
        else:
            i += 1

    return results


#!/usr/bin/env python3
"""
Plot a SPARTA-style surface file with 'Points' and 'Triangles' sections.

Usage:
    python plot_surface3d.py -i flatSurface.txt [-o out.png]

This script reads the surface file, parses the points and triangles,
and renders a 3D triangulated mesh using matplotlib.

Author: ChatGPT
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_sparta_surface(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    try:
        pts_idx = lines.index("Points") + 1
        tri_idx = lines.index("Triangles") + 1
    except ValueError as e:
        raise RuntimeError("Could not find 'Points' or 'Triangles' section") from e

    pts = []
    i = pts_idx
    while i < tri_idx - 1:
        parts = lines[i].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                x, y, z = map(float, parts[1:])
                pts.append((x, y, z))
            except ValueError:
                pass
        i += 1

    tris = []
    for j in range(tri_idx, len(lines)):
        parts = lines[j].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                i1, i2, i3 = map(int, parts[1:])
                tris.append((i1 - 1, i2 - 1, i3 - 1))
            except ValueError:
                pass

    if not pts or not tris:
        raise RuntimeError("Parsed zero points or zero triangles; check file format.")
    return np.array(pts, dtype=float), np.array(tris, dtype=int)

def plot_surface(pts, tris, save=None, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    ranges = xyz_max - xyz_min
    max_range = max(ranges)
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface mesh (SPARTA)")
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
if __name__ == "__main__":

    exec_lmp =   '/Users/42d/OpenEdge/build/src/spa_mac_mpi'
    input_file = 'in.test'
    open_edge_dir = 'output'

    pts, tris = parse_sparta_surface("surfaces/flatSurface.txt")

    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111, projection='3d')

    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)

#    if save:
#        plt.savefig(save, dpi=300, bbox_inches="tight")
#    else:
    plt.show()
