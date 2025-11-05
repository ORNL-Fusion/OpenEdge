# Copyright 2023–2025, OpenEdge contributors
# Author: Abdou Diaw 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional


# ----------------------------
# Parsing
# ----------------------------
def parse_data_openedge(file_path: str, n_species: int) -> pd.DataFrame:
    """
    Parse OpenEdge 'tmp.grid.*' text output into a tidy DataFrame.
    Columns: Timestep, Cell ID, xc, yc, f_1[1..n_species]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    cur_ts = None
    reading_cells = False

    for i, line in enumerate(lines):
        if line.startswith('ITEM: TIMESTEP'):
            reading_cells = False
            try:
                cur_ts = int(lines[i + 1].strip())
            except Exception:
                cur_ts = None
        elif line.startswith('ITEM: CELLS'):
            reading_cells = True
        elif reading_cells:
            parts = line.split()
            if len(parts) >= 4:
                cell_id, xc, yc, *rest = parts
                # take only n_species columns (OpenEdge may print more)
                fvals = [float(x) for x in rest[:n_species]]
                data.append([cur_ts, int(cell_id), float(xc), float(yc)] + fvals)

    cols = ['Timestep', 'Cell ID', 'xc', 'yc'] + [f'f_1[{i}]' for i in range(1, n_species + 1)]
    return pd.DataFrame(data, columns=cols)

# ----------------------------
# Plotting
# ----------------------------
def plot_ionization_balance(df: pd.DataFrame,
                            material: str,
                            n_species: int,
                            dt: float,
                            save_path: Optional[str] = None,
                            title: Optional[str] = None):
    """
    Plot fractional abundances vs time (ms) from OpenEdge output.
    Colors are consistent across species; no reversal.
    """
    # LaTeX-like labels: X, X^{1+}, ..., X^{Z+}
    species_labels = [material] + [fr"${material}^{{{i}+}}$" for i in range(1, n_species)]

    # 10 distinct colors (repeat if n_species > 10)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_species)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
    ax.tick_params(axis='both', direction='in')

    # Stable normalization by the max of ground-state column (fall back to 1)
    denom_series = df.get('f_1[1]')
    denom = float(np.max(denom_series)) if denom_series is not None and np.max(denom_series) != 0 else 1.0

    t_ms = df['Timestep'].to_numpy(dtype=float) * dt * 1e3

    for s in range(n_species):
        col = colors[s]
        y = df[f'f_1[{s+1}]'].to_numpy(dtype=float) / denom
        ax.semilogx(t_ms, y, '-', lw=2, color=col)

    # Legends
    species_handles = [Line2D([0], [0], color=colors[s], lw=2) for s in range(n_species)]
    leg_species = ax.legend(handles=species_handles, labels=species_labels,
                            fontsize=14, frameon=False, loc='lower left')
    ax.add_artist(leg_species)

    style_handles = [Line2D([0], [0], color='k', lw=2, linestyle='-', label='OpenEdge')]
    ax.legend(handles=style_handles, fontsize=14, frameon=False, loc='upper right')

    ax.set_xlabel('Time (ms)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Fractional abundance (-)', fontsize=14, fontname='Times New Roman')
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if title:
        ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

# ----------------------------
# Selection helper
# ----------------------------
ELEMENTS = {
    # tweak dt and n_species for your runs here
    'O': dict(file='tmp.grid.oxygen',  n_species=9,  dt=1e-7),
    'W': dict(file='tmp.grid.tungsten', n_species=10, dt=1e-9),
}


def load_and_plot(element: str,
                  base_path: str,
                  n_species: Optional[int] = None,
                  dt: Optional[float] = None,
                  save_path: Optional[str] = None):
    """
    Choose which element to load/plot. You can override n_species and dt.
    """
    element = element.upper()
    if element not in ELEMENTS:
        raise ValueError(f"Unknown element '{element}'. Options: {list(ELEMENTS)}")

    cfg = ELEMENTS[element].copy()
    if n_species is not None:
        cfg['n_species'] = n_species
    if dt is not None:
        cfg['dt'] = dt

    fpath = os.path.join(base_path, cfg['file'])
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Could not find file: {fpath}")

    df = parse_data_openedge(fpath, cfg['n_species'])
    plot_ionization_balance(df,
                            material=element,
                            n_species=cfg['n_species'],
                            dt=cfg['dt'],
                            save_path=save_path,
                            title=f"{element} ionization balance")


if __name__ == "__main__":
    base = "output"
    
    # load and plot data
#    load_and_plot('O', base_path=base)

    load_and_plot('W', base_path=base, n_species=10, dt=1e-10)
