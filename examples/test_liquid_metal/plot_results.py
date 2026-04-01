#!/usr/bin/env python3
"""
Plot fix liquid_metal per-surface outputs from OpenEdge dump.

Reads dump.surf.* and plots:
  - Surface temperature vs arc length
  - Evaporation flux vs arc length
  - Ad-atom flux vs arc length
  - Film thickness vs arc length
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def read_surf_dump(fname):
    """Read an OpenEdge surface dump file."""
    with open(fname) as f:
        lines = f.readlines()

    # find ITEM: SURFS header
    data_start = None
    ncols = None
    col_names = []
    for i, line in enumerate(lines):
        if "ITEM: SURFS" in line:
            col_names = line.split("ITEM: SURFS")[1].strip().split()
            data_start = i + 1
            break

    if data_start is None:
        print(f"Could not parse {fname}")
        return None, None

    data = []
    for line in lines[data_start:]:
        vals = line.strip().split()
        if vals:
            data.append([float(v) for v in vals])

    return col_names, np.array(data)


def main():
    # find latest dump file (check current dir and output/)
    dumps = sorted(glob.glob("dump.surf.*")) + sorted(glob.glob("output/dump.surf.*"))
    if not dumps:
        print("No dump.surf.* files found. Run OpenEdge first.")
        sys.exit(1)

    fname = dumps[-1]
    print(f"Reading {fname}")
    col_names, data = read_surf_dump(fname)
    if data is None:
        sys.exit(1)

    print(f"  Columns: {col_names}")
    print(f"  {len(data)} surface elements")

    # map column names to indices
    ci = {name: i for i, name in enumerate(col_names)}

    # sort by surface ID (arc length proxy)
    sid = data[:, ci.get("id", 0)]
    order = np.argsort(sid)
    x = np.arange(len(data))  # surface index as x-axis

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("fix liquid_metal: per-surface outputs", fontsize=14)

    # Tsurf
    if "s_Tsurf_lm" in ci:
        ax = axes[0, 0]
        ax.plot(x, data[order, ci["s_Tsurf_lm"]], "r-", lw=1.5)
        ax.set_ylabel("Tsurf [°C]")
        ax.set_title("Surface Temperature")
        ax.grid(True, alpha=0.3)

    # evap flux
    if "s_evap_lm" in ci:
        ax = axes[0, 1]
        ax.plot(x, data[order, ci["s_evap_lm"]], "b-", lw=1.5)
        ax.set_ylabel("Evap flux [atoms/m²/s]")
        ax.set_title("Evaporation Flux (Antoine+HK)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # adatom flux
    if "s_adatom_lm" in ci:
        ax = axes[1, 0]
        ax.plot(x, data[order, ci["s_adatom_lm"]], "g-", lw=1.5)
        ax.set_ylabel("Adatom flux [atoms/m²/s]")
        ax.set_title("Ad-atom Flux")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # film thickness
    if "s_h_lm" in ci:
        ax = axes[1, 1]
        ax.plot(x, data[order, ci["s_h_lm"]] * 1e3, "k-", lw=1.5)
        ax.set_ylabel("Film thickness [mm]")
        ax.set_title("Film Thickness")
        ax.grid(True, alpha=0.3)

    # incident heat flux (derive from Tsurf profile)
    # Q_incident = qss * Qs0 — for constant source this is uniform,
    # but we can back-compute net heat flux from Tsurf gradient
    if "s_Tsurf_lm" in ci and "s_evap_lm" in ci:
        ax = axes[2, 0]
        Tsurf = data[order, ci["s_Tsurf_lm"]]
        evap = data[order, ci["s_evap_lm"]]
        # evaporative cooling: Q_vapor = H_vap * evap_flux / N_A
        H_vap = 145920.0  # J/mol
        N_A = 6.022e23
        Q_evap = H_vap * evap / N_A  # W/m²
        ax.plot(x, Q_evap / 1e6, "m-", lw=1.5)
        ax.set_ylabel("Q_evap [MW/m²]")
        ax.set_title("Evaporative Cooling")
        ax.grid(True, alpha=0.3)

    # total source flux = evap + adatom
    if "s_evap_lm" in ci and "s_adatom_lm" in ci:
        ax = axes[2, 1]
        evap = data[order, ci["s_evap_lm"]]
        adatom = data[order, ci["s_adatom_lm"]]
        ax.plot(x, evap, "b-", lw=1.5, label="Evaporation")
        ax.plot(x, adatom, "g-", lw=1.5, label="Ad-atoms")
        ax.plot(x, evap + adatom, "k--", lw=1.5, label="Total")
        ax.set_ylabel("Flux [atoms/m²/s]")
        ax.set_title("Total Li Source (evap + adatom)")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Surface element index")

    plt.tight_layout()
    plt.savefig("liquid_metal_profiles.png", dpi=150, bbox_inches="tight")
    print("Saved: liquid_metal_profiles.png")
    plt.show()


if __name__ == "__main__":
    main()
