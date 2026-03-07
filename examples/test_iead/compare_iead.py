#!/usr/bin/env python3
"""
Compare OpenEdge IEAD results against sheath_tracker.f90 output.

Reads:
  - output/vanish_alpha{N}.rank*  (OpenEdge surf_collide vanish CSV logs)
  - iead_alpha{N}.csv             (Fortran sheath_tracker output)

Produces:
  - output/iead_comparison.png    (side-by-side 2D IEAD)
  - output/iead_1d_energy.png     (1D energy distributions)
  - output/iead_1d_angle.png      (1D angle distributions)
  - output/iead_summary.txt       (mean energy/angle table)

Usage:
    python3 compare_iead.py [--fortran-dir /path/to/fortran/output]
"""

import argparse
import glob as globmod
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ANGLES = [0, 45, 85]

# Histogram parameters
EMAX     = 100.0    # eV
NBINS_E  = 100
AMAX     = 90.0     # deg
NBINS_A  = 90

QE  = 1.602176634e-19
AMU = 1.660539067e-27

# Species mass (all Ta)
Ta_amu = 180.948
Ta_mass = Ta_amu * AMU

# Species charges by 1-indexed type (matching plasma.species order)
SPECIES_Z = {1: 2, 2: 3, 3: 4}  # type 1=Ta2+, type 2=Ta3+, type 3=Ta4+


def read_openedge_vanish_logs(prefix):
    """Read vanish CSV log files (one per MPI rank).

    Files are named: prefix.rank0, prefix.rank1, ...
    Columns: timestep,rank,surf_index,surf_id,surf_hit_count,
             id,ispecies,x,y,z,vx,vy,vz,nx,ny,nz,
             mass,radius,temp,weight,dtremain,erot,evib

    Returns arrays of (energy_eV, angle_deg) for all particles.
    """
    pattern = f"{prefix}.rank*"
    files = sorted(globmod.glob(pattern))
    if not files:
        # Try without rank suffix (single-proc)
        if os.path.exists(prefix):
            files = [prefix]
        else:
            return None, None

    all_energy = []
    all_angle = []

    for fname in files:
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("timestep"):
                    continue
                parts = line.split(",")
                if len(parts) < 16:
                    continue
                try:
                    ispecies = int(parts[6])
                    vx = float(parts[10])
                    vy = float(parts[11])
                    vz = float(parts[12])
                    nx = float(parts[13])
                    ny = float(parts[14])
                    nz = float(parts[15])
                except (ValueError, IndexError):
                    continue

                # Kinetic energy in eV (mass from species, not log)
                v2 = vx*vx + vy*vy + vz*vz
                energy_eV = 0.5 * Ta_mass * v2 / QE

                # Angle from surface normal
                # Normal points inward (-z typically), velocity points toward wall (+z)
                # Use |v . n| / |v| to get cos(angle from normal)
                vmag = math.sqrt(v2)
                if vmag < 1e-30:
                    continue
                nmag = math.sqrt(nx*nx + ny*ny + nz*nz)
                if nmag < 1e-30:
                    continue
                cos_theta = abs(vx*nx + vy*ny + vz*nz) / (vmag * nmag)
                cos_theta = min(cos_theta, 1.0)
                angle_deg = math.degrees(math.acos(cos_theta))

                all_energy.append(energy_eV)
                all_angle.append(angle_deg)

    if not all_energy:
        return None, None
    return np.array(all_energy), np.array(all_angle)


def read_fortran_csv(fname):
    """Read Fortran sheath_tracker CSV: species_Z,energy_eV,angle_deg"""
    if not os.path.exists(fname):
        return None, None
    data = np.loadtxt(fname, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return None, None
    energies = data[:, 1]
    angles   = data[:, 2]
    return energies, angles


def bin_data(energies, angles):
    """Bin per-particle data into 2D histogram."""
    e_edges = np.linspace(0, EMAX, NBINS_E + 1)
    a_edges = np.linspace(0, AMAX, NBINS_A + 1)
    hist, _, _ = np.histogram2d(angles, energies, bins=[a_edges, e_edges])
    return hist


def plot_2d_comparison(oe_hists, ft_hists, out_png):
    """3-column figure: one column per angle, top=OpenEdge, bottom=Fortran."""
    e_edges = np.linspace(0, EMAX, NBINS_E + 1)
    a_edges = np.linspace(0, AMAX, NBINS_A + 1)

    n_angles = len(ANGLES)
    fig, axes = plt.subplots(2, n_angles, figsize=(5 * n_angles, 8),
                             sharex=True, sharey=True)
    if n_angles == 1:
        axes = axes.reshape(2, 1)

    for col, adeg in enumerate(ANGLES):
        for row, (label, hists) in enumerate([("OpenEdge", oe_hists),
                                               ("Fortran", ft_hists)]):
            ax = axes[row, col]
            h = hists.get(adeg)
            if h is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(f"{label}  \u03b1={adeg}\u00b0")
                continue

            h_plot = np.ma.masked_where(h == 0, h)
            vmax = max(h.max(), 1)
            im = ax.pcolormesh(e_edges, a_edges, h_plot,
                               norm=LogNorm(vmin=1, vmax=vmax),
                               cmap="plasma", shading="auto")
            total = h.sum()
            if total > 0:
                e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
                a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
                mean_e = np.sum(h * e_centers[None, :]) / total
                mean_a = np.sum(h * a_centers[:, None]) / total
                ax.set_title(f"{label}  \u03b1={adeg}\u00b0\n"
                             f"<E>={mean_e:.1f} eV  <\u03b8>={mean_a:.1f}\u00b0  N={int(total)}")
            else:
                ax.set_title(f"{label}  \u03b1={adeg}\u00b0")

            if col == 0:
                ax.set_ylabel("Angle [deg]")
            if row == 1:
                ax.set_xlabel("Energy [eV]")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def plot_1d_energy(oe_hists, ft_hists, out_png):
    """1D energy distribution comparison, one panel per angle."""
    e_edges = np.linspace(0, EMAX, NBINS_E + 1)
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])

    fig, axes = plt.subplots(1, len(ANGLES), figsize=(5 * len(ANGLES), 4),
                             sharey=False)
    if len(ANGLES) == 1:
        axes = [axes]

    for col, adeg in enumerate(ANGLES):
        ax = axes[col]
        for label, hists, ls in [("OpenEdge", oe_hists, "-"),
                                  ("Fortran", ft_hists, "--")]:
            h = hists.get(adeg)
            if h is None:
                continue
            e_dist = h.sum(axis=0)  # sum over angle bins
            total = e_dist.sum()
            if total > 0:
                e_dist = e_dist / (total * (e_edges[1] - e_edges[0]))
            ax.plot(e_centers, e_dist, ls, lw=1.8, label=label)

        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("PDF [1/eV]")
        ax.set_title(f"\u03b1 = {adeg}\u00b0")
        ax.legend(framealpha=0.9)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_xlim(0, EMAX)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def plot_1d_angle(oe_hists, ft_hists, out_png):
    """1D angle distribution comparison, one panel per angle."""
    a_edges = np.linspace(0, AMAX, NBINS_A + 1)
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])

    fig, axes = plt.subplots(1, len(ANGLES), figsize=(5 * len(ANGLES), 4),
                             sharey=False)
    if len(ANGLES) == 1:
        axes = [axes]

    for col, adeg in enumerate(ANGLES):
        ax = axes[col]
        for label, hists, ls in [("OpenEdge", oe_hists, "-"),
                                  ("Fortran", ft_hists, "--")]:
            h = hists.get(adeg)
            if h is None:
                continue
            a_dist = h.sum(axis=1)  # sum over energy bins
            total = a_dist.sum()
            if total > 0:
                a_dist = a_dist / (total * (a_edges[1] - a_edges[0]))
            ax.plot(a_centers, a_dist, ls, lw=1.8, label=label)

        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("PDF [1/deg]")
        ax.set_title(f"\u03b1 = {adeg}\u00b0")
        ax.legend(framealpha=0.9)
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_xlim(0, 90)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def write_summary(oe_hists, ft_hists, out_txt):
    """Write a summary table of mean energy and angle."""
    e_edges = np.linspace(0, EMAX, NBINS_E + 1)
    a_edges = np.linspace(0, AMAX, NBINS_A + 1)
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])

    lines = []
    lines.append(f"{'alpha':>6s}  {'source':>10s}  {'N':>8s}  {'<E> eV':>10s}  "
                 f"{'<theta> deg':>12s}  {'sigma_E':>10s}  {'sigma_th':>10s}")
    lines.append("-" * 75)

    for adeg in ANGLES:
        for label, hists in [("OpenEdge", oe_hists), ("Fortran", ft_hists)]:
            h = hists.get(adeg)
            if h is None:
                lines.append(f"{adeg:>6d}  {label:>10s}  {'---':>8s}")
                continue
            total = h.sum()
            if total == 0:
                lines.append(f"{adeg:>6d}  {label:>10s}  {0:>8d}")
                continue
            mean_e = np.sum(h * e_centers[None, :]) / total
            mean_a = np.sum(h * a_centers[:, None]) / total
            var_e  = np.sum(h * (e_centers[None, :] - mean_e)**2) / total
            var_a  = np.sum(h * (a_centers[:, None] - mean_a)**2) / total
            lines.append(f"{adeg:>6d}  {label:>10s}  {int(total):>8d}  "
                         f"{mean_e:>10.2f}  {mean_a:>12.2f}  "
                         f"{np.sqrt(var_e):>10.2f}  {np.sqrt(var_a):>10.2f}")

    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    print(f"\n{text}")
    print(f"\n  Saved {out_txt}")


def main():
    parser = argparse.ArgumentParser(description="Compare OpenEdge vs Fortran IEAD")
    parser.add_argument("--fortran-dir", type=str, default=".",
                        help="Directory containing Fortran iead_alpha*.csv files")
    parser.add_argument("--openedge-dir", type=str, default="output",
                        help="Directory containing OpenEdge vanish_alpha* log files")
    args = parser.parse_args()

    print("=" * 60)
    print("  IEAD comparison: OpenEdge vs sheath_tracker.f90")
    print("=" * 60)

    oe_hists = {}
    ft_hists = {}

    for adeg in ANGLES:
        # Read OpenEdge vanish CSV logs
        oe_prefix = os.path.join(args.openedge_dir, f"vanish_alpha{adeg}")
        en, ang = read_openedge_vanish_logs(oe_prefix)
        if en is not None:
            oe_hists[adeg] = bin_data(en, ang)
            print(f"  Loaded OpenEdge \u03b1={adeg}\u00b0: {len(en)} particles, "
                  f"<E>={en.mean():.1f} eV, <\u03b8>={ang.mean():.1f}\u00b0")
        else:
            print(f"  Missing OpenEdge \u03b1={adeg}\u00b0: {oe_prefix}.rank*")

        # Read Fortran
        ft_file = os.path.join(args.fortran_dir, f"iead_alpha{adeg}.csv")
        en, ang = read_fortran_csv(ft_file)
        if en is not None:
            ft_hists[adeg] = bin_data(en, ang)
            print(f"  Loaded Fortran  \u03b1={adeg}\u00b0: {len(en)} particles, "
                  f"<E>={en.mean():.1f} eV, <\u03b8>={ang.mean():.1f}\u00b0")
        else:
            print(f"  Missing Fortran \u03b1={adeg}\u00b0: {ft_file}")

    if not oe_hists and not ft_hists:
        print("\nNo data found. Run simulations first.")
        return

    os.makedirs("output", exist_ok=True)

    if oe_hists or ft_hists:
        plot_2d_comparison(oe_hists, ft_hists, "output/iead_comparison.png")
        plot_1d_energy(oe_hists, ft_hists, "output/iead_1d_energy.png")
        plot_1d_angle(oe_hists, ft_hists, "output/iead_1d_angle.png")
        write_summary(oe_hists, ft_hists, "output/iead_summary.txt")


if __name__ == "__main__":
    main()
