#!/usr/bin/env python3
"""Quick-look plotter for plasma.h5 (SOLPS-derived OpenEdge background).

Usage:
    python3 plot_plasma.py [path_to_plasma.h5]

Renders Te, Ti, ne, ni, |E|, |B|, and psi_norm on the unstructured SOLPS mesh
(plus psi from the equilibrium grid). One PNG per panel.
"""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

H5 = sys.argv[1] if len(sys.argv) > 1 else "plasma.h5"

with h5py.File(H5, "r") as f:
    vr = f["/mesh/vtx_r"][:]
    vz = f["/mesh/vtx_z"][:]
    tris = f["/mesh/triangles"][:]
    cell_idx = f["/mesh/cell_index"][:]
    te = f["/mesh/temp_e"][:][cell_idx]
    ti = f["/mesh/temp_i"][:][cell_idx]
    ne = f["/mesh/dens_e"][:][cell_idx]
    ni = f["/mesh/dens_i"][:][cell_idx]
    er = f["/mesh/e_r"][:][cell_idx]
    ez = f["/mesh/e_z"][:][cell_idx]
    et = f["/mesh/e_t"][:][cell_idx]
    bmag = f["/mesh/vtx_bmag"][:]
    if "/equilibrium/psi" in f:
        eq_r = f["/equilibrium/r"][:]
        eq_z = f["/equilibrium/z"][:]
        psi = f["/equilibrium/psi"][:]
        psib = float(f["/equilibrium/psib"][()])
    else:
        eq_r = eq_z = psi = None

triang = mtri.Triangulation(vr, vz, tris)
emag = np.sqrt(er * er + ez * ez + et * et)


def panel(name, val, log=False, label=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5.5, 9))
    if log:
        v = np.where(val > 0, val, np.nan)
        norm = LogNorm(vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99))
        im = ax.tripcolor(triang, val, shading="flat", cmap=cmap, norm=norm)
    else:
        im = ax.tripcolor(triang, val, shading="flat", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]", fontsize=14)
    ax.set_ylabel(r"$Z$ [m]", fontsize=14)
    ax.set_title(label or name, fontsize=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.ax.tick_params(labelsize=12)
    out = f"output/plasma_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


print(f"Reading {H5}: {len(tris)} tris, {len(vr)} vtx, {len(te)} per-tri values")
panel("Te", te, log=True, label=r"$T_e$ [eV]")
panel("Ti", ti, log=True, label=r"$T_i$ [eV]")
panel("ne", ne, log=True, label=r"$n_e$ [m$^{-3}$]")
panel("ni", ni, log=True, label=r"$n_i$ [m$^{-3}$]")
panel("Emag", emag, log=False, label=r"$|E|$ [V/m]", cmap="magma")
# Bmag is per-vertex; tripcolor with shading='gouraud' interpolates linearly
fig, ax = plt.subplots(figsize=(5.5, 9))
im = ax.tripcolor(triang, bmag, shading="gouraud", cmap="plasma")
ax.set_aspect("equal")
ax.set_xlabel(r"$R$ [m]", fontsize=14)
ax.set_ylabel(r"$Z$ [m]", fontsize=14)
ax.set_title(r"$|B|$ [T]", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.7)
cb.ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig("output/plasma_Bmag.png", dpi=150)
plt.close(fig)
print("  -> output/plasma_Bmag.png")

if psi is not None:
    psi_axis = float(psi.min())
    psin = (psi - psi_axis) / (psib - psi_axis) if abs(psib - psi_axis) > 1e-30 else psi
    fig, ax = plt.subplots(figsize=(5.5, 9))
    cs = ax.contourf(eq_r, eq_z, psin, levels=30, cmap="RdBu_r")
    ax.contour(eq_r, eq_z, psin, levels=[0.85, 1.0], colors=["k", "w"], linewidths=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]", fontsize=14)
    ax.set_ylabel(r"$Z$ [m]", fontsize=14)
    ax.set_title(r"$\psi_{\rm norm}$  (black=0.85, white=LCFS)", fontsize=14)
    fig.colorbar(cs, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig("output/plasma_psinorm.png", dpi=150)
    plt.close(fig)
    print("  -> output/plasma_psinorm.png")

print("done")
