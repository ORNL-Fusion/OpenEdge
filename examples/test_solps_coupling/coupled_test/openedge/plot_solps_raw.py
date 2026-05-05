#!/usr/bin/env python3
"""Plot raw SOLPS B2 quad-mesh data from a SOLPS run dir, no interpolation.

Usage:
    python3 plot_solps_raw.py /home/cloud/li/v0_1
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path("/home/cloud/OpenEdge/tools/converters")))
from convert_solps_plasma import b2f_read_dims, b2f_extract, read_b2fgmtry, _cell_polygons_from_corners
from read_fort31 import read_fort31

run = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/cloud/li/v0_1")
gmtry = run / "b2fgmtry"
fort31 = run / "fort.31"
nx, ny, ns_geom = b2f_read_dims(gmtry)
print(f"b2fgmtry: nx={nx} ny={ny} ns={ns_geom}")

geom = read_b2fgmtry(gmtry)
crx = geom["crx"].astype(float)
cry = geom["cry"].astype(float)
polys = _cell_polygons_from_corners(crx, cry, nx, ny)   # (Nc, 4, 2)
Nc = polys.shape[0]
print(f"  {Nc} cells (incl. 2 ghost rings)")

# Determine ns_plasma from fort.31 (count blocks until success)
# Common WEST cases: ns_plasma = 5..10. Try a few.
ft31 = None
for ns_try in (geom.get("ns", [None])[0], 14, 12, 10, 8, 6, 5, 4, 3, 2):
    if ns_try is None:
        continue
    try:
        ft31 = read_fort31(fort31, nx=nx, ny=ny, ns_plasma=int(ns_try))
        if not ft31.truncated:
            print(f"  fort.31 OK with ns_plasma={ns_try}")
            break
    except Exception:
        continue
if ft31 is None:
    raise SystemExit("could not parse fort.31")

# Plasma fields, shape (nx+2, ny+2). teb/tib already in eV from read_fort31.
te = ft31.teb.reshape(-1, order="F")
ti = ft31.tib.reshape(-1, order="F")
ne = ft31.dnib[:, :, 0].reshape(-1, order="F") if ft31.dnib.ndim == 3 else ft31.dnib.reshape(-1, order="F")
# ni: total ion density (sum over species of dnib if 3D)
if ft31.dnib.ndim == 3:
    ni = ft31.dnib.sum(axis=2).reshape(-1, order="F")
else:
    ni = ne.copy()


def panel(name, val, log=True, label=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6.5, 9.5))
    if log:
        v = np.where(val > 0, val, np.nan)
        norm = LogNorm(vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99))
    else:
        norm = None
    pc = PolyCollection(polys, array=val, cmap=cmap, norm=norm, edgecolors="none")
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]", fontsize=14)
    ax.set_ylabel(r"$Z$ [m]", fontsize=14)
    ax.set_title(label or name, fontsize=14)
    cb = fig.colorbar(pc, ax=ax, shrink=0.7)
    cb.ax.tick_params(labelsize=12)
    out = f"output/solps_raw_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


print(f"Plotting raw SOLPS B2 fields:")
panel("Te", te, log=True, label=r"$T_e$ [eV] (raw SOLPS)")
panel("Ti", ti, log=True, label=r"$T_i$ [eV] (raw SOLPS)")
panel("ne", ne, log=True, label=r"$n_e$ [m$^{-3}$] (raw SOLPS)")
panel("ni_total", ni, log=True, label=r"$\sum n_i$ [m$^{-3}$] (raw SOLPS)")

# also overlay the SOLPS mesh edge for orientation
fig, ax = plt.subplots(figsize=(6.5, 9.5))
pc = PolyCollection(polys, facecolors="lightgrey", edgecolors="k", linewidths=0.1)
ax.add_collection(pc)
ax.autoscale_view()
ax.set_aspect("equal")
ax.set_xlabel(r"$R$ [m]", fontsize=14)
ax.set_ylabel(r"$Z$ [m]", fontsize=14)
ax.set_title("SOLPS B2 mesh footprint", fontsize=14)
fig.tight_layout()
fig.savefig("output/solps_raw_mesh.png", dpi=150)
plt.close(fig)
print("  -> output/solps_raw_mesh.png")
print("done")
