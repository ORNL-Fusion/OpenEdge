#!/usr/bin/env python3
"""Focused ne + Te plot with native/extrapolated triangle classification.

Shows three things for each of n_e and T_e:
  1. Values on native EIRENE triangles (fort.35 -> B2 cell direct lookup)
  2. Values on projected triangles (within 5 cm of a sheath-edge cell, B2.5/EIRENE-style)
  3. Wall polygon outline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "converters"))
from convert_solps_plasma import b2f_read_dims  # noqa: E402


def _parse_wall_surf(path: Path):
    pts, segs, mode = [], [], None
    for L in path.read_text().splitlines():
        s = L.strip()
        if not s:
            continue
        if s == "Points":
            mode = "P"; continue
        if s == "Lines":
            mode = "L"; continue
        p = s.split()
        if not p or not p[0].lstrip("-").isdigit():
            continue
        if mode == "P" and len(p) >= 3:
            pts.append((float(p[2]), float(p[1])))
        elif mode == "L" and len(p) >= 3:
            segs.append((int(p[1]) - 1, int(p[2]) - 1))
    return np.array(pts), np.array(segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plasma-h5", type=Path, required=True)
    ap.add_argument("--wall-surf", type=Path, required=True)
    ap.add_argument("--fort35",
                    default="/home/cloud/solps-runs/diii-d/runners_d2/run_lore2023_reference/fort.35",
                    type=Path,
                    help="fort.35 to distinguish native vs projected triangles.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with h5py.File(args.plasma_h5) as f:
        vr = f["mesh/vtx_r"][:]
        vz = f["mesh/vtx_z"][:]
        tri = f["mesh/triangles"][:]
        ci = f["mesh/cell_index"][:]
        ne = f["mesh/dens_e"][:]
        te = f["mesh/temp_e"][:]
    ncell = len(ne); ntri = len(tri); nvtx = len(vr)

    # Triangle classification: native = fort.35 has positive ix+iy
    native_ixiy = np.zeros(ntri, dtype=bool)
    with args.fort35.open() as ff:
        _ = int(ff.readline())
        for i, line in enumerate(ff):
            if i >= ntri:
                break
            p = line.split()
            if len(p) >= 12 and int(p[10]) > 0 and int(p[11]) > 0:
                native_ixiy[i] = True
    # Triangles after ntri_native are our added mesh-extension triangles
    # (and anything else our converter appended). For simplicity classify:
    #   native: fort.35 mapped
    #   projected: not native but cell_index >= 0 (came from sheath projection)
    #   vacuum: cell_index < 0 (unmapped)
    valid = (ci >= 0) & (ci < ncell)
    native = np.zeros(ntri, dtype=bool)
    native[:len(native_ixiy)] = native_ixiy
    native &= valid   # must also have a valid cell
    projected = valid & ~native
    vacuum = ~valid
    print(f"triangles: {ntri} total")
    print(f"  native (fort.35 mapped)    : {int(native.sum())}")
    print(f"  projected (sheath nearest) : {int(projected.sum())}")
    print(f"  vacuum (unmapped)          : {int(vacuum.sum())}")

    # Build per-triangle values
    ne_t = np.full(ntri, np.nan)
    te_t = np.full(ntri, np.nan)
    ne_t[valid] = ne[ci[valid]]
    te_t[valid] = te[ci[valid]]

    # Two-layer render to avoid Gouraud smear at the native<->projected seam:
    #   layer 1: Gouraud-shade NATIVE triangles only (smooth SOL interior).
    #            Vertex averaging uses native triangles *only*, so shared
    #            vertices between a hot SOL tri and a cold projected tri
    #            no longer inherit the hot value at all.
    #   layer 2: overlay projected triangles with flat shading (each tri
    #            painted its exact cell_index value -- no interpolation).
    v0 = np.column_stack([vr[tri[:, 0]], vz[tri[:, 0]]])
    v1 = np.column_stack([vr[tri[:, 1]], vz[tri[:, 1]]])
    v2 = np.column_stack([vr[tri[:, 2]], vz[tri[:, 2]]])
    tri_area = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))

    def to_vtx(tri_vals, mask):
        good = np.isfinite(tri_vals) & mask
        wsum = np.zeros(nvtx); vsum = np.zeros(nvtx)
        for k in range(3):
            idx = tri[good, k]
            w = tri_area[good]
            np.add.at(wsum, idx, w)
            np.add.at(vsum, idx, w * tri_vals[good])
        out = np.full(nvtx, np.nan)
        nz = wsum > 0
        out[nz] = vsum[nz] / wsum[nz]
        return out

    # Vertex values from NATIVE only -> no bleed into the projected band.
    ne_v = to_vtx(ne_t, native)
    te_v = to_vtx(te_t, native)

    triang_native = Triangulation(vr, vz, tri)
    # Gouraud layer: draw only native tris, and drop any that touch an
    # un-averaged vertex (lonely native tris with projected-only neighbors).
    vert_ok_ne = np.isfinite(ne_v)
    vert_ok_te = np.isfinite(te_v)
    mask_native_ne = ~native | ~np.all(vert_ok_ne[tri], axis=1)
    mask_native_te = ~native | ~np.all(vert_ok_te[tri], axis=1)

    # Flat overlay uses its own triangulation containing only projected tris.
    proj_idx = np.flatnonzero(projected)
    triang_proj = Triangulation(vr, vz, tri[proj_idx])

    # Wall
    pts, segs = _parse_wall_surf(args.wall_surf)

    def _draw(ax, vals_v, vals_t_proj, title, cmap, log=False, units=""):
        finite_v = vals_v[np.isfinite(vals_v)]
        finite_p = vals_t_proj[np.isfinite(vals_t_proj)]
        stacked = np.concatenate([finite_v, finite_p])
        if log:
            stacked = stacked[stacked > 0]
        vmin = np.nanpercentile(stacked, 1)
        vmax = np.nanpercentile(stacked, 99)
        if log:
            norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin * 1.1))
        else:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Layer 1: native tris, Gouraud
        vv = vals_v.copy(); vv[~np.isfinite(vv)] = vmin
        tri_n = Triangulation(vr, vz, tri)
        tri_n.set_mask(mask_native_ne if units == "m$^{-3}$" else mask_native_te)
        ax.tripcolor(tri_n, vv, shading="gouraud", cmap=cmap, norm=norm)

        # Layer 2: projected tris, flat (one color per triangle)
        if len(proj_idx):
            ax.tripcolor(triang_proj, facecolors=vals_t_proj,
                         shading="flat", cmap=cmap, norm=norm,
                         edgecolors="none")

        ax.add_collection(LineCollection(
            [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs],
            colors="black", lw=0.5, zorder=10))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, pad=0.02).set_label(units, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    ne_proj = ne_t[proj_idx]
    te_proj = te_t[proj_idx]
    _draw(axes[0], ne_v, ne_proj, r"$n_e$ (native Gouraud + projected flat)",
          "viridis", log=True, units="m$^{-3}$")
    _draw(axes[1], te_v, te_proj, r"$T_e$ (native Gouraud + projected flat, log)",
          "inferno", log=True, units="eV")
    for ax in axes:
        ax.set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    plt.suptitle(
        f"{args.plasma_h5.name}   |   "
        f"native {int(native.sum())}  projected {int(projected.sum())}  "
        f"vacuum {int(vacuum.sum())}  (of {ntri})",
        y=0.995)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
