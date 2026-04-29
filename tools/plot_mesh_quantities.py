#!/usr/bin/env python3
"""Plot scalar quantities on the EIRENE mesh from plasma.h5.

Renders native triangles with Gouraud shading and filled-vacuum triangles
with flat shading, then writes one or more PNG pages under --out-prefix.
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.tri import Triangulation


def _parse_wall_surf(path: Path):
    """Return wall (R, Z) points + segments. Auto-detects axi (col1=Z, col2=R)
    vs cart (col1=R, col2=Z) by which column has any negative values:
    R is always positive on a tokamak wall; Z usually crosses 0."""
    pts_raw, segs, mode = [], [], None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "Points":
            mode = "P"
            continue
        if s == "Lines":
            mode = "L"
            continue
        parts = s.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        if mode == "P" and len(parts) >= 3:
            pts_raw.append((float(parts[1]), float(parts[2])))
        elif mode == "L" and len(parts) >= 3:
            segs.append((int(parts[1]) - 1, int(parts[2]) - 1))
    arr = np.asarray(pts_raw)
    col1_has_neg = (arr[:, 0] < 0).any()
    col2_has_neg = (arr[:, 1] < 0).any()
    if col1_has_neg and not col2_has_neg:
        # axi layout: col1=Z (negative allowed), col2=R (>0)
        pts = arr[:, [1, 0]]
    else:
        # cart layout: col1=R (>0), col2=Z
        pts = arr
    return pts, np.asarray(segs)


FIELD_SPECS = [
    ("dens_e", r"$n_e$", "m$^{-3}$", "viridis", "log"),
    ("temp_e", r"$T_e$", "eV", "inferno", "log"),
    ("dens_i", r"$n_i$", "m$^{-3}$", "viridis", "log"),
    ("temp_i", r"$T_i$", "eV", "inferno", "log"),
    ("parr_flow", r"$u_\parallel$", "m/s", "RdBu_r", "signed"),
    ("pob", r"$\phi$", "V", "RdBu_r", "signed"),
    ("e_par", r"$E_\parallel$", "V/m", "RdBu_r", "signed"),
    ("bmag", r"$|B|$", "T", "cividis", "linear"),
    ("bpol", r"$B_{pol}$", "T", "cividis", "linear"),
    ("btor", r"$B_{tor}$", "T", "RdBu_r", "signed"),
    ("e_r", r"$E_R$", "V/m", "RdBu_r", "signed"),
    ("e_z", r"$E_Z$", "V/m", "RdBu_r", "signed"),
    ("e_t", r"$E_\phi$", "V/m", "RdBu_r", "signed"),
    ("grad_te_r", r"$\partial_R T_e$", "eV/m", "RdBu_r", "signed"),
    ("grad_te_z", r"$\partial_Z T_e$", "eV/m", "RdBu_r", "signed"),
    ("grad_ti_r", r"$\partial_R T_i$", "eV/m", "RdBu_r", "signed"),
    ("grad_ti_z", r"$\partial_Z T_i$", "eV/m", "RdBu_r", "signed"),
    ("zeff", r"$Z_{eff}$", "", "magma", "linear"),
    ("volb", r"$V_{cell}$", "m$^3$", "plasma", "log"),
    ("rrb", r"$R_{cell}$", "m", "viridis", "linear"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plasma-h5", type=Path, required=True)
    ap.add_argument("--wall-surf", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--per-page", type=int, default=9)
    args = ap.parse_args()

    with h5py.File(args.plasma_h5) as f:
        mesh = f["mesh"]
        vr = mesh["vtx_r"][:]
        vz = mesh["vtx_z"][:]
        tri = mesh["triangles"][:]
        ci = mesh["cell_index"][:]
        tri_source_kind = mesh["tri_source_kind"][:] if "tri_source_kind" in mesh else None
        wall_gap_mask = mesh["wall_gap_mask"][:] if "wall_gap_mask" in mesh else None
        scalar_fields = {name for name, *_ in FIELD_SPECS} - {"bmag", "bpol", "btor"}
        field_data = {name: mesh[name][:] for name in scalar_fields if name in mesh}
        vtx_field_data = {
            name[4:]: mesh[name][:]
            for name in ("vtx_br", "vtx_bz", "vtx_bt", "vtx_bmag", "vtx_bpol", "vtx_btor")
            if name in mesh
        }

    ntri = len(tri)
    ncell = max((len(v) for v in field_data.values()), default=0)
    nvtx = len(vr)
    valid = (ci >= 0) & (ci < ncell)
    if tri_source_kind is None:
        tri_source_kind = np.zeros(ntri, dtype=np.int8)
        tri_source_kind[valid] = 1
    if wall_gap_mask is None:
        wall_gap_mask = tri_source_kind == 2
    native = tri_source_kind == 1
    filled = tri_source_kind == 2
    vacuum = ~valid

    v0 = np.column_stack([vr[tri[:, 0]], vz[tri[:, 0]]])
    v1 = np.column_stack([vr[tri[:, 1]], vz[tri[:, 1]]])
    v2 = np.column_stack([vr[tri[:, 2]], vz[tri[:, 2]]])
    tri_area = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))

    pts, segs = _parse_wall_surf(args.wall_surf)
    triang_proj = Triangulation(vr, vz, tri[np.flatnonzero(filled)]) if filled.any() else None
    tri_br = tri_bz = tri_bt = None
    if all(k in vtx_field_data for k in ("br", "bz", "bt")):
        tri_br = vtx_field_data["br"][tri].mean(axis=1)
        tri_bz = vtx_field_data["bz"][tri].mean(axis=1)
        tri_bt = vtx_field_data["bt"][tri].mean(axis=1)
        tri_bmag = np.sqrt(tri_br**2 + tri_bz**2 + tri_bt**2)
        tri_bmag = np.where(tri_bmag > 1e-30, tri_bmag, 1e-30)
        if all(k in field_data for k in ("e_r", "e_z", "e_t")):
            e_par = np.full(ntri, np.nan)
            e_r_t = np.full(ntri, np.nan)
            e_z_t = np.full(ntri, np.nan)
            e_t_t = np.full(ntri, np.nan)
            e_r_t[valid] = field_data["e_r"][ci[valid]]
            e_z_t[valid] = field_data["e_z"][ci[valid]]
            e_t_t[valid] = field_data["e_t"][ci[valid]]
            e_par[valid] = (e_r_t[valid] * tri_br[valid] +
                            e_z_t[valid] * tri_bz[valid] +
                            e_t_t[valid] * tri_bt[valid]) / tri_bmag[valid]
            field_data["e_par"] = e_par

    def to_tri(cell_vals):
        out = np.full(ntri, np.nan)
        out[valid] = cell_vals[ci[valid]]
        return out

    def to_vtx(tri_vals):
        good = np.isfinite(tri_vals) & native
        wsum = np.zeros(nvtx)
        vsum = np.zeros(nvtx)
        for k in range(3):
            idx = tri[good, k]
            np.add.at(wsum, idx, tri_area[good])
            np.add.at(vsum, idx, tri_area[good] * tri_vals[good])
        out = np.full(nvtx, np.nan)
        mask = wsum > 0
        out[mask] = vsum[mask] / wsum[mask]
        return out

    def build_norm(mode, native_v, filled_t):
        vals = np.concatenate([native_v[np.isfinite(native_v)], filled_t[np.isfinite(filled_t)]])
        if mode == "log":
            vals = vals[vals > 0]
            vmin = np.nanpercentile(vals, 1)
            vmax = np.nanpercentile(vals, 99)
            return LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin * 1.1))
        if mode == "signed":
            vmax = np.nanpercentile(np.abs(vals), 99)
            vmax = max(float(vmax), 1e-30)
            return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        vmin = np.nanpercentile(vals, 1)
        vmax = np.nanpercentile(vals, 99)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        return Normalize(vmin=float(vmin), vmax=float(vmax))

    def draw_tri_field(ax, vals_t, vals_v, title, units, cmap, mode):
        norm = build_norm(mode, vals_v, vals_t[filled])
        tri_n = Triangulation(vr, vz, tri)
        vert_ok = np.isfinite(vals_v)
        tri_n.set_mask(~native | ~np.all(vert_ok[tri], axis=1))
        base = vals_v.copy()
        finite = np.isfinite(base)
        base[~finite] = np.nanmin(base[finite])
        ax.tripcolor(tri_n, base, shading="gouraud", cmap=cmap, norm=norm)
        if triang_proj is not None:
            ax.tripcolor(triang_proj, facecolors=vals_t[filled], shading="flat",
                         cmap=cmap, norm=norm, edgecolors="none")
        ax.add_collection(LineCollection(
            [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs],
            colors="black", lw=0.4, zorder=10))
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        if units:
            cb.set_label(units, fontsize=8)

    def draw_field(ax, name, title, units, cmap, mode):
        if name == "e_par" and name in field_data:
            vals_t = np.asarray(field_data[name], dtype=np.float64)
            vals_v = to_vtx(vals_t)
            draw_tri_field(ax, vals_t, vals_v, title, units, cmap, mode)
            return

        if name in vtx_field_data:
            vals_v = np.asarray(vtx_field_data[name], dtype=np.float64)
            finite = vals_v[np.isfinite(vals_v)]
            if mode == "log":
                finite = finite[finite > 0]
                norm = LogNorm(vmin=max(np.nanpercentile(finite, 1), 1e-30),
                               vmax=max(np.nanpercentile(finite, 99), 1e-30))
            elif mode == "signed":
                vmax = max(float(np.nanpercentile(np.abs(finite), 99)), 1e-30)
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            else:
                vmin = float(np.nanpercentile(finite, 1))
                vmax = float(np.nanpercentile(finite, 99))
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
            tri_all = Triangulation(vr, vz, tri)
            base = vals_v.copy()
            base[~np.isfinite(base)] = np.nanmin(finite)
            ax.tripcolor(tri_all, base, shading="gouraud", cmap=cmap, norm=norm)
            ax.add_collection(LineCollection(
                [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs],
                colors="black", lw=0.4, zorder=10))
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
            if units:
                cb.set_label(units, fontsize=8)
            return

        vals_t = to_tri(field_data[name])
        vals_v = to_vtx(vals_t)
        if not np.isfinite(vals_v).any():
            ax.set_axis_off()
            return
        draw_tri_field(ax, vals_t, vals_v, title, units, cmap, mode)

    def draw_mask(ax):
        class_vals = np.full(ntri, np.nan, dtype=float)
        class_vals[vacuum] = 0.0
        class_vals[native] = 1.0
        class_vals[filled] = 2.0
        cmap = ListedColormap(["white", "#4477aa", "#66c2a5"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        tri_cls = Triangulation(vr, vz, tri)
        ax.tripcolor(tri_cls, facecolors=class_vals, shading="flat", cmap=cmap, norm=norm)
        ax.add_collection(LineCollection(
            [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs],
            colors="black", lw=0.4, zorder=10))
        ax.set_title("Provenance", fontsize=10)
        ax.set_aspect("equal")

    items = [(name, title, units, cmap, mode) for name, title, units, cmap, mode in FIELD_SPECS if name in field_data]
    items.append(("__mask__", "Provenance", "", "", ""))
    per_page = max(1, args.per_page)
    n_pages = ceil(len(items) / per_page)
    ncols = 3
    nrows = ceil(per_page / ncols)

    for page in range(n_pages):
        subset = items[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, item in zip(axes, subset):
            if item[0] == "__mask__":
                draw_mask(ax)
            else:
                draw_field(ax, *item)
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
        for ax in axes[len(subset):]:
            ax.set_axis_off()
        fig.suptitle(
            f"{args.plasma_h5.name} | native {int(native.sum())} | filled {int(filled.sum())} | vacuum {int(vacuum.sum())}",
            y=0.995
        )
        fig.tight_layout()
        out = args.out_prefix.parent / f"{args.out_prefix.name}_p{page + 1}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
