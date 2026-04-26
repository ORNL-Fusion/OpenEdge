#!/usr/bin/env python3
"""
Side-by-side validation of fort.31 vs our b2fstate-derived /mesh/ plasma.

Reads plasma.h5 (which now carries both the legacy /mesh/ datasets AND
the full SOLPS-ITER /fort31/ group) and plots per-B2-cell values from
each source. If the two agree, we can safely switch /mesh/* consumers to
read /fort31/* instead.

Fields compared:
    n_e    : fort.31 dnib[:,:,0]  vs  /mesh/dens_e  (indexed by cell_index)
    T_e    : fort.31 teb          vs  /mesh/temp_e
    T_i    : fort.31 tib          vs  /mesh/temp_i
    |B|    : fort.31 bfeldb       vs  reconstructed from /equilibrium/psi
    phi    : fort.31 pob          vs  no direct /mesh/ equivalent

Usage:
    python3 tools/compare_fort31_vs_mesh.py \
        --plasma-h5 examples/test_diii_d_neutrals/input/plasma.h5 \
        --b2fgmtry /home/cloud/solps-runs/diii-d/runners_d2/baserun/b2fgmtry \
        --out examples/test_diii_d_neutrals/output/fort31_vs_mesh.png
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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PolyCollection

# Reuse the b2fgmtry reader
_TOOLS = Path(__file__).resolve().parent / "converters"
sys.path.insert(0, str(_TOOLS))
from convert_solps_plasma import b2f_read_dims, read_b2fgmtry  # noqa: E402


def _cell_polys(crx4, cry4, nx, ny):
    """Build polygons (nx+2, ny+2, 4, 2) per-cell from corners, corner
    order [0,1,3,2] so the quad doesn't self-intersect."""
    order = [0, 1, 3, 2]
    polys = np.stack(
        (crx4[:, :, order], cry4[:, :, order]), axis=-1
    )  # (nx+2, ny+2, 4, 2)
    return polys


def _panel(ax, polys_flat, values_flat, title, cmap, log=False, units="",
           sym=False):
    """Draw a PolyCollection of per-B2-cell polygons colored by values."""
    good = np.isfinite(values_flat)
    v = values_flat.copy()
    if log:
        v = np.where(v > 0, v, np.nan)
        good_pos = np.isfinite(v)
        if not good_pos.any():
            ax.set_title(f"{title}\n(no positive values)")
            return
        vmin = np.nanpercentile(v, 1)
        vmax = np.nanpercentile(v, 99)
        norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin * 1.1))
    elif sym:
        vmax = np.nanpercentile(np.abs(v[good]), 99)
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        vmin = np.nanpercentile(v[good], 1)
        vmax = np.nanpercentile(v[good], 99)
        norm = Normalize(vmin=vmin, vmax=vmax)
    pc = PolyCollection(polys_flat, array=v, cmap=cmap, norm=norm,
                        edgecolors="none")
    ax.add_collection(pc)
    ax.autoscale_view()
    cbar = plt.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(units, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def _reconstruct_bmag_on_b2(psi, r_eq, z_eq, btf, rtf, rc, zc):
    """Eval |B| at B2 cell centers (rc, zc) from equilibrium psi."""
    if psi.shape == (len(r_eq), len(z_eq)):
        psi_rz = psi
    else:
        psi_rz = psi.T
    dpsi_dr, dpsi_dz = np.gradient(psi_rz, r_eq, z_eq)
    ir = np.clip(np.searchsorted(r_eq, rc), 0, len(r_eq) - 1)
    iz = np.clip(np.searchsorted(z_eq, zc), 0, len(z_eq) - 1)
    rsafe = np.maximum(rc, 1e-6)
    br = -dpsi_dz[ir, iz] / rsafe
    bz = dpsi_dr[ir, iz] / rsafe
    bt = btf * rtf / rsafe
    return np.sqrt(br * br + bz * bz + bt * bt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plasma-h5", type=Path, required=True)
    ap.add_argument("--b2fgmtry", type=Path, required=True,
                    help="SOLPS b2fgmtry for cell-corner coordinates.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Geometry
    nx, ny, _ = b2f_read_dims(args.b2fgmtry)
    gmtry = read_b2fgmtry(str(args.b2fgmtry))
    crx4 = gmtry["crx"].reshape(nx + 2, ny + 2, 4, order="F")
    cry4 = gmtry["cry"].reshape(nx + 2, ny + 2, 4, order="F")
    rc = crx4.mean(axis=2)   # (nx+2, ny+2) cell centroids
    zc = cry4.mean(axis=2)

    polys = _cell_polys(crx4, cry4, nx, ny)   # (nx+2, ny+2, 4, 2)
    # Match the converter's F-order flattening: flat_idx = iy*(nx+2)+ix.
    # Transpose to (ny+2, nx+2, 4, 2) then C-flatten achieves that.
    polys_flat = np.transpose(polys, (1, 0, 2, 3)).reshape(-1, 4, 2)
    rc_flat = rc.reshape(-1, order="F")
    zc_flat = zc.reshape(-1, order="F")

    # Load plasma.h5
    with h5py.File(args.plasma_h5, "r") as f:
        if "fort31" not in f:
            print("ERROR: /fort31 group missing from plasma.h5 — regenerate with updated converter.")
            sys.exit(1)
        ft = f["fort31"]
        ft_ne = np.asarray(ft["dnib"])[:, :, 0]
        ft_te = np.asarray(ft["teb"])
        ft_ti = np.asarray(ft["tib"])
        ft_bmag = np.asarray(ft["bfeldb"])
        ft_pob = np.asarray(ft["pob"])
        # /mesh/* indexed by cell_index
        cell_index = np.asarray(f["mesh/cell_index"])
        m_ne_by_cell = np.asarray(f["mesh/dens_e"])
        m_te_by_cell = np.asarray(f["mesh/temp_e"])
        m_ti_by_cell = np.asarray(f["mesh/temp_i"])
        # Equilibrium for |B| reconstruction
        psi = np.asarray(f["equilibrium/psi"])
        r_eq = np.asarray(f["equilibrium/r"])
        z_eq = np.asarray(f["equilibrium/z"])
        btf = float(f["equilibrium/btf"][()])
        rtf = float(f["equilibrium/rtf"][()])

    # Flatten (nx+2, ny+2) in Fortran order for alignment with /mesh/
    # cell_index (which was built as iy*(nx+2) + ix, i.e. F-order).
    ft_ne_f = ft_ne.reshape(-1, order="F")
    ft_te_f = ft_te.reshape(-1, order="F")
    ft_ti_f = ft_ti.reshape(-1, order="F")
    ft_bmag_f = ft_bmag.reshape(-1, order="F")
    ft_pob_f = ft_pob.reshape(-1, order="F")

    # For each B2 cell, pull the matching /mesh/ value. /mesh/dens_e is
    # indexed by cell_index, which is a PER-TRIANGLE flat B2 cell
    # pointer — but we want the reverse: for each B2 cell (flat_idx),
    # what value does /mesh/ store? Since cell_index maps triangles ->
    # B2 cells, the answer is: /mesh/dens_e[flat_idx_in_array] IS the
    # B2 cell's value (it's directly per-B2-cell, the naming is
    # misleading). So a direct lookup suffices.
    ncell = (nx + 2) * (ny + 2)
    # m_ne_by_cell etc. have length ncell already (our converter stored
    # the per-B2-cell plasma flat-indexed the same way). Just use them.
    if len(m_ne_by_cell) != ncell:
        print(f"WARNING: mesh/dens_e length ({len(m_ne_by_cell)}) != ncell ({ncell}); "
              f"falling back to cell_index gather.")
        mesh_ne_f = np.zeros(ncell)
        mesh_te_f = np.zeros(ncell)
        mesh_ti_f = np.zeros(ncell)
        valid = (cell_index >= 0) & (cell_index < len(m_ne_by_cell))
        # last-writer wins; all tris sharing a cell produce same value
        mesh_ne_f[cell_index[valid]] = m_ne_by_cell[cell_index[valid]]
        mesh_te_f[cell_index[valid]] = m_te_by_cell[cell_index[valid]]
        mesh_ti_f[cell_index[valid]] = m_ti_by_cell[cell_index[valid]]
    else:
        mesh_ne_f = m_ne_by_cell
        mesh_te_f = m_te_by_cell
        mesh_ti_f = m_ti_by_cell

    # |B| on the B2 grid reconstructed from psi (same code path our
    # sanity plotter uses for per-triangle B).
    bmag_psi = _reconstruct_bmag_on_b2(psi, r_eq, z_eq, btf, rtf, rc, zc)
    bmag_psi_f = bmag_psi.reshape(-1, order="F")

    # Filter out the outer ghost ring so we only compare real B2 cells.
    # Ghost cells are iy=0, iy=ny+1, ix=0, ix=nx+1. Construct (ix, iy)
    # grids as (nx+2, ny+2) then F-flatten to match the iy*(nx+2)+ix
    # convention the converter uses.
    ix_grid, iy_grid = np.meshgrid(np.arange(nx + 2), np.arange(ny + 2),
                                    indexing="ij")
    ix_flat = ix_grid.reshape(-1, order="F")
    iy_flat = iy_grid.reshape(-1, order="F")
    ghost = (ix_flat == 0) | (ix_flat == nx + 1) | \
            (iy_flat == 0) | (iy_flat == ny + 1)

    def _rms_rel(ref, chk):
        good = ~ghost & np.isfinite(ref) & np.isfinite(chk) & (np.abs(ref) > 0)
        if not good.any():
            return float("nan")
        return float(np.sqrt(np.mean(((chk[good] - ref[good]) /
                                        ref[good]) ** 2)))

    print(f"Per-B2-cell comparison (excluding ghost ring):")
    print(f"  ne   : RMS rel = {_rms_rel(ft_ne_f, mesh_ne_f):.3e}")
    print(f"  Te   : RMS rel = {_rms_rel(ft_te_f, mesh_te_f):.3e}")
    print(f"  Ti   : RMS rel = {_rms_rel(ft_ti_f, mesh_ti_f):.3e}")
    print(f"  |B|  : RMS rel = {_rms_rel(ft_bmag_f, bmag_psi_f):.3e}")

    # Mask ghost ring for plotting too (polys stay, just paint NaN there
    # so PolyCollection leaves them uncolored).
    def _mask_ghost(arr):
        out = arr.astype(float).copy()
        out[ghost] = np.nan
        return out

    # ---------- plot: 2 rows × 3 cols, left-to-right rows show ne, Te, |B|
    # each row has (fort.31) | (current mesh or psi) | (diff)
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharex=True, sharey=True)

    # Row 0: ne
    _panel(axes[0, 0], polys_flat, _mask_ghost(ft_ne_f),
           r"fort.31  $n_e$ (dnib)", "viridis", log=True, units="m$^{-3}$")
    _panel(axes[0, 1], polys_flat, _mask_ghost(mesh_ne_f),
           r"mesh  $n_e$ (dens_e)", "viridis", log=True, units="m$^{-3}$")
    _panel(axes[0, 2], polys_flat,
           _mask_ghost((mesh_ne_f - ft_ne_f) / np.where(np.abs(ft_ne_f) > 0, ft_ne_f, np.nan)),
           r"$(mesh - fort31)/fort31$  $n_e$", "RdBu_r", sym=True, units="rel")

    # Row 1: Te
    _panel(axes[1, 0], polys_flat, _mask_ghost(ft_te_f),
           r"fort.31  $T_e$ (teb)", "inferno", units="eV")
    _panel(axes[1, 1], polys_flat, _mask_ghost(mesh_te_f),
           r"mesh  $T_e$ (temp_e)", "inferno", units="eV")
    _panel(axes[1, 2], polys_flat,
           _mask_ghost((mesh_te_f - ft_te_f) / np.where(np.abs(ft_te_f) > 0, ft_te_f, np.nan)),
           r"$(mesh - fort31)/fort31$  $T_e$", "RdBu_r", sym=True, units="rel")

    # Row 2: Ti
    _panel(axes[2, 0], polys_flat, _mask_ghost(ft_ti_f),
           r"fort.31  $T_i$ (tib)", "inferno", units="eV")
    _panel(axes[2, 1], polys_flat, _mask_ghost(mesh_ti_f),
           r"mesh  $T_i$ (temp_i)", "inferno", units="eV")
    _panel(axes[2, 2], polys_flat,
           _mask_ghost((mesh_ti_f - ft_ti_f) / np.where(np.abs(ft_ti_f) > 0, ft_ti_f, np.nan)),
           r"$(mesh - fort31)/fort31$  $T_i$", "RdBu_r", sym=True, units="rel")

    # Row 3: |B| fort31 vs psi-reconstructed
    _panel(axes[3, 0], polys_flat, _mask_ghost(ft_bmag_f),
           r"fort.31  $|B|$ (bfeldb)", "plasma", units="T")
    _panel(axes[3, 1], polys_flat, _mask_ghost(bmag_psi_f),
           r"$\psi$-reconstruction  $|B|$", "plasma", units="T")
    _panel(axes[3, 2], polys_flat,
           _mask_ghost((bmag_psi_f - ft_bmag_f) / np.where(np.abs(ft_bmag_f) > 0, ft_bmag_f, np.nan)),
           r"$(\psi - fort31)/fort31$  $|B|$", "RdBu_r", sym=True, units="rel")

    for ax in axes.flat:
        ax.set_xlim(rc.min() - 0.05, rc.max() + 0.05)
        ax.set_ylim(zc.min() - 0.05, zc.max() + 0.05)
    for ax in axes[-1]:
        ax.set_xlabel("R [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Z [m]")

    plt.suptitle(f"fort.31 vs current mesh — {args.plasma_h5.name}", y=0.995)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
