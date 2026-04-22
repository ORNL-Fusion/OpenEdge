#!/usr/bin/env python3
"""Plot neutral densities and plasma-frame source moments from a
`fix chem/adas` summary-mode dump.

Reads the multi-snapshot text dump produced by `in.diii_d_neutrals`
(15 cols:  id xc yc xlo ylo xhi yhi  nD nD2  Sp Smx Smy Smz Qe Qi),
takes the last snapshot (steady-state), and produces two figures:

  neutral_densities.png  -- D and D2 density on the native adaptive grid
                            (PolyCollection of cell quads, no smoothing).
  neutral_sources.png    -- Sp particle, Qe electron energy, Qi ion energy
                            source densities, regridded to a regular
                            (R, Z) mesh and Gaussian-smoothed.

Native SPARTA cell bounds (xlo, ylo, xhi, yhi) are overlaid as thin
gray lines on every panel so you can see which parts of the grid were
refined by `fix adapt`.

Usage:
  python3 tools/make_neutral_plots.py \
     --dump    examples/test_diii_d_neutrals/output/diii_d_neutrals.grid \
     --wall    examples/test_diii_d_neutrals/input/wall.surf \
     --out-dir examples/test_diii_d_neutrals/plots

Axi coordinate convention in this deck: SPARTA x = Z (axial),
SPARTA y = R (radial). All plots are drawn with R on the horizontal
axis and Z on the vertical axis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import AsinhNorm, LogNorm, SymLogNorm
from matplotlib.path import Path as MplPath
from matplotlib.tri import Triangulation
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

import h5py


# Dump schema (1-based col indices map to 0-based array slices)
COL_ID  = 0
COL_XC  = 1   # SPARTA x = Z
COL_YC  = 2   # SPARTA y = R
COL_XLO = 3
COL_YLO = 4
COL_XHI = 5
COL_YHI = 6
COL_ND  = 7
COL_ND2 = 8
COL_SP  = 9
COL_SMX = 10
COL_SMY = 11
COL_SMZ = 12
COL_QE  = 13
COL_QI  = 14
NCOLS   = 15


def read_last_snapshot(path: Path) -> np.ndarray:
    """Stream the text dump and return the last snapshot's numeric block."""
    lines = path.read_text().splitlines()
    # Each snapshot is: ITEM: TIMESTEP / <step> / ITEM: NUMBER OF CELLS /
    # <ncell> / ITEM: BOX BOUNDS ... / 3 bound lines / ITEM: CELLS ... /
    # <ncell> data rows.
    starts = [i for i, L in enumerate(lines) if L.startswith("ITEM: TIMESTEP")]
    if not starts:
        raise RuntimeError(f"{path}: no snapshots found")
    last = starts[-1]
    step = int(lines[last + 1])
    ncell = int(lines[last + 3])
    # Find the CELLS header after `last`
    for i in range(last, len(lines)):
        if lines[i].startswith("ITEM: CELLS"):
            data_start = i + 1
            break
    else:
        raise RuntimeError(f"{path}: no ITEM: CELLS header in last snapshot")
    rows = lines[data_start:data_start + ncell]
    if len(rows) < ncell:
        raise RuntimeError(f"{path}: truncated ({len(rows)}/{ncell} rows)")
    arr = np.fromstring(" ".join(rows), sep=" ").reshape(ncell, NCOLS)
    print(f"  read step={step:>7}  ncell={ncell}", file=sys.stderr)
    return arr


def parse_wall_surf(path: Path):
    """Parse SPARTA wall.surf and return (line_segments, points_RZ, poly_RZ).

    `line_segments` is a list of 2-point tuples for LineCollection.
    `points_RZ` is the raw (N, 2) points in file order.
    `poly_RZ` is the polygon-ordered walk (vertices in adjacency order so
    matplotlib.Path.contains_points gives a correct inside/outside test);
    falls back to `points_RZ` if the segment graph doesn't form a single
    closed loop.
    """
    pts, segs, mode = [], [], None
    for L in path.read_text().splitlines():
        s = L.strip()
        if not s:
            continue
        if s == "Points":
            mode = "P"
            continue
        if s == "Lines":
            mode = "L"
            continue
        p = s.split()
        if not p[0].lstrip("-").isdigit():
            continue
        if mode == "P" and len(p) >= 3:
            # file column 1 = x = Z, column 2 = y = R  -> store (R, Z)
            pts.append((float(p[2]), float(p[1])))
        elif mode == "L" and len(p) >= 3:
            segs.append((int(p[1]) - 1, int(p[2]) - 1))
    pts = np.array(pts)
    segs = np.array(segs)
    lines = [[(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])] for a, b in segs]

    # Build polygon-walk order.  Each interior vertex must connect to
    # exactly two segments; walk adjacency starting from seg[0].  If the
    # graph is not a single closed loop (e.g. multiple pieces), fall back
    # to raw order.
    poly = pts
    try:
        adj = {i: [] for i in range(len(pts))}
        for a, b in segs:
            adj[a].append(b)
            adj[b].append(a)
        start = int(segs[0][0])
        walked = [start]
        prev, cur = None, start
        while True:
            nbrs = [n for n in adj[cur] if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start:
                break
            walked.append(nxt)
            prev, cur = cur, nxt
        if len(walked) >= 3:
            poly = pts[np.array(walked)]
    except Exception:
        pass
    return lines, pts, poly


def cell_quads_RZ(arr: np.ndarray) -> np.ndarray:
    """Return (ncell, 4, 2) quads in (R, Z) order for PolyCollection."""
    xlo = arr[:, COL_XLO]
    ylo = arr[:, COL_YLO]
    xhi = arr[:, COL_XHI]
    yhi = arr[:, COL_YHI]
    # axi: x = Z, y = R. RZ plot -> (R=y, Z=x).
    quads = np.empty((len(arr), 4, 2), dtype=float)
    quads[:, 0, 0] = ylo;  quads[:, 0, 1] = xlo
    quads[:, 1, 0] = yhi;  quads[:, 1, 1] = xlo
    quads[:, 2, 0] = yhi;  quads[:, 2, 1] = xhi
    quads[:, 3, 0] = ylo;  quads[:, 3, 1] = xhi
    return quads


def overlay_grid(ax, quads, alpha=0.15):
    lc = PolyCollection(quads, facecolors="none",
                        edgecolors=(0.3, 0.3, 0.3, alpha),
                        linewidths=0.2, zorder=5)
    ax.add_collection(lc)


def overlay_wall(ax, wall_lines):
    ax.add_collection(LineCollection(wall_lines, colors="black",
                                     linewidths=0.8, zorder=10))


def plot_native_field(ax, quads, values, *, cmap, norm, label,
                      wall_lines, R_range, Z_range, title):
    pc = PolyCollection(quads, array=values, cmap=cmap, norm=norm,
                        edgecolors="none", zorder=1)
    ax.add_collection(pc)
    overlay_grid(ax, quads)
    overlay_wall(ax, wall_lines)
    ax.set_xlim(*R_range)
    ax.set_ylim(*Z_range)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    cb = plt.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(label)
    return pc


def regrid_gaussian(arr: np.ndarray, values: np.ndarray, nR=400, nZ=600,
                    sigma=2.0, core_polygon_RZ=None, wall_polygon_RZ=None):
    """Regrid cell-center values onto a regular (R, Z) mesh and smooth.

    `core_polygon_RZ` (N,2 array of points in (R,Z) order): regular-grid
    points inside this polygon are set to NaN after smoothing so the core
    renders transparent.
    `wall_polygon_RZ`: regular-grid points OUTSIDE this polygon are set
    to NaN (vacuum region beyond the vessel).  Without this, griddata
    would interpolate from wall-adjacent cells into the vacuum, producing
    apparent "events outside the vessel" that are actually plotter
    artifacts, not real tallies.
    """
    R = arr[:, COL_YC]
    Z = arr[:, COL_XC]
    Rmin, Rmax = R.min(), R.max()
    Zmin, Zmax = Z.min(), Z.max()
    R_grid = np.linspace(Rmin, Rmax, nR)
    Z_grid = np.linspace(Zmin, Zmax, nZ)
    RR, ZZ = np.meshgrid(R_grid, Z_grid)
    V = griddata((R, Z), values, (RR, ZZ), method="linear", fill_value=0.0)
    V = gaussian_filter(V, sigma=sigma, mode="nearest")

    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    if wall_polygon_RZ is not None and len(wall_polygon_RZ) >= 3:
        wpath = MplPath(wall_polygon_RZ)
        outside_wall = ~wpath.contains_points(pts).reshape(RR.shape)
        V = np.where(outside_wall, np.nan, V)
    if core_polygon_RZ is not None and len(core_polygon_RZ) >= 3:
        cpath = MplPath(core_polygon_RZ)
        inside_core = cpath.contains_points(pts).reshape(RR.shape)
        V = np.where(inside_core, np.nan, V)
    return R_grid, Z_grid, V


def plot_smoothed_field(ax, arr, values, quads, *, cmap, norm, label,
                        wall_lines, R_range, Z_range, title,
                        core_polygon_RZ=None, wall_polygon_RZ=None, sigma=2.0):
    R_grid, Z_grid, V = regrid_gaussian(arr, values, sigma=sigma,
                                        core_polygon_RZ=core_polygon_RZ,
                                        wall_polygon_RZ=wall_polygon_RZ)
    # Force NaN cells (core mask) to render transparent: copy the colormap
    # so set_bad doesn't mutate the global, and use shading='nearest' which
    # doesn't interpolate across NaNs (the default 'auto' path falls through
    # to gouraud-like quad blending that smears masked values).
    import copy as _copy
    cmap_masked = _copy.copy(plt.get_cmap(cmap)) if isinstance(cmap, str) \
                  else _copy.copy(cmap)
    cmap_masked.set_bad(color=(0, 0, 0, 0))
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(R_grid, Z_grid, V, cmap=cmap_masked, norm=norm,
                         shading="nearest", zorder=1)
    # Outline the core polygon in black so the masked region is visually
    # obvious (not just "missing data").
    if core_polygon_RZ is not None and len(core_polygon_RZ) >= 3:
        poly_lines = [[tuple(core_polygon_RZ[i]),
                       tuple(core_polygon_RZ[(i + 1) % len(core_polygon_RZ)])]
                      for i in range(len(core_polygon_RZ))]
        ax.add_collection(LineCollection(poly_lines, colors="black",
                                         linewidths=0.6, zorder=9))
    overlay_grid(ax, quads)
    overlay_wall(ax, wall_lines)
    ax.set_xlim(*R_range)
    ax.set_ylim(*Z_range)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    cb = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(label)
    return mesh


def load_plasma_fields(path: Path):
    """Read Te, ne on the plasma-mesh triangulation from plasma.h5.

    Returns (vr, vz, triangles, tri_te, tri_ne) where tri_te / tri_ne are
    per-triangle values (flat, one eV and one m^-3 value per triangle,
    resolved via mesh/cell_index).  Triangles with invalid cell_index get
    zero (they're outside the plasma mesh, e.g. vacuum / private-flux).
    """
    with h5py.File(path, "r") as f:
        vr  = f["mesh/vtx_r"][:]
        vz  = f["mesh/vtx_z"][:]
        tri = f["mesh/triangles"][:]
        ci  = f["mesh/cell_index"][:]
        ne  = f["mesh/dens_e"][:]
        te  = f["mesh/temp_e"][:]
    ncell = len(ne)
    valid = (ci >= 0) & (ci < ncell)
    tri_te = np.where(valid, te[np.clip(ci, 0, ncell - 1)], 0.0)
    tri_ne = np.where(valid, ne[np.clip(ci, 0, ncell - 1)], 0.0)
    return vr, vz, tri, tri_te, tri_ne


def field_at_cells(cell_R, cell_Z, vr, vz, tri, tri_vals):
    """Return plasma field values at (cell_R, cell_Z) by looking up which
    triangle each point lies in (flat per-triangle coloring).  Returns
    zero for points outside the mesh."""
    mtri = Triangulation(vr, vz, tri)
    finder = mtri.get_trifinder()
    trid = finder(cell_R, cell_Z)
    out = np.zeros_like(cell_R, dtype=float)
    inside = trid >= 0
    out[inside] = tri_vals[trid[inside]]
    return out


def load_adas_power_table(adas_h5: Path, kind: str):
    """kind: 'LineRadiation' or 'RecombRadiation'.  Returns (logTe, logNe,
    logP_q0) where logP_q0 is the 2D (nT, nD) table for charge state 0
    (neutral → ion line rad for PLT; ion → neutral recomb rad for PRB)."""
    with h5py.File(adas_h5, "r") as f:
        logTe = f[f"gridTemperature_{kind}"][:]     # log10 eV
        logNe = f[f"gridDensity_{kind}"][:]         # log10 cm^-3
        logP  = f[f"{kind}PowerCoeff"][0]           # log10 W cm^3, shape (nT, nD)
    return logTe, logNe, logP


def eval_adas_power(Te_eV, ne_m3, logTe, logNe, logP):
    """Bilinear log-interpolate the ADAS power coefficient; return the
    value in SI units W m^3.  Out-of-range queries clip to table edges."""
    logTe_q = np.log10(np.maximum(Te_eV, 1e-9))
    logNe_q = np.log10(np.maximum(ne_m3 * 1e-6, 1e-30))  # m^-3 -> cm^-3
    interp = RegularGridInterpolator((logTe, logNe), logP,
                                      method="linear",
                                      bounds_error=False,
                                      fill_value=None)   # None = clamp
    logP_q = interp(np.column_stack([logTe_q.ravel(), logNe_q.ravel()]))
    P_Wcm3 = 10.0 ** logP_q
    return (P_Wcm3 * 1e-6).reshape(Te_eV.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump",    type=Path, required=True)
    ap.add_argument("--wall",    type=Path, required=True)
    ap.add_argument("--core",    type=Path, default=None,
                    help="Optional core.surf; regions inside this closed polygon "
                         "are masked out of the smoothed source plots so the "
                         "griddata interpolation does not bleed across the core "
                         "gap.  Recommended when the deck uses a psi-contour "
                         "absorb surface.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sigma",   type=float, default=2.0,
                    help="Gaussian smoothing sigma on regridded source fields")
    ap.add_argument("--plasma-h5", type=Path, default=None,
                    help="plasma.h5 with /mesh/dens_e, /mesh/temp_e; needed "
                         "for the optional radiation figure (P_line = "
                         "n_D * n_e * PLT(Te, ne))")
    ap.add_argument("--adas-rates", type=Path, default=None,
                    help="ADAS_Rates_1.h5 carrying LineRadiationPowerCoeff "
                         "(PLT) and RecombRadiationPowerCoeff (PRB); needed "
                         "for the radiation figure")
    args = ap.parse_args()

    arr = read_last_snapshot(args.dump)
    wall_lines, _wall_raw, wall_polygon = parse_wall_surf(args.wall)
    print(f"  wall mask: {len(wall_polygon)} polygon vertices "
          f"(walked adjacency order)", file=sys.stderr)
    core_polygon = None
    if args.core is not None:
        _core_lines, _core_raw, core_polygon = parse_wall_surf(args.core)
        print(f"  core mask: {len(core_polygon)} polygon vertices",
              file=sys.stderr)
    quads = cell_quads_RZ(arr)

    R = arr[:, COL_YC]
    Z = arr[:, COL_XC]
    # Clip R to (0.98, 2.8): anything at smaller R is inside the core-absorb
    # boundary (fix reflect/psi psi_norm 0.95 action absorb), and has no
    # meaningful neutral-transport signal.
    R_range = (0.98, 2.4)
    Z_range = (Z.min() - 0.02, Z.max() + 0.02)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: densities ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    density_panels = [
        (COL_ND,  "D",    r"$n_D$ [m$^{-3}$]",    "D density"),
        (COL_ND2, "D$_2$", r"$n_{D_2}$ [m$^{-3}$]", "D$_2$ density"),
    ]
    for ax, (col, _name, label, title) in zip(axes, density_panels):
        v = np.maximum(arr[:, col], 0.0)
        finite = v[v > 0]
        if len(finite) > 0:
            vmin = max(np.nanpercentile(finite, 1), 1e6)
            vmax = np.nanpercentile(finite, 99)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        else:
            norm = None
        plot_native_field(ax, quads, v,
                          cmap="RdBu_r", norm=norm,
                          label=label,
                          wall_lines=wall_lines,
                          R_range=R_range, Z_range=Z_range,
                          title=title)
    fig.tight_layout()
    out = args.out_dir / "neutral_densities.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- Figure 2: source moments, 2x2 (Sp, Sm, Qe, Qi) ----
    # Sm here is the signed scalar sum of the three vector components
    # (Sm_x + Sm_y + Sm_z).  Signed is fine and matches the convention
    # Gkeyll/SOLEDGE expect for coupling diagnostics.
    Sm_total = arr[:, COL_SMX] + arr[:, COL_SMY] + arr[:, COL_SMZ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)
    ax_sp, ax_sm, ax_qe, ax_qi = axes.flat

    def _symlog(v, pct=99.5, lin_frac=0.03):
        """SymLogNorm covers a wide signed dynamic range with a linear
        threshold so RdBu_r gives an explicit white band at |v| < linthresh
        and log-compressed tails beyond that."""
        finite = v[np.isfinite(v) & (v != 0)]
        if len(finite) == 0:
            return None
        absmax = np.nanpercentile(np.abs(finite), pct)
        lt     = max(absmax * lin_frac, 1e-30)
        return SymLogNorm(linthresh=lt, vmin=-absmax, vmax=absmax, base=10)

    panels = [
        (ax_sp, arr[:, COL_SP], r"$S_p$ [m$^{-3}$ s$^{-1}$]",            "particle source"),
        (ax_sm, Sm_total,       r"$S_m$ (x+y+z) [kg m$^{-2}$ s$^{-2}$]", "momentum source"),
        (ax_qe, arr[:, COL_QE], r"$Q_e$ [W m$^{-3}$]",                   "electron energy source"),
        (ax_qi, arr[:, COL_QI], r"$Q_i$ [W m$^{-3}$]",                   "ion energy source"),
    ]
    for ax, v, label, title in panels:
        plot_smoothed_field(ax, arr, v, quads,
                            cmap="RdBu_r", norm=_symlog(v), label=label,
                            wall_lines=wall_lines,
                            R_range=R_range, Z_range=Z_range,
                            title=title,
                            core_polygon_RZ=core_polygon,
                            wall_polygon_RZ=wall_polygon, sigma=args.sigma)

    fig.tight_layout()
    out = args.out_dir / "neutral_sources.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- Figure 3 (optional): line + recomb radiation power density ----
    # Only emitted when both --plasma-h5 and --adas-rates are supplied.
    # Computes P_line  = n_D   * n_e * PLT(Te, ne)   [W/m^3]
    #          P_recb  = n_D+  * n_e * PRB(Te, ne)   [W/m^3]   (optional)
    # For this deck `fix chem/adas mode neutral` has no D+ kinetic particles,
    # so n_D+ is taken from the plasma mesh (ne with charge=1 = n_D+ for pure
    # hydrogenic plasma) and PRB adds only a small contribution.
    if args.plasma_h5 is not None and args.adas_rates is not None:
        vr, vz, tri, tri_te, tri_ne = load_plasma_fields(args.plasma_h5)
        cell_R = arr[:, COL_YC]
        cell_Z = arr[:, COL_XC]
        Te_cell = field_at_cells(cell_R, cell_Z, vr, vz, tri, tri_te)
        ne_cell = field_at_cells(cell_R, cell_Z, vr, vz, tri, tri_ne)
        nD_cell = arr[:, COL_ND]
        # PLT (line radiation, hydrogenic)
        logTe_plt, logNe_plt, logPLT = load_adas_power_table(
            args.adas_rates, "LineRadiation")
        PLT_Wm3 = eval_adas_power(Te_cell, ne_cell,
                                  logTe_plt, logNe_plt, logPLT)
        P_line = nD_cell * ne_cell * PLT_Wm3            # [W/m^3]
        # PRB (recombination + bremsstrahlung, uses n_e * n_D+; for a pure
        # hydrogenic plasma n_D+ = ne on the mesh, so this is ne^2 * PRB).
        logTe_prb, logNe_prb, logPRB = load_adas_power_table(
            args.adas_rates, "RecombRadiation")
        PRB_Wm3 = eval_adas_power(Te_cell, ne_cell,
                                  logTe_prb, logNe_prb, logPRB)
        P_recb = ne_cell * ne_cell * PRB_Wm3            # [W/m^3]
        P_tot  = P_line + P_recb

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
        panels = [
            (P_line, r"$P_\mathrm{line}$ [W m$^{-3}$]",  "line radiation"),
            (P_recb, r"$P_\mathrm{rec+brem}$ [W m$^{-3}$]", "recomb + brems"),
            (P_tot,  r"$P_\mathrm{rad}$ [W m$^{-3}$]",  "total radiation"),
        ]
        for ax, (v, label, title) in zip(axes, panels):
            finite = v[v > 0]
            if len(finite) == 0:
                norm = None
            else:
                vmax = np.nanpercentile(finite, 99.5)
                vmin = max(np.nanpercentile(finite, 20), vmax * 1e-4)
                norm = LogNorm(vmin=vmin, vmax=vmax)
            plot_smoothed_field(ax, arr, v, quads,
                                cmap="inferno", norm=norm, label=label,
                                wall_lines=wall_lines,
                                R_range=R_range, Z_range=Z_range,
                                title=title,
                                core_polygon_RZ=core_polygon,
                                wall_polygon_RZ=wall_polygon,
                                sigma=args.sigma)
        fig.tight_layout()
        out = args.out_dir / "neutral_radiation.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"wrote {out}")
        # Also print domain-integrated totals (rough, volume-weighted)
        xlo, ylo, xhi, yhi = (arr[:, c] for c in
                              (COL_XLO, COL_YLO, COL_XHI, COL_YHI))
        vol = (xhi - xlo) * np.pi * (yhi ** 2 - ylo ** 2)
        print(f"  P_line integrated: {(P_line * vol).sum():.3e} W",
              file=sys.stderr)
        print(f"  P_recb integrated: {(P_recb * vol).sum():.3e} W",
              file=sys.stderr)


if __name__ == "__main__":
    main()
