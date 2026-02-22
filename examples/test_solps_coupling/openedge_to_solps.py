#!/usr/bin/env python3
"""
Build SOLPS source2d file from OpenEdge mass_loss dump.

Typical use (steady test with last OpenEdge source):
  python3 openedge_to_solps.py \
    --mass-loss mass_loss.txt \
    --b2fgmtry "/Users/42d/CAT_CASE_LI/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up/b2fgmtry" \
    --out "/Users/42d/CAT_CASE_LI/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up/source2d.00001" \
    --oe-dt 1e-5 \
    --mode last
"""

import argparse
import re
from pathlib import Path

import numpy as np

AMU_KG = 1.66053906660e-27
LI_AMU = 6.94
LI_ATOM_MASS_KG = LI_AMU * AMU_KG


def parse_mass_loss(path: Path):
    blocks = []
    timestep = None
    n_cells = None
    i = 0
    with path.open() as f:
        lines = f.read().splitlines()

    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue
        if line == "ITEM: NUMBER OF CELLS":
            n_cells = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 4
            continue
        if line.startswith("ITEM: CELLS"):
            headers = lines[i].split()[2:]
            if timestep is None or n_cells is None:
                raise RuntimeError("Incomplete mass_loss block before CELLS.")
            rows = []
            for k in range(n_cells):
                vals = lines[i + 1 + k].split()
                if len(vals) < len(headers):
                    continue
                rows.append([float(vals[j]) for j in range(len(headers))])
            blocks.append({"timestep": timestep, "headers": headers, "data": np.asarray(rows, dtype=float)})
            i += 1 + n_cells
            continue
        i += 1

    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def extract_atoms_per_step(block):
    headers = block["headers"]
    data = block["data"]
    idx = {name: j for j, name in enumerate(headers)}
    if "xc" not in idx or "yc" not in idx:
        raise RuntimeError("mass_loss block missing xc/yc columns.")
    r = data[:, idx["xc"]]
    z = data[:, idx["yc"]]

    dm_idx = [j for j, name in enumerate(headers) if ("[1]" in name) or ("dm" in name.lower())]
    dn_idx = [j for j, name in enumerate(headers) if ("[2]" in name) or ("dn" in name.lower())]

    if dm_idx:
        dm = np.zeros(len(data), dtype=float)
        for j in dm_idx:
            dm += data[:, j]
        dm = np.where(np.isfinite(dm) & (dm > 0.0), dm, 0.0)
        atoms_step = dm / LI_ATOM_MASS_KG
        method = "mass_to_atoms"
    elif dn_idx:
        atoms_step = np.zeros(len(data), dtype=float)
        for j in dn_idx:
            atoms_step += data[:, j]
        atoms_step = np.where(np.isfinite(atoms_step) & (atoms_step > 0.0), atoms_step, 0.0)
        method = "atoms_direct"
    else:
        raise RuntimeError("No recognizable Li loss column found (need [1]/dm or [2]/dn).")

    return r, z, atoms_step, method


def infer_uniform_cell_size(coord):
    u = np.unique(np.round(np.asarray(coord, dtype=float), decimals=12))
    if u.size < 2:
        raise RuntimeError("Cannot infer cell size.")
    du = np.diff(np.sort(u))
    du = du[du > 0.0]
    if du.size == 0:
        raise RuntimeError("Cannot infer positive cell size.")
    return float(np.median(du))


def build_openedge_source(blocks, oe_dt: float, mode: str, last_n: int):
    atoms_all = []
    r_ref = z_ref = None
    method_ref = None
    for b in blocks:
        r, z, atoms_step, method = extract_atoms_per_step(b)
        if r_ref is None:
            r_ref, z_ref, method_ref = r, z, method
        atoms_all.append(atoms_step)
    atoms_all = np.asarray(atoms_all, dtype=float)

    per_block_totals = atoms_all.sum(axis=1)

    if mode == "last":
        atoms_step_sel = atoms_all[-1]
    elif mode == "last_nonzero":
        nz = np.where(per_block_totals > 0.0)[0]
        atoms_step_sel = atoms_all[nz[-1]] if nz.size else atoms_all[-1]
    elif mode == "mean":
        atoms_step_sel = atoms_all.mean(axis=0)
    elif mode == "mean_last_n":
        n = max(1, min(last_n, atoms_all.shape[0]))
        atoms_step_sel = atoms_all[-n:].mean(axis=0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    d_r = infer_uniform_cell_size(r_ref)
    d_z = infer_uniform_cell_size(z_ref)
    vol = 2.0 * np.pi * r_ref * d_r * d_z
    s_oe = np.where(vol > 0.0, atoms_step_sel / (oe_dt * vol), 0.0)
    s_oe = np.where(np.isfinite(s_oe) & (s_oe >= 0.0), s_oe, 0.0)
    return r_ref, z_ref, s_oe, atoms_step_sel, method_ref, d_r, d_z


def parse_b2fgmtry(path: Path, nx: int, ny: int, return_corners: bool = False):
    nxg, nyg = nx + 2, ny + 2
    with path.open() as fh:
        lines = fh.readlines()

    float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
    header_idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("*cf:")]

    datasets = {}
    for k, i0 in enumerate(header_idx):
        i1 = header_idx[k + 1] if (k + 1) < len(header_idx) else len(lines)
        parts = lines[i0].split()
        try:
            dtype = parts[1].lower()
            count = int(parts[2])
            name = parts[3]
        except (IndexError, ValueError):
            continue
        if dtype not in ("real", "int"):
            continue
        vals = []
        for ln in lines[i0 + 1 : i1]:
            for tok in float_re.findall(ln):
                vals.append(float(tok.replace("D", "E").replace("d", "e")))
        if len(vals) < count:
            raise RuntimeError(f"Dataset parse failed for {name}: need {count}, got {len(vals)}")
        datasets[name] = np.asarray(vals[:count], dtype=float)

    for req in ("crx", "cry", "vol"):
        if req not in datasets:
            raise KeyError(f"{req} not found in {path}")

    crx = datasets["crx"].reshape(4, nxg, nyg, order="F")
    cry = datasets["cry"].reshape(4, nxg, nyg, order="F")
    r_s = crx.mean(axis=0).T
    z_s = cry.mean(axis=0).T
    vol = datasets["vol"].reshape(nxg, nyg, order="F").T
    if not return_corners:
        return r_s, z_s, vol
    # (ny+2, nx+2, 4): corners for each cell
    crx_cell = np.transpose(crx, (2, 1, 0))
    cry_cell = np.transpose(cry, (2, 1, 0))
    return r_s, z_s, vol, crx_cell, cry_cell


def interpolate_to_solps(r_oe, z_oe, s_oe, r_s, z_s):
    try:
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree

        pts = np.column_stack((r_oe, z_oe))
        grid = np.column_stack((r_s.ravel(), z_s.ravel()))
        s_lin = griddata(pts, s_oe, grid, method="linear")
        nan = ~np.isfinite(s_lin)
        if np.any(nan):
            tree = cKDTree(pts)
            _, nn = tree.query(grid[nan], k=1)
            s_lin[nan] = s_oe[nn]
        s_s = s_lin.reshape(r_s.shape)
    except Exception:
        import matplotlib.tri as mtri

        pts = np.column_stack((r_oe, z_oe))
        grid = np.column_stack((r_s.ravel(), z_s.ravel()))
        tri = mtri.Triangulation(r_oe, z_oe)
        lin = mtri.LinearTriInterpolator(tri, s_oe)
        s_lin = np.asarray(lin(grid[:, 0], grid[:, 1]).filled(np.nan), dtype=float)
        nan = ~np.isfinite(s_lin)
        if np.any(nan):
            q = grid[nan]
            fill = np.zeros(q.shape[0], dtype=float)
            for i in range(0, q.shape[0], 256):
                qb = q[i : i + 256]
                dr = qb[:, None, 0] - pts[None, :, 0]
                dz = qb[:, None, 1] - pts[None, :, 1]
                j = np.argmin(dr * dr + dz * dz, axis=1)
                fill[i : i + 256] = s_oe[j]
            s_lin[nan] = fill
        s_s = s_lin.reshape(r_s.shape)

    s_s = np.where(np.isfinite(s_s) & (s_s >= 0.0), s_s, 0.0)
    return s_s


def write_source2d(outpath: Path, s_s, vol, ns: int, li0_idx: int):
    nyg, nxg = vol.shape
    src = (s_s * vol).clip(0.0)
    zero_line = " ".join(["0.0e+00"] * nxg) + "\n"

    with outpath.open("w") as fh:
        for ispec in range(ns):
            if ispec == li0_idx:
                for iy in range(nyg):
                    fh.write(" ".join(f"{v:.8e}" for v in src[iy, :]) + "\n")
            else:
                for _ in range(nyg):
                    fh.write(zero_line)
    return float(src.sum())


def plot_solps_source_poly(
    r_s,
    z_s,
    s_solps,
    out_png: Path,
    crx_cell=None,
    cry_cell=None,
    log_scale: bool = True,
    vmin=None,
    vmax=None,
    zero_transparent: bool = True,
    focus_source: bool = True,
    focus_margin: float = 0.2,
    show_grid: bool = True,
    ax_xlim=None,
    ax_ylim=None,
):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.collections import PolyCollection

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.2), dpi=220, constrained_layout=True)
    mappable = None
    if crx_cell is not None and cry_cell is not None:
        polys = []
        vals = []
        bounds = []
        nyg, nxg = s_solps.shape
        for iy in range(nyg):
            for ix in range(nxg):
                xs = crx_cell[iy, ix, :]
                ys = cry_cell[iy, ix, :]
                if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                    continue
                poly = np.column_stack((xs, ys))
                polys.append(poly)
                vals.append(float(s_solps[iy, ix]))
                bounds.append((float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))))
        vals = np.asarray(vals, dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        bounds = np.asarray(bounds, dtype=float) if bounds else np.empty((0, 4), dtype=float)

        # Base SOLPS mesh (Jeremy-style context)
        if show_grid:
            mesh = PolyCollection(
                polys,
                facecolors="none",
                edgecolors="0.25",
                linewidths=0.08,
                antialiased=False,
                zorder=0,
                alpha=0.45,
            )
            ax.add_collection(mesh)

        if zero_transparent:
            vals_plot = vals.copy()
            vals_plot[vals_plot <= 0.0] = np.nan
        else:
            vals_plot = vals

        finite_pos = vals[np.isfinite(vals) & (vals > 0.0)]
        if finite_pos.size == 0:
            finite_pos = np.array([1.0e-30], dtype=float)

        if vmin is None:
            vmin_eff = float(np.percentile(finite_pos, 2.0)) if log_scale else float(np.nanmin(vals_plot))
        else:
            vmin_eff = float(vmin)
        if vmax is None:
            vmax_eff = float(np.percentile(finite_pos, 99.5)) if log_scale else float(np.nanmax(vals_plot))
        else:
            vmax_eff = float(vmax)
        if not np.isfinite(vmin_eff) or vmin_eff <= 0.0:
            vmin_eff = float(np.min(finite_pos))
        if not np.isfinite(vmax_eff) or vmax_eff <= vmin_eff:
            vmax_eff = max(vmin_eff * 10.0, vmin_eff + 1.0)

        norm = colors.LogNorm(vmin=vmin_eff, vmax=vmax_eff) if log_scale else colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
        coll = PolyCollection(
            polys,
            array=vals_plot,
            cmap="inferno",
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
            norm=norm,
            zorder=2,
        )
        ax.add_collection(coll)
        ax.autoscale_view()
        mappable = coll

        if focus_source and vals.size > 0:
            mask = vals > 0.0
            if np.any(mask) and bounds.shape[0] == vals.shape[0]:
                b = bounds[mask]
                xmin = float(np.min(b[:, 0]))
                xmax = float(np.max(b[:, 1]))
                ymin = float(np.min(b[:, 2]))
                ymax = float(np.max(b[:, 3]))
                dx = max(xmax - xmin, 1e-6)
                dy = max(ymax - ymin, 1e-6)
                ax.set_xlim(xmin - focus_margin * dx, xmax + focus_margin * dx)
                ax.set_ylim(ymin - focus_margin * dy, ymax + focus_margin * dy)
    else:
        vals = s_solps.ravel()
        finite_pos = vals[np.isfinite(vals) & (vals > 0.0)]
        if finite_pos.size == 0:
            finite_pos = np.array([1.0e-30], dtype=float)
        if vmin is None:
            vmin_eff = float(np.percentile(finite_pos, 2.0)) if log_scale else float(np.nanmin(vals))
        else:
            vmin_eff = float(vmin)
        if vmax is None:
            vmax_eff = float(np.percentile(finite_pos, 99.5)) if log_scale else float(np.nanmax(vals))
        else:
            vmax_eff = float(vmax)
        if not np.isfinite(vmin_eff) or vmin_eff <= 0.0:
            vmin_eff = float(np.min(finite_pos))
        if not np.isfinite(vmax_eff) or vmax_eff <= vmin_eff:
            vmax_eff = max(vmin_eff * 10.0, vmin_eff + 1.0)
        norm = colors.LogNorm(vmin=vmin_eff, vmax=vmax_eff) if log_scale else colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
        sc = ax.scatter(
            r_s.ravel(),
            z_s.ravel(),
            c=vals,
            s=10,
            marker="s",
            cmap="inferno",
            linewidths=0,
            norm=norm,
        )
        mappable = sc

    cbar = fig.colorbar(mappable, ax=ax, pad=0.01)
    cbar.set_label("Li source (atoms/m$^3$/s)")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")
    if ax_xlim is not None:
        ax.set_xlim(ax_xlim)
    if ax_ylim is not None:
        ax.set_ylim(ax_ylim)
    ax.set_title("Source passed to SOLPS grid")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_solps_source_quixote(r_s, z_s, s_solps, out_png: Path, shot_obj_path: Path, xlim=None, ylim=None):
    import matplotlib.pyplot as plt
    import pickle

    # Use the same call pattern requested by user.
    from quixote import GridDataPlot  # type: ignore

    with shot_obj_path.open("rb") as fh:
        shot_ddnu = pickle.load(fh)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    GridDataPlot(
        shot_ddnu,
        "grid",
        canvas=ax1,
        xlim=xlim if xlim is not None else [float(np.nanmin(r_s)), float(np.nanmax(r_s))],
        ylim=ylim if ylim is not None else [float(np.nanmin(z_s)), float(np.nanmax(z_s))],
        lw=0.2,
    )
    sc = ax1.scatter(
        r_s.ravel(),
        z_s.ravel(),
        c=s_solps.ravel(),
        s=7,
        cmap="magma",
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("Li source (atoms/m$^3$/s)")
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Source passed to SOLPS grid (quixote)")
    fig.savefig(out_png, bbox_inches="tight")
#    plt.close(fig)
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Build SOLPS source2d from OpenEdge mass_loss.")
    p.add_argument("--mass-loss", default="mass_loss.txt", help="OpenEdge mass_loss dump file.")
    p.add_argument("--b2fgmtry", required=True, help="Path to SOLPS b2fgmtry.")
    p.add_argument("--out", required=True, help="Output source2d file path.")
    p.add_argument("--oe-dt", type=float, default=1e-5, help="OpenEdge timestep used by mass_loss [s].")
    p.add_argument("--mode", choices=["last", "last_nonzero", "mean", "mean_last_n"], default="last_nonzero")
    p.add_argument("--last-n", type=int, default=10, help="Used when --mode mean_last_n.")
    p.add_argument("--nx", type=int, default=170)
    p.add_argument("--ny", type=int, default=36)
    p.add_argument("--ns", type=int, default=17)
    p.add_argument("--li0-idx", type=int, default=13)
    p.add_argument("--solps-dt", type=float, default=1e-6, help="SOLPS timestep [s], for reporting only.")
    p.add_argument("--plot-out", default="", help="Optional sanity plot PNG on SOLPS grid.")
    p.add_argument("--plot-backend", choices=["poly", "quixote"], default="poly")
    p.add_argument("--quixote-shot", default="", help="Pickle file containing shot_ddnu object for GridDataPlot.")
    p.add_argument("--plot-xlim", nargs=2, type=float, default=None, help="Optional xlim for quixote plot.")
    p.add_argument("--plot-ylim", nargs=2, type=float, default=None, help="Optional ylim for quixote plot.")
    p.add_argument("--plot-log", choices=["yes", "no"], default="yes", help="Log-scale color map for poly backend.")
    p.add_argument("--plot-vmin", type=float, default=None, help="Optional manual color min.")
    p.add_argument("--plot-vmax", type=float, default=None, help="Optional manual color max.")
    p.add_argument(
        "--plot-zero-transparent",
        choices=["yes", "no"],
        default="yes",
        help="Hide zero-source cells in poly backend.",
    )
    p.add_argument(
        "--plot-focus-source",
        choices=["yes", "no"],
        default="no",
        help="Auto-zoom around nonzero source cells in poly backend.",
    )
    p.add_argument("--plot-focus-margin", type=float, default=0.35, help="Relative margin for auto-zoom window.")
    p.add_argument("--plot-show-grid", choices=["yes", "no"], default="yes", help="Draw SOLPS mesh lines.")
    p.add_argument("--plot-ax-xlim", nargs=2, type=float, default=None, help="Force axis x-limits.")
    p.add_argument("--plot-ax-ylim", nargs=2, type=float, default=None, help="Force axis y-limits.")
    args = p.parse_args()

    mass_loss = Path(args.mass_loss).expanduser().resolve()
    b2fgmtry = Path(args.b2fgmtry).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    blocks = parse_mass_loss(mass_loss)
    r_oe, z_oe, s_oe, atoms_step_sel, method, d_r, d_z = build_openedge_source(
        blocks, args.oe_dt, args.mode, args.last_n
    )
    r_s, z_s, vol_s, crx_cell, cry_cell = parse_b2fgmtry(
        b2fgmtry, args.nx, args.ny, return_corners=True
    )
    s_solps = interpolate_to_solps(r_oe, z_oe, s_oe, r_s, z_s)

    out.parent.mkdir(parents=True, exist_ok=True)
    total_solps_atoms_s = write_source2d(out, s_solps, vol_s, args.ns, args.li0_idx)
    total_oe_atoms_s = float(np.sum(atoms_step_sel) / args.oe_dt)

    print(f"[mass_loss] {mass_loss}")
    print(f"[mode] {args.mode}  method={method}")
    print(f"[OE grid] dR={d_r:.6g} m  dZ={d_z:.6g} m")
    print(f"[OE rate] {total_oe_atoms_s:.6e} atoms/s")
    print(f"[SOLPS rate] {total_solps_atoms_s:.6e} atoms/s")
    if total_oe_atoms_s > 0:
        print(f"[ratio SOLPS/OE] {total_solps_atoms_s / total_oe_atoms_s:.5f}")
    print(f"[write] {out}")
    if args.plot_out:
        plot_out = Path(args.plot_out).expanduser().resolve()
        if args.plot_backend == "quixote":
            if args.quixote_shot:
                shot_path = Path(args.quixote_shot).expanduser().resolve()
                try:
                    plot_solps_source_quixote(
                        r_s, z_s, s_solps, plot_out, shot_path, xlim=args.plot_xlim, ylim=args.plot_ylim
                    )
                except Exception as e:
                    print(f"[warn] quixote plotting failed ({e}); falling back to poly plot.")
                    plot_solps_source_poly(r_s, z_s, s_solps, plot_out, crx_cell=crx_cell, cry_cell=cry_cell)
            else:
                print("[warn] --plot-backend quixote requested but --quixote-shot not provided; using poly plot.")
                plot_solps_source_poly(r_s, z_s, s_solps, plot_out, crx_cell=crx_cell, cry_cell=cry_cell)
        else:
            plot_solps_source_poly(
                r_s,
                z_s,
                s_solps,
                plot_out,
                crx_cell=crx_cell,
                cry_cell=cry_cell,
                log_scale=(args.plot_log == "yes"),
                vmin=args.plot_vmin,
                vmax=args.plot_vmax,
                zero_transparent=(args.plot_zero_transparent == "yes"),
                focus_source=(args.plot_focus_source == "yes"),
                focus_margin=args.plot_focus_margin,
                show_grid=(args.plot_show_grid == "yes"),
                ax_xlim=args.plot_ax_xlim,
                ax_ylim=args.plot_ax_ylim,
            )
        print(f"[plot] {plot_out}")
    print(f"[timesteps] OpenEdge dt={args.oe_dt:.3e} s, SOLPS dt={args.solps_dt:.3e} s")
    print("[note] Source is in atoms/m^3/s, so dt mismatch is naturally handled by SOLPS time integration.")
    print("[note] For steady test, keep one file source2d.00001 and fixed sources_time_switch.")


if __name__ == "__main__":
    main()
