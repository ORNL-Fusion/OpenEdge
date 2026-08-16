"""Plot helpers for the CAT liquid-metal divertor workflow.

All figures use a paper-ready style: LaTeX-style labels, big fonts, units in ().
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PAPER_RC = {
    "font.size": 17,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
}


def use_paper_style():
    plt.rcParams.update(PAPER_RC)


def bump_notebook_fonts(px=18):
    """Inject CSS so notebook markdown + code render at `px`. Call once."""
    from IPython.display import HTML, display
    css = f"""
    <style>
      .jp-RenderedMarkdown, .jp-RenderedHTMLCommon {{ font-size: {px}px !important; line-height: 1.55 !important; }}
      .jp-RenderedMarkdown h1 {{ font-size: {px+12}px !important; }}
      .jp-RenderedMarkdown h2 {{ font-size: {px+8}px  !important; }}
      .jp-RenderedMarkdown h3 {{ font-size: {px+4}px  !important; }}
      .jp-CodeMirrorEditor, .CodeMirror, .jp-OutputArea-output pre {{
          font-size: {px-1}px !important;
      }}
    </style>
    """
    display(HTML(css))


def _front_segments(pts, lines, n_arc_lines):
    segs = []
    for lid, p1, p2, *_ in lines:
        if lid <= n_arc_lines:
            segs.append([pts[p1], pts[p2]])
    return segs


def _back_segments(pts, lines, n_arc_lines):
    segs = []
    for lid, p1, p2, *_ in lines:
        if lid > n_arc_lines:
            segs.append([pts[p1], pts[p2]])
    return segs


def _draw_back(ax, back_segs, color="0.75", lw=1.5, zorder=1):
    for (x1, y1), (x2, y2) in back_segs:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=zorder)


def _draw_front_black(ax, front_segs, lw=2.6, zorder=3):
    for (x1, y1), (x2, y2) in front_segs:
        ax.plot([x1, x2], [y1, y2], color="k", lw=lw, zorder=zorder)


def plot_geometry(pts, lines, n_arc_lines, ax=None, title="divertor slab"):
    """Front arc in steelblue, 3 back-cap segments in crimson."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for lid, p1, p2, *_ in lines:
        x1, y1 = pts[p1]
        x2, y2 = pts[p2]
        color = "steelblue" if lid <= n_arc_lines else "crimson"
        ax.plot([x1, x2], [y1, y2], "-", color=color, lw=1.0, zorder=2)
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    return ax


def plot_tsurf_profile(s_mid, T_strip, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_mid, T_strip, "-", lw=2, color="firebrick")
    ax.set_xlabel(r"arc length $s$ (m)")
    ax.set_ylabel(r"$T_\mathrm{surf}$ ($^\circ$C)")
    ax.set_title("LM MHD model surface temperature along outer divertor")
    ax.grid(True, ls=":", alpha=0.5)
    return ax


def plot_solps_profiles(s_mid, Wtot, Gamma_D):
    """Two-panel: total heat flux (MW/m^2) and D+ ion flux (log)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(s_mid, Wtot * 1e-6, "-", lw=2, color="firebrick")
    axes[0].set_xlabel(r"arc length $s$ (m)")
    axes[0].set_ylabel(r"$W_\mathrm{tot}$ (MW m$^{-2}$)")
    axes[0].set_title("SOLPS total heat flux")
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].semilogy(s_mid, Gamma_D, "-", lw=2, color="navy")
    axes[1].set_xlabel(r"arc length $s$ (m)")
    axes[1].set_ylabel(r"$\Gamma_{D^+}$ (m$^{-2}$ s$^{-1}$)")
    axes[1].set_title(r"SOLPS D$^+$ ion flux to target")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    return fig, axes


def plot_li_fluxes(s_mid, G_evap, G_adat, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(s_mid, G_evap, "-", lw=2, color="firebrick",
                label=r"$\Gamma_\mathrm{evap}$ (Antoine + Hertz-Knudsen)")
    ax.semilogy(s_mid, G_adat, "-", lw=2, color="navy",
                label=r"$\Gamma_\mathrm{adatom}$ (Arrhenius $\times$ $\Gamma_{D^+}$)")
    ax.set_xlabel(r"arc length $s$ (m)")
    ax.set_ylabel(r"Li flux (atoms m$^{-2}$ s$^{-1}$)")
    ax.set_title("evaporation vs. adatom desorption from outer divertor")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    return ax


def _wall_normal(x_mid, y_mid):
    """Unit chord-tangent and outward normal (rotated -90 deg)."""
    tx = x_mid[-1] - x_mid[0]
    ty = y_mid[-1] - y_mid[0]
    L = np.hypot(tx, ty)
    tx, ty = tx / L, ty / L
    return tx, ty, -ty, tx


def plot_tsurf_on_wall(pts, lines, n_arc_lines, x_mid, y_mid, T_plot,
                       cmap="plasma", side=-1.0, base_offset=0.0,
                       amp=0.015, title="Surface Temperature"):
    """Front face in black; T(s) drawn as a colored ribbon offset to one side."""
    front_segs = _front_segments(pts, lines, n_arc_lines)
    back_segs = _back_segments(pts, lines, n_arc_lines)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_back(ax, back_segs)
    _draw_front_black(ax, front_segs)

    _, _, nx, ny = _wall_normal(x_mid, y_mid)

    T0 = float(np.min(T_plot))
    dT = float(np.max(T_plot) - T0) or 1.0
    offset = side * (base_offset + amp * (T_plot - T0) / dT)
    x_prof = x_mid + offset * nx
    y_prof = y_mid + offset * ny

    norm = mpl.colors.Normalize(vmin=float(np.min(T_plot)),
                                vmax=float(np.max(T_plot)))
    prof_segs = [[(x_prof[i], y_prof[i]), (x_prof[i+1], y_prof[i+1])]
                 for i in range(len(x_prof) - 1)]
    lc = LineCollection(prof_segs, cmap=cmap, norm=norm, linewidths=3.2, zorder=4)
    lc.set_array(0.5 * (T_plot[:-1] + T_plot[1:]))
    ax.add_collection(lc)

    for i in range(0, len(x_mid), max(1, len(x_mid) // 24)):
        ax.plot([x_mid[i], x_prof[i]], [y_mid[i], y_prof[i]],
                color="0.55", alpha=0.35, lw=0.9, zorder=2)

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label(r"$T_{\mathrm{surf}}$ ($^\circ$C)", fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.8, alpha=0.25)
    ax.tick_params(width=1.1, length=5)
    plt.tight_layout()
    return fig, ax


def _best_ordered_b2_polys(polys, r_center, z_center, nx, ny):
    """Choose polygon ordering that best matches stored B2 cell centers."""
    polys = np.asarray(polys, dtype=float)
    centers = np.column_stack([np.asarray(r_center, dtype=float), np.asarray(z_center, dtype=float)])

    candidates = []
    candidates.append(polys)
    ncell = (nx + 2) * (ny + 2)
    if polys.shape[0] == ncell:
        candidates.append(polys.reshape(nx + 2, ny + 2, 4, 2, order='C').reshape(-1, 4, 2, order='F'))
        candidates.append(polys.reshape(nx + 2, ny + 2, 4, 2, order='F').reshape(-1, 4, 2, order='C'))

    best = candidates[0]
    best_score = np.inf
    for cand in candidates:
        cent = np.nanmean(cand, axis=1)
        score = float(np.nanmedian(np.hypot(cent[:, 0] - centers[:, 0], cent[:, 1] - centers[:, 1])))
        if score < best_score:
            best = cand
            best_score = score
    return best


def _poly_panel(ax, polys, values, title, cmap, *, log=False, symmetric=False):
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    vals = np.asarray(values, dtype=float).reshape(-1)
    valid_poly = np.all(np.isfinite(polys), axis=(1, 2))
    vals = vals[valid_poly]
    poly_use = polys[valid_poly]

    if log:
        vals = np.where(vals > 0.0, vals, np.nan)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([1.0])
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if not np.isfinite(vmin) or vmin <= 0.0:
            vmin = max(float(np.nanmin(finite)), 1e-30)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif symmetric:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([0.0])
        vlim = float(np.nanpercentile(np.abs(finite), 99))
        if not np.isfinite(vlim) or vlim <= 0.0:
            vlim = 1.0
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    else:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([0.0])
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

    coll = PolyCollection(poly_use, array=vals, cmap=cmap, norm=norm,
                          edgecolors='none', linewidths=0.0, antialiased=False)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'$R$ (m)')
    ax.set_ylabel(r'$Z$ (m)')
    return coll


def plot_plasma_h5_b2(h5_path, pts=None, lines=None, n_arc_lines=None,
                      zoom=None, with_psi=False, save=None):
    """Plot plasma.h5 fields on the native B2 quadrilateral grid."""
    import h5py

    with h5py.File(h5_path, 'r') as f:
        polys = f['/b2/cell_polygons'][:]
        nx = int(f['/b2/nx'][()])
        ny = int(f['/b2/ny'][()])
        rc = f['/b2/r_center'][:]
        zc = f['/b2/z_center'][:]
        polys = _best_ordered_b2_polys(polys, rc, zc, nx, ny)

        ne = f['/b2/dens_e'][:]
        te = f['/b2/temp_e'][:]
        ti = f['/b2/temp_i'][:]
        upar = f['/b2/parr_flow'][:]
        er = f['/b2/e_r'][:]
        ez = f['/b2/e_z'][:]
        et = f['/b2/e_t'][:]
        bmag = f['/b2/b_mag'][:]
        eq = None
        if with_psi and '/equilibrium/psi' in f:
            eq_r = f['/equilibrium/r'][:]
            eq_z = f['/equilibrium/z'][:]
            eq_psi = f['/equilibrium/psi'][:]
            if eq_psi.shape == (len(eq_r), len(eq_z)):
                eq_psi_plot = eq_psi.T
            elif eq_psi.shape == (len(eq_z), len(eq_r)):
                eq_psi_plot = eq_psi
            else:
                raise ValueError(
                    f'Unexpected equilibrium/psi shape {eq_psi.shape} for r={len(eq_r)}, z={len(eq_z)}'
                )
            eq = (eq_r, eq_z, eq_psi_plot)

    emag = np.sqrt(er * er + ez * ez + et * et)
    panels = [
        ('ne', ne, True,  r'$n_e$ (m$^{-3}$)', 'viridis', False),
        ('Te', te, True,  r'$T_e$ (eV)', 'inferno', False),
        ('Ti', ti, True,  r'$T_i$ (eV)', 'inferno', False),
        ('upar', upar, False, r'$u_\parallel$ (m s$^{-1}$)', 'RdBu_r', True),
        ('Emag', emag, False, r'$|E|$ (V m$^{-1}$)', 'magma', False),
        ('Bmag', bmag, False, r'$|B|$ (T)', 'plasma', False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (_, val, log, label, cmap, symmetric) in zip(axes.flat, panels):
        coll = _poly_panel(ax, polys, val, label, cmap, log=log, symmetric=symmetric)
        if eq is not None:
            ax.contour(eq[0], eq[1], eq[2], levels=14, colors='white', linewidths=0.4, alpha=0.4)
        if pts is not None and lines is not None and n_arc_lines is not None:
            front = _front_segments(pts, lines, n_arc_lines)
            back = _back_segments(pts, lines, n_arc_lines)
            _draw_back(ax, back, color='0.85', lw=1.0, zorder=4)
            _draw_front_black(ax, front, lw=2.0, zorder=5)
        if zoom is not None:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
        cb = fig.colorbar(coll, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    return fig, axes


def read_surf_flux(path_glob, columns=("c_cevap", "c_cadat", "c_cpmiLi")):
    """Parse SPARTA `dump surf` snapshots into time-averaged per-surf values.

    Returns: (sids, dict[col -> ndarray])
      sids[k]   = surf id of segment k (sorted)
      data[col] = time-mean of column `col` for segment k
    """
    import glob
    snaps = []
    for fp in sorted(glob.glob(str(path_glob))):
        with open(fp) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].startswith("ITEM: TIMESTEP"):
                i += 1; continue
            i += 2  # step value
            assert lines[i].startswith("ITEM: NUMBER OF SURFS")
            n = int(lines[i+1]); i += 2
            assert lines[i].startswith("ITEM: BOX BOUNDS")
            i += 4
            assert lines[i].startswith("ITEM: SURFS")
            cols = lines[i].split()[2:]
            i += 1
            arr = np.array([lines[i+k].split() for k in range(n)], dtype=float)
            i += n
            snaps.append((cols, arr))

    if not snaps:
        return np.array([]), {c: np.array([]) for c in columns}
    cols = snaps[0][0]
    id_col = cols.index("id")
    # collect per-id rows across snapshots, then mean
    by_id = {}
    for _, arr in snaps:
        for row in arr:
            sid = int(row[id_col])
            by_id.setdefault(sid, []).append(row)
    sids = np.array(sorted(by_id))
    data = {}
    for c in columns:
        if c not in cols:
            continue
        ci = cols.index(c)
        data[c] = np.array([np.mean([r[ci] for r in by_id[s]]) for s in sids])
    return sids, data


def plot_oedge_fluxes_1d(s_seg, fluxes, ax=None,
                         title="OpenEdge surface sources"):
    """1D semilogy plot of per-segment fluxes vs arc length, one line each.

    s_seg   : arc-length s of each segment midpoint, shape (N,)
    fluxes  : dict {label: array(N,)}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    # SPARTA surf IDs aren't in arc-length order in general (especially
    # under MPI), so sort to keep the line monotonic in s.
    s_seg = np.asarray(s_seg, float)
    order = np.argsort(s_seg)
    s_seg = s_seg[order]
    colors = ["firebrick", "navy", "darkgreen", "darkorange", "purple"]
    for (label, flux), c in zip(fluxes.items(), colors):
        flux = np.asarray(flux, float)[order]
        ax.semilogy(s_seg, np.where(flux > 0, flux, np.nan),
                    "-", lw=2, color=c, label=label)
    ax.set_xlabel(r"arc length $s$ (m)")
    ax.set_ylabel(r"Li flux (atoms m$^{-2}$ s$^{-1}$)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    return ax


def plot_oedge_flux_on_wall(pts, lines, n_arc_lines, x_mid, y_mid, fluxes,
                            cmap="plasma", side=-1.0, base_offset=0.010,
                            amp=0.012, title="OpenEdge surface source",
                            label=r"$\Gamma$ (atoms m$^{-2}$ s$^{-1}$)"):
    """Same ribbon-on-wall layout as `plot_total_flux_on_wall`, but for an
    arbitrary per-segment flux array measured in the simulation."""
    flux = np.asarray(fluxes, float)
    pos = flux[flux > 0]
    if pos.size == 0:
        raise ValueError("flux has no positive values")
    floor = pos.min()
    F = np.clip(flux, floor, None)

    front_segs = _front_segments(pts, lines, n_arc_lines)
    back_segs  = _back_segments(pts, lines, n_arc_lines)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_back(ax, back_segs)
    _draw_front_black(ax, front_segs)

    _, _, nx, ny = _wall_normal(x_mid, y_mid)
    logF = np.log10(F)
    g0 = logF.min(); dg = logF.max() - g0
    shape = np.zeros_like(logF) if dg == 0 else (logF - g0) / dg
    offset = side * (base_offset + amp * shape)
    x_prof = x_mid + offset * nx
    y_prof = y_mid + offset * ny

    norm = mpl.colors.LogNorm(vmin=floor, vmax=F.max())
    prof_segs = [[(x_prof[i], y_prof[i]), (x_prof[i+1], y_prof[i+1])]
                 for i in range(len(x_prof) - 1)]
    lc = LineCollection(prof_segs, cmap=cmap, norm=norm, linewidths=3.8, zorder=5)
    lc.set_array(np.sqrt(F[:-1] * F[1:]))
    ax.add_collection(lc)

    for i in range(0, len(x_mid), max(1, len(x_mid) // 24)):
        ax.plot([x_mid[i], x_prof[i]], [y_mid[i], y_prof[i]],
                color="0.55", alpha=0.25, lw=0.8, zorder=2)

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$R$ (m)"); ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.8, alpha=0.25)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, ax


def read_trajectories(dump_path):
    """Parse a SPARTA `dump particle` text file into per-id trajectories.

    Returns: (cols, trajs) where
      cols  : list of column names from the ITEM: ATOMS header (e.g. 'id','x','y','radius','temp')
      trajs : dict id -> structured numpy array with one row per snapshot, fields = cols + 'step'
    """
    snapshots = []  # list of (step, cols, ndarray)
    with open(dump_path) as f:
        while True:
            tag = f.readline()
            if not tag:
                break
            if not tag.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            n = int(f.readline())
            assert f.readline().startswith("ITEM: BOX BOUNDS")
            for _ in range(3):
                f.readline()
            header = f.readline().strip()
            assert header.startswith("ITEM: ATOMS")
            cols = header.split()[2:]
            rows = [f.readline().split() for _ in range(n)]
            arr = np.array(rows, dtype=float) if rows else np.zeros((0, len(cols)))
            snapshots.append((step, cols, arr))

    if not snapshots:
        return [], {}

    cols = snapshots[0][1]
    id_col = cols.index("id")
    by_id = {}  # id -> list of (step, row)
    for step, _, arr in snapshots:
        for row in arr:
            pid = int(row[id_col])
            by_id.setdefault(pid, []).append((step, row))

    fields = cols + ["step"]
    trajs = {}
    for pid, lst in by_id.items():
        lst.sort(key=lambda sr: sr[0])
        out = np.zeros(len(lst), dtype=[(c, float) for c in fields])
        for k, (step, row) in enumerate(lst):
            for j, c in enumerate(cols):
                out[c][k] = row[j]
            out["step"][k] = step
        trajs[pid] = out
    return cols, trajs


def plot_trajectories_on_wall(pts, lines, n_arc_lines, trajs,
                              color_by="radius", min_len=2,
                              cmap="viridis", lw=0.9, alpha=0.85,
                              show_radius_dots=True, dot_every=5,
                              dot_scale=4e5, title="Droplet trajectories",
                              axisymmetric=False):
    """Overlay particle trajectories on the slab geometry.

    color_by : 'radius' (shrinkage) or 'temp' (heating) — column name in the dump.
    show_radius_dots : draw circles whose area scales with droplet radius along
                       each trajectory every `dot_every` snapshots.
    dot_scale : marker-area = dot_scale * radius (m). Tune visually.
    axisymmetric : interpret dump and wall coordinates as (Z, R).
    """
    front_segs = _front_segments(pts, lines, n_arc_lines)
    back_segs = _back_segments(pts, lines, n_arc_lines)
    if axisymmetric:
        front_segs = [[(y, x) for x, y in segment] for segment in front_segs]
        back_segs = [[(y, x) for x, y in segment] for segment in back_segs]

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_back(ax, back_segs)
    _draw_front_black(ax, front_segs)

    valid = [t for t in trajs.values() if len(t) >= min_len]
    if not valid:
        ax.text(0.5, 0.5, "no trajectories", transform=ax.transAxes, ha="center")
        return fig, ax

    vals = np.concatenate([t[color_by] for t in valid])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmax == vmin:
        vmax = vmin + 1e-30
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    all_segs, all_c = [], []
    for t in valid:
        x, y = (t["y"], t["x"]) if axisymmetric else (t["x"], t["y"])
        c = t[color_by]
        for i in range(len(t) - 1):
            all_segs.append([(x[i], y[i]), (x[i+1], y[i+1])])
            all_c.append(0.5 * (c[i] + c[i+1]))
    lc = LineCollection(all_segs, cmap=cmap, norm=norm, linewidths=lw,
                        alpha=alpha, zorder=4)
    lc.set_array(np.asarray(all_c))
    ax.add_collection(lc)

    if show_radius_dots and "radius" in valid[0].dtype.names:
        xs, ys, rs, cs = [], [], [], []
        for t in valid:
            idx = np.arange(0, len(t), dot_every)
            x, y = (t["y"], t["x"]) if axisymmetric else (t["x"], t["y"])
            xs.append(x[idx]); ys.append(y[idx])
            rs.append(t["radius"][idx]); cs.append(t[color_by][idx])
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        rs = np.concatenate(rs); cs = np.concatenate(cs)
        sizes = np.maximum(dot_scale * rs, 2.0)
        ax.scatter(xs, ys, s=sizes, c=cs, cmap=cmap, norm=norm,
                   edgecolors="k", linewidths=0.3, alpha=0.9, zorder=5)

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    label = {"radius": r"droplet radius (m)",
             "temp":   r"droplet $T$ (K)"}.get(color_by, color_by)
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.8, alpha=0.25)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, ax


def plot_total_flux_on_wall(pts, lines, n_arc_lines, x_mid, y_mid, G_evap, G_adat,
                            cmap="plasma", side=-1.0, base_offset=0.010,
                            amp=0.012, title="Total Eroded Flux"):
    """Front face in black; (G_evap + G_adat)(s) as a log-colored ribbon."""
    G_total = np.asarray(G_evap, float) + np.asarray(G_adat, float)
    pos = G_total[G_total > 0]
    if pos.size == 0:
        raise ValueError("Gamma_total has no positive values")
    floor = pos.min()
    G_safe = np.clip(G_total, floor, None)

    front_segs = _front_segments(pts, lines, n_arc_lines)
    back_segs = _back_segments(pts, lines, n_arc_lines)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_back(ax, back_segs)
    _draw_front_black(ax, front_segs)

    _, _, nx, ny = _wall_normal(x_mid, y_mid)

    logG = np.log10(G_safe)
    g0 = logG.min()
    dg = logG.max() - g0
    shape = np.zeros_like(logG) if dg == 0 else (logG - g0) / dg
    offset = side * (base_offset + amp * shape)
    x_prof = x_mid + offset * nx
    y_prof = y_mid + offset * ny

    norm = mpl.colors.LogNorm(vmin=floor, vmax=G_safe.max())
    prof_segs = [[(x_prof[i], y_prof[i]), (x_prof[i+1], y_prof[i+1])]
                 for i in range(len(x_prof) - 1)]
    lc = LineCollection(prof_segs, cmap=cmap, norm=norm, linewidths=3.8, zorder=5)
    lc.set_array(np.sqrt(G_safe[:-1] * G_safe[1:]))
    ax.add_collection(lc)

    for i in range(0, len(x_mid), max(1, len(x_mid) // 24)):
        ax.plot([x_mid[i], x_prof[i]], [y_mid[i], y_prof[i]],
                color="0.55", alpha=0.25, lw=0.8, zorder=2)

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label(r"$\Gamma_\mathrm{evap} + \Gamma_\mathrm{adatom}$ "
                   r"(atoms m$^{-2}$ s$^{-1}$)", fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.8, alpha=0.25)
    ax.tick_params(width=1.1, length=5)
    plt.tight_layout()
    return fig, ax
