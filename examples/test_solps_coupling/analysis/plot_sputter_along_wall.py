#!/usr/bin/env python3
"""Plot Li erosion flux on the lower divertor vs (L - L_sep). The lower
divertor has two legs (inner small-R, outer large-R) sharing one
continuous arc coordinate. The strike per leg is the segment with the
smallest |psi - psib| within that R-half (PFR and SOL share the same
sign of psi-psib, so a sign-flip finder does not work).

Two subplots: inner leg on the left, outer leg on the right.

Reads:
  ../coupled_test/openedge/sputter.dump   (surf-style SPARTA dump, last frame)
  ../coupled_test/openedge/plasma.h5      (/equilibrium/{r, z, psi, psib})
  ../input/wall.surf                       (axi: line endpoints in (Z, R))

Usage: python3 plot_sputter_along_wall.py [sputter.dump]
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator


def parse_wall(path):
    text = Path(path).read_text().splitlines()
    pts, lns, sec = {}, [], None
    for line in text:
        s = line.strip()
        if s == "Points": sec = "p"; continue
        if s == "Lines":  sec = "l"; continue
        if not s or not s[0].isdigit(): continue
        tok = s.split()
        if sec == "p":
            pts[int(tok[0])] = (float(tok[1]), float(tok[2]))   # (Z, R)
        elif sec == "l":
            lns.append((int(tok[0]), int(tok[1]), int(tok[2])))
    return pts, lns


def parse_surf_dump_last_frame(path):
    text = Path(path).read_text().splitlines()
    frames, cur = [], []
    for line in text:
        if line.startswith("ITEM: TIMESTEP") and cur:
            frames.append(cur); cur = []
        cur.append(line)
    if cur:
        frames.append(cur)
    if not frames:
        return {}, None
    last = frames[-1]
    out, t, in_data, col = {}, None, False, {}
    for i, line in enumerate(last):
        s = line.strip()
        if s.startswith("ITEM: TIMESTEP"):
            t = int(last[i + 1]); continue
        if s.startswith("ITEM: SURFS") or s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            col = {k: j for j, k in enumerate(cols)}
            in_data = True; continue
        if s.startswith("ITEM:"):
            in_data = False; continue
        if in_data and s and s[0].isdigit():
            tok = s.split()
            sid = int(tok[col["id"]])
            out[sid] = float(tok[-1])
    return out, t


def build_strip(div_ids, flux, lns, pts):
    """Order segments by ID (≈ arc order), compute centroid + arc length."""
    seg = []
    for sid in sorted(div_ids):
        pid_a, pid_b = next((p1, p2) for sidx, p1, p2 in lns if sidx == sid)
        za, ra = pts[pid_a]; zb, rb = pts[pid_b]
        zc, rc = 0.5 * (za + zb), 0.5 * (ra + rb)
        L = float(np.hypot(zb - za, rb - ra))
        seg.append((sid, zc, rc, L, flux[sid]))
    arc = np.zeros(len(seg))
    for k in range(1, len(seg)):
        arc[k] = arc[k - 1] + 0.5 * (seg[k - 1][3] + seg[k][3])
    return np.array([s[0] for s in seg]), \
           np.array([s[1] for s in seg]), \
           np.array([s[2] for s in seg]), \
           arc, \
           np.array([s[4] for s in seg])


def find_strike_per_leg(arc, Rc, psi_minus_psib):
    """Split strip into inner-R / outer-R halves; in each, the strike is the
    segment with the smallest |psi - psib|. PFR and SOL are on the same side
    of psib (psi > psib in both for the standard convention) so a sign-flip
    finder fails — but each leg's |psi - psib| has a clear local minimum.

    Returns a dict with 'inner' / 'outer' keys, each holding (L_sep, R_sep,
    bool-mask of segments belonging to that leg).
    """
    # Split by R midpoint of the strip
    R_split = 0.5 * (Rc.min() + Rc.max())
    is_inner = Rc < R_split
    is_outer = ~is_inner

    abs_dpsi = np.abs(psi_minus_psib)
    out = {}
    for side, mask in [("inner", is_inner), ("outer", is_outer)]:
        if not mask.any():
            out[side] = None
            continue
        idx = np.where(mask)[0]
        k = idx[int(np.argmin(abs_dpsi[idx]))]
        out[side] = (float(arc[k]), float(Rc[k]), mask)
    return out


def main():
    base = Path(__file__).resolve().parent
    project = base.parent
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        project / "coupled_test" / "openedge" / "sputter.dump"
    plasma_h5 = project / "coupled_test" / "openedge" / "plasma.h5"
    wall = project / "input" / "wall.surf"
    if not dump.exists():
        sys.exit(f"no sputter dump at {dump}")
    if not plasma_h5.exists():
        sys.exit(f"no plasma.h5 at {plasma_h5} (need equilibrium for L_sep)")

    pts, lns = parse_wall(wall)
    flux, tstep = parse_surf_dump_last_frame(dump)
    if not flux:
        sys.exit(f"no surface records in {dump}")

    # Equilibrium psi(r, z) for psi-norm
    with h5py.File(plasma_h5, "r") as f:
        eq_r = f["equilibrium/r"][:]
        eq_z = f["equilibrium/z"][:]
        eq_psi = f["equilibrium/psi"][:]   # shape (nz, nr)
        psib = float(f["equilibrium/psib"][()])
    psi_interp = RegularGridInterpolator(
        (eq_z, eq_r), eq_psi, bounds_error=False, fill_value=np.nan)

    # Lower divertor only (the only Li-coated part of the wall).
    lower_ids = [s for s in flux
                 if pts[next(p1 for sid, p1, p2 in lns if sid == s)][0] < 0]
    if not lower_ids:
        sys.exit("no lower-divertor segments in dump (all Z >= 0)")

    # Independent y-axes — inner and outer can differ by 5+ decades
    # (different Te / Γ regimes), so sharey hides the smaller subplot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    ax_inner, ax_outer = axes[0], axes[1]
    ax_inner.set_title("Inner leg")
    ax_outer.set_title("Outer leg")
    for ax in axes:
        ax.set_xlabel(r"$L - L_{\rm sep}$ (m)")
        ax.set_ylabel(r"$\Gamma$ (atoms/m$^2$/s)")
        ax.grid(alpha=0.3)
        ax.axvline(0, color="0.5", lw=0.7, ls="--")

    sids, Zc, Rc, arc, gamma = build_strip(lower_ids, flux, lns, pts)
    psi_at = psi_interp(np.column_stack([Zc, Rc]))
    psi_minus_psib = psi_at - psib

    strikes = find_strike_per_leg(arc, Rc, psi_minus_psib)

    # Optional overlay of SOLPS b2plot target output + predicted-sputter
    # curve (D_flux_ion x Eckstein D_on_Li yield(E, theta) using SOLPS Te
    # / Ti / Bangle, matching OpenEdge compute surface/physical/sputter).
    b2pl_dir = project / "coupled_test" / "solps" / "v0_1" / "b2pl.exe.dir"
    b2fgmtry_mat = project / "coupled_test" / "solps" / "v0_1" / "b2fgmtry.mat"
    solps_data = {}
    pol_to_prl = {}     # ix-major pol-to-parallel ratio per target iy
    if b2pl_dir.exists():
        try:
            from read_solps_target import read_target
            for tgt in ("inner", "outer"):
                try:
                    solps_data[tgt] = read_target(b2pl_dir, tgt)
                except (FileNotFoundError, ValueError):
                    pass
        except ImportError:
            pass

    # B-from-vertex sampler matching what OpenEdge's compute uses
    # (plasma.h5 mesh/vtx_b{r,z,t}). Avoids the b2fgmtry.mat / pol_to_prl
    # path which gives a slightly different angle and, at grazing
    # incidence, a yield that's wrong by orders of magnitude.
    plasma_h5_path = project / "coupled_test" / "openedge" / "plasma.h5"
    sample_B_at = None
    if solps_data and plasma_h5_path.exists():
        try:
            import h5py as _h5
            from scipy.spatial import cKDTree as _KDTree
            with _h5.File(plasma_h5_path, 'r') as _f:
                _vr = _f['mesh/vtx_r'][:]; _vz = _f['mesh/vtx_z'][:]
                _tri = _f['mesh/triangles'][:]
                _Br = _f['mesh/vtx_br'][:]; _Bz = _f['mesh/vtx_bz'][:]
                _Bt = _f['mesh/vtx_bt'][:]
            _bary = np.column_stack([
                (_vr[_tri[:, 0]] + _vr[_tri[:, 1]] + _vr[_tri[:, 2]]) / 3.0,
                (_vz[_tri[:, 0]] + _vz[_tri[:, 1]] + _vz[_tri[:, 2]]) / 3.0])
            _tree = _KDTree(_bary)
            def sample_B_at(R_arr, Z_arr):
                _, idx = _tree.query(np.column_stack([R_arr, Z_arr]))
                v = _tri[idx]
                Br_ = (_Br[v[:, 0]] + _Br[v[:, 1]] + _Br[v[:, 2]]) / 3.0
                Bz_ = (_Bz[v[:, 0]] + _Bz[v[:, 1]] + _Bz[v[:, 2]]) / 3.0
                Bt_ = (_Bt[v[:, 0]] + _Bt[v[:, 1]] + _Bt[v[:, 2]]) / 3.0
                return Br_, Bz_, Bt_
        except Exception as e:
            print(f"  [vtx_B sampler] {e}")

    # KDTree over divertor wall-segment centroids so we can pull each
    # SOLPS-target row's (Rclfc,Zclfc) onto the nearest wall.surf normal.
    seg_normals = None
    if pts and lns:
        seg_records = []      # (sid, Rc, Zc, n_R, n_Z)
        for sid, p1, p2 in lns:
            z1, r1 = pts[p1]; z2, r2 = pts[p2]
            zc, rc = 0.5 * (z1 + z2), 0.5 * (r1 + r2)
            tan_z, tan_r = z2 - z1, r2 - r1
            nlen = np.hypot(tan_z, tan_r)
            if nlen <= 0:
                continue
            n_r = -tan_z / nlen
            n_z =  tan_r / nlen
            seg_records.append((sid, rc, zc, n_r, n_z))
        seg_records = np.array(seg_records, dtype=float)
        if seg_records.size:
            from scipy.spatial import cKDTree as _KDTree2
            seg_tree = _KDTree2(seg_records[:, [1, 2]])
            seg_normals = (seg_records, seg_tree)

    def vertex_B_theta_deg(R_arr, Z_arr):
        """Return θ_from_normal (deg) at each (R, Z) using plasma.h5 vertex
        B and wall.surf nearest-segment normals (matches OE pipeline)."""
        if sample_B_at is None or seg_normals is None:
            return np.zeros_like(R_arr)
        Br, Bz, Bt = sample_B_at(np.asarray(R_arr), np.asarray(Z_arr))
        Bmag = np.sqrt(Br**2 + Bz**2 + Bt**2)
        recs, tree = seg_normals
        _, idx = tree.query(np.column_stack([R_arr, Z_arr]))
        n_r = recs[idx, 3]; n_z = recs[idx, 4]
        # B has no toroidal component along an axisymmetric wall normal,
        # so |B.n| = |Br*n_r + Bz*n_z|.
        sin_alpha = np.abs(Br * n_r + Bz * n_z) / np.maximum(Bmag, 1e-30)
        sin_alpha = np.clip(sin_alpha, 0.0, 1.0)
        return 90.0 - np.degrees(np.arcsin(sin_alpha))

    AMU = 1.66053906660e-27
    ME = 9.1093837015e-31

    def eckstein_d_on_li_yield(E_eV_arr, theta_deg_arr,
                                Q=0.16, Eth=6.92, ETF=209.0):
        """Vectorised Bohdansky-1993 + Yamamura angular for D-on-Li.
        Matches OpenEdge eckstein_sputter.h: sputter_yield()."""
        E = np.asarray(E_eV_arr, dtype=float)
        th = np.clip(np.asarray(theta_deg_arr, dtype=float), 0, 89.5)
        c = np.maximum(np.cos(np.radians(th)), 1e-3)
        # Bohdansky normal-incidence yield
        Y0 = np.zeros_like(E)
        ok = E > Eth
        if ok.any():
            r = Eth / E[ok]
            eps = E[ok] / ETF
            f1 = Q * (1 - r**(2.0/3.0)) * (1 - r)**2
            num = 0.5 * np.log1p(1.2288 * eps)
            den = eps + 0.1728 * np.sqrt(eps) + 0.008 * eps**0.1504
            se = num / den
            Y0[ok] = np.maximum(f1 * se, 0.0)
        # Yamamura angular factor
        arg = -2.0 * np.log(c) + 2.0 * (1.0 - 1.0 / c) * 0.26
        arg = np.clip(arg, -500, 500)
        Y = Y0 * np.exp(arg)
        return np.maximum(Y, 0.0)

    def impact_energy_eV(Te, Ti, Z_proj=1, mi_bg_amu=2.014):
        """Chankin 2014 sheath formula, matching compute_surface_physical_sputter."""
        Te = np.maximum(Te, 1e-30)
        ti_ratio = Ti / Te
        mi_me = (mi_bg_amu * AMU) / ME
        psi_over_Te = 0.5 * np.log(mi_me / (2.0 * np.pi * (1.0 + ti_ratio)))
        return 2.0 * Ti + Z_proj * psi_over_Te * Te

    summary = []
    for side, ax in [("inner", ax_inner), ("outer", ax_outer)]:
        info = strikes[side]
        if info is None:
            ax.text(0.5, 0.5, f"no segments in {side} half",
                    transform=ax.transAxes, ha="center", va="center")
            continue
        L_sep, R_sep, mask = info
        ax.plot(arc[mask] - L_sep, gamma[mask], "-o", lw=1.5, ms=4,
                color="C3", label=r"OpenEdge $\Gamma_{\rm Li}$ (sputter)")
        gamma_pred_max = None
        if side in solps_data:
            sd = solps_data[side]
            x = sd['x']
            # SOLPS Li-ion impact flux (different physics — for reference).
            gli = sd.get('Li_flux_ion')
            if gli is not None:
                pos = gli > 0
                #ax.plot(x[pos], gli[pos], "--", lw=1.0, color="C0", alpha=0.6,
                #        label=r"SOLPS $\Gamma_{\rm Li^+}$ deposition")
            # Predicted sputter from SOLPS data: D_flux_ion x Y(D->Li, E, theta).
            # Use plasma values sampled from plasma.h5 at the SOLPS-target
            # (Rclfc, Zclfc) — same cells OE samples — so the curve is
            # apples-to-apples vs OE's red. (Previously used SOLPS-target-file
            # Te which sits at a slightly different cell, causing ~25x gap.)
            d_flux = sd.get('D_flux_ion')
            Te = sd.get('Te'); Ti = sd.get('Ti')
            if 'Rclfc' in sd and sample_B_at is not None:
                # Re-sample Te / ni from plasma.h5 at the SAME (Rclfc, Zclfc)
                # the OE compute would have used.
                try:
                    import h5py as _h5b
                    from scipy.spatial import cKDTree as _KD
                    with _h5b.File(plasma_h5_path, 'r') as _ff:
                        _vr2 = _ff['mesh/vtx_r'][:]; _vz2 = _ff['mesh/vtx_z'][:]
                        _tri2 = _ff['mesh/triangles'][:]
                        _ci2 = _ff['mesh/cell_index'][:]
                        _Te = _ff['mesh/temp_e'][:]
                        _Ni = _ff['mesh/dens_i'][:]
                    _bary2 = np.column_stack([
                        (_vr2[_tri2[:, 0]] + _vr2[_tri2[:, 1]] + _vr2[_tri2[:, 2]]) / 3,
                        (_vz2[_tri2[:, 0]] + _vz2[_tri2[:, 1]] + _vz2[_tri2[:, 2]]) / 3])
                    _tk = _KD(_bary2)
                    _, _idx = _tk.query(np.column_stack([sd['Rclfc'], sd['Zclfc']]))
                    _ce = _ci2[_idx]
                    Te = _Te[_ce]
                    Ti = _Te[_ce]            # plasma.h5 has only Te
                    ni_oe = _Ni[_ce]
                    cs = np.sqrt(np.maximum((Te + Ti) * 1.602176634e-19 /
                                              (2.014 * AMU), 0.0))
                    # sin_alpha already encoded via theta below; for Bohm
                    # flux we need it explicitly. Recompute here using same
                    # vertex-B sampler that drives theta.
                    _Br_, _Bz_, _Bt_ = sample_B_at(np.asarray(sd['Rclfc']),
                                                    np.asarray(sd['Zclfc']))
                    _Bm = np.sqrt(_Br_**2 + _Bz_**2 + _Bt_**2)
                    _recs, _t = seg_normals
                    _, _ii = _t.query(np.column_stack([sd['Rclfc'], sd['Zclfc']]))
                    n_r = _recs[_ii, 3]; n_z = _recs[_ii, 4]
                    sin_a = np.abs(_Br_*n_r + _Bz_*n_z) / np.maximum(_Bm, 1e-30)
                    sin_a = np.clip(sin_a, 0, 1)
                    d_flux = ni_oe * cs * sin_a
                except Exception as _e:
                    print(f"  [plasma.h5 re-sample] {_e}; using SOLPS-file values")
            if d_flux is not None and Te is not None and Ti is not None:
                E_imp = impact_energy_eV(Te, Ti)
                # theta = angle of incidence from surface normal.
                # Bangle = 90 - asin(1/pol_to_prl) is the B-target angle from
                # the target plane; the angle from the surface normal is then
                # 90 - Bangle = asin(1/pol_to_prl).
                # Angle of incidence — sample plasma.h5 vertex B at the
                # SOLPS target (Rclfc, Zclfc) and project onto the
                # wall.surf segment normal nearest each row. This is
                # exactly the geometry OpenEdge's compute uses internally,
                # so the predicted curve becomes apples-to-apples vs the
                # OE red curve. Falls back to b2fgmtry.mat pol_to_prl if
                # the vertex-B path isn't available.
                if 'Rclfc' in sd and 'Zclfc' in sd and sample_B_at is not None:
                    theta = vertex_B_theta_deg(sd['Rclfc'], sd['Zclfc'])
                elif side in pol_to_prl and pol_to_prl[side].size == Te.size:
                    pp = pol_to_prl[side]
                    theta = 90.0 - np.degrees(np.arcsin(np.clip(1.0/pp, 0, 1)))
                else:
                    theta = np.zeros_like(Te)
                Y = eckstein_d_on_li_yield(E_imp, theta)
                gamma_pred = d_flux * Y
                pos = gamma_pred > 0
                #if pos.any():
                    #ax.plot(x[pos], gamma_pred[pos], "--", lw=1.8, color="C2",
                     #       label=r"SOLPS-predicted $\Gamma_{\rm Li}$ "
                     #             r"($\Gamma_{\rm D^+}{\cdot}Y_{\rm D{\to}Li}$)")
                gamma_pred_max = gamma_pred.max() if gamma_pred.size else 0.0
        # Annotate the panel with the physical reason for zero sputter,
        # if applicable (Te everywhere below D-on-Li threshold).
        if gamma[mask].max() == 0 and side in solps_data:
            sd = solps_data[side]
            te_max = float(sd['Te'].max())
            ax.text(0.04, 0.96,
                    f"$\\Gamma_{{\\rm sputter}}\\equiv 0$:  "
                    f"max $T_e$ = {te_max:.2f} eV "
                    f"$<E_{{\\rm th}}({{\\rm D{{\\to}}Li}})=6.92$ eV",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
        ax.legend(fontsize=7, loc="best")
        line = (f"  {side} leg:  L_sep = {L_sep:.3f} m  (at R = {R_sep:.3f} m)  "
                f"max Γ_OE = {gamma[mask].max():.3e} atoms/m^2/s")
        if gamma_pred_max is not None:
            line += f"   max Γ_pred = {gamma_pred_max:.3e}"
        summary.append(line)

    if gamma.max() > 0:
        ax_inner.set_yscale("log")
    fig.suptitle(f"sputter at step {tstep}  ({len(sids)} divertor segs)", y=1.0)

    out = base / "sputter_along_wall.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
