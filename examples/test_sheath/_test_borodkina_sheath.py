#!/usr/bin/env python3
"""
Standalone Borodkina-style sheath model test.

This script is for fast model verification before wiring sheath physics into
OpenEdge runtime code.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
AMU = 1.66053906660e-27
ME = 9.1093837015e-31


def fd_poly_deg(alpha_deg: float) -> float:
    a = float(alpha_deg)
    fd = (
        0.98992
        + 5.1220e-3 * a
        - 7.0040e-4 * a * a
        + 3.3591e-5 * a**3
        - 8.2917e-7 * a**4
        + 9.5856e-9 * a**5
        - 4.2682e-11 * a**6
    )
    return float(np.clip(fd, 0.0, 1.0))


def angle_to_normal_rad(alpha_deg: float, angle_def: str) -> float:
    if angle_def == "normal":
        return math.radians(alpha_deg)
    if angle_def == "surface":
        return math.radians(90.0 - alpha_deg)
    raise ValueError(f"Unsupported angle_def={angle_def}")


def plasma_scales(
    te_eV: float,
    ti_eV: float,
    ne_m3: float,
    b_T: float,
    mi_amu: float,
) -> tuple[float, float, float, float]:
    mi_kg = mi_amu * AMU
    lambda_D = math.sqrt(EPS0 * te_eV / (max(ne_m3, 1e-99) * QE))
    # User-requested sheath sound speed for D:
    # c_s = sqrt((Te + Ti) * e / (2 m_D))
    cs = math.sqrt(max(te_eV + ti_eV, 0.0) * QE / max(2.0 * mi_kg, 1e-99))
    omega_ci = QE * b_T / max(mi_kg, 1e-99)
    rho_i = cs / max(abs(omega_ci), 1e-99)
    return lambda_D, cs, omega_ci, rho_i


def borodkina_profile(
    d_m: np.ndarray,
    te_eV: float,
    ti_eV: float,
    ne0_m3: float,
    b_T: float,
    alpha_deg: float,
    mi_amu: float,
    angle_def: str,
    pot_mult: float,
) -> dict[str, np.ndarray | float]:
    d = np.maximum(np.asarray(d_m, dtype=float), 0.0)
    lambda_D, cs, omega_ci, rho_i = plasma_scales(te_eV, ti_eV, ne0_m3, b_T, mi_amu)
    # Keep Debye length fixed at sheath entrance values.
    lambda_D_se = lambda_D
    alpha_n = angle_to_normal_rad(alpha_deg, angle_def)
    sin_a = abs(math.sin(alpha_n))
    sin_floor = 0.05
    l_mps = rho_i / max(sin_a, sin_floor)

    fd = fd_poly_deg(alpha_deg)
    phi0 = pot_mult * te_eV
    e_ds = np.exp(-d / max(2.0 * lambda_D_se, 1e-99))
    e_mps = np.exp(-d / max(l_mps, 1e-99))

    phi = -phi0 * (fd * e_ds + (1.0 - fd) * e_mps)
    e_ds_vpm = phi0 * fd * e_ds / max(2.0 * lambda_D_se, 1e-99)
    e_mps_vpm = phi0 * (1.0 - fd) * e_mps / max(l_mps, 1e-99)
    emag = np.abs(e_ds_vpm + e_mps_vpm)

    x = np.clip(phi / max(te_eV, 1e-99), -100.0, 50.0)
    ne = ne0_m3 * np.exp(x)

    return {
        "phi": phi,
        "E": emag,
        "E_ds": e_ds_vpm,
        "E_mps": e_mps_vpm,
        "ne": ne,
        "lambda_D": lambda_D,
        "lambda_D_se": lambda_D_se,
        "cs": cs,
        "omega_ci": omega_ci,
        "rho_i": rho_i,
        "L_mps": l_mps,
        "alpha_n_rad": alpha_n,
    }


def estimate_entrance_from_phi(d: np.ndarray, phi: np.ndarray, te_eV: float, target_mult: float) -> float:
    phi_target = -target_mult * te_eV
    idx = int(np.argmin(np.abs(phi - phi_target)))
    return float(d[idx])


def check_chodura_bohm(u_par: float, cs: float, alpha_n_rad: float) -> dict[str, float | bool]:
    u_par_abs = abs(u_par)
    u_n = u_par_abs * abs(math.cos(alpha_n_rad))
    return {
        "u_par": u_par_abs,
        "u_n": u_n,
        "cs": cs,
        "chodura_ok": u_par_abs >= cs,
        "bohm_ok_from_par_projection": u_n >= cs,
    }


def _boundary_mask(valid: np.ndarray) -> np.ndarray:
    vp = np.pad(valid, 1, mode="constant", constant_values=False)
    c = vp[1:-1, 1:-1]
    n = vp[:-2, 1:-1]
    s = vp[2:, 1:-1]
    w = vp[1:-1, :-2]
    e = vp[1:-1, 2:]
    return c & (~(n & s & w & e))


def _dilate_4(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mp = np.pad(mask, 1, mode="constant", constant_values=False)
    c = mp[1:-1, 1:-1]
    n = mp[:-2, 1:-1]
    s = mp[2:, 1:-1]
    w = mp[1:-1, :-2]
    e = mp[1:-1, 2:]
    out = c | n | s | w | e
    return out & valid


def _near_wall_mask(valid: np.ndarray, layers: int) -> np.ndarray:
    layers_i = max(int(layers), 1)
    b = _boundary_mask(valid)
    acc = b.copy()
    for _ in range(layers_i - 1):
        acc = _dilate_4(acc, valid)
    return acc


def load_west_sheath_entrance(
    plasma_h5: Path,
    mi_amu: float,
    mach_min: float = 0.8,
    mach_max: float = 1.2,
    near_layers: int = 3,
) -> dict[str, float]:
    with h5py.File(plasma_h5, "r") as f:
        te = np.asarray(f["temp_e"][...], dtype=float)
        ti = np.asarray(f["temp_i"][...], dtype=float)
        ne = np.asarray(f["dens_e"][...], dtype=float)
        upar = np.asarray(f["parr_flow"][...], dtype=float)
        r = np.asarray(f["r"][...], dtype=float) if "r" in f else None
        z = np.asarray(f["z"][...], dtype=float) if "z" in f else None

    mi_kg = mi_amu * AMU
    cs = np.sqrt(np.maximum(te + ti, 0.0) * QE / max(2.0 * mi_kg, 1e-99))

    valid = np.isfinite(te) & np.isfinite(ti) & np.isfinite(ne) & np.isfinite(upar)
    valid &= (te > 0.0) & (ti >= 0.0) & (ne > 0.0) & (cs > 0.0)
    if not np.any(valid):
        raise RuntimeError(f"No valid plasma cells found in {plasma_h5}")

    mach = np.full_like(te, np.nan, dtype=float)
    mach[valid] = np.abs(upar[valid]) / cs[valid]
    near = _near_wall_mask(valid, near_layers)
    band = valid & near & (mach >= mach_min) & (mach <= mach_max)

    # Sheath-entrance estimate from masked candidates first.
    err = np.full_like(mach, np.inf, dtype=float)
    err[band] = np.abs(mach[band] - 1.0)
    source = "band"
    if not np.any(np.isfinite(err)):
        err[:] = np.inf
        err[near & valid] = np.abs(mach[near & valid] - 1.0)
        source = "near_wall"
    if not np.any(np.isfinite(err)):
        err[:] = np.inf
        err[valid] = np.abs(mach[valid] - 1.0)
        source = "global_valid"

    ij = np.unravel_index(int(np.argmin(err)), err.shape)
    i, j = int(ij[0]), int(ij[1])

    out = {
        "te_eV": float(te[i, j]),
        "ti_eV": float(ti[i, j]),
        "ne_m3": float(ne[i, j]),
        "u_par": float(upar[i, j]),
        "cs": float(cs[i, j]),
        "mach": float(mach[i, j]),
        "gamma_par": float(ne[i, j] * upar[i, j]),
        "valid_count": float(np.count_nonzero(valid)),
        "mach_ge_1_frac": float(np.count_nonzero((mach >= 1.0) & np.isfinite(mach)) / np.count_nonzero(valid)),
        "i": float(i),
        "j": float(j),
        "mach_min": float(mach_min),
        "mach_max": float(mach_max),
        "near_layers": float(near_layers),
        "selection_source": source,
        "near_count": float(np.count_nonzero(near)),
        "band_count": float(np.count_nonzero(band)),
    }

    if r is not None and z is not None and te.ndim == 2:
        out["R_m"] = float(r[j]) if j < r.size else float("nan")
        out["Z_m"] = float(z[i]) if i < z.size else float("nan")
    else:
        out["R_m"] = float("nan")
        out["Z_m"] = float("nan")
    return out


def plot_sheath_entrance_rz(
    plasma_h5: Path,
    mi_amu: float,
    entrance: dict[str, float],
    out_png: Path,
    mach_min: float,
    mach_max: float,
    near_layers: int,
) -> None:
    with h5py.File(plasma_h5, "r") as f:
        te = np.asarray(f["temp_e"][...], dtype=float)
        ti = np.asarray(f["temp_i"][...], dtype=float)
        ne = np.asarray(f["dens_e"][...], dtype=float)
        upar = np.asarray(f["parr_flow"][...], dtype=float)
        r = np.asarray(f["r"][...], dtype=float)
        z = np.asarray(f["z"][...], dtype=float)

    mi_kg = mi_amu * AMU
    cs = np.sqrt(np.maximum(te + ti, 0.0) * QE / max(2.0 * mi_kg, 1e-99))
    mach = np.full_like(te, np.nan, dtype=float)
    valid = np.isfinite(te) & np.isfinite(ti) & np.isfinite(ne) & np.isfinite(upar)
    valid &= (te > 0.0) & (ti >= 0.0) & (ne > 0.0) & (cs > 0.0)
    mach[valid] = np.abs(upar[valid]) / cs[valid]
    near = _near_wall_mask(valid, near_layers)
    band = valid & near & (mach >= mach_min) & (mach <= mach_max)

    rr, zz = np.meshgrid(r, z)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0), dpi=220, constrained_layout=True)
    cmap = "viridis"
    pcm = ax.pcolormesh(rr, zz, np.ma.masked_invalid(mach), shading="auto", cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax, pad=0.01)
    cb.set_label(r"$|u_{\parallel}|/c_s$")

    # Mach=1 contour approximates sheath-entrance locus.
    try:
        ax.contour(rr, zz, np.nan_to_num(mach, nan=np.nan), levels=[1.0], colors="white", linewidths=1.4)
    except Exception:
        pass
    # Near-wall mask boundary and Mach-band candidates.
    try:
        ax.contour(rr, zz, near.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
    except Exception:
        pass
    if np.any(band):
        ax.scatter(rr[band], zz[band], s=6, c="yellow", alpha=0.55, edgecolors="none", label="Mach-band near-wall")

    ax.plot([entrance["R_m"]], [entrance["Z_m"]], marker="o", ms=7, color="red", mec="black", mew=0.6, label="selected sheath entrance")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("WEST plasma: sheath entrance on R-Z grid")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(framealpha=0.95, loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def parse_wall_segments(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    npts = nlines = None
    i_points = i_lines = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        p = s.split()
        if len(p) == 2 and p[1] == "points":
            npts = int(p[0])
            continue
        if len(p) == 2 and p[1] == "lines":
            nlines = int(p[0])
            continue
        if s == "points":
            i_points = i + 1
            continue
        if s == "lines":
            i_lines = i + 1
            continue
    if i_points is None or i_lines is None or npts is None or nlines is None:
        raise RuntimeError(f"Could not parse wall surface file: {path}")

    pts = {}
    k = i_points
    while k < len(lines) and len(pts) < npts:
        ln = lines[k].strip()
        k += 1
        if not ln:
            continue
        p = ln.split()
        if len(p) < 3:
            continue
        try:
            pid = int(p[0])
            pts[pid] = (float(p[1]), float(p[2]))
        except ValueError:
            continue

    r1s, z1s, r2s, z2s = [], [], [], []
    k = i_lines
    nread = 0
    while k < len(lines) and nread < nlines:
        ln = lines[k].strip()
        k += 1
        if not ln:
            continue
        p = ln.split()
        if len(p) < 3:
            continue
        try:
            p1 = int(p[1])
            p2 = int(p[2])
        except ValueError:
            continue
        if p1 in pts and p2 in pts:
            r1, z1 = pts[p1]
            r2, z2 = pts[p2]
            r1s.append(r1)
            z1s.append(z1)
            r2s.append(r2)
            z2s.append(z2)
        nread += 1

    r1 = np.asarray(r1s, dtype=float)
    z1 = np.asarray(z1s, dtype=float)
    r2 = np.asarray(r2s, dtype=float)
    z2 = np.asarray(z2s, dtype=float)
    dr = r2 - r1
    dz = z2 - z1
    ds = np.sqrt(dr * dr + dz * dz)
    ds = np.maximum(ds, 1.0e-12)
    nr = dz / ds
    nz = -dr / ds
    rm = 0.5 * (r1 + r2)
    zm = 0.5 * (z1 + z2)
    arc = np.zeros_like(ds)
    for i in range(1, ds.size):
        arc[i] = arc[i - 1] + ds[i - 1]

    return {"r1": r1, "z1": z1, "r2": r2, "z2": z2, "rm": rm, "zm": zm, "nr": nr, "nz": nz, "ds": ds, "arc": arc}


def eirene_sheath_ev(
    te_eV: np.ndarray,
    dens_m3: np.ndarray,
    upar_ms: np.ndarray,
    z_charge: np.ndarray,
    gamma_see: float = 0.0,
    cur_A_m2: float = 0.0,
) -> np.ndarray:
    te = np.asarray(te_eV, dtype=float)
    ni = np.asarray(dens_m3, dtype=float)
    ui = np.asarray(upar_ms, dtype=float)
    zi = np.asarray(z_charge, dtype=float)[:, None]

    de = np.sum(zi * ni, axis=0) + 1.0e-60
    ueff = np.sum((zi * ni / de) * np.abs(ui), axis=0)
    ce = np.sqrt(np.maximum(te, 1.0e-12) * QE / ME)
    fac = math.sqrt(2.0 * math.pi) / max(1.0 - gamma_see, 1.0e-12)
    arg = fac * (ueff / ce - cur_A_m2 / (QE * de * ce))

    out = 2.8 * te
    pos = arg > 0.0
    lg = np.zeros_like(te)
    lg[pos] = np.log(arg[pos])
    good = pos & (lg < 0.0)
    out[good] = -te[good] * lg[good]
    return out


def compute_wall_sheath_from_west(
    plasma_h5: Path,
    bfield_h5: Path,
    wall_file: Path,
    mi_amu: float,
    gamma_see: float = 0.0,
    cur_A_m2: float = 0.0,
) -> dict[str, np.ndarray]:
    wall = parse_wall_segments(wall_file)
    pts = np.column_stack([wall["zm"], wall["rm"]])  # (z,r)

    with h5py.File(plasma_h5, "r") as f:
        r = np.asarray(f["r"][...], dtype=float)
        z = np.asarray(f["z"][...], dtype=float)
        te = np.asarray(f["temp_e"][...], dtype=float)
        ti = np.asarray(f["temp_i"][...], dtype=float)
        ne = np.asarray(f["dens_e"][...], dtype=float)
        upar_main = np.asarray(f["parr_flow"][...], dtype=float)

        has_multi = all(k in f for k in ["ions/dens", "ions/parr_flow", "ion_species/charge_state_z"])
        if has_multi:
            ni_all_grid = np.asarray(f["ions/dens"][...], dtype=float)
            ui_all_grid = np.asarray(f["ions/parr_flow"][...], dtype=float)
            zi_all = np.asarray(f["ion_species/charge_state_z"][...], dtype=float)
        else:
            ni_main = np.asarray(f["dens_i"][...], dtype=float)
            ni_all_grid = ni_main[None, :, :]
            ui_all_grid = upar_main[None, :, :]
            zi_all = np.array([1.0], dtype=float)

    with h5py.File(bfield_h5, "r") as f:
        br = np.asarray(f["br"][...], dtype=float)
        bz = np.asarray(f["bz"][...], dtype=float)
        bt = np.asarray(f["bt"][...], dtype=float)

    interp_te = RegularGridInterpolator((z, r), te, bounds_error=False, fill_value=np.nan)
    interp_ti = RegularGridInterpolator((z, r), ti, bounds_error=False, fill_value=np.nan)
    interp_ne = RegularGridInterpolator((z, r), ne, bounds_error=False, fill_value=np.nan)
    interp_upar = RegularGridInterpolator((z, r), upar_main, bounds_error=False, fill_value=np.nan)
    interp_br = RegularGridInterpolator((z, r), br, bounds_error=False, fill_value=np.nan)
    interp_bz = RegularGridInterpolator((z, r), bz, bounds_error=False, fill_value=np.nan)
    interp_bt = RegularGridInterpolator((z, r), bt, bounds_error=False, fill_value=np.nan)

    te_w = interp_te(pts)
    ti_w = interp_ti(pts)
    ne_w = interp_ne(pts)
    upar_w = interp_upar(pts)
    br_w = interp_br(pts)
    bz_w = interp_bz(pts)
    bt_w = interp_bt(pts)

    ns = ni_all_grid.shape[0]
    ni_w = np.zeros((ns, pts.shape[0]), dtype=float)
    ui_w = np.zeros((ns, pts.shape[0]), dtype=float)
    for k in range(ns):
        ni_w[k] = RegularGridInterpolator((z, r), ni_all_grid[k], bounds_error=False, fill_value=np.nan)(pts)
        ui_w[k] = RegularGridInterpolator((z, r), ui_all_grid[k], bounds_error=False, fill_value=np.nan)(pts)

    mi_kg = mi_amu * AMU
    cs = np.sqrt(np.maximum(te_w + ti_w, 0.0) * QE / max(2.0 * mi_kg, 1.0e-99))
    mach_par = np.abs(upar_w) / np.maximum(cs, 1.0e-99)

    bmag = np.sqrt(br_w * br_w + bz_w * bz_w + bt_w * bt_w)
    bhatr = np.where(bmag > 0.0, br_w / bmag, 0.0)
    bhatz = np.where(bmag > 0.0, bz_w / bmag, 0.0)
    bdotn = bhatr * wall["nr"] + bhatz * wall["nz"]
    alpha_deg = np.degrees(np.arccos(np.clip(np.abs(bdotn), 0.0, 1.0)))
    un = np.abs(upar_w) * np.abs(bdotn)
    mach_n = un / np.maximum(cs, 1.0e-99)

    esh = eirene_sheath_ev(te_w, ni_w, ui_w, zi_all, gamma_see=gamma_see, cur_A_m2=cur_A_m2)
    phi_norm = -esh / np.maximum(te_w, 1.0e-99)

    out = dict(wall)
    out.update(
        {
            "te": te_w, "ti": ti_w, "ne": ne_w, "upar": upar_w,
            "br": br_w, "bz": bz_w, "bt": bt_w, "bdotn": bdotn, "alpha_deg": alpha_deg,
            "cs": cs, "un": un, "mach_par": mach_par, "mach_n": mach_n,
            "e_sheath_eV": esh, "phi_norm": phi_norm,
        }
    )
    return out


def plot_west_eirene_sheath(result: dict[str, np.ndarray], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), dpi=220, constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flat
    r1, z1, r2, z2 = result["r1"], result["z1"], result["r2"], result["z2"]
    arc = result["arc"]

    v = result["e_sheath_eV"]
    if np.any(np.isfinite(v)):
        vmin, vmax = np.nanpercentile(v, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    for i in range(len(r1)):
        c = plt.cm.inferno((np.clip(v[i], vmin, vmax) - vmin) / max(vmax - vmin, 1.0e-12))
        ax0.plot([r1[i], r2[i]], [z1[i], z2[i]], color=c, lw=2.0)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax0, pad=0.01)
    cb.set_label("E_sheath [eV]")
    ax0.set_title("EIRENE-like sheath on wall segments")
    ax0.set_xlabel("R [m]")
    ax0.set_ylabel("Z [m]")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, ls="--", alpha=0.25)

    ax1.plot(arc, result["e_sheath_eV"], lw=1.8)
    ax1.set_title("E_sheath vs wall arc")
    ax1.set_xlabel("wall arc [m]")
    ax1.set_ylabel("E_sheath [eV]")
    ax1.grid(True, ls="--", alpha=0.3)

    ax2.plot(arc, result["mach_par"], lw=1.6, label="M_parallel")
    ax2.plot(arc, result["mach_n"], lw=1.6, label="M_normal")
    ax2.axhline(1.0, color="k", lw=1.0, ls="--")
    ax2.set_title("Chodura/Bohm diagnostics")
    ax2.set_xlabel("wall arc [m]")
    ax2.set_ylabel("Mach")
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.legend(framealpha=0.95)

    ax3.plot(arc, result["alpha_deg"], lw=1.6, label=r"$\alpha$ [deg]")
    ax3.plot(arc, np.abs(result["bdotn"]), lw=1.6, label=r"$|b\cdot n|$")
    ax3.set_title("Wall/B-field geometry")
    ax3.set_xlabel("wall arc [m]")
    ax3.set_ylabel("deg / unitless")
    ax3.grid(True, ls="--", alpha=0.3)
    ax3.legend(framealpha=0.95)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def save_west_eirene_sheath_csv(result: dict[str, np.ndarray], out_csv: Path) -> None:
    cols = np.column_stack(
        [
            result["arc"], result["rm"], result["zm"], result["nr"], result["nz"],
            result["te"], result["ti"], result["ne"], result["upar"], result["cs"],
            result["mach_par"], result["mach_n"], result["bdotn"], result["alpha_deg"],
            result["e_sheath_eV"], result["phi_norm"],
        ]
    )
    hdr = (
        "arc_m,R_m,Z_m,nR,nZ,Te_eV,Ti_eV,ne_m-3,upar_m-s,cs_m-s,"
        "Mach_par,Mach_n,bdotn,alpha_deg,Esheath_eV,phi_norm_eDeltaPhi_over_kTe"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, cols, delimiter=",", header=hdr, comments="")


def plot_case(
    out_png: Path,
    d: np.ndarray,
    model: dict[str, np.ndarray | float],
    d_se_phi1: float,
    d_se_phi3: float,
    xnorm: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    phi = model["phi"]
    ne = model["ne"]
    E = model["E"]
    rho_i = float(model["rho_i"])
    lambda_D = float(model["lambda_D"])
    if xnorm == "rho_i":
        x = d / max(rho_i, 1e-99)
        xlabel = r"$d/\rho_i$"
    elif xnorm == "lambda_D":
        x = d / max(lambda_D, 1e-99)
        xlabel = r"$d/\lambda_D$"
    else:
        raise ValueError(f"unsupported xnorm={xnorm}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=220, constrained_layout=True)
    ax0, ax1 = axes

    ax0.plot(x, phi, lw=2.0, label=r"$\phi$")
    norm_len = rho_i if xnorm == "rho_i" else lambda_D
    ax0.axvline(d_se_phi1 / max(norm_len, 1e-99), ls="--", lw=1.3, color="k", label=r"SE from $\phi\approx-T_e$")
    ax0.axvline(d_se_phi3 / max(norm_len, 1e-99), ls="--", lw=1.3, color="r", label=r"SE from $\phi\approx-3T_e$")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(r"$\phi$ [V]")
    ax0.set_title("Borodkina Potential")
    ax0.grid(True, ls="--", alpha=0.3)
    ax0.legend(framealpha=0.9)

    ax1.plot(x, ne, lw=2.0, label=r"$n_e$")
    ax1.plot(x, E, lw=1.8, label=r"$|E|$")
    ax1.set_xlabel(xlabel)
    ax1.set_title("Electron Density and Field")
    ax1.set_yscale("log")
    ax1.grid(True, ls="--", alpha=0.3)
    ax1.legend(framealpha=0.9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def paper_anchor_curve() -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate anchor points read from Borodkina Fig.2 analytical (red) curve.
    Used only for quick sanity metrics.
    """
    x = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], dtype=float)
    y = np.array([-2.50, -1.95, -1.45, -1.00, -0.73, -0.55, -0.43, -0.34, -0.27, -0.21, -0.16], dtype=float)
    return x, y


def plot_paper_compare(
    out_png: Path,
    x_model: np.ndarray,
    phi_norm_model: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), dpi=220, constrained_layout=True)
    ax.plot(x_model, phi_norm_model, lw=2.2, color="crimson", label="model (analytical)")
    ax.plot(x_ref, y_ref, "o", ms=4, color="black", label="paper anchors (Fig.2)")
    ax.set_xlabel(r"$y/r_d$")
    ax.set_ylabel(r"$e(\phi-\phi_p)/kT_e$")
    ax.set_title("Borodkina Fig.2-style comparison")
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(-3.1, 0.05)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(framealpha=0.95)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_alpha_sweep(
    out_png: Path,
    d: np.ndarray,
    te_eV: float,
    ti_eV: float,
    ne0_m3: float,
    b_T: float,
    alphas_deg: list[float],
    mi_amu: float,
    angle_def: str,
    pot_mult: float,
    xnorm: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=220, constrained_layout=True)
    ax0, ax1 = axes

    for a in alphas_deg:
        model = borodkina_profile(
            d_m=d,
            te_eV=te_eV,
            ti_eV=ti_eV,
            ne0_m3=ne0_m3,
            b_T=b_T,
            alpha_deg=a,
            mi_amu=mi_amu,
            angle_def=angle_def,
            pot_mult=pot_mult,
        )
        if xnorm == "rho_i":
            x = d / max(float(model["rho_i"]), 1e-99)
            xlabel = r"$d/\rho_i$"
        else:
            x = d / max(float(model["lambda_D"]), 1e-99)
            xlabel = r"$d/\lambda_D$"

        phi = np.asarray(model["phi"], dtype=float)
        phi_p = float(phi[-1])
        phi_norm = (phi - phi_p) / max(te_eV, 1e-99)
        ax0.plot(x, phi_norm, lw=2.0, label=rf"$\alpha={a:.0f}^\circ$")
        ax1.plot(x, model["ne"], lw=2.0, label=rf"$\alpha={a:.0f}^\circ$")

    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(r"$e(\phi-\phi_p)/kT_e$")
    ax0.set_title("Normalized Potential")
    ax0.set_xlim(0.0, 10.0)
    ax0.grid(True, ls="--", alpha=0.3)
    ax0.legend(framealpha=0.95)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"$n_e$ [m$^{-3}$]")
    ax1.set_title("Electron Density")
    ax1.set_xlim(0.0, 10.0)
    ax1.set_yscale("log")
    ax1.grid(True, ls="--", alpha=0.3)
    ax1.legend(framealpha=0.95)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Borodkina sheath model standalone tester.")
    p.add_argument(
        "--plasma-h5",
        type=Path,
        default=Path("../test_west_axi/input/plasma.h5"),
        help="WEST plasma file used to extract Te/Ti/ne/u_par sheath-entrance values",
    )
    p.add_argument(
        "--bfield-h5",
        type=Path,
        default=Path("../test_west_axi/input/bfield.h5"),
        help="WEST magnetic-field file for wall-local Chodura geometry diagnostics",
    )
    p.add_argument(
        "--wall-file",
        type=Path,
        default=Path("../test_west_axi/input/wall.txt"),
        help="WEST wall surface file (2D lines)",
    )
    p.add_argument(
        "--use-plasma-h5",
        choices=["yes", "no"],
        default="yes",
        help="Use plasma.h5 values (yes) or CLI scalar values (no)",
    )
    p.add_argument("--paper-case", action="store_true", help="Use Borodkina Fig.2 case values")
    p.add_argument(
        "--paper-compare",
        action="store_true",
        help="Generate a Fig.2-style comparison in y/r_d with anchor-point error metrics",
    )
    p.add_argument("--te", type=float, default=30.0, help="Electron temperature [eV]")
    p.add_argument("--ti", type=float, default=30.0, help="Ion temperature [eV]")
    p.add_argument("--ne", type=float, default=1.0e14, help="Electron density [m^-3]")
    p.add_argument("--b", type=float, default=3.21, help="Magnetic field [T]")
    p.add_argument("--alpha", type=float, default=88.0, help="Incidence angle [deg]")
    p.add_argument(
        "--angle-def",
        choices=["normal", "surface"],
        default="normal",
        help="Interpretation of alpha",
    )
    p.add_argument("--mi-amu", type=float, default=2.0, help="Ion mass [amu]")
    p.add_argument("--u-par", type=float, default=None, help="Parallel ion speed along B [m/s]")
    p.add_argument("--gamma-par", type=float, default=None, help="Parallel ion flux Gamma_parallel [m^-2 s^-1]")
    p.add_argument("--ni", type=float, default=None, help="Ion density [m^-3] used with --gamma-par (default: ne)")
    p.add_argument("--pot-mult", type=float, default=3.0, help="Surface potential multiplier")
    p.add_argument(
        "--alpha-sweep",
        type=str,
        default="",
        help="Comma-separated alpha list for 2-panel sweep plot, e.g. '0,45,85'",
    )
    p.add_argument(
        "--alpha-sweep-out",
        type=Path,
        default=Path("output/borodkina_alpha_sweep.png"),
        help="Output PNG for alpha sweep plot",
    )
    p.add_argument(
        "--xnorm",
        choices=["rho_i", "lambda_D"],
        default="rho_i",
        help="Distance normalization used in plots/csv",
    )
    p.add_argument("--dmax", type=float, default=0.03, help="Distance extent from surface [m]")
    p.add_argument("--npts", type=int, default=800, help="Points in profile")
    p.add_argument(
        "--near-power",
        type=float,
        default=1.0,
        help="Grid clustering power near wall (1=uniform, >1 clusters near d=0)",
    )
    p.add_argument("--out", type=Path, default=Path("output/borodkina_sheath_test.png"), help="Output PNG")
    p.add_argument(
        "--plot-rz",
        choices=["yes", "no"],
        default="yes",
        help="Plot WEST R-Z Mach map with selected sheath-entrance marker (requires --use-plasma-h5 yes)",
    )
    p.add_argument(
        "--rz-out",
        type=Path,
        default=Path("output/sheath_entrance_rz.png"),
        help="Output PNG for R-Z sheath-entrance diagnostic",
    )
    p.add_argument(
        "--mach-band",
        type=str,
        default="0.8,1.2",
        help="Mach band for sheath-entrance candidates, format 'min,max'",
    )
    p.add_argument(
        "--near-layers",
        type=int,
        default=3,
        help="Near-wall mask thickness in grid-cell layers",
    )
    p.add_argument(
        "--eirene-wall",
        choices=["yes", "no"],
        default="yes",
        help="Run EIRENE-like wall-local sheath model using WEST plasma/bfield/wall inputs",
    )
    p.add_argument(
        "--eirene-out",
        type=Path,
        default=Path("output/eirene_sheath_west.png"),
        help="Output PNG for EIRENE-like wall-local sheath diagnostics",
    )
    p.add_argument(
        "--eirene-csv",
        type=Path,
        default=Path("output/eirene_sheath_west.csv"),
        help="Output CSV for EIRENE-like wall-local sheath diagnostics",
    )
    p.add_argument("--gamma-see", type=float, default=0.0, help="Secondary electron emission coefficient for EIRENE-like sheath model")
    p.add_argument("--cur-A-m2", type=float, default=0.0, help="Net current density [A/m^2] for EIRENE-like sheath model")
    args = p.parse_args()

    if args.paper_case:
        args.te = 30.0
        args.ti = 30.0
        args.b = 3.21
        args.ne = 1.0e20  # 1e14 cm^-3
        args.alpha = 88.0
        args.angle_def = "normal"
        args.mi_amu = 1.0  # H+
        if args.pot_mult == 3.0:
            args.pot_mult = 2.5
        if args.paper_compare:
            args.xnorm = "lambda_D"

    entrance = None
    mparts = [v.strip() for v in args.mach_band.split(",") if v.strip()]
    if len(mparts) != 2:
        raise ValueError(f"Invalid --mach-band='{args.mach_band}', expected 'min,max'")
    mach_min = float(mparts[0])
    mach_max = float(mparts[1])
    if mach_max < mach_min:
        mach_min, mach_max = mach_max, mach_min

    if args.use_plasma_h5 == "yes":
        if not args.plasma_h5.exists():
            raise FileNotFoundError(f"plasma.h5 not found: {args.plasma_h5}")
        entrance = load_west_sheath_entrance(
            args.plasma_h5,
            args.mi_amu,
            mach_min=mach_min,
            mach_max=mach_max,
            near_layers=args.near_layers,
        )
        args.te = entrance["te_eV"]
        args.ti = entrance["ti_eV"]
        args.ne = entrance["ne_m3"]
        args.u_par = entrance["u_par"]
        args.ni = entrance["ne_m3"]

    s = np.linspace(0.0, 1.0, args.npts)
    pwr = max(args.near_power, 1.0)
    d = args.dmax * np.power(s, pwr)
    model = borodkina_profile(
        d_m=d,
        te_eV=args.te,
        ti_eV=args.ti,
        ne0_m3=args.ne,
        b_T=args.b,
        alpha_deg=args.alpha,
        mi_amu=args.mi_amu,
        angle_def=args.angle_def,
        pot_mult=args.pot_mult,
    )

    ni = args.ni if args.ni is not None else args.ne
    if args.u_par is not None:
        u_par = args.u_par
        gamma_par = ni * u_par
    elif args.gamma_par is not None:
        gamma_par = args.gamma_par
        u_par = gamma_par / max(ni, 1e-99)
    else:
        # Analytic default for quick checks
        gamma_par = 3.0e19
        u_par = gamma_par / max(ni, 1e-99)

    d_se_phi1 = estimate_entrance_from_phi(d, model["phi"], args.te, target_mult=1.0)
    d_se_phi3 = estimate_entrance_from_phi(d, model["phi"], args.te, target_mult=3.0)
    checks = check_chodura_bohm(u_par, float(model["cs"]), float(model["alpha_n_rad"]))

    plot_case(args.out, d, model, d_se_phi1, d_se_phi3, args.xnorm)

    # Export comparison-ready profile: x and normalized potential
    phi = np.asarray(model["phi"], dtype=float)
    if args.xnorm == "rho_i":
        x = d / max(float(model["rho_i"]), 1e-99)
    else:
        x = d / max(float(model["lambda_D"]), 1e-99)
    # General normalization (kept for CSV): wall-referenced
    # e(phi - phi_w)/kTe = (phi - phi_w)/Te_eV
    phi_w = float(phi[0])
    phi_norm = (phi - phi_w) / max(args.te, 1e-99)
    csv_path = args.out.with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack([x, phi_norm, phi]),
        delimiter=",",
        header="x_norm,phi_norm_e(phi-phi_w)/kTe,phi_V",
        comments="",
    )

    print("Borodkina test summary")
    if entrance is not None:
        print(f"  plasma.h5      = {args.plasma_h5.resolve()}")
        print(
            "  sheath-entry cell (closest Mach=1): "
            f"(i={int(entrance['i'])}, j={int(entrance['j'])}), "
            f"R={entrance['R_m']:.6g} m, Z={entrance['Z_m']:.6g} m"
        )
        print(
            f"  selection      = {entrance['selection_source']} "
            f"(band={int(entrance['band_count'])}, near={int(entrance['near_count'])})"
        )
        print(f"  source Mach    = {entrance['mach']:.6e}")
        print(f"  frac(Mach>=1)  = {entrance['mach_ge_1_frac']:.6f}")
    print(f"  lambda_D [m]   = {model['lambda_D']:.6e}")
    print(f"  L_mps [m]      = {model['L_mps']:.6e}")
    print(f"  rho_i [m]      = {model['rho_i']:.6e}")
    print(f"  c_s [m/s]      = {model['cs']:.6e}")
    print(f"  n_i [m^-3]     = {ni:.6e}")
    print(f"  Gamma_par [m^-2 s^-1] = {gamma_par:.6e}")
    print(f"  u_par [m/s]    = {u_par:.6e}")
    print(f"  sheath entrance from phi=-Te  [m] = {d_se_phi1:.6e}")
    print(f"  sheath entrance from phi=-3Te [m] = {d_se_phi3:.6e}")
    print(f"  Chodura check (|u_par|>=c_s): {checks['chodura_ok']} (u_par={checks['u_par']:.3e})")
    print(
        "  Bohm from projection (|u_n|>=c_s): "
        f"{checks['bohm_ok_from_par_projection']} (u_n={checks['u_n']:.3e})"
    )
    print(f"  wrote: {args.out.resolve()}")
    print(f"  wrote: {csv_path.resolve()}")

    if args.paper_compare:
        x_ref, y_ref = paper_anchor_curve()
        # Paper x-axis is y/r_d, interpreted here as Debye normalization.
        x_paper = d / max(float(model["lambda_D"]), 1e-99)
        # Fig.2-style y-axis is plasma-referenced, negative at wall, ~0 in plasma:
        # e(phi - phi_p)/kTe
        phi_p = float(phi[-1])
        phi_paper = (phi - phi_p) / max(args.te, 1e-99)
        y_interp = np.interp(x_ref, x_paper, phi_paper)
        err = y_interp - y_ref
        rmse = float(np.sqrt(np.mean(err * err)))
        max_abs = float(np.max(np.abs(err)))
        mae = float(np.mean(np.abs(err)))
        cmp_png = args.out.with_name(args.out.stem + ".paper_compare.png")
        plot_paper_compare(cmp_png, x_paper, phi_paper, x_ref, y_ref)
        print("Paper comparison (anchor points from Fig.2 red curve):")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  max|err| = {max_abs:.4f}")
        print(f"  wrote: {cmp_png.resolve()}")

    if args.alpha_sweep.strip():
        alphas = [float(v.strip()) for v in args.alpha_sweep.split(",") if v.strip()]
        if len(alphas) == 0:
            raise ValueError("alpha-sweep provided but no valid values parsed")
        plot_alpha_sweep(
            out_png=args.alpha_sweep_out,
            d=d,
            te_eV=args.te,
            ti_eV=args.ti,
            ne0_m3=args.ne,
            b_T=args.b,
            alphas_deg=alphas,
            mi_amu=args.mi_amu,
            angle_def=args.angle_def,
            pot_mult=args.pot_mult,
            xnorm=args.xnorm,
        )
        print(f"  wrote: {args.alpha_sweep_out.resolve()}")

    if args.plot_rz == "yes" and entrance is not None:
        plot_sheath_entrance_rz(
            args.plasma_h5,
            args.mi_amu,
            entrance,
            args.rz_out,
            mach_min=mach_min,
            mach_max=mach_max,
            near_layers=args.near_layers,
        )
        print(f"  wrote: {args.rz_out.resolve()}")

    if args.eirene_wall == "yes" and args.use_plasma_h5 == "yes":
        if not args.bfield_h5.exists():
            raise FileNotFoundError(f"bfield.h5 not found: {args.bfield_h5}")
        if not args.wall_file.exists():
            raise FileNotFoundError(f"wall file not found: {args.wall_file}")
        eres = compute_wall_sheath_from_west(
            plasma_h5=args.plasma_h5,
            bfield_h5=args.bfield_h5,
            wall_file=args.wall_file,
            mi_amu=args.mi_amu,
            gamma_see=args.gamma_see,
            cur_A_m2=args.cur_A_m2,
        )
        plot_west_eirene_sheath(eres, args.eirene_out)
        save_west_eirene_sheath_csv(eres, args.eirene_csv)
        print("EIRENE-like wall-local sheath summary:")
        print(f"  wall segments   = {len(eres['rm'])}")
        print(f"  Esheath [eV]    min={np.nanmin(eres['e_sheath_eV']):.6e} max={np.nanmax(eres['e_sheath_eV']):.6e}")
        print(f"  M_par           min={np.nanmin(eres['mach_par']):.6e} max={np.nanmax(eres['mach_par']):.6e}")
        print(f"  M_n             min={np.nanmin(eres['mach_n']):.6e} max={np.nanmax(eres['mach_n']):.6e}")
        print(f"  wrote: {args.eirene_out.resolve()}")
        print(f"  wrote: {args.eirene_csv.resolve()}")


if __name__ == "__main__":
    main()
