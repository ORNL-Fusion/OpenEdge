"""Visualize OML droplet charging in 2D space and time.

Creates a 2x2 subplot figure:
- 3 spatial panels: particle positions (R,Z) colored by Zd at early/mid/late time
- 1 time panel: Zd min/avg/max vs time

Reads:
- particle dump (default: case.oml)
- plasma fields via h5dump from ../test_evaporation/input/plasma.h5
"""

import argparse
import os
import re
import subprocess
from dataclasses import dataclass

import numpy as np


# Physical constants (SI)
QE = 1.60217646e-19
ME = 9.10938215e-31
PROTON_MASS = 1.6726219e-27
EPS0 = 8.8541878128e-12
KB = 1.3806503e-23


@dataclass
class OMLParams:
    ion_mass_amu: float = 2.0
    thermionic: bool = True
    richardson_A: float = 1.2e6
    work_function_eV: float = 2.9
    fallback_temp_K: float = 773.15


def parse_dump_particle(filename: str):
    """Parse SPARTA dump particle file with ITEM headers.

    Expected ATOMS columns for in.oml_charging:
    id type x y z vx vy vz temp radius
    """
    t, x, y, temp, radius, pid, pcharge = [], [], [], [], [], [], []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = 0
    ts = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            ts = int(lines[i + 1])
            i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1])
            i += 2
        elif s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            colmap = {name: idx for idx, name in enumerate(cols)}
            req = ["id", "x", "y"]
            for r in req:
                if r not in colmap:
                    raise RuntimeError(f"Dump is missing required column '{r}' in: {s}")
            for _ in range(n):
                p = lines[i + 1].split()
                pid.append(int(p[colmap["id"]]))
                t.append(ts)
                x.append(float(p[colmap["x"]]))  # R
                y.append(float(p[colmap["y"]]))  # Z
                temp.append(float(p[colmap["temp"]]) if "temp" in colmap else np.nan)
                radius.append(float(p[colmap["radius"]]) if "radius" in colmap else np.nan)
                pcharge.append(float(p[colmap["v_pcharge"]]) if "v_pcharge" in colmap else np.nan)
                i += 1
            i += 1
        else:
            i += 1

    return {
        "timestep": np.asarray(t, dtype=np.int64),
        "R": np.asarray(x, dtype=float),
        "Z": np.asarray(y, dtype=float),
        "temp_K": np.asarray(temp, dtype=float),
        "radius_m": np.asarray(radius, dtype=float),
        "pid": np.asarray(pid, dtype=np.int64),
        "pcharge": np.asarray(pcharge, dtype=float),
    }


def _extract_numbers_from_h5dump(text: str):
    # Keep only inside DATA block
    m = re.search(r"DATA\s*\{(.*)\}\s*\}\s*$", text, flags=re.S)
    if not m:
        raise RuntimeError("Could not locate DATA block in h5dump output")
    data = m.group(1)

    # Remove row/element indices like "(0,0):" and "(12):"
    data = re.sub(r"\([^\)]*\)\s*:\s*", " ", data)

    # Float regex with optional exponent
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data)
    if not nums:
        raise RuntimeError("No numeric values found in h5dump DATA block")
    return np.asarray([float(v) for v in nums], dtype=float)


def read_h5_dataset_via_h5dump(h5file: str, dset: str):
    cmd = ["h5dump", "-d", dset, h5file]
    out = subprocess.check_output(cmd, text=True)
    shape_match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s*\(\s*([0-9,\s]+)\s*\)", out)
    if not shape_match:
        raise RuntimeError(f"Could not parse DATASPACE for dataset '{dset}'")
    dims = tuple(int(x.strip()) for x in shape_match.group(1).split(","))
    vals = _extract_numbers_from_h5dump(out)
    expected = int(np.prod(dims))
    if vals.size != expected:
        raise RuntimeError(f"Dataset '{dset}' size mismatch: got {vals.size}, expected {expected}")
    return vals.reshape(dims)


def load_plasma_fields(plasma_h5: str):
    r = read_h5_dataset_via_h5dump(plasma_h5, "r")
    z = read_h5_dataset_via_h5dump(plasma_h5, "z")
    te = read_h5_dataset_via_h5dump(plasma_h5, "temp_e")
    ti = read_h5_dataset_via_h5dump(plasma_h5, "temp_i")
    ne = read_h5_dataset_via_h5dump(plasma_h5, "dens_e")
    ni = read_h5_dataset_via_h5dump(plasma_h5, "dens_i")
    return r, z, te, ti, ne, ni


def bilinear_interp(rg, zg, field, r, z):
    # Clamp inside table bounds
    r = np.clip(r, rg[0], rg[-1])
    z = np.clip(z, zg[0], zg[-1])

    ir = np.searchsorted(rg, r, side="right") - 1
    iz = np.searchsorted(zg, z, side="right") - 1
    ir = np.clip(ir, 0, len(rg) - 2)
    iz = np.clip(iz, 0, len(zg) - 2)

    r1 = rg[ir]
    r2 = rg[ir + 1]
    z1 = zg[iz]
    z2 = zg[iz + 1]

    tr = (r - r1) / np.maximum(r2 - r1, 1.0e-30)
    tz = (z - z1) / np.maximum(z2 - z1, 1.0e-30)

    f11 = field[iz, ir]
    f21 = field[iz, ir + 1]
    f12 = field[iz + 1, ir]
    f22 = field[iz + 1, ir + 1]

    return (
        (1.0 - tr) * (1.0 - tz) * f11
        + tr * (1.0 - tz) * f21
        + (1.0 - tr) * tz * f12
        + tr * tz * f22
    )


def solve_phi_oml(te_eV, ti_eV, ne_m3, ni_m3, td_K, rd_m, prm: OMLParams):
    if te_eV <= 0.0 or ti_eV <= 0.0 or ne_m3 <= 0.0 or ni_m3 <= 0.0 or rd_m <= 0.0:
        return np.nan

    mi = prm.ion_mass_amu * PROTON_MASS
    area_coll = np.pi * rd_m * rd_m
    area_emit = 4.0 * np.pi * rd_m * rd_m

    Ce = QE * area_coll * ne_m3 * np.sqrt(8.0 * (te_eV * QE) / (np.pi * ME))
    Ci = QE * area_coll * ni_m3 * np.sqrt(8.0 * (ti_eV * QE) / (np.pi * mi))

    Ith = 0.0
    if prm.thermionic and td_K > 0.0 and prm.richardson_A > 0.0 and prm.work_function_eV > 0.0:
        W_inf_J = prm.work_function_eV * QE
        dW = (QE * QE) / (16.0 * np.pi * EPS0 * rd_m)
        W_rd = max(0.0, W_inf_J - dW)
        Ith = area_emit * prm.richardson_A * td_K * td_K * np.exp(-W_rd / (KB * td_K))

    def f(phi):
        Ie = -Ce * np.exp(phi / te_eV)
        Ii = Ci * (1.0 - phi / ti_eV)
        return Ie + Ii + Ith

    scale = max(te_eV, ti_eV)
    lo = -80.0 * scale
    hi = 20.0 * scale
    flo = f(lo)
    fhi = f(hi)

    for _ in range(12):
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0.0:
            break
        lo *= 2.0
        hi *= 2.0
        flo = f(lo)
        fhi = f(hi)

    if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0.0:
        return np.nan

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return np.nan
        if abs(fm) <= 1.0e-10 * max(1.0, abs(Ci + Ith)):
            return mid
        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
        if abs(hi - lo) <= 1.0e-10 * max(1.0, abs(mid)):
            return 0.5 * (lo + hi)

    return 0.5 * (lo + hi)


def compute_charges(data, rg, zg, te_g, ti_g, ne_g, ni_g, prm: OMLParams):
    n = data["R"].size
    if "pcharge" in data and np.any(np.isfinite(data["pcharge"])):
        return np.asarray(data["pcharge"], dtype=float)

    zd = np.full(n, np.nan, dtype=float)

    te = bilinear_interp(rg, zg, te_g, data["R"], data["Z"])
    ti = bilinear_interp(rg, zg, ti_g, data["R"], data["Z"])
    ne = bilinear_interp(rg, zg, ne_g, data["R"], data["Z"])
    ni = bilinear_interp(rg, zg, ni_g, data["R"], data["Z"])

    td = np.where(data["temp_K"] > 0.0, data["temp_K"], prm.fallback_temp_K)

    for i in range(n):
        rd = data["radius_m"][i]
        phi = solve_phi_oml(te[i], ti[i], ne[i], ni[i], td[i], rd, prm)
        if np.isfinite(phi):
            qd = 4.0 * np.pi * EPS0 * rd * phi
            zd[i] = qd / QE

    return zd


def plot_subplots(data, zd, dt_s, out_png):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    ts_all = np.unique(data["timestep"])
    if ts_all.size == 0:
        raise RuntimeError("No timesteps found in dump file")

    snap_ids = [0, ts_all.size // 2, ts_all.size - 1]
    snap_ts = [ts_all[i] for i in snap_ids]

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
#    axs = axs.ravel()

    finite = np.isfinite(zd)
    if np.any(finite):
        vmin = np.nanpercentile(zd[finite], 5.0)
        vmax = np.nanpercentile(zd[finite], 95.0)
        if vmin == vmax:
            vmin, vmax = np.nanmin(zd[finite]), np.nanmax(zd[finite])
    else:
        vmin, vmax = -1.0, 1.0

#    for k, ts in enumerate(snap_ts):
#        ax = axs[k]
#        m = data["timestep"] == ts
#        sc = ax.scatter(data["R"][m], data["Z"][m], c=zd[m], s=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
#        ax.set_title(f"Spatial charge map at step {ts}")
#        ax.set_xlabel("R [m]")
#        ax.set_ylabel("Z [m]")
#        ax.grid(alpha=0.2, linestyle="--")
#        cbar = fig.colorbar(sc, ax=ax)
#        cbar.set_label("Zd = q/e")
#
#    ax = axs[3]
    t_plot = ts_all.astype(float) * dt_s
    zmin = []
    zavg = []
    zmax = []
    for ts in ts_all:
        m = (data["timestep"] == ts) & np.isfinite(zd)
        if np.any(m):
            zmin.append(np.min(zd[m]))
            zavg.append(np.mean(zd[m]))
            zmax.append(np.max(zd[m]))
        else:
            zmin.append(np.nan)
            zavg.append(np.nan)
            zmax.append(np.nan)

    ax.plot(t_plot, zmin, "-", lw=1.5, label="Zd min")
    ax.plot(t_plot, zavg, "-", lw=1.8, label="Zd avg")
    ax.plot(t_plot, zmax, "-", lw=1.5, label="Zd max")
    ax.set_title("Charge evolution in time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Zd = q/e")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()

    fig.suptitle("OML Charging: 2D spatial maps and time evolution", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_png}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot OML charging in space/time from particle dump")
    parser.add_argument("--dump", default="case.oml", help="SPARTA particle dump file")
    parser.add_argument("--plasma", default="../test_evaporation/input/plasma.h5", help="plasma.h5 path")
    parser.add_argument("--dt", type=float, default=1.0e-6, help="timestep in seconds")
    parser.add_argument("--ion-mass-amu", type=float, default=2.0)
    parser.add_argument("--thermionic", choices=["yes", "no"], default="yes")
    parser.add_argument("--richardson-A", type=float, default=1.2e6)
    parser.add_argument("--work-function-eV", type=float, default=2.9)
    parser.add_argument("--fallback-temp", type=float, default=773.15)
    parser.add_argument("--out", default="Figs/oml_charging_subplots.png")
    args = parser.parse_args()

    if not os.path.exists(args.dump):
        raise FileNotFoundError(f"Missing dump file: {args.dump}")
    if not os.path.exists(args.plasma):
        raise FileNotFoundError(f"Missing plasma file: {args.plasma}")

    prm = OMLParams(
        ion_mass_amu=args.ion_mass_amu,
        thermionic=(args.thermionic == "yes"),
        richardson_A=args.richardson_A,
        work_function_eV=args.work_function_eV,
        fallback_temp_K=args.fallback_temp,
    )

    print("Parsing particle dump...")
    data = parse_dump_particle(args.dump)
    print(f"Loaded {data['R'].size} particle samples across {np.unique(data['timestep']).size} timesteps")

    print("Loading plasma fields via h5dump...")
    rg, zg, te, ti, ne, ni = load_plasma_fields(args.plasma)

    print("Computing OML charges per sample...")
    zd = compute_charges(data, rg, zg, te, ti, ne, ni, prm)

    print("Generating subplot figure...")
    plot_subplots(data, zd, args.dt, args.out)


if __name__ == "__main__":
    main()
