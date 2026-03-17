#!/usr/bin/env python3
"""Plot parallel flow direction in R-Z from WEST-axisymmetric plasma/B-field inputs.

This script recreates an "old-style" interpretation plot:
- quiver arrows in R-Z projected from v_parallel * b_hat
- arrow color = signed Mach-like value M = v_parallel / c_s
- optional wall overlay from input/wall.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required: pip install h5py") from exc

QE = 1.602176634e-19
AMU = 1.66053906660e-27


def _read_ds(h5f: h5py.File, key: str):
    return np.array(h5f[key]) if key in h5f else None


def _to_rz(arr: np.ndarray, nr: int, nz: int, name: str) -> np.ndarray:
    # Output shape is (nz, nr) for meshgrid(indexing='xy') compatibility.
    if arr.shape == (nr, nz):
        return arr.T
    if arr.shape == (nz, nr):
        return arr
    raise ValueError(f"Unexpected shape for {name}: {arr.shape}, expected {(nr, nz)} or {(nz, nr)}")


def _safe_unit(vr: np.ndarray, vz: np.ndarray, eps: float = 1.0e-30):
    mag = np.hypot(vr, vz)
    return vr / np.maximum(mag, eps), vz / np.maximum(mag, eps)


def _read_wall_points(path: Path):
    if not path.exists():
        return None, None
    lines = path.read_text().splitlines()
    npts = None
    for ln in lines:
        s = ln.strip().lower()
        if s.endswith("points"):
            parts = s.split()
            if parts and parts[0].isdigit():
                npts = int(parts[0])
                break
    if npts is None:
        return None, None
    try:
        i0 = lines.index("Points") + 1
    except ValueError:
        return None, None
    rr, zz = [], []
    for ln in lines[i0 : i0 + npts]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        rr.append(float(parts[1]))
        zz.append(float(parts[2]))
    if not rr:
        return None, None
    return np.asarray(rr), np.asarray(zz)


def _main_ion_mass_kg(pf: h5py.File) -> float:
    masses = _read_ds(pf, "ion_species/mass_amu")
    i0 = 0
    if "ion_species/main_ion_spec_index" in pf:
        m = np.array(pf["ion_species/main_ion_spec_index"]).reshape(-1)
        if len(m):
            i0 = int(m[0])
    if masses is None:
        # conservative default: deuterium
        return 2.0 * AMU
    i0 = max(0, min(i0, masses.shape[0] - 1))
    return float(masses[i0]) * AMU


def main():
    ap = argparse.ArgumentParser(description="Plot v_parallel flow direction in R-Z.")
    ap.add_argument("--bfield", default="input/bfield.h5", help="Path to bfield.h5")
    ap.add_argument("--plasma", default="input/plasma.h5", help="Path to plasma.h5")
    ap.add_argument("--wall", default="input/wall.txt", help="Optional wall geometry file")
    ap.add_argument("--out", default="input/Figs/vpar_flow_direction_rz.png", help="Output image")
    ap.add_argument("--stride", type=int, default=6, help="Quiver downsample stride")
    ap.add_argument("--ion-index", type=int, default=0, help="Ion index if using /ions/* fields")
    ap.add_argument(
        "--vector-source",
        choices=["auto", "stored", "projected"],
        default="auto",
        help="Use stored parr_flow_r/z vectors or project v_parallel on B",
    )
    ap.add_argument("--sample-frac", type=float, default=0.12, help="Random fraction of arrows to plot")
    ap.add_argument("--sample-seed", type=int, default=1234, help="RNG seed for deterministic sampling")
    ap.add_argument("--title", default="Parallel Flow Direction Check in R-Z", help="Figure title")
    args = ap.parse_args()

    bpath = Path(args.bfield)
    ppath = Path(args.plasma)
    wpath = Path(args.wall)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(bpath, "r") as bf, h5py.File(ppath, "r") as pf:
        r = np.array(bf["r"]) if "r" in bf else np.array(pf["r"])
        z = np.array(bf["z"]) if "z" in bf else np.array(pf["z"])

        br = np.array(bf["br"])
        bt = np.array(bf["bt"])
        bz = np.array(bf["bz"])

        parr = _read_ds(pf, "parr_flow")
        if parr is None:
            parr = _read_ds(pf, "n_i/parr_flow")
        if parr is None and "ions/parr_flow" in pf:
            ip = np.array(pf["ions/parr_flow"])
            i = max(0, min(args.ion_index, ip.shape[0] - 1))
            parr = ip[i]
        if parr is None:
            raise KeyError("Missing parallel flow field in plasma.h5")

        parr_r = _read_ds(pf, "parr_flow_r")
        parr_z = _read_ds(pf, "parr_flow_z")
        if parr_r is None and "ions/parr_flow_r" in pf:
            pr = np.array(pf["ions/parr_flow_r"])
            pz = np.array(pf["ions/parr_flow_z"])
            i = max(0, min(args.ion_index, pr.shape[0] - 1))
            parr_r, parr_z = pr[i], pz[i]

        te = _read_ds(pf, "temp_e")
        if te is None:
            te = _read_ds(pf, "n_e/temp")
        ti = _read_ds(pf, "temp_i")
        if ti is None:
            ti = _read_ds(pf, "n_i/temp")
        if ti is None and "ions/temp" in pf:
            it = np.array(pf["ions/temp"])
            i = max(0, min(args.ion_index, it.shape[0] - 1))
            ti = it[i]

        mi = _main_ion_mass_kg(pf)

    nr, nz = len(r), len(z)
    br = _to_rz(br, nr, nz, "br")
    bt = _to_rz(bt, nr, nz, "bt")
    bz = _to_rz(bz, nr, nz, "bz")
    parr = _to_rz(parr, nr, nz, "parr_flow")

    if te is not None:
        te = _to_rz(te, nr, nz, "temp_e")
    if ti is not None:
        ti = _to_rz(ti, nr, nz, "temp_i")
    if parr_r is not None:
        parr_r = _to_rz(parr_r, nr, nz, "parr_flow_r")
    if parr_z is not None:
        parr_z = _to_rz(parr_z, nr, nz, "parr_flow_z")

    bmag = np.sqrt(br * br + bt * bt + bz * bz)
    vr_proj = parr * br / np.maximum(bmag, 1.0e-30)
    vz_proj = parr * bz / np.maximum(bmag, 1.0e-30)

    use_stored = (parr_r is not None and parr_z is not None)
    if args.vector_source == "stored" and not use_stored:
        raise KeyError("Requested --vector-source stored but parr_flow_r/parr_flow_z not found")
    if args.vector_source == "auto":
        use_stored = use_stored
    elif args.vector_source == "projected":
        use_stored = False

    if use_stored:
        vr, vz = parr_r, parr_z
        src_label = "stored parr_flow_r/z"
    else:
        vr, vz = vr_proj, vz_proj
        src_label = "projected from v_parallel * b_hat"

    ur, uz = _safe_unit(vr, vz)

    # Mach-like scaling from local sound speed; fallback to |parr| max if temperatures missing.
    if (te is not None) and (ti is not None):
        cs = np.sqrt(np.maximum((te + ti) * QE / mi, 1.0e-12))
        mach = parr / cs
    else:
        vmax = max(float(np.nanmax(np.abs(parr))), 1.0)
        mach = parr / vmax

    R, Z = np.meshgrid(r, z, indexing="xy")
    s = max(1, int(args.stride))

    # Subsample and mask near-zero flow for cleaner arrows.
    Rs = R[::s, ::s]
    Zs = Z[::s, ::s]
    Urs = ur[::s, ::s]
    Uzs = uz[::s, ::s]
    Ms = mach[::s, ::s]
    speed = np.hypot(vr, vz)
    Sp = speed[::s, ::s]
    cut = 0.03 * float(np.nanmax(speed))
    mask = Sp > cut
    # Random thinning to mimic old sparse quiver style.
    rng = np.random.default_rng(args.sample_seed)
    keep = rng.random(mask.shape) < max(0.0, min(1.0, args.sample_frac))
    mask &= keep

    rr_wall, zz_wall = _read_wall_points(wpath)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0), dpi=180)
    q = ax.quiver(
        Rs[mask],
        Zs[mask],
        Urs[mask],
        Uzs[mask],
        Ms[mask],
        cmap="viridis",
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=18,
        width=0.003,
        headwidth=3.0,
        headlength=4.0,
    )

    if rr_wall is not None:
        ax.plot(rr_wall, zz_wall, color="k", linewidth=2.2)

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{args.title}\\n({src_label})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)

    cbar = fig.colorbar(q, ax=ax, pad=0.02)
    cbar.set_label("Mach number (vpar/cs)")

    # quick scalar checks for sign by lower/upper half
    zmid = 0.5 * (z.min() + z.max())
    lower = Z < zmid
    upper = ~lower
    print(f"Vector source: {src_label}")
    print(f"Mean vR (lower Z half): {float(np.nanmean(vr[lower])):.3e} m/s")
    print(f"Mean vR (upper Z half): {float(np.nanmean(vr[upper])):.3e} m/s")

    fig.savefig(out, dpi=220)
    print(f"Wrote: {out}")
    plt.show()


if __name__ == "__main__":
    main()
