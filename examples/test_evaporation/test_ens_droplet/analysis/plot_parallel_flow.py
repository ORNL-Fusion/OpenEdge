#!/usr/bin/env python3
"""Plot parallel flow direction in the R-Z plane from plasma and magnetic fields."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


QE = 1.602176634e-19
MP = 1.6726219e-27


def read_wall_points(wall_surf_path: str):
    pts = []
    with open(wall_surf_path, "r", encoding="utf-8") as f:
        in_points = False
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "Points":
                in_points = True
                continue
            if s == "Lines":
                break
            if in_points:
                tok = s.split()
                if len(tok) >= 3:
                    pts.append((float(tok[1]), float(tok[2])))
    if not pts:
        return np.array([]), np.array([])
    arr = np.asarray(pts, dtype=float)
    return arr[:, 0], arr[:, 1]


def sample_indices(n_total: int, stride: int):
    if stride < 1:
        stride = 1
    return np.arange(0, n_total, stride, dtype=int)


def is_effectively_zero(arr: np.ndarray, tol: float = 1.0e-14) -> bool:
    return np.nanmax(np.abs(arr)) <= tol


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    case_root = os.path.dirname(here)
    evap_input = os.path.normpath(os.path.join(case_root, "..", "test_evaporation", "input"))

    parser = argparse.ArgumentParser(description="Plot parallel flow direction from plasma/bfield files")
    parser.add_argument("--plasma", default=os.path.join(evap_input, "plasma.h5"), help="Path to plasma.h5")
    parser.add_argument("--bfield", default=os.path.join(evap_input, "bfield.h5"), help="Path to bfield.h5")
    parser.add_argument("--wall", default=os.path.join(case_root, "wall.surf"), help="Path to wall.surf (optional overlay)")
    parser.add_argument("--stride", type=int, default=8, help="Quiver sampling stride")
    parser.add_argument("--min-pol-mach", type=float, default=0.03, help="Mask vectors with |M_pol| below this value")
    parser.add_argument(
        "--vector-source",
        choices=("from_bfield", "from_plasma"),
        default="from_bfield",
        help="Use vpar*Bhat poloidal projection (from_bfield) or parr_flow_r/z datasets (from_plasma)",
    )
    parser.add_argument(
        "--unit-vectors",
        action="store_true",
        help="Normalize arrows to unit vectors (off by default to preserve poloidal speed magnitude)",
    )
    parser.add_argument("--out", default="Figs/FlowDirection.png", help="Output PNG path")
    args = parser.parse_args()

    with h5py.File(args.plasma, "r") as plasma, h5py.File(args.bfield, "r") as bfield:
        r = np.asarray(plasma["r"][:], dtype=float)
        z = np.asarray(plasma["z"][:], dtype=float)

        # Prefer grouped datasets when available.
        ti = np.asarray(plasma["n_i"]["temp"][:], dtype=float) if "n_i" in plasma else np.asarray(plasma["temp_i"][:], dtype=float)
        te = np.asarray(plasma["n_e"]["temp"][:], dtype=float) if "n_e" in plasma else np.asarray(plasma["temp_e"][:], dtype=float)
        vpar = (
            np.asarray(plasma["n_i"]["parr_flow"][:], dtype=float)
            if "n_i" in plasma and "parr_flow" in plasma["n_i"]
            else np.asarray(plasma["parr_flow"][:], dtype=float)
        )

        # bfield uses br/bt/bz names.
        b_r = np.asarray(bfield["br"][:], dtype=float)
        b_phi = np.asarray(bfield["bt"][:], dtype=float)
        b_z = np.asarray(bfield["bz"][:], dtype=float)

    bmag = np.sqrt(b_r**2 + b_z**2 + b_phi**2)
    bmag = np.where(bmag > 0.0, bmag, 1.0)
    b_r_unit = b_r / bmag
    b_z_unit = b_z / bmag

    cs = np.sqrt(QE * (ti + te) / (2.0 * MP))
    cs = np.where(cs > 0.0, cs, 1.0)
    mach = vpar / cs

    if args.vector_source == "from_plasma":
        with h5py.File(args.plasma, "r") as plasma:
            if "parr_flow_r" not in plasma or "parr_flow_z" not in plasma:
                raise KeyError(
                    "vector-source=from_plasma requires datasets 'parr_flow_r' and 'parr_flow_z' in plasma.h5"
                )
            u_r = np.asarray(plasma["parr_flow_r"][:], dtype=float)
            u_z = np.asarray(plasma["parr_flow_z"][:], dtype=float)
        if is_effectively_zero(u_r) and np.nanmax(np.abs(b_r)) > 0.0 and np.nanmax(np.abs(vpar)) > 0.0:
            print(
                "Warning: parr_flow_r is effectively zero while br and vpar are non-zero. "
                "Falling back to projection from bfield (vpar * Br/|B|, vpar * Bz/|B|)."
            )
            u_r = vpar * b_r_unit
            u_z = vpar * b_z_unit
    else:
        # Poloidal projection of parallel flow: vpar * (Br,Bz)/|B|.
        u_r = vpar * b_r_unit
        u_z = vpar * b_z_unit

    m_r = u_r / cs
    m_z = u_z / cs
    m_pol = np.sqrt(m_r**2 + m_z**2)

    if args.unit_vectors:
        v2 = np.sqrt(m_r**2 + m_z**2)
        v2 = np.where(v2 > 0.0, v2, 1.0)
        U_all = m_r / v2
        V_all = m_z / v2
    else:
        U_all = m_r
        V_all = m_z

    R, Z = np.meshgrid(r, z)
    ridx = sample_indices(R.shape[1], args.stride)
    zidx = sample_indices(R.shape[0], args.stride)

    RR = R[np.ix_(zidx, ridx)].ravel()
    ZZ = Z[np.ix_(zidx, ridx)].ravel()
    U = U_all[np.ix_(zidx, ridx)].ravel()
    V = V_all[np.ix_(zidx, ridx)].ravel()
    M = mach[np.ix_(zidx, ridx)].ravel()
    MPOL = m_pol[np.ix_(zidx, ridx)].ravel()

    norm = Normalize(vmin=np.nanmin(mach), vmax=np.nanmax(mach))
    colors = plt.cm.viridis(norm(M))
    keep = np.isfinite(U) & np.isfinite(V) & np.isfinite(MPOL) & (MPOL >= args.min_pol_mach)

    RR = RR[keep]
    ZZ = ZZ[keep]
    U = U[keep]
    V = V[keep]
    colors = colors[keep]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2), dpi=300)
    ax.quiver(
        RR,
        ZZ,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=6 if not args.unit_vectors else 20,
        pivot="middle",
        color=colors,
        linewidth=0.8,
    )

    rw, zw = read_wall_points(args.wall)
    if rw.size and zw.size:
        ax.plot(rw, zw, "k", linewidth=1.0)
        ax.set_xlim(np.min(rw) - 0.1, np.max(rw) + 0.1)
        ax.set_ylim(np.min(zw) - 0.1, np.max(zw) + 0.1)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cbar.set_label("Mach number (vpar/cs)")

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Parallel Flow Direction (R-Z Projection)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
