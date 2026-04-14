#!/usr/bin/env python3
"""Build a surf_react pmi HDF5 table for W->W using the Eckstein 2007
revised Bohdansky-Yamamura sputter-yield fit, while preserving the
angular shape from the source F-TRIDYN table.

Differences vs build_west_pmi_surface.py:
  * spyld energy-dependence comes from Eckstein 2007 (normal incidence),
    not the F-TRIDYN source table.
  * E grid is extended from [10, 1000] eV (linear) to [10, 10000] eV
    (log-spaced) so divertor-scale impact energies are covered.
  * Sub-threshold noise is clamped to 0.
  * E_bind set to the Eckstein reference W surface binding 8.79 eV.
  * Angular shape is lifted from the source table as a per-angle factor
    averaged over energies where the source table is reliable
    (500 eV <= E <= 1000 eV), then applied to the Eckstein Y(E, 0).
  * RN is interpolated from the source onto the new E grid, with flat
    extrapolation above 1 keV.
  * RE is left at a constant 0.8 for now (explicit tech debt --
    should be replaced with an Eckstein reflection model later).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# Eckstein 2007 revised Bohdansky parameters for W->W self-sputtering
# (matches analysis/compare_eckstein_w_on_w.py).
DEFAULT_Z = 74.0
DEFAULT_M = 183.84
DEFAULT_ESB = 8.79
DEFAULT_ETH = 65.6
DEFAULT_Q = 18.08
DEFAULT_LAM = 0.8
DEFAULT_MU = 1.5


def lindhard_screening_length_A(Z1: float, Z2: float) -> float:
    a0 = 0.5292
    return 0.8853 * a0 / np.sqrt(Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0))


def reduced_energy(E_eV: np.ndarray, Z1: float, Z2: float,
                   M1: float, M2: float) -> np.ndarray:
    aL = lindhard_screening_length_A(Z1, Z2)
    k = 14.3996  # e^2/(4*pi*eps0) in eV*A
    return E_eV * (M2 / (M1 + M2)) * aL / (Z1 * Z2 * k)


def sn_KrC(eps: np.ndarray) -> np.ndarray:
    num = 0.5 * np.log(1.0 + 1.2288 * eps)
    den = eps + 0.1728 * np.sqrt(eps) + 0.008 * eps ** 0.1504
    return np.where(eps > 0.0, num / den, 0.0)


def eckstein_yield_normal(E_eV: np.ndarray,
                          Eth: float, q: float, lam: float, mu: float,
                          Z1: float = DEFAULT_Z, Z2: float = DEFAULT_Z,
                          M1: float = DEFAULT_M, M2: float = DEFAULT_M) -> np.ndarray:
    E = np.asarray(E_eV, dtype=np.float64)
    eps = reduced_energy(E, Z1, Z2, M1, M2)
    sn = sn_KrC(eps)
    above = E > Eth
    x = np.zeros_like(E)
    x[above] = (E[above] / Eth - 1.0) ** mu
    Y = q * sn * x / (lam + x)
    Y = np.where(above, Y, 0.0)
    return np.clip(Y, 0.0, None)


def angular_shape_from_source(E_src: np.ndarray, A_src: np.ndarray,
                              Y_src: np.ndarray,
                              E_lo: float = 500.0, E_hi: float = 1000.0) -> np.ndarray:
    """Extract a 1D angular factor a(theta) from the source table.

    Uses energies in [E_lo, E_hi] where the source yield is well above
    the sub-threshold noise floor. Returns a(theta) normalized to 1 at
    theta=0, with length len(A_src).
    """
    mask = (E_src >= E_lo) & (E_src <= E_hi)
    block = np.clip(Y_src[mask, :], 0.0, None)          # (nE', nA)
    y0 = block[:, 0]                                    # (nE',)
    good = y0 > 0.0
    ratios = block[good, :] / y0[good, None]            # (nE_good, nA)
    a = np.mean(ratios, axis=0)
    a[0] = 1.0  # enforce exact 1.0 at theta=0
    return a


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path,
                   default=root.parents[1].parent / "database" / "surface" / "74_on_74.h5",
                   help="Source F-TRIDYN table (needed for angular shape and RN).")
    p.add_argument("--output", type=Path,
                   default=root.parent / "input" / "74_on_74_pmi.h5")
    p.add_argument("--E-min", type=float, default=10.0)
    p.add_argument("--E-max", type=float, default=10000.0)
    p.add_argument("--nE", type=int, default=250)
    p.add_argument("--Eth", type=float, default=DEFAULT_ETH)
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument("--mu", type=float, default=DEFAULT_MU)
    p.add_argument("--ebind", type=float, default=DEFAULT_ESB,
                   help="Surface binding energy [eV].")
    p.add_argument("--re", type=float, default=0.8,
                   help="Constant energy reflection coefficient (tech debt).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.source, "r") as src:
        E_src = np.asarray(src["E"][:], dtype=np.float64)
        A_src = np.asarray(src["A"][:], dtype=np.float64)
        spyld_src = np.asarray(src["spyld"][:], dtype=np.float64)
        rn_src = np.asarray(src["rfyld"][:], dtype=np.float64)

    # ---- new E grid (log-spaced, extended to 10 keV) ----
    E_new = np.logspace(np.log10(args.E_min), np.log10(args.E_max), args.nE)

    # ---- angular shape from source ----
    a_theta = angular_shape_from_source(E_src, A_src, spyld_src)
    # A grid is unchanged
    A_new = A_src.copy()

    # ---- Eckstein yield on new E grid, then splay across angles ----
    Y0 = eckstein_yield_normal(E_new, args.Eth, args.q, args.lam, args.mu)
    spyld_new = Y0[:, None] * a_theta[None, :]
    spyld_new = np.clip(spyld_new, 0.0, None)

    # ---- RN: interpolate source onto new E grid per angle, flat beyond src range ----
    rn_src_clipped = np.clip(rn_src, 0.0, 1.0)
    rn_new = np.empty((len(E_new), len(A_new)), dtype=np.float64)
    for ja in range(len(A_new)):
        rn_new[:, ja] = np.interp(E_new, E_src, rn_src_clipped[:, ja],
                                  left=rn_src_clipped[0, ja],
                                  right=rn_src_clipped[-1, ja])

    # ---- RE: constant (explicit tech debt) ----
    re_new = np.full((len(E_new), len(A_new)), float(args.re), dtype=np.float64)

    # ---- diagnostics ----
    print(f"Output: {args.output}")
    print(f"E grid: {args.E_min:.1f} -> {args.E_max:.1f} eV, {args.nE} log points")
    print(f"A grid: {A_new.min():.2f} -> {A_new.max():.2f} deg, {len(A_new)} points")
    print(f"E_bind: {args.ebind} eV   (was 11.75)")
    print(f"Eckstein params: Eth={args.Eth} eV, q={args.q}, "
          f"lam={args.lam}, mu={args.mu}")
    print()
    print("Angular factor samples:")
    for theta in [0, 15, 30, 45, 60, 75, 85]:
        ja = int(np.argmin(np.abs(A_new - theta)))
        print(f"  theta={A_new[ja]:5.1f}  a(theta)={a_theta[ja]:.3f}")
    print()
    print("Y(E, theta=0) on new grid:")
    for Ev in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        ie = int(np.argmin(np.abs(E_new - Ev)))
        print(f"  E={E_new[ie]:8.1f} eV   Y={spyld_new[ie, 0]:.4e}   "
              f"RN={rn_new[ie, 0]:.4e}")

    # ---- write ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as dst:
        dst.create_dataset("E", data=E_new)
        dst.create_dataset("A", data=A_new)
        dst.create_dataset("spyld", data=spyld_new)
        dst.create_dataset("RN", data=rn_new)
        dst.create_dataset("RE", data=re_new)
        dst.create_dataset("E_bind", data=np.float64(args.ebind))
        dst.attrs["source"] = "Eckstein 2007 revised Bohdansky fit (W->W)"
        dst.attrs["Eth_eV"] = float(args.Eth)
        dst.attrs["q"] = float(args.q)
        dst.attrs["lambda"] = float(args.lam)
        dst.attrs["mu"] = float(args.mu)
        dst.attrs["E_sb_eV"] = float(args.ebind)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
