#!/usr/bin/env python3
"""Compare 74_on_74_pmi.h5 sputter yield at normal incidence against the
Eckstein 2007 revised Bohdansky-Yamamura fit for W -> W self-sputtering.

Eckstein revised Bohdansky formula (Eckstein, "Sputtering Yields", in
Behrisch & Eckstein eds., Topics Appl. Phys. 110, Springer 2007, chap. 3):

    Y(E) = q * s_n^KrC(eps) * (E/Eth - 1)^mu
                               ------------------------
                               lambda + (E/Eth - 1)^mu

with KrC reduced nuclear stopping (Wilson-Haggmark-Biersack):

    s_n^KrC(eps) = 0.5 * ln(1 + 1.2288*eps)
                   -----------------------------------------
                   eps + 0.1728*sqrt(eps) + 0.008*eps^0.1504

Parameter set below is the commonly cited Eckstein 2007 fit for W->W.
These are the defaults you can override via command-line flags if you
want to test an alternative parameter set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Eckstein 2007 revised Bohdansky parameters for W -> W self-sputtering.
# Source: Eckstein, Topics Appl. Phys. 110 (2007), Table 4.1 region.
DEFAULT_Z1 = 74.0
DEFAULT_Z2 = 74.0
DEFAULT_M1 = 183.84
DEFAULT_M2 = 183.84
DEFAULT_ESB = 8.79       # surface binding energy [eV]
DEFAULT_ETH = 65.6       # threshold energy [eV]
DEFAULT_Q = 18.08
DEFAULT_LAM = 0.8
DEFAULT_MU = 1.5


def lindhard_screening_length_A(Z1: float, Z2: float) -> float:
    """Lindhard screening length in Angstroms."""
    a0 = 0.5292  # Bohr radius [A]
    return 0.8853 * a0 / np.sqrt(Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0))


def reduced_energy(E_eV: np.ndarray, Z1: float, Z2: float,
                   M1: float, M2: float) -> np.ndarray:
    """Lindhard reduced energy epsilon_L."""
    aL = lindhard_screening_length_A(Z1, Z2)       # [A]
    # e^2/(4*pi*eps0) = 14.3996 eV*A
    k = 14.3996
    return E_eV * (M2 / (M1 + M2)) * aL / (Z1 * Z2 * k)


def sn_KrC(eps: np.ndarray) -> np.ndarray:
    """KrC reduced nuclear stopping cross section (Wilson-Haggmark-Biersack)."""
    num = 0.5 * np.log(1.0 + 1.2288 * eps)
    den = eps + 0.1728 * np.sqrt(eps) + 0.008 * eps ** 0.1504
    return np.where(eps > 0.0, num / den, 0.0)


def eckstein_yield(E_eV: np.ndarray,
                   Z1: float = DEFAULT_Z1, Z2: float = DEFAULT_Z2,
                   M1: float = DEFAULT_M1, M2: float = DEFAULT_M2,
                   Eth: float = DEFAULT_ETH,
                   q: float = DEFAULT_Q,
                   lam: float = DEFAULT_LAM,
                   mu: float = DEFAULT_MU) -> np.ndarray:
    """Eckstein 2007 revised Bohdansky sputter yield at normal incidence."""
    E = np.asarray(E_eV, dtype=np.float64)
    eps = reduced_energy(E, Z1, Z2, M1, M2)
    sn = sn_KrC(eps)
    x = np.where(E > Eth, (E / Eth - 1.0) ** mu, 0.0)
    Y = q * sn * x / (lam + x)
    return np.where(E > Eth, Y, 0.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=root.parent / "input" / "74_on_74_pmi.h5")
    p.add_argument("--output", type=Path,
                   default=root / "eckstein_w_on_w_compare.png")
    p.add_argument("--Eth", type=float, default=DEFAULT_ETH)
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument("--mu", type=float, default=DEFAULT_MU)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.input, "r") as f:
        E_tab = np.asarray(f["E"][:])
        A_tab = np.asarray(f["A"][:])
        spyld = np.asarray(f["spyld"][:])

    ia0 = int(np.argmin(np.abs(A_tab - 0.0)))
    Y_tab = spyld[:, ia0]
    Y_tab_clipped = np.clip(Y_tab, 0.0, None)

    # Evaluate Eckstein fit on the same E grid + an extended grid to 10 keV.
    E_ext = np.logspace(np.log10(30.0), np.log10(1.0e4), 400)
    Y_eck_tab = eckstein_yield(E_tab, Eth=args.Eth, q=args.q,
                               lam=args.lam, mu=args.mu)
    Y_eck_ext = eckstein_yield(E_ext, Eth=args.Eth, q=args.q,
                               lam=args.lam, mu=args.mu)

    print(f"Eckstein parameters used: Eth={args.Eth} eV, q={args.q}, "
          f"lambda={args.lam}, mu={args.mu}")
    print()
    print(f"{'E [eV]':>10} {'Y_table':>12} {'Y_eck':>12} {'ratio':>10}")
    for Ev in [50, 100, 200, 300, 500, 700, 1000]:
        ie = int(np.argmin(np.abs(E_tab - Ev)))
        yt = max(Y_tab[ie], 0.0)
        ye = Y_eck_tab[ie]
        ratio = yt / ye if ye > 0 else float('nan')
        print(f"{E_tab[ie]:10.1f} {yt:12.4e} {ye:12.4e} {ratio:10.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(E_tab, Y_tab_clipped, "o-", label="table (74_on_74_pmi.h5)",
               markersize=3, linewidth=1.2)
    ax1.loglog(E_ext, Y_eck_ext, "-", label="Eckstein 2007 fit",
               linewidth=1.5, color="C3")
    ax1.axvline(1000.0, color="gray", linestyle=":", alpha=0.5,
                label="table E_max")
    ax1.set_xlabel("Impact energy [eV]")
    ax1.set_ylabel("Y (W on W, theta=0)")
    ax1.set_title("Sputter yield vs impact energy")
    ax1.set_xlim(30, 1e4)
    ax1.set_ylim(1e-4, 3.0)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="lower right")

    mask = (E_tab >= args.Eth * 1.2) & (E_tab <= 1000.0)
    ratio_arr = np.where(Y_eck_tab > 0,
                         Y_tab_clipped / np.maximum(Y_eck_tab, 1e-30),
                         np.nan)
    ax2.semilogx(E_tab[mask], ratio_arr[mask], "o-", markersize=3)
    ax2.axhline(1.0, color="gray", linestyle="--")
    ax2.set_xlabel("Impact energy [eV]")
    ax2.set_ylabel("Y_table / Y_Eckstein")
    ax2.set_title("Ratio (normal incidence)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(0, 3)

    fig.suptitle("W -> W self-sputter: OpenEdge table vs Eckstein 2007 fit",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
