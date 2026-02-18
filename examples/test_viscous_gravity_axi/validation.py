#!/usr/bin/env python3
"""
Validate SPARTA/OpenEdge viscous+gravity droplet test cases against the
closed-form model for constant Epstein drag frequency:

    dvz/dt = g - nuE * (vz - u_g)

Solution:
    v_inf = u_g + g/nuE
    vz(t) = (v0 - v_inf) * exp(-nuE * t) + v_inf

The script reads SPARTA dump files (case.1..case.4), compares simulation vs model,
prints error metrics, and saves a 2x2 figure.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- Physical constants ---
MP = 1.67262192369e-27      # proton mass [kg]
ECHARGE = 1.602176634e-19   # electron charge [C]


@dataclass
class CaseSpec:
    path: str
    dt: float
    g: float
    nuE: float
    u_g: float
    v0_override: float | None
    title: str


def epstein_nue(Ni: float, Ti_eV: float, rd_m: float, rho_d: float,
                alphaE: float = 1.26, A_background: float = 2.0) -> float:
    """Constant Epstein drag frequency used by these validation cases."""
    mi = A_background * MP
    vth = math.sqrt(8.0 * Ti_eV * ECHARGE / (math.pi * mi))
    rho_g = Ni * mi
    return alphaE * rho_g * vth / (rho_d * rd_m)


def parse_dump_first_particle(path: str, dt: float):
    """
    Parse a SPARTA dump and return time, vz, z for the first particle id.

    Supports ITEM: ATOMS with dynamic column order by reading header names.
    """
    timesteps = []
    ids = []
    vz_vals = []
    z_vals = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    timestep = None
    n_atoms = None
    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue

        if line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue

        if line.startswith("ITEM: ATOMS"):
            if timestep is None or n_atoms is None:
                raise RuntimeError(f"Malformed dump near line {i+1} in {path}")

            cols = line.split()[2:]
            colmap = {name: idx for idx, name in enumerate(cols)}
            required = ["id", "z", "vz"]
            missing = [k for k in required if k not in colmap]
            if missing:
                raise RuntimeError(f"Missing required columns {missing} in {path}")

            for j in range(n_atoms):
                row = lines[i + 1 + j].split()
                timesteps.append(timestep)
                ids.append(int(row[colmap["id"]]))
                z_vals.append(float(row[colmap["z"]]))
                vz_vals.append(float(row[colmap["vz"]]))

            i += 1 + n_atoms
            continue

        i += 1

    if not ids:
        raise RuntimeError(f"No particles parsed from {path}")

    ids = np.asarray(ids)
    pick = np.unique(ids)[0]
    mask = (ids == pick)

    t = np.asarray(timesteps, dtype=float)[mask] * dt
    vz = np.asarray(vz_vals, dtype=float)[mask]
    z = np.asarray(z_vals, dtype=float)[mask]
    return t, vz, z


def analytic_vz(t: np.ndarray, v0: float, g: float, nuE: float, u_g: float):
    t_rel = t - t[0]
    v_inf = u_g + g / nuE
    return (v0 - v_inf) * np.exp(-nuE * t_rel) + v_inf, v_inf


def error_metrics(v_sim: np.ndarray, v_model: np.ndarray):
    err = v_sim - v_model
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_abs = float(np.max(np.abs(err)))
    return rmse, max_abs


def annotate_case(ax, t: np.ndarray, v_sim: np.ndarray, v_model: np.ndarray, v_inf: float):
    tail = max(5, len(v_sim) // 10)
    sim_end = float(np.median(v_sim[-tail:]))
    model_end = float(np.median(v_model[-tail:]))
    text = (
        rf"$v_z(t_{{end}})$ sim = {sim_end:.2f} m/s" + "\n" +
        rf"$v_z(t_{{end}})$ model = {model_end:.2f} m/s" + "\n" +
        rf"$v_\infty$ = {v_inf:.2f} m/s"
    )
    ax.text(
        0.04,
        0.06,
        text,
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )


def make_plot(cases: list[CaseSpec], out_png: Path):
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300, constrained_layout=True)
    tags = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (ax, spec) in enumerate(zip(axes.ravel(), cases)):
        t, vz, _z = parse_dump_first_particle(spec.path, spec.dt)
        v0 = spec.v0_override if spec.v0_override is not None else float(vz[0])
        vz_model, v_inf = analytic_vz(t, v0, spec.g, spec.nuE, spec.u_g)

        rmse, max_abs = error_metrics(vz, vz_model)
        print(
            f"{Path(spec.path).name}: RMSE={rmse:.3e} m/s, "
            f"max|err|={max_abs:.3e} m/s, v_inf={v_inf:.3f} m/s"
        )

        ax.plot(
            t,
            vz,
            linestyle="None",
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            markeredgewidth=1.1,
            label="simulation",
        )
        ax.plot(t, vz_model, "-", linewidth=1.8, label="analytic")

        annotate_case(ax, t, vz, vz_model, v_inf)
        ax.text(0.02, 0.98, tags[idx], transform=ax.transAxes, va="top", ha="left", fontsize=14)
        ax.set_title(spec.title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$v_z$ (m/s)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.legend(frameon=True, framealpha=0.9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Validate viscous+gravity test cases.")
    parser.add_argument("--dt", type=float, default=1.0e-5, help="Simulation timestep [s].")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Figs/viscous_gravity_validation.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    # Baseline parameters used in current validation setup
    nuE_case1 = epstein_nue(Ni=1.5746e20, Ti_eV=10.0, rd_m=50e-6, rho_d=534.0)
    nuE_case4 = epstein_nue(Ni=3.1492e20, Ti_eV=10.0, rd_m=50e-6, rho_d=534.0)

    cases = [
        CaseSpec(
            path="case.1",
            dt=args.dt,
            g=-9.81,
            nuE=nuE_case1,
            u_g=0.0,
            v0_override=0.0,
            title=r"$v_z(0)=0$, $u_g=0$, $\nu_E=0.868$ s$^{-1}$",
        ),
        CaseSpec(
            path="case.2",
            dt=args.dt,
            g=-9.81,
            nuE=nuE_case1,
            u_g=0.0,
            v0_override=0.0,
            title=r"$v\perp g$, $u_g=0$, $\nu_E=0.868$ s$^{-1}$",
        ),
        CaseSpec(
            path="case.3",
            dt=args.dt,
            g=-9.81,
            nuE=nuE_case1,
            u_g=1.0,
            v0_override=0.0,
            title=r"$v_z(0)=0$, $u_g=1$ m/s, $\nu_E=0.868$ s$^{-1}$",
        ),
        CaseSpec(
            path="case.4",
            dt=args.dt,
            g=-9.81,
            nuE=nuE_case4,
            u_g=1.0,
            v0_override=0.0,
            title=r"$v_z(0)=0$, $u_g=1$ m/s, $\nu_E=1.736$ s$^{-1}$",
        ),
    ]

    make_plot(cases, args.out)


if __name__ == "__main__":
    main()
