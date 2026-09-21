#!/usr/bin/env python3
"""Analyze the deterministic ST40 ballistic and plasma-physics smoke cases."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
DT_S = 2.0e-6
PSI_CORE = 0.51
FINAL_STEP = 375000
WALL_CLASSIFY_M = 0.02

CASES = {
    "ballistic": ROOT / "output_core_transit_ballistic",
    "dustt2005_corefill": ROOT / "output_core_transit_cohort",
}
ANALYSIS = ROOT / "analysis"


def read_wall(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    i = lines.index("Points") + 1
    while not lines[i].strip():
        i += 1
    points = []
    while i < len(lines) and lines[i].strip():
        _, z, r = lines[i].split()
        points.append((float(z), float(r)))
        i += 1
    return np.asarray(points)


def read_dump(path: Path, dt_s: float = DT_S) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    rows: list[dict[str, float]] = []
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        count = int(lines[i + 3])
        names = lines[i + 8].split()[2:]
        for raw in lines[i + 9 : i + 9 + count]:
            row = dict(zip(names, map(float, raw.split())))
            row["step"] = step
            rows.append(row)
        i += 9 + count
    if not rows:
        raise RuntimeError(f"no particle rows in {path}")
    data = pd.DataFrame(rows)
    data["id"] = data["id"].astype(int)
    data["time_s"] = data["step"] * dt_s
    data = data.rename(
        columns={
            "x": "Z_m",
            "y": "R_m",
            "vx": "vZ_m_s",
            "vy": "vR_m_s",
            "radius": "radius_m",
            "temp": "temperature_K",
            "p_particulate_charge": "charge_e",
            "p_droplet_heating_q": "heating_W_m2",
            "p_droplet_a_over_lambdaD": "a_over_lambdaD",
            "p_droplet_oml_weight": "oml_weight",
        }
    )
    return data


def equilibrium_interpolator(path: Path) -> RegularGridInterpolator:
    with h5py.File(path, "r") as h5:
        r = h5["equilibrium/r"][:]
        z = h5["equilibrium/z"][:]
        psi = h5["equilibrium/psi"][:]
        axis = float(h5["equilibrium/psi_axis"][()])
        boundary = float(h5["equilibrium/psib"][()])
    return RegularGridInterpolator(
        (z, r), (psi - axis) / (boundary - axis),
        bounds_error=False, fill_value=np.nan,
    )


def closest_wall(point_zr: np.ndarray, wall_zr: np.ndarray) -> tuple[float, int]:
    a = wall_zr
    b = np.roll(wall_zr, -1, axis=0)
    ab = b - a
    frac = np.clip(np.sum((point_zr - a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0, 1)
    distance = np.linalg.norm(point_zr - (a + frac[:, None] * ab), axis=1)
    index = int(np.argmin(distance))
    return float(distance[index]), index + 1


def wall_reactions(path: Path) -> int:
    text = (path / "log.openedge").read_text()
    matches = re.findall(r"reaction droplet --> Li \[A:absorb\]:\s+(\d+)", text)
    if not matches:
        raise RuntimeError(f"wall-reaction tally not found in {path / 'log.openedge'}")
    return int(matches[-1])


def crossing_flags(track: pd.DataFrame) -> tuple[bool, bool]:
    track = track.sort_values("step")
    psi = track["psi_n"].to_numpy()
    z = track["Z_m"].to_numpy()
    entered = bool(np.nanmin(psi) < PSI_CORE)
    lower_exit = bool(
        np.any((psi[:-1] < PSI_CORE) & (psi[1:] >= PSI_CORE) & (z[1:] < 0.0))
    )
    return entered, lower_exit


def parse_species() -> tuple[float, float, float]:
    grain = next(
        line.split() for line in (ROOT / "input/grains.species").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    atom = next(
        line.split() for line in (ROOT / "input/atoms.species").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return float(grain[2]), float(grain[10]), float(atom[2])


def read_evaporation(path: Path) -> tuple[float, float]:
    data = np.loadtxt(path, comments="#")
    last = np.atleast_2d(data)[-1]
    return float(last[1]), float(last[2])


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT / "input/core_transit_cohort.csv")
    wall = read_wall(ROOT / "input/st40_wall_axi.surf")
    psi_at = equilibrium_interpolator(ROOT / "input/plasma_st40_solps_corefill.h5")

    trajectories = []
    summaries = []
    reaction_counts: dict[str, int] = {}

    for case, output in CASES.items():
        data = read_dump(output / "grain_state")
        data["case"] = case
        data["psi_n"] = psi_at(data[["Z_m", "R_m"]].to_numpy())
        data = data.merge(manifest, on="id", how="left", validate="many_to_one")
        trajectories.append(data)
        reaction_counts[case] = wall_reactions(output)

        for particle_id, track in data.groupby("id", sort=True):
            track = track.sort_values("step")
            last = track.iloc[-1]
            wall_distance, wall_segment = closest_wall(
                np.array([last.Z_m, last.R_m]), wall
            )
            entered, lower_exit = crossing_flags(track)
            alive_end = int(last.step) == FINAL_STEP
            wall_hit = (not alive_end) and wall_distance < WALL_CLASSIFY_M

            if alive_end:
                terminal_fate = "alive_at_end"
            elif wall_hit and last.Z_m < -0.2:
                terminal_fate = "lower_wall"
            elif wall_hit:
                terminal_fate = "upper_or_side_wall"
            elif entered:
                terminal_fate = "evaporated_inside_core"
            else:
                terminal_fate = "evaporated_in_SOL"

            summaries.append(
                {
                    "case": case,
                    "id": particle_id,
                    "dropper_line": int(last.dropper_line),
                    "speed_m_s": last.speed_m_s,
                    "angle_deg": last.angle_deg,
                    "entered_core": entered,
                    "crossed_lower_core_alive": lower_exit,
                    "terminal_fate": terminal_fate,
                    "last_time_s": last.time_s,
                    "last_Z_m": last.Z_m,
                    "last_R_m": last.R_m,
                    "min_Z_m": track.Z_m.min(),
                    "min_psi_n": track.psi_n.min(),
                    "last_radius_um": last.radius_m * 1.0e6,
                    "last_temperature_K": last.temperature_K,
                    "wall_distance_mm": wall_distance * 1.0e3,
                    "closest_wall_segment": wall_segment,
                    "max_a_over_lambdaD": (
                        track.a_over_lambdaD.max() if "a_over_lambdaD" in track else np.nan
                    ),
                    "fraction_samples_a_over_lambdaD_gt_1": (
                        float((track.a_over_lambdaD > 1.0).mean())
                        if "a_over_lambdaD" in track else np.nan
                    ),
                }
            )

    trajectory = pd.concat(trajectories, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Exact gravity oracle before the terminal wall hit.
    ballistic = trajectory[trajectory.case == "ballistic"].copy()
    t = ballistic.time_s.to_numpy()
    z_expected = (
        ballistic.z0_m.to_numpy() + ballistic.vz0_m_s.to_numpy() * t
        - 0.5 * 9.8 * t * t
    )
    r_expected = ballistic.r0_m.to_numpy() + ballistic.vr0_m_s.to_numpy() * t
    vz_expected = ballistic.vz0_m_s.to_numpy() - 9.8 * t
    ballistic_position_error = np.hypot(
        ballistic.Z_m.to_numpy() - z_expected,
        ballistic.R_m.to_numpy() - r_expected,
    )
    ballistic_velocity_error = np.hypot(
        ballistic.vZ_m_s.to_numpy() - vz_expected,
        ballistic.vR_m_s.to_numpy() - ballistic.vr0_m_s.to_numpy(),
    )

    grain_mass, initial_radius, atom_mass = parse_species()
    physics_summary = summary[summary.case == "dustt2005_corefill"]
    wall_rows = physics_summary[
        physics_summary.terminal_fate.isin(["lower_wall", "upper_or_side_wall"])
    ]
    wall_atoms_estimate = float(
        np.sum(grain_mass * (wall_rows.last_radius_um.to_numpy() * 1e-6 / initial_radius) ** 3)
        / atom_mass
    )
    evaporated_atoms, terminal_remainder_atoms = read_evaporation(
        CASES["dustt2005_corefill"] / "evaporation_inventory"
    )
    initial_atoms = len(manifest) * grain_mass / atom_mass
    closure = (
        evaporated_atoms + terminal_remainder_atoms + wall_atoms_estimate
    ) / initial_atoms

    fate_counts = {
        case: {k: int(v) for k, v in group.terminal_fate.value_counts().items()}
        for case, group in summary.groupby("case")
    }
    entered_counts = {
        case: int(group.entered_core.sum()) for case, group in summary.groupby("case")
    }
    lower_exit_counts = {
        case: int(group.crossed_lower_core_alive.sum())
        for case, group in summary.groupby("case")
    }
    classified_wall_counts = {
        case: int(group.terminal_fate.str.contains("wall").sum())
        for case, group in summary.groupby("case")
    }

    acceptance = {
        "status": "PASS" if (
            ballistic_position_error.max() < 1.0e-8
            and ballistic_velocity_error.max() < 1.0e-8
            and all(classified_wall_counts[c] == reaction_counts[c] for c in CASES)
            and abs(closure - 1.0) < 0.01
        ) else "FAIL",
        "cohort_size": len(manifest),
        "ballistic_max_position_error_m": float(ballistic_position_error.max()),
        "ballistic_max_velocity_error_m_s": float(ballistic_velocity_error.max()),
        "wall_reaction_counts": reaction_counts,
        "classified_wall_counts": classified_wall_counts,
        "entered_core_counts": entered_counts,
        "crossed_lower_core_alive_counts": lower_exit_counts,
        "terminal_fate_counts": fate_counts,
        "physics_inventory": {
            "initial_atoms": initial_atoms,
            "continuously_evaporated_atoms": evaporated_atoms,
            "dustt_terminal_remainder_atoms": terminal_remainder_atoms,
            "wall_retained_atoms_estimate_from_1ms_last_state": wall_atoms_estimate,
            "closure_fraction": closure,
        },
        "classification_note": (
            "Wall IDs are selected by terminal distance <20 mm and are required "
            "to equal the independent OpenEdge surface-reaction tally. Wall "
            "inventory uses the final 1-ms particle sample and is approximate."
        ),
    }

    trajectory.to_csv(ANALYSIS / "core_transit_trajectories.csv", index=False)
    summary.to_csv(ANALYSIS / "core_transit_cohort_summary.csv", index=False)
    (ANALYSIS / "core_transit_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )

    print(json.dumps(acceptance, indent=2))
    if acceptance["status"] != "PASS":
        raise SystemExit("FAIL: ST40 core-transit smoke acceptance")


if __name__ == "__main__":
    main()
