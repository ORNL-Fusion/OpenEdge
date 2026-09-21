#!/usr/bin/env python3
"""Analyze ST40 grain-size response and outer-timestep convergence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_core_transit_smoke import (
    closest_wall,
    equilibrium_interpolator,
    read_dump,
    read_evaporation,
    read_wall,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
WALL_CLASSIFY_M = 0.02

CASES = {
    "dt2us": {
        "output": ROOT / "output_core_transit_size_scan_dt2us",
        "dt_s": 2.0e-6,
        "steps": 375000,
        "dump_steps": 500,
    },
    "dt1us": {
        "output": ROOT / "output_core_transit_size_scan_dt1us",
        "dt_s": 1.0e-6,
        "steps": 750000,
        "dump_steps": 1000,
    },
    "dt0p5us": {
        "output": ROOT / "output_core_transit_size_scan_dt0p5us",
        "dt_s": 0.5e-6,
        "steps": 1500000,
        "dump_steps": 2000,
    },
}


def wall_reactions(path: Path) -> int:
    matches = re.findall(r"reaction all:\s+(\d+)", (path / "log.openedge").read_text())
    if not matches:
        raise RuntimeError(f"surface-reaction total not found in {path / 'log.openedge'}")
    return int(matches[-1])


def classify_track(track: pd.DataFrame, final_step: int, wall: np.ndarray) -> dict:
    track = track.sort_values("step")
    last = track.iloc[-1]
    min_psi = float(track.psi_n.min())
    wall_distance, wall_segment = closest_wall(
        np.array([last.Z_m, last.R_m]), wall
    )
    alive_end = int(last.step) == final_step
    wall_hit = (not alive_end) and wall_distance < WALL_CLASSIFY_M
    entered_lcfs = min_psi < 1.0
    entered_core = min_psi < 0.51

    if alive_end:
        fate = "alive_at_end"
    elif wall_hit and last.Z_m < -0.2:
        fate = "lower_wall"
    elif wall_hit:
        fate = "upper_or_side_wall"
    elif entered_core:
        fate = "evaporated_inside_core"
    elif entered_lcfs:
        fate = "evaporated_inside_lcfs"
    else:
        fate = "evaporated_in_SOL"

    return {
        "entered_lcfs": entered_lcfs,
        "entered_core": entered_core,
        "terminal_fate": fate,
        "last_time_s": float(last.time_s),
        "last_Z_m": float(last.Z_m),
        "last_R_m": float(last.R_m),
        "min_Z_m": float(track.Z_m.min()),
        "min_psi_n": min_psi,
        "last_radius_um": float(last.radius_m * 1.0e6),
        "minimum_radius_um": float(track.radius_m.min() * 1.0e6),
        "maximum_temperature_K": float(track.temperature_K.max()),
        "wall_distance_mm": wall_distance * 1.0e3,
        "closest_wall_segment": wall_segment,
        "max_a_over_lambdaD": float(track.a_over_lambdaD.max()),
    }


def comparison(summary: pd.DataFrame, coarse: str, fine: str) -> dict:
    keys = ["diameter_um", "dropper_line", "speed_m_s", "angle_deg"]
    a = summary[summary.case == coarse].set_index(keys).sort_index()
    b = summary[summary.case == fine].set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise RuntimeError(f"{coarse}/{fine}: launch keys differ")

    dr_mm = 1.0e3 * np.hypot(
        a.last_R_m.to_numpy() - b.last_R_m.to_numpy(),
        a.last_Z_m.to_numpy() - b.last_Z_m.to_numpy(),
    )
    dt_ms = 1.0e3 * np.abs(a.last_time_s.to_numpy() - b.last_time_s.to_numpy())
    radius_delta = np.abs(
        a.last_radius_um.to_numpy() - b.last_radius_um.to_numpy()
    ) / a.index.get_level_values("diameter_um").to_numpy()

    return {
        "coarse": coarse,
        "fine": fine,
        "fate_match_fraction": float(
            (a.terminal_fate.to_numpy() == b.terminal_fate.to_numpy()).mean()
        ),
        "lcfs_entry_match_fraction": float(
            (a.entered_lcfs.to_numpy() == b.entered_lcfs.to_numpy()).mean()
        ),
        "core_entry_match_fraction": float(
            (a.entered_core.to_numpy() == b.entered_core.to_numpy()).mean()
        ),
        "terminal_position_delta_mm_median": float(np.median(dr_mm)),
        "terminal_position_delta_mm_p95": float(np.percentile(dr_mm, 95)),
        "terminal_position_delta_mm_max": float(np.max(dr_mm)),
        "last_sample_time_delta_ms_p95": float(np.percentile(dt_ms, 95)),
        "last_sample_time_delta_ms_max": float(np.max(dt_ms)),
        "last_radius_delta_over_initial_p95": float(np.percentile(radius_delta, 95)),
        "last_radius_delta_over_initial_max": float(np.max(radius_delta)),
    }


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT / "input/core_transit_size_scan.csv")
    wall = read_wall(ROOT / "input/st40_wall_axi.surf")
    psi_at = equilibrium_interpolator(ROOT / "input/plasma_st40_solps_corefill.h5")

    trajectories: list[pd.DataFrame] = []
    summaries: list[dict] = []
    integrity: dict[str, dict] = {}

    for case, config in CASES.items():
        output = config["output"]
        data = read_dump(output / "grain_state", dt_s=config["dt_s"])
        data["case"] = case
        data["psi_n"] = psi_at(data[["Z_m", "R_m"]].to_numpy())
        data = data.merge(manifest, on="id", how="left", validate="many_to_one")
        trajectories.append(data)

        for particle_id, track in data.groupby("id", sort=True):
            first = track.iloc[0]
            summaries.append(
                {
                    "case": case,
                    "id": int(particle_id),
                    "diameter_um": int(first.diameter_um),
                    "initial_radius_um": float(first.radius_um),
                    "initial_mass_kg": float(first.mass_kg),
                    "dropper_line": int(first.dropper_line),
                    "speed_m_s": float(first.speed_m_s),
                    "angle_deg": float(first.angle_deg),
                    **classify_track(track, int(config["steps"]), wall),
                }
            )

        reaction_count = wall_reactions(output)
        evaporated, remainder = read_evaporation(output / "evaporation_inventory")
        integrity[case] = {
            "wall_reactions": reaction_count,
            "continuously_evaporated_atoms": evaporated,
            "dustt_terminal_remainder_atoms": remainder,
        }

    trajectory = pd.concat(trajectories, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Close each inventory using the last sampled state of wall-hit/alive grains.
    for case in CASES:
        rows = summary[summary.case == case]
        retained = rows[rows.terminal_fate.str.contains("wall|alive", regex=True)]
        retained_atoms = float(
            np.sum(
                retained.initial_mass_kg.to_numpy()
                * (retained.last_radius_um.to_numpy()
                   / retained.initial_radius_um.to_numpy()) ** 3
                / (6.94 * 1.66053906660e-27)
            )
        )
        initial_atoms = float(manifest.atoms_per_grain.sum())
        ledger = integrity[case]
        ledger["retained_atoms_last_sample_estimate"] = retained_atoms
        ledger["initial_atoms"] = initial_atoms
        ledger["closure_fraction"] = (
            ledger["continuously_evaporated_atoms"]
            + ledger["dustt_terminal_remainder_atoms"]
            + retained_atoms
        ) / initial_atoms
        ledger["classified_wall_events"] = int(
            rows.terminal_fate.str.contains("wall").sum()
        )

    response = (
        summary.assign(
            lower_wall=lambda x: x.terminal_fate.eq("lower_wall"),
            evaporated=lambda x: x.terminal_fate.str.startswith("evaporated"),
        )
        .groupby(["case", "diameter_um"], as_index=False)
        .agg(
            grains=("id", "size"),
            entered_lcfs=("entered_lcfs", "sum"),
            entered_core=("entered_core", "sum"),
            lower_wall=("lower_wall", "sum"),
            evaporated=("evaporated", "sum"),
            median_min_psi_n=("min_psi_n", "median"),
            median_terminal_Z_m=("last_Z_m", "median"),
            median_max_temperature_K=("maximum_temperature_K", "median"),
        )
    )
    for column in ("entered_lcfs", "entered_core", "lower_wall", "evaporated"):
        response[f"{column}_fraction"] = response[column] / response.grains

    convergence = [
        comparison(summary, "dt2us", "dt1us"),
        comparison(summary, "dt1us", "dt0p5us"),
    ]
    finest = convergence[-1]
    status = "PASS" if (
        finest["fate_match_fraction"] == 1.0
        and finest["lcfs_entry_match_fraction"] == 1.0
        and finest["core_entry_match_fraction"] == 1.0
        and finest["terminal_position_delta_mm_p95"] < 2.0
        and finest["last_radius_delta_over_initial_p95"] < 0.01
        and all(
            item["wall_reactions"] == item["classified_wall_events"]
            and abs(item["closure_fraction"] - 1.0) < 0.01
            for item in integrity.values()
        )
    ) else "FAIL"

    acceptance = {
        "status": status,
        "description": (
            "Outer-timestep convergence for the deterministic diagnostic "
            "diameter scan; thermal adaptive substeps remain active."
        ),
        "diameters_um": sorted(manifest.diameter_um.unique().astype(int).tolist()),
        "launches_per_diameter": int(
            manifest.groupby("diameter_um").size().iloc[0]
        ),
        "total_grains_per_case": len(manifest),
        "cases": {
            key: {
                name: (
                    str(value.relative_to(ROOT))
                    if isinstance(value, Path)
                    else value
                )
                  for name, value in config.items()}
            for key, config in CASES.items()
        },
        "integrity": integrity,
        "convergence": convergence,
        "interpretation_guard": (
            "Diameter bins have equal diagnostic launch coverage and are not "
            "an experimental ST40 particle-size distribution."
        ),
    }

    trajectory.to_csv(ANALYSIS / "core_transit_size_scan_trajectories.csv", index=False)
    summary.to_csv(ANALYSIS / "core_transit_size_scan_summary.csv", index=False)
    response.to_csv(ANALYSIS / "core_transit_size_response.csv", index=False)
    (ANALYSIS / "core_transit_size_scan_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )
    print(json.dumps(acceptance, indent=2))
    print(response.to_string(index=False))
    if status != "PASS":
        raise SystemExit("FAIL: ST40 grain-size/timestep acceptance")


if __name__ == "__main__":
    main()
