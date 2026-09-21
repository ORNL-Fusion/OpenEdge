#!/usr/bin/env python3
"""Analyze the deterministic OML heat-scale bracket for ST40 grains."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_core_transit_smoke import (
    equilibrium_interpolator,
    read_dump,
    read_evaporation,
    read_wall,
)
from analyze_core_transit_size_scan import classify_track, wall_reactions


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FINAL_STEP = 375000
DT_S = 2.0e-6
CASES = {
    "hs000": 0.0,
    "hs025": 0.25,
    "hs050": 0.50,
    "hs075": 0.75,
    "hs100": 1.00,
    "hs150": 1.50,
    "hs200": 2.00,
}


def wilson(k: pd.Series, n: pd.Series, z: float = 1.959963984540054) -> tuple[pd.Series, pd.Series]:
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - half, center + half


def wall_reactions_by_diameter(path: Path) -> dict[int, int]:
    text = (path / "log.openedge").read_text()
    return {
        int(diameter): int(count)
        for diameter, count in re.findall(
            r"reaction grain_d(\d+)um --> Li \[A:absorb\]:\s+(\d+)", text
        )
    }


def reconcile_wall_events(rows: list[dict], expected: dict[int, int]) -> tuple[list[dict], float | None]:
    """Assign exact surface-ledger events to nearest terminal tracks per size.

    A 1-ms trajectory dump can precede a fast wall impact by several cm.  The
    surface ledger is exact but lacks particle IDs, so within each diameter we
    assign its known event count to the terminal tracks closest to the wall.
    A separation gap is reported to make this deterministic association auditable.
    """
    frame = pd.DataFrame(rows)
    gaps: list[float] = []
    for diameter, group in frame.groupby("diameter_um"):
        indices = group.sort_values("wall_distance_mm").index.to_numpy()
        nwall = int(expected.get(int(diameter), 0))
        wall_indices = set(indices[:nwall].tolist())
        if 0 < nwall < len(indices):
            distances = frame.loc[indices, "wall_distance_mm"].to_numpy()
            gaps.append(float(distances[nwall] - distances[nwall - 1]))
        for index in indices:
            row = frame.loc[index]
            if index in wall_indices:
                frame.at[index, "terminal_fate"] = (
                    "lower_wall" if row.last_Z_m < -0.2 else "upper_or_side_wall"
                )
            elif row.entered_core:
                frame.at[index, "terminal_fate"] = "evaporated_inside_core"
            elif row.entered_lcfs:
                frame.at[index, "terminal_fate"] = "evaporated_inside_lcfs"
            else:
                frame.at[index, "terminal_fate"] = "evaporated_in_SOL"
    return frame.to_dict("records"), (min(gaps) if gaps else None)


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT / "input/core_transit_size_scan.csv")
    wall = read_wall(ROOT / "input/st40_wall_axi.surf")
    psi_at = equilibrium_interpolator(ROOT / "input/plasma_st40_solps_corefill.h5")
    initial_atoms = float(manifest.atoms_per_grain.sum())

    summaries: list[dict] = []
    integrity: dict[str, dict] = {}
    for case, scale in CASES.items():
        output = ROOT / f"output_core_transit_heat_scan_{case}"
        data = read_dump(output / "grain_state", dt_s=DT_S)
        data["psi_n"] = psi_at(data[["Z_m", "R_m"]].to_numpy())
        data = data.merge(manifest, on="id", how="left", validate="many_to_one")
        case_rows: list[dict] = []
        for particle_id, track in data.groupby("id", sort=True):
            first = track.iloc[0]
            case_rows.append(
                {
                    "case": case,
                    "heat_scale": scale,
                    "id": int(particle_id),
                    "diameter_um": int(first.diameter_um),
                    "initial_radius_um": float(first.radius_um),
                    "initial_mass_kg": float(first.mass_kg),
                    "dropper_line": int(first.dropper_line),
                    "speed_m_s": float(first.speed_m_s),
                    "angle_deg": float(first.angle_deg),
                    **classify_track(track, FINAL_STEP, wall),
                }
            )

        reaction_count = wall_reactions(output)
        reaction_by_diameter = wall_reactions_by_diameter(output)
        case_rows, minimum_gap = reconcile_wall_events(case_rows, reaction_by_diameter)
        summaries.extend(case_rows)
        evaporated, remainder = read_evaporation(output / "evaporation_inventory")
        integrity[case] = {
            "heat_scale": scale,
            "wall_reactions": reaction_count,
            "wall_reactions_by_diameter_um": reaction_by_diameter,
            "minimum_wall_assignment_gap_mm": minimum_gap,
            "continuously_evaporated_atoms": evaporated,
            "dustt_terminal_remainder_atoms": remainder,
        }

    summary = pd.DataFrame(summaries)
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
        .groupby(["case", "heat_scale", "diameter_um"], as_index=False)
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
        lo, hi = wilson(response[column], response.grains)
        response[f"{column}_lo95"] = lo
        response[f"{column}_hi95"] = hi

    aggregate = (
        summary.assign(
            lower_wall=lambda x: x.terminal_fate.eq("lower_wall"),
            evaporated=lambda x: x.terminal_fate.str.startswith("evaporated"),
        )
        .groupby(["case", "heat_scale"], as_index=False)
        .agg(
            grains=("id", "size"),
            entered_lcfs=("entered_lcfs", "sum"),
            entered_core=("entered_core", "sum"),
            lower_wall=("lower_wall", "sum"),
            evaporated=("evaporated", "sum"),
        )
    )
    for column in ("entered_lcfs", "entered_core", "lower_wall", "evaporated"):
        aggregate[f"{column}_fraction"] = aggregate[column] / aggregate.grains
        lo, hi = wilson(aggregate[column], aggregate.grains)
        aggregate[f"{column}_lo95"] = lo
        aggregate[f"{column}_hi95"] = hi

    baseline = summary[summary.case == "hs100"].set_index("id").sort_index()
    reference = pd.read_csv(ANALYSIS / "core_transit_size_scan_summary.csv")
    reference = reference[reference.case == "dt2us"].set_index("id").sort_index()
    baseline_match = bool(
        baseline.index.equals(reference.index)
        and (baseline.terminal_fate == reference.terminal_fate).all()
        and (baseline.entered_lcfs == reference.entered_lcfs).all()
        and (baseline.entered_core == reference.entered_core).all()
    )
    integrity_pass = all(
        row["wall_reactions"] == row["classified_wall_events"]
        and abs(row["closure_fraction"] - 1.0) < 0.01
        and (row["minimum_wall_assignment_gap_mm"] is None
             or row["minimum_wall_assignment_gap_mm"] > 2.0)
        for row in integrity.values()
    )
    status = "PASS" if baseline_match and integrity_pass else "FAIL"
    acceptance = {
        "status": status,
        "description": (
            "Deterministic DUSTT-2005 OML heat-scale sensitivity at the "
            "accepted 2-us outer timestep. The scale multiplies net OML "
            "surface heating and does not modify Te or ne."
        ),
        "heat_scales": list(CASES.values()),
        "grains_per_scale": len(manifest),
        "baseline_reproduces_dt2us": baseline_match,
        "integrity": integrity,
        "uncertainty": "Wilson 95% binomial intervals over deterministic launch states",
    }

    summary.to_csv(ANALYSIS / "core_transit_heat_scale_summary.csv", index=False)
    response.to_csv(ANALYSIS / "core_transit_heat_scale_response.csv", index=False)
    aggregate.to_csv(ANALYSIS / "core_transit_heat_scale_aggregate.csv", index=False)
    (ANALYSIS / "core_transit_heat_scale_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )
    print(json.dumps(acceptance, indent=2))
    print(aggregate.to_string(index=False))
    if status != "PASS":
        raise SystemExit("FAIL: heat-scale scan acceptance")


if __name__ == "__main__":
    main()
