#!/usr/bin/env python3
"""Build engine-sampled background and spatial evaporation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from analyze_core_transit_smoke import equilibrium_interpolator, read_wall


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
GRID_DUMP = ROOT / "output_core_transit_background_probe/background_grid"
AMU_KG = 1.66053906660e-27
LI_MASS_AMU = 6.94


def read_grid_dump(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS"))
    names = lines[header].split()[2:]
    rows = [line.split() for line in lines[header + 1 :] if line.strip()]
    data = pd.DataFrame(rows, columns=names).astype(float)
    data = data.rename(
        columns={
            "c_cbg[1]": "te_eV",
            "c_cbg[2]": "ne_m3",
            "c_cbg[3]": "q_par_W_m2",
            "c_cbg[4]": "q_perp_W_m2",
            "c_cbg[5]": "q_mag_W_m2",
        }
    )
    # Native axisymmetric OpenEdge slots are x=Z and y=R.
    data["Z_m"] = 0.5 * (data.xlo + data.xhi)
    data["R_m"] = 0.5 * (data.ylo + data.yhi)
    data["dZ_m"] = data.xhi - data.xlo
    data["dR_m"] = data.yhi - data.ylo
    wall_zr = read_wall(ROOT / "input/st40_wall_axi.surf")
    vessel = MplPath(np.column_stack([wall_zr[:, 1], wall_zr[:, 0]]), closed=True)
    data["inside_vessel"] = vessel.contains_points(data[["R_m", "Z_m"]])
    return data


def evaporation_segments() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trajectories = pd.read_csv(ANALYSIS / "core_transit_size_scan_trajectories.csv")
    summaries = pd.read_csv(ANALYSIS / "core_transit_size_scan_summary.csv")
    trajectories = trajectories[trajectories.case == "dt0p5us"].copy()
    summaries = summaries[summaries.case == "dt0p5us"].set_index("id")
    accepted = json.loads(
        (ANALYSIS / "core_transit_size_scan_acceptance.json").read_text()
    )
    exact_atoms = accepted["integrity"]["dt0p5us"]["continuously_evaporated_atoms"]

    rows: list[dict] = []
    terminal: list[dict] = []
    for particle_id, track in trajectories.groupby("id", sort=True):
        track = track.sort_values("step")
        initial_atoms = float(track.atoms_per_grain.iloc[0])
        initial_radius = float(track.radius_um.iloc[0]) * 1.0e-6
        remaining = initial_atoms * (track.radius_m.to_numpy() / initial_radius) ** 3
        for i, loss in enumerate(np.maximum(remaining[:-1] - remaining[1:], 0.0)):
            if loss <= 0.0:
                continue
            a, b = track.iloc[i], track.iloc[i + 1]
            rows.append(
                {
                    "id": int(particle_id),
                    "diameter_um": int(a.diameter_um),
                    "R_m": 0.5 * (a.R_m + b.R_m),
                    "Z_m": 0.5 * (a.Z_m + b.Z_m),
                    "time_s": 0.5 * (a.time_s + b.time_s),
                    "evaporated_atoms_raw": float(loss),
                    "segment_kind": "resolved_interval",
                }
            )

        fate = str(summaries.loc[particle_id, "terminal_fate"])
        if fate.startswith("evaporated"):
            cutoff_atoms = initial_atoms * 0.1**3
            terminal_loss = max(float(remaining[-1] - cutoff_atoms), 0.0)
            last = track.iloc[-1]
            if terminal_loss > 0.0:
                rows.append(
                    {
                        "id": int(particle_id),
                        "diameter_um": int(last.diameter_um),
                        "R_m": float(last.R_m),
                        "Z_m": float(last.Z_m),
                        "time_s": float(last.time_s),
                        "evaporated_atoms_raw": terminal_loss,
                        "segment_kind": "last_sample_to_dustt_cutoff",
                    }
                )
            terminal.append(
                {
                    "id": int(particle_id),
                    "diameter_um": int(last.diameter_um),
                    "R_m": float(last.R_m),
                    "Z_m": float(last.Z_m),
                    "time_s": float(last.time_s),
                    "terminal_fate": fate,
                    "initial_atoms": initial_atoms,
                }
            )

    segments = pd.DataFrame(rows)
    terminals = pd.DataFrame(terminal)
    reconstructed = float(segments.evaporated_atoms_raw.sum())
    normalization = exact_atoms / reconstructed
    segments["evaporated_atoms"] = segments.evaporated_atoms_raw * normalization
    segments["fraction_of_exact_vapor"] = segments.evaporated_atoms / exact_atoms
    audit = {
        "status": "PASS" if abs(normalization - 1.0) < 1.0e-3 else "FAIL",
        "source_case": "dt0p5us",
        "dump_interval_s": 1.0e-3,
        "terminal_evaporation_events": len(terminals),
        "exact_continuous_vapor_atoms": exact_atoms,
        "trajectory_reconstructed_atoms_raw": reconstructed,
        "trajectory_capture_fraction": reconstructed / exact_atoms,
        "map_normalization_factor": normalization,
        "weighting_guard": (
            "The map is Li-atom weighted within an equal-count diagnostic "
            "diameter scan; it is not weighted by a measured ST40 size distribution."
        ),
    }
    return segments, terminals, audit


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    background = read_grid_dump(GRID_DUMP)
    segments, terminals, audit = evaporation_segments()

    psi_at = equilibrium_interpolator(ROOT / "input/plasma_st40_solps_corefill.h5")
    segments["psi_n"] = psi_at(segments[["Z_m", "R_m"]].to_numpy())
    weights = segments.evaporated_atoms.to_numpy()
    psi = segments.psi_n.to_numpy()
    total_weight = float(weights.sum())
    audit["spatial_inventory"] = {
        "atom_weighted_centroid_R_m": float(np.average(segments.R_m, weights=weights)),
        "atom_weighted_centroid_Z_m": float(np.average(segments.Z_m, weights=weights)),
        "vapor_fraction_in_SOL_psi_ge_1": float(weights[psi >= 1.0].sum() / total_weight),
        "vapor_fraction_inside_LCFS_0p51_to_1": float(
            weights[(psi < 1.0) & (psi >= 0.51)].sum() / total_weight
        ),
        "vapor_fraction_inside_core_psi_lt_0p51": float(
            weights[psi < 0.51].sum() / total_weight
        ),
    }

    inside = background[background.inside_vessel]
    core = inside[
        inside.R_m.between(0.45, 0.55) & inside.Z_m.between(-0.10, 0.10)
    ]
    with h5py.File(ROOT / "input/plasma_st40_solps_corefill.h5", "r") as h5:
        imported_q = [
            name for name in ("q_par", "q_perp", "mesh/q_par", "mesh/q_perp")
            if name in h5
        ]
    q_mag = inside.q_mag_W_m2.to_numpy()
    q_identity_error = float(
        np.max(
            np.abs(
                q_mag
                - np.hypot(
                    inside.q_par_W_m2.to_numpy(),
                    inside.q_perp_W_m2.to_numpy(),
                )
            )
        )
    )
    expected_q = ["q_par", "q_perp", "mesh/q_par", "mesh/q_perp"]
    background_pass = (
        len(core) > 0
        and bool((core.te_eV > 0.0).all())
        and bool((core.ne_m3 > 0.0).all())
        and imported_q == expected_q
        and bool(np.isfinite(q_mag).all())
        and bool((q_mag >= 0.0).all())
        and int(np.count_nonzero(q_mag)) > 1000
        and float(q_mag.max()) > 1.0e8
        and q_identity_error < 1.0e-8
    )
    audit["background_query"] = {
        "status": "PASS" if background_pass else "FAIL",
        "adaptive_cells_total": len(background),
        "cells_inside_vessel": len(inside),
        "central_audit_cells": len(core),
        "central_te_eV_range": [float(core.te_eV.min()), float(core.te_eV.max())],
        "central_ne_m3_range": [float(core.ne_m3.min()), float(core.ne_m3.max())],
        "imported_heatflux_datasets": imported_q,
        "q_mag_W_m2_range_inside": [float(q_mag.min()), float(q_mag.max())],
        "q_mag_W_m2_quantiles_inside": {
            str(p): float(np.quantile(q_mag, p))
            for p in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "q_nonzero_cells_inside": int(np.count_nonzero(q_mag)),
        "q_vector_identity_max_abs_error_W_m2": q_identity_error,
        "central_q_mag_W_m2_range": [
            float(core.q_mag_W_m2.min()),
            float(core.q_mag_W_m2.max()),
        ],
        "interpretation": (
            "The native SOLPS q field was restored after the equilibrium-only "
            "conversion dropped it. The central regular-grid carrier is an "
            "explicitly labelled harmonic continuation. Accepted trajectories "
            "use OML heating, so prescribed q remains a diagnostic."
        ),
    }
    if not background_pass:
        audit["status"] = "FAIL"

    background.to_csv(ANALYSIS / "core_transit_background_openedge.csv", index=False)
    segments.to_csv(ANALYSIS / "core_transit_evaporation_segments.csv", index=False)
    terminals.to_csv(ANALYSIS / "core_transit_terminal_evaporation.csv", index=False)
    (ANALYSIS / "core_transit_map_acceptance.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    report = {
        **audit,
        "background_cells_inside_vessel": len(inside),
        "te_eV_range_inside": [float(inside.te_eV.min()), float(inside.te_eV.max())],
        "ne_m3_range_inside": [float(inside.ne_m3.min()), float(inside.ne_m3.max())],
        "q_mag_W_m2_range_inside": [float(q_mag.min()), float(q_mag.max())],
    }
    print(json.dumps(report, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("FAIL: spatial evaporation reconstruction")


if __name__ == "__main__":
    main()
