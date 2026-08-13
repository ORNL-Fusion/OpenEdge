#!/usr/bin/env python3
"""Fail-fast preflight checks for the RFPIE OpenEdge case."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import nbformat
import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
PROCESS_DATABASE = CASE_DIR.parents[1] / "database/processes.h5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def dump_columns(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    require(len(lines) >= 9, f"truncated dump: {path}")
    return lines[8].split()[2:]


def main() -> None:
    cfg = json.loads((CASE_DIR / "case_config.json").read_text())
    plasma_cfg = cfg["plasma"]
    transport_cfg = cfg["transport"]
    require(plasma_cfg["source_lp_position_unit"] == "cm",
            "LP position must follow the source workbook's Radius (cm) axis")

    plasma_path = CASE_DIR / "input/plasma_he.h5"
    require(plasma_path.exists(), "missing input/plasma_he.h5")
    with h5py.File(plasma_path, "r") as h5:
        r = h5["r"][:]
        z = h5["z"][:]
        ne = h5["dens_e"][:]
        te = h5["temp_e"][:]
        ti = h5["temp_i"][:]
        measured_edge = float(h5["audit/folded_r_m"][-1])
        require(h5.attrs["source_position_unit"] == "cm", "HDF5 unit audit is not cm")
        require(measured_edge >= r[-1], "OpenEdge radial grid extends beyond LP data")
        require(ne.shape == (len(z), len(r)), "unexpected electron-density shape")
        require(te.shape == ne.shape and ti.shape == ne.shape, "temperature shape mismatch")
        require(np.all(np.isfinite(ne)) and np.all(ne > 0), "invalid electron density")
        require(np.all(np.isfinite(te)) and np.all(te > 0), "invalid electron temperature")
        require(np.allclose(ti, plasma_cfg["ti_eV"]), "ion temperature differs from config")
        require(np.max(np.abs(np.diff(te[0], 2))) < 2.0e-3,
                "electron-temperature reconstruction has a sharp radial kink")

    grid_cells = np.asarray(transport_cfg["grid_cells"], dtype=int)
    require(grid_cells.shape == (3,) and np.all(grid_cells > 0),
            "transport grid must contain three positive dimensions")
    cell_widths = np.array([0.0602, 0.0602, 0.0602]) / grid_cells
    qe = 1.602176634e-19
    amu = 1.66053906660e-27
    vmax_w = np.sqrt(2 * 80 * qe / (184 * amu))
    step_fraction = vmax_w * transport_cfg["timestep_s"] / cell_widths.min()
    substep_fraction = step_fraction / transport_cfg["pusher_subcycles"]
    require(substep_fraction < 0.25,
            "80 eV W moves more than one quarter of the smallest cell per pusher substep")
    require(0 <= transport_cfg["balance_switch_steps"] < transport_cfg["run_steps"],
            "RCB switch must occur within the requested run")
    rf_period = 1 / cfg["target_sheath"]["frequency_hz"]
    steps_per_rf = rf_period / transport_cfg["timestep_s"]
    if cfg["target_sheath"]["vrf_v"] != 0:
        require(steps_per_rf >= 20,
                "time-resolved RF transport needs at least 20 steps per RF period")

    require(PROCESS_DATABASE.exists(), f"missing {PROCESS_DATABASE}")
    with h5py.File(PROCESS_DATABASE, "r") as h5:
        table = h5["surface/sputter/he_on_w"]
        energy = table["E"][:]
        angle = table["theta"][:]
        sputter_yield = table["Y"][:]
        require(sputter_yield.shape == (len(energy), len(angle)),
                "He-on-W RustBCA table shape mismatch")
        require(np.all(np.isfinite(sputter_yield)) and np.all(sputter_yield >= 0),
                "invalid He-on-W sputter yield")

    notebook = nbformat.read(CASE_DIR / "rfpie_w.ipynb", as_version=4)
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    require(not errors, "executed notebook contains error output")

    density_dump = CASE_DIR / "output/rfpie_w_density.1000.dump"
    if density_dump.exists():
        columns = dump_columns(density_dump)
        require(all(f"f_fWdens[{index}]" in columns for index in range(1, 6)),
                "charge-resolved W density fields missing from grid dump")

    print("RFPIE preflight passed")
    print(f"  LP measured radius: {measured_edge*1e3:.1f} mm; "
          f"OpenEdge radius: {r[-1]*1e3:.1f} mm")
    print(f"  ne: {ne.min():.3e} .. {ne.max():.3e} m^-3")
    print(f"  Te: {te.min():.3f} .. {te.max():.3f} eV; Ti: {ti[0,0]:.3f} eV")
    print(f"  RustBCA He-on-W: {sputter_yield.shape}, "
          f"Y={sputter_yield.min():.4g}..{sputter_yield.max():.4g}")
    print(f"  transport grid: {tuple(grid_cells)}; dt: "
          f"{transport_cfg['timestep_s']*1e9:.1f} ns; "
          f"80 eV W full-step/cell: {step_fraction:.3f}; "
          f"substep/cell: {substep_fraction:.3f}")
    print(f"  run: {transport_cfg['run_steps']} steps = "
          f"{transport_cfg['run_steps']*transport_cfg['timestep_s']*1e6:.1f} us; "
          f"steps/RF period: {steps_per_rf:.2f}")
    print(f"  notebook cells: {len(notebook.cells)}, errors: {len(errors)}")
    print(f"  charge-resolved transport dump: {'present' if density_dump.exists() else 'not run yet'}")


if __name__ == "__main__":
    main()
