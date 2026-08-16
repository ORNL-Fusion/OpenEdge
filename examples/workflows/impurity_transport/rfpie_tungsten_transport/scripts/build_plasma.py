#!/usr/bin/env python3
"""Build an OpenEdge HDF5 He+ background from the RFPIE LP scans.

The two measurements in experimental_data/lp/data.txt are averaged and then
folded about the column axis.  The source workbook labels position in cm; this
is important because a legacy script treated the same numbers as mm.  The
measurement supplies radial shape only, so the first case deliberately assumes
no axial variation.  Beyond the measured radius, density is continued with a
slope-matched exponential tail.
The folded electron-temperature measurements are fit with an even polynomial
and continued with a slope-matched exponential approach to the configured
far-edge value.  Both transitions are C1 continuous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator


QE = 1.602176634e-19
AMU = 1.66053906660e-27
HE_MASS_AMU = 4.002602
CASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CASE_DIR.parents[3]
LENGTH_TO_M = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}


def folded_mean(values: np.ndarray, errors: np.ndarray,
                signed_r: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold both sides/scans and retain a conservative standard error."""
    means = []
    sigmas = []
    for radius in radii:
        rows = np.isclose(np.abs(signed_r), radius)
        samples = values[rows].ravel()
        sample_errors = errors[rows].ravel()
        means.append(samples.mean())
        spread = samples.var(ddof=1) if len(samples) > 1 else 0.0
        sigmas.append(np.sqrt(np.mean(sample_errors**2) + spread) / np.sqrt(len(samples)))
    return np.asarray(means), np.asarray(sigmas)


def folded_lp_profile(path: Path, position_unit: str) -> dict[str, np.ndarray]:
    raw = np.loadtxt(path, comments="#")
    try:
        signed_r = raw[:, 0] * LENGTH_TO_M[position_unit]
    except KeyError as exc:
        raise ValueError(f"unsupported LP position unit: {position_unit}") from exc
    ne_scans = raw[:, [1, 5]]
    ne_errors = raw[:, [2, 6]]
    te_scans = raw[:, [3, 7]]
    te_errors = raw[:, [4, 8]]
    ne_signed = ne_scans.mean(axis=1)
    te_signed = te_scans.mean(axis=1)
    ne_signed_sigma = np.sqrt(np.sum(ne_errors**2, axis=1)) / 2.0
    te_signed_sigma = np.sqrt(np.sum(te_errors**2, axis=1)) / 2.0
    radii = np.unique(np.abs(signed_r))
    ne, ne_sigma = folded_mean(ne_scans, ne_errors, signed_r, radii)
    te, te_sigma = folded_mean(te_scans, te_errors, signed_r, radii)
    return {
        "r": radii, "ne": ne, "ne_sigma": ne_sigma,
        "te": te, "te_sigma": te_sigma, "signed_r": signed_r,
        "ne_signed": ne_signed, "ne_signed_sigma": ne_signed_sigma,
        "te_signed": te_signed, "te_signed_sigma": te_signed_sigma,
    }


def reconstruct_profiles(r: np.ndarray, measured: dict[str, np.ndarray],
                         pcfg: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    """Return smooth radial ne/Te and sufficient audit data to reproduce them."""
    rm = measured["r"]
    edge = rm[-1]

    # Interpolate log(ne-floor) inside the scan with a monotone C1 PCHIP.
    # Continue the endpoint logarithmic slope outside, which makes both ne
    # and dne/dr continuous at the last measured point.
    ne_floor = float(pcfg["density_floor_m3"])
    if np.any(measured["ne"] <= ne_floor):
        raise ValueError("all folded densities must exceed density_floor_m3")
    log_ne_excess = np.log(measured["ne"] - ne_floor)
    ne_interp = PchipInterpolator(rm, log_ne_excess)
    log_ne_edge = float(ne_interp(edge))
    log_ne_slope = float(ne_interp.derivative()(edge))
    lo_ne, hi_ne = map(float, pcfg["density_tail_scale_bounds_m"])
    if log_ne_slope >= 0.0:
        raise ValueError("edge density does not fall; cannot construct exponential tail")
    ne_tail_scale = float(-1.0 / log_ne_slope)
    if not lo_ne <= ne_tail_scale <= hi_ne:
        raise ValueError("derived density tail scale falls outside configured bounds")
    inside = r <= edge
    log_excess_r = np.empty_like(r)
    log_excess_r[inside] = ne_interp(r[inside])
    log_excess_r[~inside] = log_ne_edge - (r[~inside] - edge) / ne_tail_scale
    ne_r = ne_floor + np.exp(log_excess_r)

    # A polynomial in r^2 respects cylindrical symmetry (dTe/dr=0 on axis).
    # Weight with probe uncertainty so the asymmetric 45 mm point is visible
    # in the audit plot but does not create an artificial shoulder.
    order = int(pcfg["temperature_even_polynomial_order"])
    te_coeff = np.polyfit(rm**2, measured["te"], order,
                          w=1.0 / measured["te_sigma"])
    te_poly = np.poly1d(te_coeff)
    te_edge = float(te_poly(edge**2))
    te_slope = float(np.polyder(te_poly)(edge**2) * 2.0 * edge)
    te_asymptote = float(pcfg["temperature_asymptote_eV"])
    if te_slope <= 0.0 or te_asymptote <= te_edge:
        raise ValueError("temperature fit/asymptote must give a rising edge tail")
    lo_te, hi_te = map(float, pcfg["temperature_tail_scale_bounds_m"])
    te_tail_scale = float((te_asymptote - te_edge) / te_slope)
    if not lo_te <= te_tail_scale <= hi_te:
        raise ValueError("derived temperature tail scale falls outside configured bounds")
    te_r = np.empty_like(r)
    te_r[inside] = te_poly(r[inside]**2)
    dr = r[~inside] - edge
    te_r[~inside] = te_asymptote - (te_asymptote - te_edge) * np.exp(
        -dr / te_tail_scale)

    audit = {
        "density_tail_scale_m": ne_tail_scale,
        "density_edge_log_slope_per_m": -1.0 / ne_tail_scale,
        "temperature_polynomial_coefficients": te_coeff,
        "temperature_edge_eV": te_edge,
        "temperature_edge_slope_eV_per_m": te_slope,
        "temperature_tail_scale_m": te_tail_scale,
    }
    return ne_r, te_r, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CASE_DIR / "input/config.json")
    parser.add_argument("--output", type=Path, default=CASE_DIR / "input/plasma_he.h5")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    pcfg = cfg["plasma"]
    lp_path = REPO_DIR / pcfg["source_lp_data"]

    measured = folded_lp_profile(lp_path, pcfg["source_lp_position_unit"])
    r = np.linspace(0.0, float(pcfg["r_max_m"]), int(pcfg["radial_points"]))
    z = np.linspace(float(pcfg["z_min_m"]), float(pcfg["z_max_m"]),
                    int(pcfg["axial_points"]))
    ne_r, te_r, fit_audit = reconstruct_profiles(r, measured, pcfg)
    ti_r = np.full_like(te_r, float(pcfg["ti_eV"]))

    # Reproduce the first reconstruction for audit: it treated the workbook's
    # cm coordinates as mm, putting the apparent data edge at 5 rather than
    # 50 mm and creating the screenshot's kink there.
    legacy_r = measured["r"] * 0.1
    legacy_edge = legacy_r[-1]
    legacy_ne_r = np.interp(np.minimum(r, legacy_edge), legacy_r, measured["ne"])
    legacy_outside = r > legacy_edge
    legacy_ne_r[legacy_outside] = float(pcfg["density_floor_m3"]) + (
        measured["ne"][-1] - float(pcfg["density_floor_m3"])) * np.exp(
            -(r[legacy_outside] - legacy_edge) / 0.005)
    legacy_te_r = np.interp(np.minimum(r, legacy_edge), legacy_r, measured["te"])
    legacy_te_r[legacy_outside] = measured["te"][-1]

    ne = np.broadcast_to(ne_r, (len(z), len(r))).copy()
    te = np.broadcast_to(te_r, ne.shape).copy()
    ti = np.broadcast_to(ti_r, ne.shape).copy()
    ni = ne.copy()
    cs_r = np.sqrt((te_r + ti_r) * QE / (HE_MASS_AMU * AMU))
    upar_r = -cs_r
    upar = np.broadcast_to(upar_r, ne.shape).copy()
    grad_te_r = np.broadcast_to(np.gradient(te_r, r), ne.shape).copy()
    grad_ti_r = np.broadcast_to(np.gradient(ti_r, r), ne.shape).copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        h5.attrs["case"] = "RFPIE He+ background for W sputtering"
        h5.attrs["source"] = str(lp_path)
        h5.attrs["source_position_unit"] = pcfg["source_lp_position_unit"]
        h5.attrs["source_unit_evidence"] = (
            f"{pcfg['source_lp_workbook']} chart axis is labeled Radius (cm)")
        h5.attrs["assumption_axial_profile"] = "uniform; LP data provide radial shape only"
        h5.attrs["assumption_quasineutral"] = "n_He+ = n_e"
        h5.attrs["assumption_ti"] = (
            f"uniform Ti = {pcfg['ti_eV']} eV; RF PIE hPIC2 assumption in "
            "Caughman et al., IEEE TPS 52 (2024), DOI 10.1109/TPS.2024.3374252")
        h5.attrs["density_model"] = (
            "C1 PCHIP of log(ne-floor) through folded data; endpoint-slope exponential tail")
        h5.attrs["temperature_model"] = (
            "uncertainty-weighted polynomial in r^2; C1 exponential approach to edge asymptote")
        h5.create_dataset("r", data=r)
        h5.create_dataset("z", data=z)
        h5.create_dataset("dens_e", data=ne)
        h5.create_dataset("temp_e", data=te)
        h5.create_dataset("dens_i", data=ni)
        h5.create_dataset("temp_i", data=ti)
        h5.create_dataset("parr_flow", data=upar)
        h5.create_dataset("parr_flow_r", data=np.zeros_like(ne))
        h5.create_dataset("parr_flow_t", data=np.zeros_like(ne))
        h5.create_dataset("parr_flow_z", data=upar)
        h5.create_dataset("grad_te_r", data=grad_te_r)
        h5.create_dataset("grad_te_t", data=np.zeros_like(ne))
        h5.create_dataset("grad_te_z", data=np.zeros_like(ne))
        h5.create_dataset("grad_ti_r", data=grad_ti_r)
        h5.create_dataset("grad_ti_t", data=np.zeros_like(ne))
        h5.create_dataset("grad_ti_z", data=np.zeros_like(ne))
        h5.create_dataset("epar", data=np.zeros_like(ne))

        strings = h5py.string_dtype("utf-8")
        ions = h5.create_group("ion_species")
        ions.create_dataset("names", data=np.asarray(["He+"], dtype=object), dtype=strings)
        ions.create_dataset("elements", data=np.asarray(["He"], dtype=object), dtype=strings)
        ions.create_dataset("spec_index", data=np.asarray([0], dtype=np.int32))
        ions.create_dataset("main_ion_spec_index", data=np.asarray([0], dtype=np.int32))
        ions.create_dataset("mass_amu", data=np.asarray([HE_MASS_AMU]))
        ions.create_dataset("charge_state_z", data=np.asarray([1], dtype=np.int32))
        ion_fields = h5.create_group("ions")
        ion_fields.create_dataset("dens", data=ni[None, :, :])
        ion_fields.create_dataset("temp", data=ti[None, :, :])
        ion_fields.create_dataset("parr_flow", data=upar[None, :, :])

        # Retain an explicit target wall-flux profile.  Current regular-grid
        # OpenEdge reconstructs the same Bohm flux; this group is ready for
        # direct consumption once wall_flux is carrier-independent.
        target_radius = 11.43e-3
        rw = np.linspace(0.0, target_radius, 80)
        new = np.interp(rw, r, ne_r)
        tew = np.interp(rw, r, te_r)
        tiw = np.interp(rw, r, ti_r)
        csw = np.sqrt((tew + tiw) * QE / (HE_MASS_AMU * AMU))
        wall = h5.create_group("wall_flux")
        wall.create_dataset("r", data=rw)
        wall.create_dataset("z", data=np.full_like(rw, 1.905e-3))
        wall.create_dataset("te", data=tew)
        wall.create_dataset("ti", data=tiw)
        wall.create_dataset("b_r", data=np.zeros_like(rw))
        wall.create_dataset("b_z", data=np.full_like(rw, float(pcfg["bz_t"])))
        wall.create_dataset("b_t", data=np.zeros_like(rw))
        wall.create_dataset("gamma_i", data=(new * csw)[None, :])

        audit = h5.create_group("audit")
        audit.create_dataset("lp_signed_r_m", data=measured["signed_r"])
        audit.create_dataset("lp_mean_ne_m3", data=measured["ne_signed"])
        audit.create_dataset("lp_mean_ne_sigma_m3", data=measured["ne_signed_sigma"])
        audit.create_dataset("lp_mean_te_eV", data=measured["te_signed"])
        audit.create_dataset("lp_mean_te_sigma_eV", data=measured["te_signed_sigma"])
        audit.create_dataset("folded_r_m", data=measured["r"])
        audit.create_dataset("folded_ne_m3", data=measured["ne"])
        audit.create_dataset("folded_ne_sigma_m3", data=measured["ne_sigma"])
        audit.create_dataset("folded_te_eV", data=measured["te"])
        audit.create_dataset("folded_te_sigma_eV", data=measured["te_sigma"])
        audit.create_dataset("legacy_wrong_unit_ne_r_m3", data=legacy_ne_r)
        audit.create_dataset("legacy_wrong_unit_te_r_eV", data=legacy_te_r)
        for name, value in fit_audit.items():
            audit.create_dataset(name, data=value)

    print(f"wrote {args.output}")
    print(f"  grid: nz={len(z)} nr={len(r)}")
    print(f"  ne: {ne.min():.3e} .. {ne.max():.3e} m^-3")
    print(f"  Te: {te.min():.3f} .. {te.max():.3f} eV")
    print(f"  Ti: {ti.min():.3f} eV")
    print(f"  density tail scale: {fit_audit['density_tail_scale_m']*1e3:.3f} mm")
    print(f"  temperature tail scale: {fit_audit['temperature_tail_scale_m']*1e3:.3f} mm")


if __name__ == "__main__":
    main()
