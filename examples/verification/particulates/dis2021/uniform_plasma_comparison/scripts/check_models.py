#!/usr/bin/env python3
"""Independent one-step references for the DUSTT and DIS comparison.

The formulas are transcribed from Pigarov et al. (2005), Nespoli et al.
(2021), and the corrected Smirnov et al. (2007) OML collection expression.
Only Python's standard library is used so this maintained example does not
depend on the larger research audit under examples/wip.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]

# OpenEdge/SPARTA constants used by the compiled implementation.
QE = 1.60217646e-19
ME = 9.10938215e-31
MP = 1.6726219e-27
AMU = 1.66053906660e-27
EPS0 = 8.8541878128e-12
SQPI = math.sqrt(math.pi)

TE = 60.0
TI = 60.0
NE = 1.0e19
NI = 1.0e19
ION_AMU = 2.014
RADIUS = 1.0e-6
RHO = 2340.0
FLOW = 2.0e4
DT = 1.0e-9


def read_last_frame(path: Path) -> dict[str, float]:
    frames: dict[int, list[dict[str, float]]] = {}
    with path.open() as stream:
        while line := stream.readline():
            if line.strip() != "ITEM: TIMESTEP":
                continue
            step = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise RuntimeError(f"malformed particle dump: {path}")
            count = int(stream.readline())
            if not stream.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"malformed particle dump: {path}")
            for _ in range(3):
                stream.readline()
            keys = stream.readline().split()[2:]
            frames[step] = [
                dict(zip(keys, map(float, stream.readline().split())))
                for _ in range(count)
            ]
    if not frames or len(frames[max(frames)]) != 1:
        raise RuntimeError(f"expected one grain in final frame: {path}")
    return frames[max(frames)][0]


def f_gamma(u: float, attraction: float) -> float:
    """DUSTT Eqs. (4)-(5), with its analytic stationary limit."""
    if u < 1.0e-3:
        return 2.0 * (1.0 + attraction) / SQPI
    return ((u + 0.5 / u + attraction / u) * math.erf(u)
            + math.exp(-u * u) / SQPI)


def dis_ion_current(u: float, x: float) -> float:
    """DIS Appendix B ion-current factor for the attractive branch."""
    if x < 0.0:
        raise ValueError("this comparison expects a negatively charged grain")
    if u < 1.0e-3:
        return 2.0 * (1.0 + x) / SQPI
    return (((1.0 + 2.0 * (u * u + x)) * math.erf(u)
             + 2.0 * u * math.exp(-u * u) / SQPI) / (2.0 * u))


def dis_electron_current(chi: float) -> float:
    return math.exp(-chi) if chi >= 0.0 else 1.0 - chi


def ion_mass(model: str) -> float:
    return ION_AMU * (AMU if model == "dis2021" else MP)


def solve_phi(model: str, velocity: float) -> float:
    mi = ion_mass(model)
    vti = math.sqrt(2.0 * TI * QE / mi)
    u = abs(FLOW - velocity) / vti
    ce = QE * math.pi * RADIUS**2 * NE * math.sqrt(
        8.0 * TE * QE / (math.pi * ME))
    ci = QE * math.pi * RADIUS**2 * NI * vti

    def residual(phi: float) -> float:
        if model == "dustt2005":
            return (-ce * math.exp(phi / TE)
                    + ci * max(f_gamma(u, -phi / TI), 0.0))
        chi = -phi / TE
        x = chi * TE / TI
        return (-ce * dis_electron_current(chi)
                + ci * dis_ion_current(u, x))

    # This fixture deliberately isolates the ordinary negative-grain branch.
    # Bracketing at phi=0 keeps the reference scoped to that branch instead
    # of silently adding the positive-grain formulas that are tested elsewhere.
    lo, hi = -80.0 * max(TE, TI), 0.0
    flo = residual(lo)
    if flo * residual(hi) > 0.0:
        raise RuntimeError(f"could not bracket {model} floating potential")
    for _ in range(180):
        mid = 0.5 * (lo + hi)
        fm = residual(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def coulomb_log(u: float, chi: float) -> float:
    if chi == 0.0:
        return 0.0
    tij = TI * QE
    tej = TE * QE
    mve2 = tij * (3.0 + 2.0 * u * u)
    b90 = RADIUS * abs(chi) * tej / mve2
    lambda_d = math.sqrt(EPS0 * tej / (NE * QE * QE))
    lambda_s = lambda_d / math.sqrt(1.0 + 3.0 * tej / mve2)
    eta = 1.0 + (RADIUS / lambda_s) * (
        1.0 + math.sqrt(TE / (6.0 * TI)))
    value = 0.5 * math.log(
        (b90 * b90 + (eta * lambda_s) ** 2)
        / (b90 * b90 + RADIUS**2))
    return max(value, 0.0)


def dustt_drag_factor(u: float, chi: float, ln_lambda: float) -> float:
    """Pigarov Eq. (16): collection plus orbit terms."""
    a = chi / (TI / TE)
    if u < 1.0e-3:
        return ((5.0 + 4.0 * a) / (3.0 * SQPI)
                + 4.0 * a * a * ln_lambda / (3.0 * SQPI))
    u2 = u * u
    exp_u = math.exp(-u2)
    erf_u = math.erf(u)
    collection = (
        u * (2.0 * u2 + 1.0 + 2.0 * a) * exp_u
        + 0.5 * SQPI
        * (4.0 * u2 * u2 + 2.0 * u2 - 1.0
           - 2.0 * (1.0 - 2.0 * u2) * a) * erf_u
    ) / (2.0 * u2 * u * SQPI)
    y = (erf_u - 2.0 * u * exp_u / SQPI) / (2.0 * u2)
    orbit = 2.0 * a * a * ln_lambda * y / u
    return collection + orbit


def dis_drag_factor(u: float, x: float, ln_lambda: float) -> float:
    """Smirnov Eq. (5) collection plus DIS Eq. (A5) scattering."""
    if x < 0.0:
        raise ValueError("this comparison expects the attractive branch")
    if u < 1.0e-3:
        collection = (8.0 + 4.0 * x) / (3.0 * SQPI)
        scattering = x * x * ln_lambda * 4.0 / (3.0 * SQPI)
        return collection + scattering
    u2 = u * u
    wp = u2 + x
    wm = u2 - x
    bracket = 1.0 + 2.0 * wp - (1.0 - 2.0 * wm) / (2.0 * u2)
    collection = ((1.0 + 2.0 * wp) * math.exp(-u2) / SQPI
                  + u * bracket * math.erf(u)) / (2.0 * u2)
    scattering = (x * x * ln_lambda
                  * (math.erf(u) - 2.0 * u * math.exp(-u2) / SQPI)
                  / (u * u2))
    return collection + scattering


def half_kick(model: str, velocity: float) -> tuple[float, float]:
    phi = solve_phi(model, velocity)
    mi = ion_mass(model)
    vti = math.sqrt(2.0 * TI * QE / mi)
    u = abs(FLOW - velocity) / vti
    chi = -phi / TE
    ln_lambda = coulomb_log(u, chi)
    if model == "dustt2005":
        factor = dustt_drag_factor(u, min(max(chi, 0.0), 20.0), ln_lambda)
    else:
        factor = dis_drag_factor(u, chi * TE / TI, ln_lambda)
    nu0 = 0.75 * NI * mi * vti / (RHO * RADIUS)
    s = nu0 * factor * (0.5 * DT)
    decay = 1.0 - s + 0.5 * s * s if abs(s) < 1.0e-8 else math.exp(-s)
    return FLOW + (velocity - FLOW) * decay, phi


def one_step_reference(model: str) -> tuple[float, float]:
    velocity, _ = half_kick(model, 0.0)
    return half_kick(model, velocity)


def dustt_heat(phi: float) -> float:
    """Legacy DUSTT one-sided fluxes and fixed 2.5-T energy factors."""
    chi = max(phi / TE, -20.0)
    mi = ION_AMU * AMU  # thermal fix uses the explicit SI atomic-mass unit
    electron_flux = (0.25 * NE
                     * math.sqrt(8.0 * QE * TE / (math.pi * ME))
                     * math.exp(chi))
    ion_flux = (0.25 * NI
                * math.sqrt(8.0 * QE * TI / (math.pi * mi))
                * (1.0 - chi * TE / TI))
    electron_energy = 2.5 * TE
    ion_energy = 2.5 * TI + (-chi) * TE + 13.6
    return QE * (electron_flux * electron_energy + ion_flux * ion_energy)


def dis_ion_energy(u: float, x: float) -> float:
    if x < 0.0:
        raise ValueError("this comparison expects the attractive branch")
    if u < 1.0e-3:
        return 2.0 * (2.0 + x) / SQPI
    u2 = u * u
    first = (2.0 / SQPI) * (5.0 + 2.0 * (u2 + x)) * math.exp(-u2)
    second = (3.0 + 12.0 * u2 + 4.0 * u2 * u2
              + 2.0 * x * (1.0 + 2.0 * u2)) * math.erf(u) / u
    return 0.25 * (first + second)


def dis_heat(velocity: float) -> float:
    """DIS Appendix C heat balance with emission disabled."""
    phi = solve_phi("dis2021", velocity)
    chi = -phi / TE
    mi = ION_AMU * AMU
    vti = math.sqrt(2.0 * TI * QE / mi)
    u = abs(FLOW - velocity) / vti
    x = chi * TE / TI
    fi = dis_ion_current(u, x)
    fe = dis_electron_current(chi)
    gi = dis_ion_energy(u, x)
    ge = 2.0 + chi if chi >= 0.0 else (2.0 - chi) / (1.0 - chi)
    gamma_i0 = 0.25 * NI * vti
    gamma_e0 = 0.25 * NE * math.sqrt(8.0 * QE * TE / (math.pi * ME))
    ion_current_density = QE * gamma_i0 * fi
    electron_current_density = -QE * gamma_e0 * fe
    return (QE * gamma_i0 * TI * gi
            + QE * gamma_e0 * fe * TE * ge
            + ion_current_density * 13.6
            + (ion_current_density + electron_current_density) * chi * TE)


def gate(name: str, passed: bool, evidence: str) -> bool:
    print(f"{'PASS' if passed else 'FAIL'} | {name} | {evidence}")
    return bool(passed)


def main() -> int:
    rows = {
        model: read_last_frame(CASE / "output" / f"state.{model}")
        for model in ("dustt2005", "dis2021")
    }
    refs: dict[str, dict[str, float]] = {}
    ok = True

    for model, row in rows.items():
        velocity_ref, charge_phi_ref = one_step_reference(model)
        charge_ref = 4.0 * math.pi * EPS0 * RADIUS * charge_phi_ref / QE
        heat_ref = (dustt_heat(charge_phi_ref) if model == "dustt2005"
                    else dis_heat(velocity_ref))
        refs[model] = {
            "velocity": velocity_ref,
            "charge": charge_ref,
            "heat": heat_ref,
        }
        rtol = 3.0e-4 if model == "dustt2005" else 2.0e-8
        ok &= gate(f"{model} floating charge matches its reference",
                   math.isclose(row["p_particulate_charge"], charge_ref,
                                rel_tol=rtol),
                   f"engine={row['p_particulate_charge']:.12g}, "
                   f"reference={charge_ref:.12g}")
        ok &= gate(f"{model} ion-drag kick matches its reference",
                   math.isclose(row["vz"], velocity_ref, rel_tol=rtol,
                                abs_tol=2.0e-12),
                   f"engine={row['vz']:.12g}, "
                   f"reference={velocity_ref:.12g} m/s")
        ok &= gate(f"{model} OML heat flux matches its reference",
                   math.isclose(row["p_droplet_heating_q"], heat_ref,
                                rel_tol=rtol),
                   f"engine={row['p_droplet_heating_q']:.12g}, "
                   f"reference={heat_ref:.12g} W/m2")
        ok &= gate(f"{model} retains the grain for the one-step test",
                   row["radius"] > 0.999999 * RADIUS,
                   f"radius={row['radius']:.12g} m")

    potentials = {
        model: row["p_particulate_charge"] * QE
        / (4.0 * math.pi * EPS0 * row["radius"])
        for model, row in rows.items()
    }
    charge_delta = abs(potentials["dis2021"] / potentials["dustt2005"] - 1.0)
    drag_ratio = rows["dis2021"]["vz"] / rows["dustt2005"]["vz"]
    heat_ratio = rows["dustt2005"]["p_droplet_heating_q"] / rows["dis2021"]["p_droplet_heating_q"]
    lambda_d = math.sqrt(EPS0 * TE / (NE * QE))

    ok &= gate("baseline floating potentials remain close",
               charge_delta < 0.01,
               f"relative separation={charge_delta:.3%}")
    ok &= gate("DIS and DUSTT drag closures are measurably distinct",
               drag_ratio > 1.02,
               f"DIS/DUSTT velocity kick={drag_ratio:.6g}")
    ok &= gate("DIS and DUSTT heat closures are measurably distinct",
               heat_ratio > 1.05,
               f"DUSTT/DIS heat flux={heat_ratio:.6g}")
    ok &= gate("shared grain is inside the OML small-grain range",
               RADIUS / lambda_d < 0.1,
               f"R/lambda_D={RADIUS/lambda_d:.6g}")

    with (CASE / "output" / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "potential_V", "charge_number",
                         "vz_m_s", "heat_flux_W_m2"))
        for model in ("dustt2005", "dis2021"):
            row = rows[model]
            writer.writerow((model, f"{potentials[model]:.17g}",
                             f"{row['p_particulate_charge']:.17g}",
                             f"{row['vz']:.17g}",
                             f"{row['p_droplet_heating_q']:.17g}"))

    print("PASS | DUSTT-2005 versus DIS-2021 comparison" if ok else
          "FAIL | DUSTT-2005 versus DIS-2021 comparison")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
