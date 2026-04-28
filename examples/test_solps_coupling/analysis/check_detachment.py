#!/usr/bin/env python3
"""Check detachment status of all 4 CDN targets from b2plot wlld output.

Run after `run_b2plot_wlld.sh` completes. For each target prints:
  Te @ strike, Te peak, peak D ion/atom flux, peak D2 mol pres.

Stangeby criteria:
  Te < 5 eV at strike     -> detached
  Te < 2 eV at strike     -> deeply detached / recombining
  D_flux_atm >= D_flux_ion at strike  -> high-recycling regime
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_solps_target import read_target


def main():
    here = Path(__file__).resolve().parent.parent
    bp = here / "coupled_test" / "solps" / "v0_1" / "b2pl.exe.dir"
    if not bp.exists():
        sys.exit(f"missing {bp} — run ./run_b2plot_wlld.sh first")

    print(f"{'target':>14}  {'Te_strike':>9}  {'Te_peak':>8}  "
          f"{'D_flux_ion':>11}  {'D_flux_atm':>11}  "
          f"{'D2_pres_mol':>12}  {'verdict':>20}")
    print("-" * 105)
    for tgt in ('inner', 'outer', 'inner_upper', 'outer_upper'):
        try:
            d = read_target(bp, tgt)
        except (FileNotFoundError, ValueError) as e:
            print(f"{tgt:>14}  {'missing':>9}   ({e})")
            continue
        x = d['x']
        # Strike point: where x = 0 (separatrix arc location)
        k_strike = int(np.argmin(np.abs(x)))
        Te_strike = float(d['Te'][k_strike])
        Te_peak = float(d['Te'].max())
        D_ion_peak = float(d['D_flux_ion'].max())
        D_atm_peak = float(d['D_flux_atm'].max()) if 'D_flux_atm' in d else 0.0
        D2_pres = float(d['D2_pres_mol'].max()) if 'D2_pres_mol' in d else 0.0
        if Te_strike < 2.0:
            verdict = "deeply detached"
        elif Te_strike < 5.0:
            verdict = "detached"
        elif Te_strike < 10.0:
            verdict = "marginal"
        else:
            verdict = "attached"
        # Atomic-vs-ion balance hint
        atm_dom = D_atm_peak > D_ion_peak
        if atm_dom and Te_strike < 5.0:
            verdict += " (atom-dominated)"
        print(f"{tgt:>14}  {Te_strike:>9.2f}  {Te_peak:>8.2f}  "
              f"{D_ion_peak:>11.2e}  {D_atm_peak:>11.2e}  "
              f"{D2_pres:>12.3e}  {verdict:>20}")


if __name__ == "__main__":
    main()
