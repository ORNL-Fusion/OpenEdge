#!/usr/bin/env python3
"""Add open-ADAS PLT (line radiation) and PRB (recombination + bremsstrahlung)
power coefficients to an existing database/processes.h5 without rebuilding it.

Writes /volume/radiation/{plt,prb}/<elem>/{coefficient,temperature,density}
in the same layout build_processes_h5.py uses, so fix volume/chem/adas picks
them up (its start-of-run banner then reads "SCD ACD CCD PLT PRB IP") and the
electron-energy source includes the radiated power per ionization /
recombination event.

Usage:
  python3 add_radiation_tables.py --adf11-dir /path/to/adf11 [--elements li c ...]
         [--years 96 89] [--db ../processes.h5]
The first year in --years that has both plt and prb files for an element is used.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adas import DEFAULT_ELEMENTS, parse_adf11  # noqa: E402


def write_table(fout, cls, sym, path):
    logQ, logDens, logTe, iZmin, iZmax = parse_adf11(path)
    coef = np.transpose(logQ, (2, 1, 0))          # (nZ, nTe, nDens)
    grp = fout.require_group(f"volume/radiation/{cls}/{sym}")
    for name in ("coefficient", "temperature", "density"):
        if name in grp:
            del grp[name]
    ds = grp.create_dataset("coefficient", data=coef, compression="gzip", compression_opts=6)
    ds.attrs["units"] = "log10(power) [W cm^3]"
    ds.attrs["source"] = f"open-ADAS adf11 {path.name}"
    ds.attrs["method"] = "log10 values on (log10 Te [eV], log10 ne [cm^-3])"
    grp.create_dataset("temperature", data=logTe)
    grp.create_dataset("density", data=logDens)
    return coef.shape


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adf11-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=here.parent / "processes.h5")
    ap.add_argument("--elements", nargs="+", default=list(DEFAULT_ELEMENTS))
    ap.add_argument("--years", nargs="+", default=["96", "89"])
    args = ap.parse_args()

    added = []
    with h5py.File(args.db, "r+") as f:
        for sym in args.elements:
            if f"volume/rates/scd/{sym}" not in f:
                print(f"  {sym}: no scd table in the database, skipped")
                continue
            year = next((y for y in args.years
                         if (args.adf11_dir / f"plt{y}" / f"plt{y}_{sym}.dat").exists()
                         and (args.adf11_dir / f"prb{y}" / f"prb{y}_{sym}.dat").exists()), None)
            if year is None:
                print(f"  {sym}: no plt+prb pair for years {args.years}, skipped")
                continue
            shapes = [write_table(f, cls, sym, args.adf11_dir / f"{cls}{year}" / f"{cls}{year}_{sym}.dat")
                      for cls in ("plt", "prb")]
            added.append(sym)
            print(f"  {sym}: plt/prb year {year}, shapes {shapes}")
        note = f"PLT/PRB added {datetime.datetime.utcnow():%Y-%m-%dT%H:%MZ} from {args.adf11_dir} for {' '.join(added)}"
        f.attrs["radiation_tables"] = note
    print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
