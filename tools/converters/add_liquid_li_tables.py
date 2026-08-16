#!/usr/bin/env python3
"""Import temperature-dependent liquid-Li sputter tables into processes.h5.

Reads li_liquid_tables.h5 (RustBCA/scripts/liquid_li/merge_li_liquid.py
output) and writes, per pair:

    /surface/sputter/<pair>/{E, theta, T, Y}     Y shaped (NT, NE, NTHETA)

The T axis (Kelvin) is the temperature analog of the compound tables' C
axis; the OpenEdge reader (ProcessLibrary::load_trim_sputter) must know
the T-axis extension before these tables are used — otherwise surface/pwi
falls back to the analytic model with a warning. Corrected yields (f_D
soft-collision factor already applied to d_on_li at merge time) go in Y;
raw BCA yields are stored as Y_raw for provenance.

Replaces any existing 2-D d_on_li/... group content for these pairs
(a timestamped .bak of processes.h5 is made first).

Usage:
  ./add_liquid_li_tables.py li_liquid_tables.h5            # dry run
  ./add_liquid_li_tables.py li_liquid_tables.h5 --commit
"""

from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path

import h5py
import numpy as np

DB = Path(__file__).resolve().parents[2] / "database" / "processes.h5"
SOURCE = ("RustBCA layered liquid-Li (Allain-calibrated 2026-08-14): "
          "Es(T) 1.67/1.40/0.46 eV anchors, CD(T)x1.4, 1.5A pure-Li top; "
          "d_on_li x f_D(T,E) from Allain&Brooks NF51(2011) Fig3; "
          "ne_on_li is an uncalibrated BCA prediction")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", type=Path)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--commit", action="store_true",
                    help="actually write processes.h5 (default: dry run)")
    args = ap.parse_args()

    with h5py.File(args.tables, "r") as src:
        pairs = {}
        for case in src:
            g = src[case]
            if "Y" not in g:
                continue
            pairs[case] = {
                "T": g["T"][:] + 273.15,          # C -> K
                "E": g["E"][:],
                "theta": g["theta"][:],
                "Y": g["Y"][:],
                "Y_raw": g["Y_raw"][:],
                "R_N": g["R_N"][:],
                "fD": int(g.attrs.get("fD_applied", 0)),
            }

    for p, d in pairs.items():
        nan = int(np.isnan(d["Y_raw"]).sum())
        print(f"{p}: T {d['T'].min():.0f}-{d['T'].max():.0f} K "
              f"({len(d['T'])} slices) x {len(d['E'])} E x "
              f"{len(d['theta'])} theta, fD={'yes' if d['fD'] else 'no'}, "
              f"{nan} NaN raw pts")

    if not args.commit:
        print("dry run (pass --commit to write)")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d")
    bak = args.db.with_name(args.db.name + f".before-liquid-li-{stamp}")
    if not bak.exists():
        shutil.copy2(args.db, bak)
        print(f"backup: {bak}")

    with h5py.File(args.db, "a") as db:
        for p, d in pairs.items():
            grp = f"surface/sputter/{p}"
            if grp in db:
                del db[grp]
            g = db.require_group(grp)
            g.create_dataset("E", data=d["E"])
            g.create_dataset("theta", data=d["theta"])
            g.create_dataset("T", data=d["T"])
            g.create_dataset("Y", data=np.nan_to_num(d["Y"]))
            g.create_dataset("Y_raw", data=np.nan_to_num(d["Y_raw"]))
            g.create_dataset("R_N", data=np.nan_to_num(d["R_N"]))
            g.attrs["Es_eV"] = 1.67      # room-T anchor; table owns Es(T)
            g.attrs["T_units"] = "K"
            g.attrs["source"] = SOURCE
            print(f"wrote /{grp}")
    print("done")


if __name__ == "__main__":
    main()
