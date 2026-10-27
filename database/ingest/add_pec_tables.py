#!/usr/bin/env python3
"""Add open-ADAS line PEC (adf15) and S/XB (adf13) tables for selected lines to an
existing database/processes.h5 without rebuilding it.

Writes, in the layout ProcessLibrary::load_pec_line / compute volume/emissivity/grid read:
  /volume/pec/<elem>/<year>/<ion>_<wl>_pec         excitation PEC  (x n_e x n_ion)
  /volume/pec/<elem>/<year>/<ion>_<wl>_recom_pec   recombination PEC (x n_e x n_(ion+1))
  /volume/sxb/<elem>/<year>/<ion>_<wl>_sxb         ionisations per photon
each with coefficient = log10 on (nTe, nne), temperature = log10(Te eV), density = log10(ne m^-3).
<wl> is the adf15 wavelength in Angstrom, rounded.

Usage (files from https://open.adas.ac.uk/element/<elem>):
  python3 add_pec_tables.py --elem li --year 96 --ion li0 --wl 6709.7 \\
      --adf15 pec96#li_pju#li0.dat --adf13 sxb96#li_pju#li0.dat [--db ../processes.h5]
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def read_blocks(path):
    """adf15/adf13 -> list of (wavelength A, type, Te eV [nT], ne cm^-3 [nN], values [nT, nN])."""
    tok = open(path).read().split("\n")
    blocks, i = [], 1
    hdr = re.compile(r"^\s*([\d.]+)\s*A\s+(\d+)\s+(\d+)\s*/(.*)$")
    while i < len(tok):
        m = hdr.match(tok[i])
        if not m:
            i += 1
            continue
        wl, nn, nt, rest = float(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        typ = re.search(r"TYPE\s*=\s*(\w+)", rest)
        nums, i = [], i + 1
        while len(nums) < nn + nt + nn * nt:
            nums += [float(v.replace("D", "E")) for v in tok[i].split()]
            i += 1
        ne = np.array(nums[:nn]); te = np.array(nums[nn:nn + nt])
        val = np.array(nums[nn + nt:nn + nt + nn * nt]).reshape(nn, nt).T      # file order: per density, all Te
        blocks.append((wl, typ.group(1) if typ else "SXB", te, ne, val))
    return blocks


def write(grp, te, ne, val, units, source):
    for name in ("coefficient", "temperature", "density"):
        if name in grp:
            del grp[name]
    ds = grp.create_dataset("coefficient", data=np.log10(np.maximum(val, 1e-300)),
                            compression="gzip", compression_opts=6)
    ds.attrs["units"] = units
    ds.attrs["layout"] = "(nTe, nne) row-major; axes log10(Te eV), log10(ne m^-3)"
    ds.attrs["source"] = source
    grp.create_dataset("temperature", data=np.log10(te))
    grp.create_dataset("density", data=np.log10(ne * 1.0e6))


def pick(blocks, wl, typ):
    hit = [b for b in blocks if abs(b[0] - wl) < 0.05 and b[1] == typ]
    return hit[0] if hit else None


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elem", required=True)
    ap.add_argument("--year", required=True)
    ap.add_argument("--ion", required=True, help="emitting stage, e.g. li0 or li1")
    ap.add_argument("--wl", type=float, required=True, help="adf15 wavelength [A]")
    ap.add_argument("--adf15", type=Path, required=True)
    ap.add_argument("--adf13", type=Path)
    ap.add_argument("--db", type=Path, default=here.parent / "processes.h5")
    args = ap.parse_args()

    key = f"{args.ion}_{round(args.wl)}"
    pec = read_blocks(args.adf15)
    with h5py.File(args.db, "r+") as f:
        root = f"volume/pec/{args.elem}/{args.year}"
        for typ, suffix in (("EXCIT", "_pec"), ("RECOM", "_recom_pec")):
            b = pick(pec, args.wl, typ)
            if b is None:
                print(f"  {key}: no {typ} block")
                continue
            write(f.require_group(f"{root}/{key}{suffix}"), b[2], b[3], b[4],
                  "log10(photons cm^3 s^-1)", f"open-ADAS adf15 {args.adf15.name} {b[0]} A {typ}")
            print(f"  /{root}/{key}{suffix}: {b[4].shape}, PEC {b[4].min():.2e}..{b[4].max():.2e} cm^3/s")
        if args.adf13:
            b = pick(read_blocks(args.adf13), args.wl, "SXB")
            if b is None:
                print(f"  {key}: no SXB block")
            else:
                write(f.require_group(f"volume/sxb/{args.elem}/{args.year}/{key}_sxb"), b[2], b[3], b[4],
                      "log10(ionisations per photon)", f"open-ADAS adf13 {args.adf13.name} {b[0]} A")
                print(f"  /volume/sxb/{args.elem}/{args.year}/{key}_sxb: {b[4].shape}, S/XB {b[4].min():.3g}..{b[4].max():.3g}")
        f.attrs[f"pec_{args.elem}_{key}"] = f"added {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%MZ} from {args.adf15.name}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
