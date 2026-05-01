#!/usr/bin/env python3
"""Build OpenEdge ADAS rate tables from open-ADAS adf11 text files.

Ingests per element, writing a single ADAS_Rates_<Z>.h5 per element:

  scd - ionization      sigma_v    [log10, cm^3/s]   (required)
  acd - recombination   sigma_v    [log10, cm^3/s]   (required)
  ccd - charge-exchange sigma_v    [log10, cm^3/s]   (optional)
  plt - line radiation  power      [log10, W cm^3]   (optional)
  prb - recomb+brems    power      [log10, W cm^3]   (optional)

plus ionization potentials (eV, one per charge state) when an
open-ADAS ionization_potentials file is available.

Usage (defaults process every element in DEFAULT_ELEMENTS):

  python3 adas.py
  python3 adas.py --elements h c w
  python3 adas.py --year 96 --elements h

The script looks for adf11 text files under
  ${ADAS_ADF11_DIR:-./adf11}/{scd,acd,ccd,plt,prb}<year>/<class><year>_<elt>.dat
and ionization-potential files under ./ionization_potentials/ADAS_ionization_potentials_<Elt>.

Any class whose file is missing is skipped (with a warning); only scd + acd
are required.  The charge-state grids are 0-based (neutral = 0) and use
convention [before_state, after_state] for ionization/recombination/CX and
[radiating_state, radiating_state] for plt, so the 2xN layout is uniform.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np


# Element symbol -> nuclear charge.  Hydrogen isotopes all use the adf11 'h' file.
DEFAULT_ELEMENTS: dict[str, int] = {
    "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8,
    "ne": 10, "ar": 18, "fe": 26, "kr": 36, "mo": 42, "xe": 54, "w": 74,
}

# (class, dataset_base, coeff_name, required, units)
CLASSES = [
    ("scd", "Ionization",      "RateCoeff",  True,  "log10(sigma_v) [cm^3/s]"),
    ("acd", "Recombination",   "RateCoeff",  True,  "log10(sigma_v) [cm^3/s]"),
    ("ccd", "ChargeExchange",  "RateCoeff",  False, "log10(sigma_v) [cm^3/s]"),
    ("plt", "LineRadiation",   "PowerCoeff", False, "log10(power) [W cm^3]"),
    ("prb", "RecombRadiation", "PowerCoeff", False, "log10(power) [W cm^3]"),
]


def parse_adf11(path: Path):
    """Parse an adf11 text file.

    Returns (logQ[nDens, nTe, nZ], logDens, logTe, iZmin, iZmax).
    The adf11 header reads: izmax_file  nDens  nTe  iZmin  iZmax  /ELEMENT/ source
    """
    text = path.read_text().splitlines()
    nums = [int(s) for s in text[0].split() if s.lstrip("-").isdigit()]
    if len(nums) < 5:
        raise ValueError(f"{path}: cannot parse header '{text[0]}'")
    _izmax_file, nDens, nTe, iZmin, iZmax = nums[:5]
    nZ = iZmax - iZmin + 1
    nData = nDens * nTe

    logQ = np.zeros((nDens, nTe, nZ))
    logDens = np.zeros(nDens)
    logTe = np.zeros(nTe)

    # Read logDens then logTe (both on continuation lines, starting line 2 after the dashes)
    iline = 2
    iDens = iTe = 0
    while iTe < nTe:
        flist = [float(s) for s in text[iline].split()]
        imin = 0
        if iDens < nDens:
            take = min(nDens - iDens, len(flist))
            logDens[iDens:iDens + take] = flist[:take]
            iDens += take
            imin = take
        if iDens == nDens and iTe < nTe and imin < len(flist):
            take = min(nTe - iTe, len(flist) - imin)
            logTe[iTe:iTe + take] = flist[imin:imin + take]
            iTe += take
        iline += 1

    # Per-charge-state blocks: one header line, then nData values in row-major (fortran order).
    for iZ in range(nZ):
        iline += 1  # skip per-Z header
        data = np.zeros(nData)
        pos = 0
        while pos < nData:
            flist = [float(s) for s in text[iline].split()]
            take = min(nData - pos, len(flist))
            data[pos:pos + take] = flist[:take]
            pos += take
            iline += 1
        logQ[:, :, iZ] = data.reshape((nDens, nTe), order="F")

    return logQ, logDens, logTe, iZmin, iZmax


def charge_state_grid(class_name: str, iZmin: int, iZmax: int) -> np.ndarray:
    """Build a (2, nZ) 0-based charge-state grid matching the tabulated stages.

    adf11 indexes charge states 1..Zmax where iZ=1 is the neutral atom.  We
    convert to 0-based (neutral=0) and pair [before, after] by reaction:

      scd (X^q -> X^{q+1}): before = q,   after = q+1
      acd (X^q -> X^{q-1}): before = q,   after = q-1
      ccd (X^q -> X^{q-1}): before = q,   after = q-1   (CX with neutral H)
      plt (line rad in state q):         before = after = q
      prb (brems+recomb from q -> q-1):  before = q,   after = q-1
    """
    nZ = iZmax - iZmin + 1
    if class_name == "scd":
        before = np.arange(iZmin - 1, iZmax)
        after  = np.arange(iZmin,     iZmax + 1)
    elif class_name in ("acd", "ccd", "prb"):
        before = np.arange(iZmin,     iZmax + 1)
        after  = np.arange(iZmin - 1, iZmax)
    elif class_name == "plt":
        before = np.arange(iZmin - 1, iZmax)
        after  = before.copy()
    else:
        raise ValueError(class_name)
    grid = np.vstack([before, after])
    assert grid.shape == (2, nZ), (grid.shape, nZ)
    return grid


def load_ionization_potentials(path: Path, Z: int) -> np.ndarray:
    """Parse an open-ADAS ionization_potentials text file.

    Format (S3X convention):
        ----- header comment -----
        Element = <symbol>
        Z       = <Z>

        <IP_0>
        <IP_1>
        ...
        <IP_{Z-1}>
        <footer text>

    Returns an array of Z floats (eV), one per charge state (0..Z-1).
    """
    text = path.read_text()
    vals = [float(s) for s in re.findall(r"[-+]?\d+\.\d+[eE][-+]?\d+", text)]
    if len(vals) < Z:
        raise RuntimeError(f"{path}: expected >={Z} IP values, got {len(vals)}")
    return np.asarray(vals[:Z], dtype=np.float64)


def ip_filename(symbol: str) -> str:
    """Open-ADAS uses 'D' for hydrogen; everything else uses capitalized symbol."""
    if symbol == "h":
        return "ADAS_ionization_potentials_D"
    return f"ADAS_ionization_potentials_{symbol[0].upper() + symbol[1:].lower()}"


def build_element(symbol: str, Z: int, adf11_dir: Path, ip_dir: Path,
                  year: str, out_dir: Path, verbose: bool = True) -> Path:
    out_path = out_dir / f"ADAS_Rates_{Z}.h5"
    loaded: dict[str, dict] = {}

    for class_name, base, coeff_name, required, units in CLASSES:
        path = adf11_dir / f"{class_name}{year}" / f"{class_name}{year}_{symbol}.dat"
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            if verbose:
                print(f"  [skip] {class_name}: {path} not found")
            continue
        logQ, logDens, logTe, iZmin, iZmax = parse_adf11(path)
        if verbose:
            print(f"  [ok]   {class_name}: {path.name} "
                  f"(nDens={len(logDens)}, nTe={len(logTe)}, nZ={iZmax-iZmin+1})")
        loaded[class_name] = dict(
            logQ=logQ, logDens=logDens, logTe=logTe,
            iZmin=iZmin, iZmax=iZmax,
            base=base, coeff=coeff_name, units=units,
        )

    # Ionization potentials (optional)
    ip_vals = None
    if ip_dir is not None:
        ip_path = ip_dir / ip_filename(symbol)
        if ip_path.exists():
            ip_vals = load_ionization_potentials(ip_path, Z)
            if verbose:
                print(f"  [ok]   IP  : {ip_path.name} -> {ip_vals.tolist()}")
        elif verbose:
            print(f"  [skip] IP  : {ip_path.name} not found")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("Atomic_Number", data=np.array([Z], dtype=np.int64))
        for class_name, d in loaded.items():
            base = d["base"]
            dset = f.create_dataset(f"{base}{d['coeff']}", data=d["logQ"].T)
            dset.attrs["units"] = d["units"]
            f.create_dataset(f"gridDensity_{base}",     data=d["logDens"])
            f.create_dataset(f"gridTemperature_{base}", data=d["logTe"])
            f.create_dataset(
                f"gridChargeState_{base}",
                data=charge_state_grid(class_name, d["iZmin"], d["iZmax"]),
            )
        if ip_vals is not None:
            ip = f.create_dataset("IonizationPotential", data=ip_vals)
            ip.attrs["units"] = "eV"

    if verbose:
        print(f"  wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--adf11-dir", type=Path,
                   default=Path(os.environ.get("ADAS_ADF11_DIR", here / "adf11")),
                   help="Directory with {scd,acd,ccd,plt,prb}<year>/ subdirs")
    p.add_argument("--ip-dir", type=Path,
                   default=here / "ionization_potentials",
                   help="Directory with ADAS_ionization_potentials_<Elt> files")
    p.add_argument("--year", default="89",
                   help="adf11 year suffix (default 89)")
    p.add_argument("--out-dir", type=Path, default=here,
                   help="Output directory for ADAS_Rates_<Z>.h5 files")
    p.add_argument("--elements", nargs="+", default=None,
                   help=f"Element symbols to process (default: {list(DEFAULT_ELEMENTS)})")
    args = p.parse_args(argv)

    if args.elements:
        elements = {}
        for sym in args.elements:
            s = sym.lower()
            if s not in DEFAULT_ELEMENTS:
                print(f"Unknown element {sym!r}; known: {list(DEFAULT_ELEMENTS)}", file=sys.stderr)
                return 2
            elements[s] = DEFAULT_ELEMENTS[s]
    else:
        elements = dict(DEFAULT_ELEMENTS)

    for symbol, Z in elements.items():
        print(f"\n=== {symbol.upper()} (Z={Z}) ===")
        try:
            build_element(symbol, Z, args.adf11_dir, args.ip_dir,
                          args.year, args.out_dir)
        except FileNotFoundError as e:
            print(f"  [FAIL] required file missing: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
