#!/usr/bin/env python3
"""Parse EIRENE TRIM reflection data files into per-combination HDF5.

Each EIRENE TRIM file (e.g. ``W_on_W``, ``D_on_W``) holds the reflection
response for one (projectile, target) pair as 84 blocks spanning
``INE=12`` incident energies and ``INW=7`` incident angles.  See
``modules/Eirene/.../rdtrim.f`` - the EIRENE reader - and
``eirmod_reflec.f`` which samples the quantile tables.

File layout per (ie, iw) block (Fortran list-directed, one logical
record per physical line; comment/header lines are skipped):

    # 2 comment lines
    Z1  M1  Z2  M2  E  theta  HFTR0   [RE?  INR?  0.00]      (header line)
                                                              (blank line)
    HFTR1(5)   EMINR   EMAXR                                  (5 E_out quantiles + bounds)
                                                              (blank line)
    5 lines, each: HFTR2(5)   cmin  cmax                      (cos(polar) quantiles)
                                                              (blank line)
    25 lines, each: HFTR3(5)   amin  amax                     (azimuthal quantiles)

Quantile levels are RAAR = [0.1, 0.3, 0.5, 0.7, 0.9] (hard-coded in
rdtrim.f line 146).

We store each combination as an HDF5 file with the following datasets:

    E            (12,)         incident energies [eV]
    theta        (7,)          incident angles [deg]
    R_N          (12, 7)       particle reflection probability
    Eout_q       (12, 7, 5)    quantiles of reflected particle energy [eV]
    Eout_min     (12, 7)       bound for clamping E_out from below
    Eout_max     (12, 7)       bound for clamping E_out from above
    cos_polar_q  (12, 7, 5, 5)   polar-angle cosines of outgoing v
    cos_azim_q   (12, 7, 5, 5, 5) azimuthal cosines of outgoing v
    raar         (5,)            quantile levels
    projectile   scalar attr    "W", "D", ...
    target       scalar attr    "W", "C", ...
    Z1, M1, Z2, M2              scalar attrs (from block header)

Usage:

    python3 parse_eirene_trim.py --src /path/to/TRIM \
        --out database/surface/trim
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

INE = 12
INW = 7
INR = 5
RAAR = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def parse_block(lines, start):
    """Parse one (ie, iw) block starting at lines[start].

    Returns (next_start, header_tokens, hftr1+bounds, hftr2+bounds, hftr3+bounds).
    """
    i = start
    n = len(lines)

    def next_data_line():
        nonlocal i
        while i < n:
            s = lines[i].strip()
            i += 1
            if not s:
                continue
            # skip comment-ish lines (contain any letters and aren't all-numeric)
            if re.search(r"[A-Za-z]", s):
                # but header lines start with pure numbers - we'll detect that
                # by attempting to parse; if it fails, it's a comment
                try:
                    [float(t) for t in s.split()]
                except ValueError:
                    continue
            return s
        return None

    header = next_data_line()
    if header is None:
        return None
    header_toks = [float(t) for t in header.split()]

    hftr1_line = next_data_line()
    if hftr1_line is None:
        raise RuntimeError(f"EOF mid-block at line {i}")
    hftr1_toks = [float(t) for t in hftr1_line.split()]

    hftr2 = []
    for _ in range(INR):
        row = next_data_line()
        if row is None:
            raise RuntimeError(f"EOF mid HFTR2 at line {i}")
        hftr2.append([float(t) for t in row.split()])

    hftr3 = []
    for _ in range(INR * INR):
        row = next_data_line()
        if row is None:
            raise RuntimeError(f"EOF mid HFTR3 at line {i}")
        hftr3.append([float(t) for t in row.split()])

    return i, header_toks, hftr1_toks, hftr2, hftr3


def parse_trim_file(path: Path):
    """Parse one TRIM file into dense arrays."""
    lines = path.read_text().splitlines()

    E_grid = np.zeros(INE)
    theta_grid = np.zeros(INW)
    R_N = np.zeros((INE, INW))
    Eout_q = np.zeros((INE, INW, INR))
    Eout_min = np.zeros((INE, INW))
    Eout_max = np.zeros((INE, INW))
    cos_polar_q = np.zeros((INE, INW, INR, INR))
    polar_min = np.zeros((INE, INW, INR))
    polar_max = np.zeros((INE, INW, INR))
    cos_azim_q = np.zeros((INE, INW, INR, INR, INR))

    Z1 = M1 = Z2 = M2 = 0.0

    i = 0
    for ie in range(INE):
        for iw in range(INW):
            result = parse_block(lines, i)
            if result is None:
                raise RuntimeError(
                    f"{path.name}: expected block (ie={ie},iw={iw}), got EOF")
            i, hdr, h1, h2_rows, h3_rows = result

            # Header (first 7 tokens are Z1, M1, Z2, M2, E, theta, R_N;
            # Fortran read discards anything beyond position 7).
            if len(hdr) < 7:
                raise RuntimeError(
                    f"{path.name}: header at block ({ie},{iw}) has "
                    f"only {len(hdr)} values")
            Z1 = hdr[0]; M1 = hdr[1]; Z2 = hdr[2]; M2 = hdr[3]
            E_grid[ie]    = hdr[4]
            theta_grid[iw] = hdr[5]
            R_N[ie, iw]   = hdr[6]

            # HFTR1: first 5 tokens are E_out quantiles; tokens 6,7 (if
            # present) are Eout_min, Eout_max.
            if len(h1) < INR:
                raise RuntimeError(
                    f"{path.name}: HFTR1 at ({ie},{iw}) has {len(h1)} values")
            Eout_q[ie, iw, :] = h1[:INR]
            Eout_min[ie, iw] = h1[INR] if len(h1) > INR else 0.0
            Eout_max[ie, iw] = h1[INR + 1] if len(h1) > INR + 1 else 0.0

            for ir1 in range(INR):
                row = h2_rows[ir1]
                if len(row) < INR:
                    raise RuntimeError(
                        f"{path.name}: HFTR2 row short at ({ie},{iw},{ir1})")
                cos_polar_q[ie, iw, ir1, :] = row[:INR]
                polar_min[ie, iw, ir1] = row[INR] if len(row) > INR else 0.0
                polar_max[ie, iw, ir1] = row[INR + 1] if len(row) > INR + 1 else 1.0

            for idx3, row in enumerate(h3_rows):
                ir1 = idx3 // INR
                ir2 = idx3 % INR
                if len(row) < INR:
                    raise RuntimeError(
                        f"{path.name}: HFTR3 row short at ({ie},{iw},{ir1},{ir2})")
                cos_azim_q[ie, iw, ir1, ir2, :] = row[:INR]

    return {
        "Z1": Z1, "M1": M1, "Z2": Z2, "M2": M2,
        "E_grid": E_grid,
        "theta_grid": theta_grid,
        "R_N": R_N,
        "Eout_q": Eout_q,
        "Eout_min": Eout_min,
        "Eout_max": Eout_max,
        "cos_polar_q": cos_polar_q,
        "polar_min": polar_min,
        "polar_max": polar_max,
        "cos_azim_q": cos_azim_q,
    }


def write_hdf5(out_path: Path, name: str, data: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["name"] = name
        f.attrs["Z1"] = data["Z1"]
        f.attrs["M1"] = data["M1"]
        f.attrs["Z2"] = data["Z2"]
        f.attrs["M2"] = data["M2"]
        f.attrs["source"] = "EIRENE TRIM database (rdtrim.f format)"
        f.attrs["ine"] = INE
        f.attrs["inw"] = INW
        f.attrs["inr"] = INR
        f.create_dataset("E",           data=data["E_grid"])
        f.create_dataset("theta",       data=data["theta_grid"])
        f.create_dataset("raar",        data=RAAR)
        f.create_dataset("R_N",         data=data["R_N"])
        f.create_dataset("Eout_q",      data=data["Eout_q"])
        f.create_dataset("Eout_min",    data=data["Eout_min"])
        f.create_dataset("Eout_max",    data=data["Eout_max"])
        f.create_dataset("cos_polar_q", data=data["cos_polar_q"])
        f.create_dataset("polar_min",   data=data["polar_min"])
        f.create_dataset("polar_max",   data=data["polar_max"])
        f.create_dataset("cos_azim_q",  data=data["cos_azim_q"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path,
                    default=Path("/home/cloud/local/solps/solps-iter-3.0.8-devel/"
                                 "modules/Eirene/Database/Surfacedata/TRIM"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] /
                             "database/surface/trim")
    ap.add_argument("--pattern", default="*_on_*",
                    help="Filename glob for TRIM files "
                         "(default matches base files like 'W_on_W')")
    ap.add_argument("--skip-suffixed", action="store_true", default=True,
                    help="Skip files ending in _N.NN which are alternative "
                         "binding-energy variants (default true).")
    args = ap.parse_args()

    files = sorted(args.src.glob(args.pattern))
    # drop files ending in _number.number (alternative Es variants)
    if args.skip_suffixed:
        files = [f for f in files
                 if not re.search(r"_\d+\.\d+$", f.name)]

    print(f"Found {len(files)} base TRIM files under {args.src}")

    ok, skip = 0, 0
    for f in files:
        name = f.name  # e.g. "W_on_W"
        if not re.match(r"^[A-Za-z][A-Za-z0-9]*_on_[A-Za-z][A-Za-z0-9]*$", name):
            print(f"  skip {name}: not a standard A_on_B name")
            skip += 1
            continue
        try:
            data = parse_trim_file(f)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            skip += 1
            continue

        out_file = args.out / f"{name}.h5"
        write_hdf5(out_file, name, data)

        R_N_max = float(data["R_N"].max())
        Eout_med = float(data["Eout_q"][6, 3, 2])  # E index 6 (mid), θ index 3, median q
        print(f"  {name:12s}  R_N_max={R_N_max:.3f}  "
              f"median E_out at bin(6,3)={Eout_med:.1f} eV  ->  {out_file.name}")
        ok += 1

    print(f"\n{ok} files written to {args.out}, {skip} skipped")


if __name__ == "__main__":
    main()
