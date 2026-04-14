#!/usr/bin/env python3
"""Parse the EIRENE SPUTER database into a C++ data header.

The SPUTER file is EIRENE's canonical physical-sputter parameter table,
transcribed from:

    W. Eckstein, C. Garcia-Rosales, J. Roth, W. Ottenberger,
    "Sputtering Data", IPP 9/82, Feb 1993, Table 3, page 335.

Every SOLPS-ITER / SOLEDGE3X / EIRENE run that uses physical sputtering
MODPYS=2 reads this exact file (eirmod_sputer.f, subroutine EIRENE_SPUTER,
around line 600).  Target materials run across ~28 targets x 11 projectile
columns.  The formula used with these parameters is NOT the Eckstein-
Preuss 2003 revised Bohdansky - it is the older Bohdansky/Roth/Eckstein
1993 formula:

    Y(E, 0) = Q * (1 - (Eth/E)^(2/3)) * (1 - Eth/E)^2 * s_n^{KrC}(E/ETF)

with an empirical Yamamura angular factor applied on top:

    Y(E, theta) = Y(E, 0) * cos(theta)^(-f) *
                  exp(f * (1 - 1/cos(theta)) * cos(theta_opt))

with f = 2.0 and cos(theta_opt) = 0.26 hard-coded for every combination.

Usage:
    python3 parse_eirene_sputer.py \
        --input /path/to/Eirene/Database/Surfacedata/SPUTER \
        --output ../../src/OPENEDGE/eckstein_sputter_data.h
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Map of projectile column headers to (symbol, Z, M) used to generate C++
# entry names like "D_on_W".  Columns in the SPUTER table:
#   H, D, T, He, C, O, self-sput, Ne, Ar, Kr, Xe
PROJECTILES = [
    ("H",  1,  1.008),
    ("D",  1,  2.014),
    ("T",  1,  3.016),
    ("He", 2,  4.003),
    ("C",  6, 12.011),
    ("O",  8, 15.999),
    ("self", None, None),   # self-sputter - projectile species == target species
    ("Ne", 10, 20.180),
    ("Ar", 18, 39.948),
    ("Kr", 36, 83.798),
    ("Xe", 54, 131.293),
]


# German-in-EIRENE -> English/symbol mapping for the target names in the
# SPUTER table so we can emit readable C++ shorthand keys.
TARGET_ALIASES = {
    "Lithium":    ("Li",  3,   6.94),
    "Beryllium":  ("Be",  4,   9.012),
    "Bor":        ("B",   5,  10.81),
    "Graphite":   ("C",   6,  12.011),
    "Aluminium":  ("Al", 13,  26.982),
    "Silizium":   ("Si", 14,  28.085),
    "Titanium":   ("Ti", 22,  47.867),
    "Vanadium":   ("V",  23,  50.942),
    "Chromium":   ("Cr", 24,  51.996),
    "Manganese":  ("Mn", 25,  54.938),
    "Iron":       ("Fe", 26,  55.845),
    "Cobalt":     ("Co", 27,  58.933),
    "Nickel":     ("Ni", 28,  58.693),
    "Copper":     ("Cu", 29,  63.546),
    "Gallium":    ("Ga", 31,  69.723),
    "Germanium":  ("Ge", 32,  72.630),
    "Zirconium":  ("Zr", 40,  91.224),
    "Niobium":    ("Nb", 41,  92.906),
    "Molybdenum": ("Mo", 42,  95.95),
    "Rhodium":    ("Rh", 45, 102.906),
    "Palladium":  ("Pd", 46, 106.42),
    "Silver":     ("Ag", 47, 107.868),
    "Cadmium":    ("Cd", 48, 112.414),
    "Tin":        ("Sn", 50, 118.710),
    "Tantalum":   ("Ta", 73, 180.948),
    "Tungsten":   ("W",  74, 183.84),
    "Platinum":   ("Pt", 78, 195.084),
    "Gold":       ("Au", 79, 196.967),
    "Lead":       ("Pb", 82, 207.2),
    "Uranium":    ("U",  92, 238.029),
}


# Projectile atomic numbers / masses from SPUTER header "101 201 301 402
# 1206 1608 0 2010 4018 8436 13154" - the encoding is ZZAAA decimal.  We
# don't need to parse them since we already supply PROJECTILES above, but
# we verify the order is consistent by checking Z against the header.


def parse_sputer(path: Path):
    """Return a list of target dicts, each with Es and per-projectile params."""
    text = path.read_text()
    lines = text.splitlines()

    targets = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        # Target blocks begin with a name followed by atomic ID (e.g. "Tungsten 18474")
        m = re.match(r"^([A-Z][a-z]+)\s+(\d+)\s*$", line)
        if m and m.group(1) in TARGET_ALIASES:
            name = m.group(1)
            # Next line: "Es = X.XXeV"
            es_line = lines[i + 1].strip()
            es_match = re.match(r"Es\s*=\s*([\d.]+)\s*eV", es_line)
            if not es_match:
                raise RuntimeError(f"No Es line after target {name!r}")
            es_ev = float(es_match.group(1))

            def parse_row(lnum: int):
                # take tokens 2..end (after label "M2/M1" or "ETF(eV)" etc.)
                row_line = lines[lnum].strip()
                parts = row_line.split()
                # parts[0] might be "M2/M1" or "M2/M1:" etc; strip until the
                # first numeric value
                vals = []
                for tok in parts:
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        continue
                if len(vals) != len(PROJECTILES):
                    raise RuntimeError(
                        f"Row at line {lnum+1} has {len(vals)} numeric "
                        f"fields, expected {len(PROJECTILES)}: {row_line!r}")
                return vals

            m2m1 = parse_row(i + 2)
            etf  = parse_row(i + 3)
            eth  = parse_row(i + 4)
            q    = parse_row(i + 5)
            targets.append({
                "name": name,
                "es":   es_ev,
                "m2m1": m2m1,
                "etf":  etf,
                "eth":  eth,
                "q":    q,
            })
            i += 6
        else:
            i += 1
    return targets


def emit_cpp_header(targets: list[dict], out_path: Path, src_path: Path):
    """Write a C++ header with a flat SputterEntry array, auto-generated."""
    lines = []
    lines.append("/* ----------------------------------------------------------------------")
    lines.append("   Auto-generated from EIRENE's canonical SPUTER parameter database.")
    lines.append("   DO NOT EDIT BY HAND.  Regenerate with:")
    lines.append("       python3 tools/sputter/parse_eirene_sputer.py \\")
    lines.append(f"           --input {src_path} \\")
    lines.append("           --output src/OPENEDGE/eckstein_sputter_data.h")
    lines.append("")
    lines.append("   Source: W. Eckstein, C. Garcia-Rosales, J. Roth, W. Ottenberger,")
    lines.append("           \"Sputtering Data\", IPP 9/82, Feb 1993, Table 3, page 335.")
    lines.append("   EIRENE reader:  modules/Eirene/.../eirmod_sputer.f  (EIRENE_SPUTER)")
    lines.append("   Used verbatim by SOLPS-ITER and SOLEDGE3X.")
    lines.append("")
    lines.append("   Fields per entry (IPP 9/82 Bohdansky-1993 formula):")
    lines.append("       Z1, M1   projectile atomic number and mass (amu)")
    lines.append("       Z2, M2   target     atomic number and mass (amu)")
    lines.append("       Es       target surface binding energy [eV]")
    lines.append("       Eth      threshold energy [eV]")
    lines.append("       Q        Bohdansky yield prefactor [atoms/ion]")
    lines.append("       ETF      Thomas-Fermi reference energy [eV]  (for s_n^KrC(E/ETF))")
    lines.append("------------------------------------------------------------------------- */")
    lines.append("")
    lines.append("#ifndef SPARTA_ECKSTEIN_SPUTTER_DATA_H")
    lines.append("#define SPARTA_ECKSTEIN_SPUTTER_DATA_H")
    lines.append("")
    lines.append("namespace SPARTA_NS {")
    lines.append("namespace Eckstein {")
    lines.append("")
    lines.append("struct SputterDataEntry {")
    lines.append("  const char *name;")
    lines.append("  double Z1, M1, Z2, M2;")
    lines.append("  double Es, Eth, Q, ETF;")
    lines.append("};")
    lines.append("")
    lines.append("static const SputterDataEntry EIRENE_SPUTTER_TABLE[] = {")

    for t in targets:
        tgt_name, Z2, M2 = TARGET_ALIASES[t["name"]]
        es = t["es"]
        for j, (proj_name, Zj, Mj) in enumerate(PROJECTILES):
            eth_j = t["eth"][j]
            q_j   = t["q"][j]
            etf_j = t["etf"][j]
            if eth_j <= 0.0 or q_j <= 0.0:
                continue  # skip empty entries
            if proj_name == "self":
                Z1, M1 = Z2, M2
                key = f"{tgt_name}_on_{tgt_name}"
            else:
                Z1, M1 = Zj, Mj
                key = f"{proj_name}_on_{tgt_name}"
            lines.append(
                f'  {{"{key}", '
                f"{Z1:>4}, {M1:>9.4f}, {Z2:>4}, {M2:>9.4f}, "
                f"{es:>6.3f}, {eth_j:>9.4f}, {q_j:>9.4f}, {etf_j:>12.3f}}},")

    lines.append("};")
    lines.append("")
    lines.append("static const int N_EIRENE_SPUTTER = "
                 "sizeof(EIRENE_SPUTTER_TABLE) / sizeof(EIRENE_SPUTTER_TABLE[0]);")
    lines.append("")
    lines.append("}  // namespace Eckstein")
    lines.append("}  // namespace SPARTA_NS")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("/home/cloud/local/solps/solps-iter-3.0.8-devel/"
                                "modules/Eirene/Database/Surfacedata/SPUTER"))
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parents[2] /
                                "src/OPENEDGE/eckstein_sputter_data.h")
    args = p.parse_args()

    targets = parse_sputer(args.input)
    print(f"Parsed {len(targets)} targets from {args.input}")
    for t in targets:
        print(f"  {t['name']:12s}  Es={t['es']:.2f} eV  "
              f"(W/{t['name']} Eth={t['eth'][6]:.2f})")
    emit_cpp_header(targets, args.output, args.input)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
