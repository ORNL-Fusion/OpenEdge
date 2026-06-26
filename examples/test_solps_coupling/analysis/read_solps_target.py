#!/usr/bin/env python3
"""Reader for SOLPS-ITER 'b2plot wlld' target profile files.

Combines the flux profile (fp_tg_*.dat) and load profile (ld_tg_*.dat)
files for a chosen target into a single dict keyed on the column header
names. Both files share the `x` axis (coordinate along target, m, with
PFR < 0). The reader joins on `x` and returns one record per target
radial cell.

Files written by `b2plot wlld` (run after SOLPS converges):
  fp_tg_o.dat   outer lower target  (key: 'outer')
  fp_tg_i.dat   inner lower target  (key: 'inner')
  fp_tg_ou.dat  outer upper target  (key: 'outer_upper', CDN/DDN only)
  fp_tg_iu.dat  inner upper target  (key: 'inner_upper', CDN/DDN only)

Common columns of interest:
  x          coordinate along target [m]   (>=0 SOL, <0 PFR — i.e. L-Lsep)
  Li_flux_ion incident Li-ion flux at target [1/m^2/s]
  D_flux_ion  incident D-ion flux  at target [1/m^2/s]
  Ne_flux_ion incident Ne-ion flux at target [1/m^2/s]
  ne, Te, Ti  electron / ion plasma fields at target
  Wtot        total power flux at target [W/m^2]
  Rclfc, Zclfc target (R, Z) coords
"""

from pathlib import Path
import numpy as np


def _parse_b2plot_target_file(path):
    """Parse one fp_tg_*.dat or ld_tg_*.dat file. Returns dict(name -> 1D array).

    Robust to (a) repeated header sections from b2plot append-mode runs and
    (b) data-row + header concatenation on the same line (b2plot occasionally
    writes the next-file header without a leading newline).
    """
    text = Path(path).read_text().splitlines()
    rows = []
    header_cols = None
    seen_data = False
    for raw in text:
        s = raw.rstrip()
        if not s:
            continue
        # If a stray header is appended to a data row (no newline between),
        # the line will look like:  -1.23E-01 ... # Target erosion - inner
        # Truncate at any embedded '#' so we keep only the leading numeric prefix.
        hash_pos = s.find('#')
        if hash_pos > 0 and any(c.isdigit() for c in s[:hash_pos]):
            s = s[:hash_pos].rstrip()
        if not s:
            continue
        # Comment / header lines (start with '#' or non-numeric leading tok).
        first_tok = s.split()[0].lstrip('#').lstrip()
        leading_is_number = False
        try:
            float(first_tok); leading_is_number = True
        except ValueError:
            leading_is_number = False
        if not leading_is_number:
            # Take the LAST header section before data starts: column-name rows
            # have many tokens that look like identifiers (no ':' on first tok).
            if seen_data:
                # ignore appended next-file header
                continue
            stripped = s.lstrip('#').strip()
            if stripped and ':' not in stripped.split()[0]:
                # column-name row (avoids "x : coordinate ...")
                header_cols = stripped.split()
            continue
        # Numeric data row.
        try:
            row = [float(x) for x in s.split()]
        except ValueError:
            continue
        rows.append(row)
        seen_data = True

    if header_cols is None:
        raise ValueError(f"no column-name header found in {path}")
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{path}: no numeric rows parsed")
    # If file was duplicated (appended), keep only one full copy. Each copy
    # should have the same number of rows, deduplicated by 'x' (first column).
    if arr.shape[0] >= 2:
        x_first = arr[0, 0]
        # find next occurrence of the same x (within tight tol) -> end of 1st copy
        rest = arr[1:, 0]
        match = np.where(np.abs(rest - x_first) < 1e-12)[0]
        if match.size:
            arr = arr[: match[0] + 1]
    if arr.shape[1] != len(header_cols):
        raise ValueError(
            f"{path}: header has {len(header_cols)} columns "
            f"but data has {arr.shape[1]}\n  header={header_cols}\n  "
            f"first row={arr[0].tolist()}")
    return {name: arr[:, j] for j, name in enumerate(header_cols)}


def read_target(b2pl_dir, target='outer'):
    """Read fp_tg_<key>.dat + ld_tg_<key>.dat for a target.

    Args:
      b2pl_dir: path to b2pl.exe.dir (or any dir holding the .dat files)
      target: 'outer'|'inner'|'outer_upper'|'inner_upper'
    Returns:
      dict of column-name -> 1D np.ndarray (one row per radial cell), with
      the union of fp + ld columns. The shared `x` axis is verified to
      match between the two files; if it doesn't, a ValueError is raised.
    """
    suf = {'outer': 'o', 'inner': 'i',
           'outer_upper': 'ou', 'inner_upper': 'iu'}[target]
    bp = Path(b2pl_dir)
    fp = _parse_b2plot_target_file(bp / f'fp_tg_{suf}.dat')
    ld = _parse_b2plot_target_file(bp / f'ld_tg_{suf}.dat')

    if not np.allclose(fp['x'], ld['x'], atol=1e-9):
        raise ValueError(f"x axes of fp_tg_{suf}.dat and ld_tg_{suf}.dat "
                         f"do not match")
    out = dict(fp)
    for k, v in ld.items():
        if k == 'x':
            continue
        if k in out:
            # ld keys override fp keys when names collide (e.g. xMP)
            out[k] = v
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    import sys
    bp = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/cloud/OpenEdge/examples/test_solps_coupling/coupled_test/" \
        "solps/v0_1/b2pl.exe.dir"
    for t in ('outer', 'inner'):
        try:
            d = read_target(bp, t)
            print(f"{t}: {len(d['x'])} radial cells, "
                  f"x range [{d['x'].min():+.4f}, {d['x'].max():+.4f}] m, "
                  f"max Li_flux_ion = {d['Li_flux_ion'].max():.3e} m^-2 s^-1, "
                  f"max Te = {d['Te'].max():.2f} eV")
        except FileNotFoundError as e:
            print(f"{t}: skipped ({e})")
