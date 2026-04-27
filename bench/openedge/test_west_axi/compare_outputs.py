#!/usr/bin/env python3
"""Diff CPU vs GPU runs of test_west_axi to validate the Phase D
sheath-spatial Kokkos port (and any future GPU port).

Reads the per-cell grid-density dumps:
    output/grid.dens.<tag>.west

Each dump has columns (per the in.west deck):
    id xc yc xlo ylo xhi yhi vol  f_fwgrid[1]  f_fwgrid[2]  f_fwgrid[3]
                                     total_W      W_neutral     W_plus

The dump is a multi-frame text file (one frame per Ndump steps). We read
the *last* frame from each and compare cell-by-cell.

Pass criterion (default):
    * max relative cell-wise diff < 5%   (set by --tol)
    * RMS relative diff           < 1%
    * Total mass agreement        < 1% relative

Adjust thresholds for stochastic noise — at small particle counts the
shot noise floor dominates. Increase nlaunch / Nphase for tighter checks.

Usage:
    python3 compare_outputs.py \\
        ../../../examples/test_west_axi/output/grid.dens.cpu_smoke_4.west \\
        ../../../examples/test_west_axi/output/grid.dens.gpu_smoke_4.west \\
        --col 9 --tol 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_last_frame(path: Path):
    """Return (header_cols, data_array) for the last ITEM frame in a SPARTA
    grid dump. Frames begin with `ITEM: TIMESTEP`; we keep the last one."""
    text = path.read_text()
    # Split on TIMESTEP markers; first chunk is preamble (empty).
    chunks = text.split("ITEM: TIMESTEP")
    if len(chunks) < 2:
        raise SystemExit(f"no ITEM frames in {path}")
    last = "ITEM: TIMESTEP" + chunks[-1]

    lines = last.splitlines()
    # Find the column header line: "ITEM: CELLS <names...>"
    header_cols = None
    data_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: CELLS"):
            header_cols = ln.split()[2:]   # skip "ITEM:" "CELLS"
            data_start = i + 1
            break
    if header_cols is None:
        raise SystemExit(f"no 'ITEM: CELLS' header in last frame of {path}")

    rows = []
    for ln in lines[data_start:]:
        ln = ln.strip()
        if not ln or ln.startswith("ITEM:"):
            break
        rows.append([float(x) for x in ln.split()])
    if not rows:
        raise SystemExit(f"no data rows in last frame of {path}")
    return header_cols, np.array(rows)


def align_by_id(arr_a, arr_b, id_col=0):
    """Sort both by cell id so cell N lines up — partitioning / load-balance
    can shuffle row order between CPU and GPU runs."""
    a = arr_a[arr_a[:, id_col].argsort()]
    b = arr_b[arr_b[:, id_col].argsort()]
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape} — "
                         "decks/grid must match exactly")
    if not np.allclose(a[:, id_col], b[:, id_col]):
        raise SystemExit("cell ids do not match after sorting")
    return a, b


def relative_diff(x, y, eps=None):
    """Per-cell |x - y| / max(|x|, |y|, eps). eps defaults to 1e-12 of
    the max of either field — avoids division blow-up where the signal
    is below the shot-noise floor."""
    if eps is None:
        eps = 1.0e-12 * max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0)
    denom = np.maximum.reduce([np.abs(x), np.abs(y), np.full_like(x, eps)])
    return np.abs(x - y) / denom


def report(label, x, y, vol, tol):
    rd = relative_diff(x, y)
    # Volume-weighted mass: integral of n dV
    mass_a = float(np.sum(x * vol))
    mass_b = float(np.sum(y * vol))
    mass_rel = abs(mass_a - mass_b) / max(abs(mass_a), abs(mass_b), 1.0e-30)

    rms = float(np.sqrt(np.mean(rd**2)))
    pmax = float(np.max(rd))
    p99 = float(np.percentile(rd, 99))

    print(f"\n--- {label} ---")
    print(f"  total mass CPU = {mass_a: .6e}")
    print(f"  total mass GPU = {mass_b: .6e}")
    print(f"  mass relative diff = {mass_rel*100: .3f} %")
    print(f"  cellwise rel diff:  RMS = {rms*100:.3f} %"
          f"   max = {pmax*100:.3f} %   p99 = {p99*100:.3f} %")
    cell_pass = (pmax < tol) and (rms < 0.2 * tol)
    mass_pass = (mass_rel < 0.01)
    overall = cell_pass and mass_pass
    flag = "OK" if overall else "FAIL"
    print(f"  -> {flag}  (tol_max={tol*100:.1f} %, "
          f"tol_rms={20*tol:.1f} %, tol_mass=1.0 %)")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu_dump", type=Path,
                    help="grid.dens.<tag>.west from CPU run")
    ap.add_argument("gpu_dump", type=Path,
                    help="grid.dens.<tag>.west from GPU run")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed relative diff per cell (default 5%%)")
    ap.add_argument("--cols", type=int, nargs="+", default=[8, 9, 10],
                    help="0-based column indices to compare "
                         "(default 8/9/10 = total_W / W_neutral / W_plus)")
    ap.add_argument("--vol-col", type=int, default=7,
                    help="0-based column index of cell volume (default 7)")
    args = ap.parse_args()

    print(f"CPU dump: {args.cpu_dump}")
    print(f"GPU dump: {args.gpu_dump}")

    hdr_a, raw_a = read_last_frame(args.cpu_dump)
    hdr_b, raw_b = read_last_frame(args.gpu_dump)
    if hdr_a != hdr_b:
        print(f"WARNING: header columns differ:\n  CPU: {hdr_a}\n  GPU: {hdr_b}",
              file=sys.stderr)

    a, b = align_by_id(raw_a, raw_b, id_col=0)

    print(f"\nframes aligned: {a.shape[0]} cells x {a.shape[1]} cols")
    print(f"columns: {hdr_a}")

    vol = a[:, args.vol_col]

    overall_pass = True
    for c in args.cols:
        name = hdr_a[c] if c < len(hdr_a) else f"col{c}"
        ok = report(name, a[:, c], b[:, c], vol, args.tol)
        overall_pass = overall_pass and ok

    print("\n=== SUMMARY ===")
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
