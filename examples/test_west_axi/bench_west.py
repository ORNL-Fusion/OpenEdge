#!/usr/bin/env python3
"""Benchmark harness for in.west.

Runs the WEST axisymmetric case, parses the OpenEdge log for timing breakdown
and a small physics fingerprint, appends a row to bench_results.csv tagged
with the current git SHA, and prints a delta vs the previous matching run.

Usage:
    # quick iteration (Nwarm=2000, Nphase=8000)
    python3 bench_west.py --quick --label cell-pcache-v1

    # full case as configured in in.west
    python3 bench_west.py --full --label cell-pcache-v1

Requires Intel oneAPI sourced (mpirun on PATH must be Intel MPI).
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BIN = Path("/home/cloud/buildOpenEdge/src/spa_mpi")
CSV_PATH = HERE / "bench_results.csv"
LOG_DIR = HERE / "bench_logs"

SECTIONS = ["Move", "Coll", "Sort", "Comm", "Modify", "Output", "Pcache", "Other"]

CSV_FIELDS = [
    "ts", "git_sha", "dirty", "label", "mode", "np",
    "Nwarm", "Nphase", "loop_s", "steps", "final_np",
    *(f"{s.lower()}_{k}" for s in SECTIONS for k in ("avg", "max", "var", "pct")),
    "p_moves", "surf_checks", "surf_occurs", "surf_reactions",
    "chem_total", "chem_ioniz", "chem_recomb", "chem_cx", "chem_dissoc",
    "particles_ave", "particles_max", "particles_min",
]


def git_info():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, text=True
        ).strip())
        return sha, dirty
    except subprocess.CalledProcessError:
        return "unknown", False


ONEAPI_SETVARS = "/opt/intel/oneapi/setvars.sh"


def check_intel_mpi():
    if not Path(ONEAPI_SETVARS).exists():
        sys.exit(
            f"Intel oneAPI setvars.sh not found at {ONEAPI_SETVARS}; "
            "the OpenEdge binary needs Intel MPI."
        )


def make_quick_input(nwarm, nphase, dst):
    """Copy in.west to dst, replacing the Nwarm/Nphase variable lines so we can
    pin them via the harness without colliding with -var redefinition errors."""
    src = (HERE / "in.west").read_text().splitlines()
    out = []
    for line in src:
        if re.match(r"\s*variable\s+Nwarm\s+equal\s+", line):
            out.append(f"variable Nwarm equal {nwarm}    # bench_west override")
        elif re.match(r"\s*variable\s+Nphase\s+equal\s+", line):
            out.append(f"variable Nphase equal {nphase}    # bench_west override")
        else:
            out.append(line)
    dst.write_text("\n".join(out) + "\n")


def run_case(np, input_file, binary, log_path):
    """Source Intel oneAPI inside the subshell so PATH points at Intel MPI's
    mpirun. Linuxbrew's mpirun has the wrong ABI and silently launches N
    singleton processes — visible as repeated 'Step CPU Np' blocks in the log
    instead of one combined run."""
    inner = (
        f"source {ONEAPI_SETVARS} --force >/dev/null 2>&1 && "
        f"command -v mpirun | grep -q oneapi || {{ "
        f"echo 'ERROR: mpirun is not Intel MPI after sourcing oneAPI'; exit 2; }}; "
        f"exec mpirun -np {np} {binary} -in {input_file} -log {log_path}"
    )
    print(f"$ bash -c '<source oneAPI && exec mpirun -np {np} {binary.name} ...>'")
    t0 = time.time()
    rc = subprocess.call(["bash", "-c", inner], cwd=HERE)
    wall = time.time() - t0
    if rc != 0:
        print(f"ERROR: spa_mpi exited with code {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"  wall = {wall:.1f}s (subprocess), see {log_path.name} for log breakdown")


_LOOP_RE = re.compile(
    r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+particles"
)
_SECTION_RE = re.compile(
    r"^\s*(\w+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*$"
)
_OTHER_RE = re.compile(
    r"^\s*Other\s*\|\s*\|\s*([0-9.eE+-]+)\s*\|\s*\|\s*\|\s*([0-9.eE+-]+)\s*$"
)
_INT_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z ]+?)\s*=\s*(\d+)")
_PARTICLES_HIST = re.compile(
    r"^Particles:\s+([0-9.eE+-]+)\s+ave\s+(\d+)\s+max\s+(\d+)\s+min"
)


def parse_log(log_path):
    """Parse the LAST production run block from the log (skip warmup if any)."""
    text = log_path.read_text()
    loops = list(_LOOP_RE.finditer(text))
    if not loops:
        raise RuntimeError(f"No 'Loop time of' line found in {log_path}")
    # Use the last loop block — production run
    last = loops[-1]
    loop_s = float(last.group(1))
    np_used = int(last.group(2))
    steps = int(last.group(3))
    final_np = int(last.group(4))

    block = text[last.end():]
    # next "Loop time" would bound this, but we took the last one already

    out = {
        "loop_s": loop_s, "np": np_used, "steps": steps, "final_np": final_np,
    }

    # Section timings
    for section in SECTIONS:
        out[f"{section.lower()}_avg"] = None
        out[f"{section.lower()}_max"] = None
        out[f"{section.lower()}_var"] = None
        out[f"{section.lower()}_pct"] = None

    for line in block.splitlines():
        # Other has a different format (no min/max/var)
        m = _OTHER_RE.match(line)
        if m:
            out["other_avg"] = float(m.group(1))
            out["other_pct"] = float(m.group(2))
            continue
        m = _SECTION_RE.match(line)
        if m and m.group(1) in SECTIONS:
            sect = m.group(1).lower()
            out[f"{sect}_avg"] = float(m.group(3))
            out[f"{sect}_max"] = float(m.group(4))
            out[f"{sect}_var"] = float(m.group(5))
            out[f"{sect}_pct"] = float(m.group(6))

    # Counters from SPARTA stats output (always present, robust).
    # The gas-phase reaction line is "Reactions" in current OpenEdge builds
    # and "Gas reactions" in some older variants — accept both.
    counters = {
        "Particle moves": "p_moves",
        "SurfColl checks": "surf_checks",
        "SurfColl occurs": "surf_occurs",
        "Surf reactions": "surf_reactions",
        "Reactions": "chem_total",
        "Gas reactions": "chem_total",
    }
    for line in block.splitlines():
        m = _INT_LINE.match(line)
        if m and m.group(1).strip() in counters:
            out[counters[m.group(1).strip()]] = int(m.group(2))

    # Per-type chem/adas breakdown (only present if Chem/ADAS tally block printed)
    chem = {"ioniz": 0, "recomb": 0, "cx": 0, "dissoc": 0}
    in_chem = False
    for line in block.splitlines():
        if "Chem/ADAS reaction tallies" in line:
            in_chem = True
            continue
        if in_chem:
            m = re.match(r"\s*ioniz:\s*(\d+)", line)
            if m: chem["ioniz"] = int(m.group(1)); continue
            m = re.match(r"\s*recomb:\s*(\d+)", line)
            if m: chem["recomb"] = int(m.group(1)); continue
            m = re.match(r"\s*CX:\s*(\d+)", line)
            if m: chem["cx"] = int(m.group(1)); continue
            m = re.match(r"\s*dissoc:\s*(\d+)", line)
            if m: chem["dissoc"] = int(m.group(1)); in_chem = False
    out["chem_ioniz"] = chem["ioniz"]
    out["chem_recomb"] = chem["recomb"]
    out["chem_cx"] = chem["cx"]
    out["chem_dissoc"] = chem["dissoc"]
    out.setdefault("chem_total", 0)

    # Particles histogram
    for line in block.splitlines():
        m = _PARTICLES_HIST.match(line)
        if m:
            out["particles_ave"] = float(m.group(1))
            out["particles_max"] = int(m.group(2))
            out["particles_min"] = int(m.group(3))
            break

    return out


def append_row(row):
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_csv():
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def find_prior(rows, this_row):
    """Find the most recent prior row with same (np, mode, Nwarm, Nphase)."""
    key = (this_row["np"], this_row["mode"], this_row["Nwarm"], this_row["Nphase"])
    matches = [
        r for r in rows
        if (int(r["np"]), r["mode"], int(r["Nwarm"]), int(r["Nphase"])) == key
    ]
    return matches[-1] if matches else None


def fmt_delta(curr, prev, key, fmt="{:.3f}", invert=False):
    try:
        c = float(curr[key]); p = float(prev[key])
    except (TypeError, ValueError):
        return f"{curr.get(key, '?')}"
    if p == 0:
        return f"{fmt.format(c)} (was {fmt.format(p)})"
    pct = 100.0 * (c - p) / p
    arrow = ""
    if abs(pct) >= 1.0:
        # for timing: smaller is better; for physics: should be ~zero
        better = (pct < 0) ^ invert
        arrow = " (faster)" if better and not invert else (" (slower)" if not invert else f" {pct:+.2f}%")
        if invert:
            arrow = f" {pct:+.2f}%"
        else:
            arrow = f" {pct:+.1f}%"
    return f"{fmt.format(c)} vs {fmt.format(p)}{arrow}"


def print_summary(row, prev):
    print()
    print(f"=== Run summary: {row['label']} ({row['mode']}, np={row['np']}, "
          f"Nwarm={row['Nwarm']}, Nphase={row['Nphase']}) ===")
    if prev:
        print(f"    comparing vs prior: {prev['label']} @ {prev['ts']}")
    print()
    print(f"  loop_s         : {fmt_delta(row, prev, 'loop_s')}" if prev
          else f"  loop_s         : {row['loop_s']:.3f}")

    print("\n  Section timings (avg / max / var%):")
    print(f"  {'sect':<8} {'avg':>10} {'max':>10} {'var%':>8} {'pct%':>8}"
          + ("   delta_avg" if prev else ""))
    for s in SECTIONS:
        sl = s.lower()
        avg = row.get(f"{sl}_avg")
        mx  = row.get(f"{sl}_max")
        var = row.get(f"{sl}_var")
        pct = row.get(f"{sl}_pct")
        if avg is None: continue
        line = f"  {s:<8} {avg:>10.4f} "
        line += f"{mx:>10.4f} " if mx is not None else f"{'-':>10} "
        line += f"{var:>8.1f} " if var is not None else f"{'-':>8} "
        line += f"{pct:>8.2f}" if pct is not None else f"{'-':>8}"
        if prev and prev.get(f"{sl}_avg"):
            try:
                p = float(prev[f"{sl}_avg"])
                if p > 0:
                    d = 100.0 * (avg - p) / p
                    line += f"   {d:+.1f}%"
            except ValueError:
                pass
        print(line)

    print("\n  Physics fingerprint:")
    physics = ["final_np", "p_moves", "surf_checks", "surf_occurs",
               "surf_reactions", "chem_total"]
    for k in physics:
        v = row.get(k)
        if prev and prev.get(k):
            try:
                p = float(prev[k]); c = float(v)
                if p > 0:
                    d = 100.0 * (c - p) / p
                    flag = "  <-- DRIFT" if abs(d) > 1.0 else ""
                    print(f"    {k:<16}: {v} vs {prev[k]}  ({d:+.2f}%){flag}")
                    continue
            except (TypeError, ValueError):
                pass
        print(f"    {k:<16}: {v}")

    print(f"\n    particles_max/min: {row.get('particles_max')}/{row.get('particles_min')}"
          f"  (load imbalance signal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=64, help="MPI ranks (default 64)")
    ap.add_argument("--quick", action="store_true",
                    help="Override Nwarm/Nphase for fast iteration")
    ap.add_argument("--full", action="store_true",
                    help="Run as configured in in.west (do not override)")
    ap.add_argument("--quick-warm", type=int, default=2000)
    ap.add_argument("--quick-phase", type=int, default=8000)
    ap.add_argument("--label", default=None,
                    help="Tag for this run (default: git SHA)")
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--keep-log", action="store_true",
                    help="Keep the per-run log file (default: overwrite)")
    args = ap.parse_args()

    if args.quick == args.full:
        ap.error("Pick exactly one of --quick or --full")

    if not args.bin.exists():
        sys.exit(f"Binary not found: {args.bin}")
    check_intel_mpi()

    sha, dirty = git_info()
    label = args.label or sha
    if dirty:
        label = f"{label}+dirty"

    LOG_DIR.mkdir(exist_ok=True)
    mode = "quick" if args.quick else "full"
    if args.quick:
        nwarm, nphase = args.quick_warm, args.quick_phase
    else:
        # Pull from in.west — read the variable lines so we tag accurately.
        nwarm = nphase = 0
        for line in (HERE / "in.west").read_text().splitlines():
            m = re.match(r"\s*variable\s+Nwarm\s+equal\s+(\d+)", line)
            if m: nwarm = int(m.group(1))
            m = re.match(r"\s*variable\s+Nphase\s+equal\s+(\d+)", line)
            if m: nphase = int(m.group(1))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{ts}_{label.replace('/', '_')}_{mode}_np{args.np}.log"

    if args.quick:
        input_file = LOG_DIR / f"{ts}_in.west.quick"
        make_quick_input(nwarm, nphase, input_file)
    else:
        input_file = HERE / "in.west"

    run_case(args.np, input_file, args.bin, log_path)
    parsed = parse_log(log_path)

    row = {
        "ts": ts, "git_sha": sha, "dirty": int(dirty), "label": label,
        "mode": mode, "np": args.np, "Nwarm": nwarm, "Nphase": nphase,
        **parsed,
    }

    prior_rows = load_csv()
    prev = find_prior(prior_rows, row)

    append_row(row)
    print_summary(row, prev)

    if not args.keep_log:
        # Keep just the last 5 logs to avoid disk bloat; user can pass --keep-log
        logs = sorted(LOG_DIR.glob("*.log"))
        for old in logs[:-5]:
            old.unlink()

    if prev:
        # Hard physics gate: warn if drift > 1%
        for k in ("p_moves", "surf_occurs", "chem_total"):
            try:
                c = float(row[k]); p = float(prev[k])
                if p > 0 and abs(c - p) / p > 0.01:
                    print(f"\n!! PHYSICS DRIFT: {k} changed by "
                          f"{100*(c-p)/p:+.2f}% — investigate before claiming a win.")
            except (TypeError, ValueError, KeyError):
                pass


if __name__ == "__main__":
    main()
