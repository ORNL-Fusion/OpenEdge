#!/usr/bin/env python3
"""Parse SPARTA stats and timing summary from bench logs.

Usage:
    python3 collect_timings.py logs/ -o results/timings.csv

Reads every `*.log` under the given directory; each log carries one
SPARTA run. Extracts:
    nranks            -- from filename pattern  *_n<RANKS>.log
                          or 'Running on N MPI task(s)'
    nsteps            -- from 'Loop time of ... for N steps'
    loop_time_s       -- wall time
    move_time_s, modify_time_s, comm_time_s, output_time_s,
    pcache_time_s, other_time_s    -- per-section averages
    np_final          -- particles in last stats line

Outputs one CSV row per log.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(
    r"^(Move|Coll|Sort|Comm|Modify|Output|Pcache|Other)\s*\|\s*"
    r"\S+\s*\|\s*([\d.eE+-]+)\s*\|"
)
LOOP_RE = re.compile(
    r"Loop time of\s+([\d.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps"
)
NRANKS_FROM_NAME = re.compile(r"_n(\d+)\.log$")


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="ignore")
    rec = {"file": path.name}

    m = NRANKS_FROM_NAME.search(path.name)
    if m:
        rec["nranks"] = int(m.group(1))

    m = LOOP_RE.search(text)
    if not m:
        return None  # incomplete run
    rec["loop_time_s"] = float(m.group(1))
    rec.setdefault("nranks", int(m.group(2)))
    rec["nsteps"] = int(m.group(3))

    for ln in text.splitlines():
        sm = SECTION_RE.match(ln.strip())
        if sm:
            section = sm.group(1).lower()
            avg = float(sm.group(2))
            rec[f"{section}_time_s"] = avg

    # Final particle count: last 'Particles: ... ave' line wins.
    for ln in reversed(text.splitlines()):
        if ln.lstrip().startswith("Particles:"):
            try:
                rec["np_final"] = float(ln.split()[1])
            except (ValueError, IndexError):
                pass
            break

    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    logs = sorted(args.logdir.glob("*.log"))
    if not logs:
        print(f"no *.log under {args.logdir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in logs:
        r = parse_log(p)
        if r is None:
            print(f"  skip (incomplete): {p.name}", file=sys.stderr)
            continue
        rows.append(r)
        print(f"  {p.name:>30}  ranks={r.get('nranks')!s:>4}  "
              f"loop={r['loop_time_s']:.2f} s  np={r.get('np_final', '-')}",
              file=sys.stderr)

    keys = sorted({k for r in rows for k in r})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
