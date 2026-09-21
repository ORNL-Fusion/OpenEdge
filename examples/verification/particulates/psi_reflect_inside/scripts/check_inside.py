#!/usr/bin/env python3
"""PASS/FAIL gate for the psi_reflect_inside regression.

normal run : must reach the final step and report a nonzero inside-start
             tally (scalar f_fcore) with zero absorptions (f_fcore[1]).
strict run : OE_PSI_STRICT=1 must abort with the crossing error.
"""
import re
import sys
from pathlib import Path

NSTEPS = 1000
failed = []


def read(path):
    p = Path(path)
    return p.read_text() if p.exists() else ""


log = read("output/log.normal")
exit_normal = read("output/exit.normal").strip()
rows = re.findall(r"^\s*(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$", log, re.M)
if not rows:
    failed.append("normal: no stats rows found")
else:
    if int(rows[0][1]) != 10:
        failed.append(f"normal: expected 10 particles at step 0, got {rows[0][1]}")
    step, np_, absorbed, inside = rows[-1]
    print(f"normal: last step {step}, np {np_}, absorbed {absorbed}, inside-start {inside}")
    if int(step) != NSTEPS:
        failed.append(f"normal: ended at step {step}, expected {NSTEPS}")
    if float(inside) <= 0.0:
        failed.append("normal: inside-start tally is zero")
    if float(absorbed) != 0.0:
        failed.append("normal: absorptions reported in reflect mode")
if "ERROR" in log + read("output/screen.normal") or "exit 0" not in exit_normal:
    failed.append(f"normal: solver error ({exit_normal})")

slog = read("output/log.strict") + read("output/screen.strict")
exit_strict = read("output/exit.strict").strip()
if "cannot resolve an outside-to-inside" in slog and "exit 0" not in exit_strict:
    print("strict: aborted as expected")
else:
    failed.append(f"strict: OE_PSI_STRICT=1 did not abort with the crossing error ({exit_strict})")

for f in failed:
    print("  " + f)
print("PASS" if not failed else "FAIL")
sys.exit(1 if failed else 0)
