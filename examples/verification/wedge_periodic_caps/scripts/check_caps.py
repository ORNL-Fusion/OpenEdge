#!/usr/bin/env python3
"""Gate: constant particle count, no boundary exits, no 'lost at periodic caps' line.  check_caps.py [log]"""
import re, sys
log = sys.argv[1] if len(sys.argv) > 1 else "regression.log"
txt = open(log, errors="ignore").read()
rows = [l.split() for l in txt.splitlines()]
hdr = [i for i, t in enumerate(rows) if t[:2] == ["Step", "CPU"]]
stats = [t for t in rows[hdr[-1] + 1:] if len(t) == 5 and re.fullmatch(r"\d+", t[0])]
np0, np1 = int(float(stats[0][2])), int(float(stats[-1][2]))
nscoll = int(float(stats[-1][3]))
m = re.search(r"Boundary exits\s*=\s*(\d+)", txt)
nexit = int(m.group(1)) if m else -1
m = re.search(r"Particles lost at periodic caps[^\d]*(\d+)", txt)
lost = int(m.group(1)) if m else 0
print(f"Np {np0} -> {np1}, surface collisions/step at the end {nscoll}, boundary exits {nexit}, lost at caps {lost}")
ok = np0 > 0 and np1 == np0 and nexit == 0 and lost == 0 and nscoll > 0
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
