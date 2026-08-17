#!/usr/bin/env python3
"""Inventory SPARTA/OpenEdge commands used in canonical decks and map
them to doc/*.txt pages. Prints coverage checklist for the manual."""
import glob, re
from pathlib import Path

root = Path(__file__).resolve().parents[3]
decks = [f for pat in ("examples/workflows/**/in.*",
                       "examples/verification/**/in.*")
         for f in glob.glob(str(root / pat), recursive=True)]
cmds, styles = set(), set()
for d in decks:
    joined, cont = [], ""
    for raw in open(d, errors="ignore"):
        raw = raw.split('#')[0].rstrip()
        if raw.endswith('&'):
            cont += raw[:-1] + ' '
            continue
        joined.append(cont + raw)
        cont = ""
    for line in joined:
        line = line.strip()
        if not line:
            continue
        tok = line.split()
        cmds.add(tok[0])
        if tok[0] in ("fix", "compute", "dump") and len(tok) > 2:
            styles.add(f"{tok[0]} {tok[2]}")
        if tok[0] in ("surf_collide", "surf_react") and len(tok) > 2:
            styles.add(f"{tok[0]} {tok[2]}")

have = {p.stem for p in (root / "doc").glob("*.txt")}
done = {p.stem for p in (root / "docs/manual").glob("*.md")}
print(f"{len(decks)} decks scanned\n")
print("== commands ==")
for c in sorted(cmds):
    page = c if c in have else "-"
    mark = "done" if c in done else ("txt" if c in have else "NO PAGE")
    print(f"  {c:18s} {mark}")
print("\n== styles (documented under docs/fixes/ or doc/) ==")
for s in sorted(styles):
    print(f"  {s}")
