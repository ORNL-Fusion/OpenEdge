#!/usr/bin/env python3
"""PASS/FAIL checks for surface/emit/source conc_scale on a depleting B layer."""
import glob, re, sys
import numpy as np

DT, NEV, BEAM, NBEAM = 1e-8, 10, 2e20, 50
key = lambda p: int(re.findall(r"\.(\d+)\.dump$", p)[0])


def table(path, tag):
    t = open(path).read().split(tag)[1].split("\n")
    rows = [list(map(float, l.split())) for l in t[1:] if l.strip()]
    return np.array(rows) if rows else np.zeros((0, len(t[0].split())))


def load(case):
    surf = {key(f): table(f, "ITEM: SURFS") for f in glob.glob(f"output/{case}.surf.*.dump")}
    surf = {k: v[np.argsort(v[:, 0])] for k, v in surf.items()}
    area = next(iter(surf.values()))[:, 1]
    wbeam = BEAM * area.sum() * DT * NEV / NBEAM
    emitted, prev = {}, set()
    for f in sorted(glob.glob(f"output/{case}.part.*.dump"), key=key):
        A = table(f, "ITEM: ATOMS")
        ids = A[:, 0].astype(int) if len(A) else np.array([], int)
        new = np.array([i not in prev for i in ids], bool)
        w = A[new, 2] if len(A) else np.array([])
        emitted[key(f)] = w[np.abs(w / wbeam - 1) > 1e-6].sum()
        prev = set(ids)
    steps = sorted(s for s in emitted if s >= NEV and s - NEV in surf)
    expected = np.array([(lambda S: (S[:, 4] * S[:, 3] * S[:, 1]).sum() * DT * NEV)(surf[s - NEV]) for s in steps])
    return surf, np.array(steps), np.array([emitted[s] for s in steps]), expected


results = []
def check(name, ok, detail):
    results.append(ok); print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}")


runs = {c: load(c) for c in ("static_no", "static_yes")}
for case, (surf, steps, em, ex) in runs.items():
    nb = len(steps) // 5
    rel = [abs(em[i:i + nb].sum() / ex[i:i + nb].sum() - 1) for i in range(0, nb * 5, nb)]
    check(f"{case}: emitted B follows the live concentration (5 windows)", max(rel) < 0.05,
          "window |emitted/expected - 1| = " + ", ".join(f"{r:.3f}" for r in rel))

surf = runs["static_no"][0]
s0, s1 = surf[min(surf)], surf[max(surf)]
ids, F = s0[:, 0].astype(int), s0[:, 4]
coated = (ids >= 1800) & (ids <= 1900)
hot = coated & (F > 1e20)
drop = s1[hot, 3].max() / s0[hot, 3].min()
check("coated faces with F > 1e20 deplete", drop < 0.05,
      f"{hot.sum()} faces, conc {s0[hot,3].min():.3f} -> max {s1[hot,3].max():.2e}")
react = (~coated) & (F > 1e19) & (s0[:, 3] == 0) & (s1[:, 3] > 1e-3)
src_react = (s1[react, 4] * s1[react, 3] * s1[react, 1]).sum() / (s1[:, 4] * s1[:, 3] * s1[:, 1]).sum()
check("bare faces switch on (conc 0 -> > 1e-3) and carry source", react.sum() > 0 and src_react > 0.01,
      f"{react.sum()} faces, {100*src_react:.1f}% of the final source")
# later redeposition makes multi-rank runs diverge run to run; compare the
# depletion-dominated start.  2%: weak faces fire probabilistically and with
# Kokkos the two runs draw from host vs device streams (pre-fix: 63% off)
early = {c: r[2][r[1] <= 400].sum() for c, r in runs.items()}
ratio = early["static_yes"] / early["static_no"]
check("static yes and static no emit the same B over steps <= 400", abs(ratio - 1) < 0.02,
      f"yes/no = {ratio:.4f}")
_, st2, em2, ex2 = load("two_writers")
dev = np.abs(em2 / ex2 - 1)
check("two writers: every event follows the live concentration", dev.max() < 0.01,
      f"max |emitted/expected - 1| = {dev.max():.4f} over {len(dev)} events")
ok = all(results)
print("PASS | surface/emit/source conc_scale" if ok else "FAIL | surface/emit/source conc_scale")
sys.exit(0 if ok else 1)
