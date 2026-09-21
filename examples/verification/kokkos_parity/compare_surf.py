#!/usr/bin/env python3
"""CPU vs GPU parity of compute surf/weighted: per-surf tallies at every dumped
step (same particles, deterministic Boris motion, absorbing target)."""
import sys, glob, numpy as np
def read_dump(path):
    L = open(path).read().splitlines()
    i = [k for k, l in enumerate(L) if l.startswith('ITEM: SURFS')][0]
    cols = L[i].split()[2:]; d = np.loadtxt(L[i+1:]).reshape(-1, len(cols))
    return cols, d[np.argsort(d[:, 0])]
fail = 0
steps = sorted(int(p.split('.')[-2]) for p in glob.glob('out_cpu/surf.*.dump'))
for st in steps:
    ca, a = read_dump(f'out_cpu/surf.{st}.dump'); cb, b = read_dump(f'out_gpu/surf.{st}.dump')
    assert ca == cb and a.shape == b.shape and np.array_equal(a[:, 0], b[:, 0]), 'surf id sets differ'
    tot = a[:, 1:].sum(0); err = np.abs(a[:, 1:] - b[:, 1:]).max(0) / np.maximum(np.abs(a[:, 1:]).max(0), 1e-300)
    hits = int((a[:, 1:].max(1) != 0).sum())
    ok = (err < 1e-9).all() and (hits > 0 or st == 0)
    print(f'step {st:3d}: surfs hit {hits:5d}  totals ' + ' '.join(f'{c}={t:.4g}' for c, t in zip(ca[1:], tot)) + f'  max rel diff {err.max():.2e}  {"OK" if ok else "FAIL"}')
    fail |= not ok
if not steps: print('no surf dumps'); fail = 1
print('SURF PARITY', 'FAIL' if fail else 'PASS'); sys.exit(1 if fail else 0)
