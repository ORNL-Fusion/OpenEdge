#!/usr/bin/env python3
"""GPU write_restart round trip: the particle dump written by the GPU just before
write_restart (out_gpu/wpart.5.dump) must be reproduced bit for bit by the
step-0 dumps of the CPU and GPU readers of that restart file (out_*/part.0.dump):
same id set, same species, positions, velocities and custom pweight."""
import sys, numpy as np
def read_dump(path):
    L = open(path).read().splitlines()
    i = [k for k, l in enumerate(L) if l.startswith('ITEM: ATOMS')][0]
    cols = L[i].split()[2:]; d = np.loadtxt(L[i+1:]).reshape(-1, len(cols))
    return cols, d[np.argsort(d[:, 0])]
def dedup(d):
    ids, n = np.unique(d[:, 0], return_counts=True); return d[np.isin(d[:, 0], ids[n == 1])]
fail = 0
wc, w = read_dump('out_gpu/wpart.5.dump'); w = dedup(w)
print(f'writer (GPU, step 5 before write_restart): {len(w)} particles, columns {wc}')
if len(w) == 0: print('  writer dump empty: FAIL'); fail = 1
for tag in ('cpu', 'gpu'):
    rc, r = read_dump(f'out_{tag}/part.0.dump'); r = dedup(r)
    assert rc == wc, (rc, wc)
    same_ids = len(r) == len(w) and np.array_equal(r[:, 0], w[:, 0])
    print(f'  reader {tag}: {len(r)} particles, id set identical: {"OK" if same_ids else "FAIL"}'); fail |= not same_ids
    if not same_ids: continue
    for j, c in enumerate(wc[1:], 1):
        eq = np.array_equal(r[:, j], w[:, j]); nz = int((w[:, j] != 0).sum())
        print(f'    {c:10s} bitwise equal {"OK" if eq else "FAIL"}  (nonzero {nz})' + (' (all zero)' if nz == 0 else ''))
        fail |= not eq or (nz == 0 and c == 'p_pweight')
print('RESTART', 'FAIL' if fail else 'PASS'); sys.exit(1 if fail else 0)
