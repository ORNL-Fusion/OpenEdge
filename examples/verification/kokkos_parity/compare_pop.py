#!/usr/bin/env python3
"""CPU vs GPU parity of fix population/control: identical particle count after
thinning and conserved, identical per-cell weighted densities."""
import sys, re, numpy as np
def read_dump(path):
    L = open(path).read().splitlines()
    i = [k for k, l in enumerate(L) if l.startswith('ITEM: CELLS')][0]
    cols = L[i].split()[2:]; d = np.loadtxt(L[i+1:]).reshape(-1, len(cols))
    return cols, d[np.argsort(d[:, 0])]
def np_series(screen):
    rows = re.findall(r'^\s+(\d+)\s+(\d+)\s*$', open(screen).read(), re.M)
    return {int(s): int(n) for s, n in rows}
fail = 0
npc, npg = np_series('screen.cpu_pop'), np_series('screen.gpu_pop')
print('particles per step  cpu', npc, ' gpu', npg)
ok = npc.get(0) == npg.get(0) and npc.get(1) == npg.get(1) and npc.get(2) == npg.get(2) and npc.get(1, 0) < npc.get(0, 0)
print(f'  count after thinning identical and reduced: {"OK" if ok else "FAIL"}'); fail |= not ok
c0, g0 = read_dump('out_cpu/pgrid.0.dump'); _, a1 = read_dump('out_cpu/pgrid.1.dump'); _, b1 = read_dump('out_gpu/pgrid.1.dump'); _, a2 = read_dump('out_cpu/pgrid.2.dump'); _, b2 = read_dump('out_gpu/pgrid.2.dump')
def rel(x, y): return np.abs(x - y).max(0) / np.maximum(np.abs(x).max(0), 1e-300)
for name, x, y in (('cpu step1 vs step0 (conservation)', g0[:, 1:], a1[:, 1:]), ('cpu vs gpu step1', a1[:, 1:], b1[:, 1:]), ('cpu vs gpu step2', a2[:, 1:], b2[:, 1:])):
    e = rel(x, y); ok = (e < 1e-9).all()
    print(f'  {name:36s} max rel diff over columns {e.max():.2e}  {"OK" if ok else "FAIL"}'); fail |= not ok
print('POP PARITY', 'FAIL' if fail else 'PASS'); sys.exit(1 if fail else 0)
