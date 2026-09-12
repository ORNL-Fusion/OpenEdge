#!/usr/bin/env python3
"""CPU vs GPU parity: grid/weighted columns at step 0 (exact state) and the
force/thermal velocity kick after one step. Exit 1 on any failure."""
import sys, glob, numpy as np
fail = 0
def read_dump(path):
    lines = open(path).read().splitlines()
    i = [k for k, l in enumerate(lines) if l.startswith('ITEM: ATOMS') or l.startswith('ITEM: CELLS')][0]
    cols = lines[i].split()[2:]
    data = np.loadtxt(lines[i+1:]) if len(lines) > i+1 else np.zeros((0, len(cols)))
    data = data.reshape(-1, len(cols))
    order = np.argsort(data[:, 0]); return cols, data[order]
def dedup(d):
    # create_particles draws random ids: drop the few duplicated ids (ambiguous to match)
    ids, counts = np.unique(d[:, 0], return_counts=True)
    keep = np.isin(d[:, 0], ids[counts == 1]); return d[keep]
def merged(a, b):
    ca, da = read_dump(a); cb, db = read_dump(b)
    assert ca == cb, (ca, cb)
    da, db = dedup(da), dedup(db)
    common = np.intersect1d(da[:, 0], db[:, 0])
    if len(common) != len(da) or len(common) != len(db):
        print(f'  id sets differ: cpu {len(da)} gpu {len(db)} common {len(common)} (particles lost on one side)'); global fail; fail = 1
    return ca, da[np.isin(da[:, 0], common)], db[np.isin(db[:, 0], common)]
# 1. grid/weighted columns, step 0 (same particles, same positions) -> tight tolerance
cols, g_cpu, g_gpu = merged('out_cpu/grid.0.dump', 'out_gpu/grid.0.dump')
print('grid/weighted columns at step 0:')
for j, c in enumerate(cols[1:], 1):
    x, y = g_cpu[:, j], g_gpu[:, j]
    scale = max(np.abs(x).max(), 1e-300); err = np.abs(x - y).max() / scale
    nz = int((x != 0).sum()); nzg = int((y != 0).sum())
    ok = err < 1e-9 and (nz > 0) == (nzg > 0)      # a column empty on both sides (empty group) is fine
    print(f'  {c:14s} max rel diff {err:.2e}  nonzero cells cpu {nz:6d} gpu {nzg:6d}  {"OK" if ok else "FAIL"}{"" if nz else " (empty)"}')
    fail |= not ok
if int((g_cpu[:, 1:] != 0).sum()) == 0: print('  no nonzero grid column at all: FAIL'); fail = 1
# 2. thermal force after one step: same particles (move no), deterministic kick
cols, p_cpu, p_gpu = merged('out_cpu/part.1.dump', 'out_gpu/part.1.dump')
cols0, p0, _ = merged('out_cpu/part.0.dump', 'out_gpu/part.0.dump')
iv = [cols.index(c) for c in ('vx', 'vy', 'vz')]
dv_cpu = p_cpu[:, iv] - p0[:, iv]; dv_gpu = p_gpu[:, iv] - p0[:, iv]
kick = np.linalg.norm(dv_cpu, axis=1); kicked = int((kick > 0).sum())
err = np.abs(dv_cpu - dv_gpu).max() / max(kick.max(), 1e-300)
ok = kicked > 0 and err < 1e-6
print(f'force/thermal kick after 1 step: particles kicked (cpu) {kicked}/{len(kick)}, max |dv| {kick.max():.3e} m/s, '
      f'max rel diff cpu-gpu {err:.2e}  {"OK" if ok else "FAIL"}')
fail |= not ok
print('PARITY', 'FAIL' if fail else 'PASS'); sys.exit(1 if fail else 0)
