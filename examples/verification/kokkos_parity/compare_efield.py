#!/usr/bin/env python3
"""CPU vs GPU parity of one Boris step under constant cylindrical E and B from
the fix background: same particle set, velocity change identical to 1e-9."""
import sys, numpy as np
def read_dump(path):
    L = open(path).read().splitlines()
    i = [k for k, l in enumerate(L) if l.startswith('ITEM: ATOMS')][0]
    cols = L[i].split()[2:]; d = np.loadtxt(L[i+1:]).reshape(-1, len(cols))
    return cols, d[np.argsort(d[:, 0])]
def dedup(d):
    ids, n = np.unique(d[:, 0], return_counts=True); return d[np.isin(d[:, 0], ids[n == 1])]
def merged(a, b):
    ca, da = read_dump(a); cb, db = read_dump(b); assert ca == cb
    da, db = dedup(da), dedup(db); common = np.intersect1d(da[:, 0], db[:, 0])
    return ca, da[np.isin(da[:, 0], common)], db[np.isin(db[:, 0], common)], len(da), len(db)
fail = 0
cols, c0, g0, nc0, ng0 = merged('out_cpu/epart.0.dump', 'out_gpu/epart.0.dump')
cols, c1, g1, nc1, ng1 = merged('out_cpu/epart.1.dump', 'out_gpu/epart.1.dump')
print(f'particles: step0 cpu {nc0} gpu {ng0}, step1 cpu {nc1} gpu {ng1}, matched at step1 {len(c1)}')
ok = nc1 == ng1 and len(c1) == nc1 and len(c1) > 0
print(f'  same particle set after one moving step: {"OK" if ok else "FAIL"}'); fail |= not ok
iv = [cols.index(c) for c in ('vx', 'vy', 'vz')]; ix = [cols.index(c) for c in ('x', 'y', 'z')]
common0 = np.isin(c0[:, 0], c1[:, 0]); c0 = c0[common0]; g0 = g0[np.isin(g0[:, 0], c1[:, 0])]
dv_c = c1[:, iv] - c0[:, iv]; dv_g = g1[:, iv] - g0[:, iv]
scale = max(np.linalg.norm(dv_c, axis=1).max(), 1e-300)
err = np.abs(dv_c - dv_g).max() / scale; kicked = int((np.linalg.norm(dv_c, axis=1) > 0).sum())
ok = kicked > 0 and err < 1e-9
print(f'  Boris dv under constant E+B: kicked {kicked}/{len(c1)}, max |dv| {scale:.3e} m/s, max rel diff cpu-gpu {err:.2e}  {"OK" if ok else "FAIL"}'); fail |= not ok
errx = np.abs(c1[:, ix] - g1[:, ix]).max() / max(np.abs(c1[:, ix]).max(), 1e-300)
ok = errx < 1e-12
print(f'  positions after one step: max rel diff {errx:.2e}  {"OK" if ok else "FAIL"}'); fail |= not ok
print('EFIELD PARITY', 'FAIL' if fail else 'PASS'); sys.exit(1 if fail else 0)
