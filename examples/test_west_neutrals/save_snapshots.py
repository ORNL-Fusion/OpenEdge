#!/usr/bin/env python3
"""Parse output/grid.west and save ALL timesteps as a single .npz archive
so plots can be remade / tweaked without re-running the simulation.

Archive fields (npz):
  steps         (ntime,)            simulation step at each snapshot
  times_ms      (ntime,)            simulation time in ms (dt=1e-7)
  x, y          (ncells,)           cell-center coordinates, in meters
  nD2, nD, nDp  (ntime, ncells)     species densities [m^-3]
  src_ion, src_rec, src_cx, src_dis (ntime, ncells)  cumulative event counts
  xg, yg        (nx+1,), (ny+1,)    histogram bin edges used in plots
"""

import os, numpy as np

DUMP = 'output/grid.west'
OUT  = 'output/snapshots.npz'
DT   = 1e-7

def parse_all(path):
    snaps = []
    current = None
    state = None
    with open(path) as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                current = {'step': None, 'rows': []}
                snaps.append(current)
                state = 'step'; continue
            if line.startswith('ITEM: NUMBER OF CELLS'):
                state = 'ncells'; continue
            if line.startswith('ITEM: BOX BOUNDS'):
                state = 'bounds'; continue
            if line.startswith('ITEM: CELLS'):
                state = 'cells'; continue
            parts = line.split()
            if not parts: continue
            if state == 'step':
                current['step'] = int(parts[0]); state = None
            elif state == 'ncells':
                state = None
            elif state == 'cells' and len(parts) >= 9:
                try:
                    current['rows'].append([float(x) for x in parts])
                except ValueError:
                    pass
    return snaps

print(f'Parsing {DUMP} ...')
snaps = parse_all(DUMP)
print(f'  got {len(snaps)} snapshots')

# Build arrays keyed on cell id (col 0) so all snapshots are aligned.
steps = np.array([s['step'] for s in snaps])
times_ms = steps * DT * 1e3

ref = np.array(snaps[-1]['rows'])
ids_ref = ref[:, 0].astype(int)
order = np.argsort(ids_ref)
ids_sorted = ids_ref[order]
ncells = len(ids_sorted)

x = ref[order, 1]
y = ref[order, 2]

def slot(col):
    out = np.full((len(snaps), ncells), np.nan)
    for i, s in enumerate(snaps):
        arr = np.array(s['rows'])
        if arr.size == 0: continue
        ids = arr[:, 0].astype(int)
        vals = arr[:, col]
        # map by id
        id_to_idx = {cid: k for k, cid in enumerate(ids_sorted)}
        for cid, v in zip(ids, vals):
            j = id_to_idx.get(cid)
            if j is not None:
                out[i, j] = v
    return out

nD2     = slot(3)
nD      = slot(4)
nDp     = slot(5)
src_ion = slot(6)
src_rec = slot(7)
src_cx  = slot(8)
src_dis = slot(9)

# Bin edges for default heatmap plots
nx, ny = 120, 160
xg = np.linspace(x.min(), x.max(), nx + 1)
yg = np.linspace(y.min(), y.max(), ny + 1)

np.savez_compressed(
    OUT,
    steps=steps, times_ms=times_ms,
    x=x, y=y,
    nD2=nD2, nD=nD, nDp=nDp,
    src_ion=src_ion, src_rec=src_rec, src_cx=src_cx, src_dis=src_dis,
    xg=xg, yg=yg,
)
print(f'Saved {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')

# Quick human-readable summary
summary = 'output/snapshots_summary.txt'
with open(summary, 'w') as f:
    f.write(f'test_west_neutrals snapshots\n')
    f.write(f'============================\n\n')
    f.write(f'Steps:  {len(snaps)}\n')
    f.write(f'Cells:  {ncells}\n')
    f.write(f'Time:   t = {times_ms[0]:.2f} .. {times_ms[-1]:.2f} ms '
            f'(dt = {DT:.1e} s)\n\n')
    f.write('Cumulative source events (last snapshot):\n')
    f.write(f'  Ionization:   {np.nansum(src_ion[-1]):.0f}\n')
    f.write(f'  Recombination:{np.nansum(src_rec[-1]):.0f}\n')
    f.write(f'  Charge Ex.:   {np.nansum(src_cx[-1]):.0f}\n')
    f.write(f'  Dissociation: {np.nansum(src_dis[-1]):.0f}\n\n')
    f.write('Species totals (last snapshot, density * volume):\n')
    f.write(f'  D2:  {np.nansum(nD2[-1]):.3e} m^-3 (sum over cells)\n')
    f.write(f'  D:   {np.nansum(nD[-1]):.3e} m^-3\n')
    f.write(f'  D+:  {np.nansum(nDp[-1]):.3e} m^-3\n')
print(f'Saved {summary}')
