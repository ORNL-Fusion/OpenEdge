#!/usr/bin/env python3
"""Parse output/grid.west and save ALL timesteps as a single .npz archive
so plots can be remade / tweaked without re-running the simulation.

Archive fields (npz):
  steps         (ntime,)            simulation step at each snapshot
  times_ms      (ntime,)            simulation time in ms (dt=1e-7)
  x, y          (ncells,)           cell-center coordinates, in meters
  nD2, nD, nDp  (ntime, ncells)     species densities [m^-3]

  Per-reaction cumulative source tallies (4 reactions each):
  src_n_*       (ntime, ncells)     count of events           [dimensionless]
  src_mx_*      (ntime, ncells)     sum of m*vx at events     [kg m/s]
  src_my_*      (ntime, ncells)     sum of m*vy
  src_mz_*      (ntime, ncells)     sum of m*vz
  src_E_*       (ntime, ncells)     sum of 0.5 m |v|^2        [J]
  (suffix in {ion, rec, cx, dis} for the four reaction types)

  xg, yg        (nx+1,), (ny+1,)    histogram bin edges used in plots

Dump columns (dump_grid f_fden[*] f_fchem[*]):
  0: id, 1: xc, 2: yc
  3: nD2,  4: nD,  5: nDp
  6..9:   count_{ion,rec,cx,dis}
  10..13: mx_{ion,rec,cx,dis}
  14..17: my_{ion,rec,cx,dis}
  18..21: mz_{ion,rec,cx,dis}
  22..25: E_{ion,rec,cx,dis}
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
            elif state == 'cells' and len(parts) >= 26:
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
# Counts
src_n_ion  = slot(6)
src_n_rec  = slot(7)
src_n_cx   = slot(8)
src_n_dis  = slot(9)
# Momentum (px, py, pz) per reaction
src_mx_ion = slot(10); src_mx_rec = slot(11); src_mx_cx = slot(12); src_mx_dis = slot(13)
src_my_ion = slot(14); src_my_rec = slot(15); src_my_cx = slot(16); src_my_dis = slot(17)
src_mz_ion = slot(18); src_mz_rec = slot(19); src_mz_cx = slot(20); src_mz_dis = slot(21)
# Energy per reaction
src_E_ion  = slot(22); src_E_rec  = slot(23); src_E_cx  = slot(24); src_E_dis  = slot(25)

# Bin edges for default heatmap plots
nx, ny = 120, 160
xg = np.linspace(x.min(), x.max(), nx + 1)
yg = np.linspace(y.min(), y.max(), ny + 1)

np.savez_compressed(
    OUT,
    steps=steps, times_ms=times_ms,
    x=x, y=y,
    nD2=nD2, nD=nD, nDp=nDp,
    # counts (backward-compatible names kept)
    src_ion=src_n_ion, src_rec=src_n_rec, src_cx=src_n_cx, src_dis=src_n_dis,
    src_n_ion=src_n_ion, src_n_rec=src_n_rec, src_n_cx=src_n_cx, src_n_dis=src_n_dis,
    # momentum components
    src_mx_ion=src_mx_ion, src_mx_rec=src_mx_rec, src_mx_cx=src_mx_cx, src_mx_dis=src_mx_dis,
    src_my_ion=src_my_ion, src_my_rec=src_my_rec, src_my_cx=src_my_cx, src_my_dis=src_my_dis,
    src_mz_ion=src_mz_ion, src_mz_rec=src_mz_rec, src_mz_cx=src_mz_cx, src_mz_dis=src_mz_dis,
    # energy
    src_E_ion=src_E_ion, src_E_rec=src_E_rec, src_E_cx=src_E_cx, src_E_dis=src_E_dis,
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
    f.write(f'  Ionization:   count={np.nansum(src_n_ion[-1]):.0f}  '
            f'Ek={np.nansum(src_E_ion[-1]):.3e} J\n')
    f.write(f'  Recombination:count={np.nansum(src_n_rec[-1]):.0f}\n')
    f.write(f'  Charge Ex.:   count={np.nansum(src_n_cx[-1]):.0f}  '
            f'Ek={np.nansum(src_E_cx[-1]):.3e} J\n')
    f.write(f'  Dissociation: count={np.nansum(src_n_dis[-1]):.0f}  '
            f'Ek={np.nansum(src_E_dis[-1]):.3e} J\n\n')
    f.write('Total momentum deposited (last snapshot, sum over cells) [kg m/s]:\n')
    for lbl, mx, my, mz in [('ion', src_mx_ion, src_my_ion, src_mz_ion),
                            ('cx',  src_mx_cx,  src_my_cx,  src_mz_cx),
                            ('dis', src_mx_dis, src_my_dis, src_mz_dis)]:
        f.write(f'  {lbl}: ({np.nansum(mx[-1]):+.3e}, '
                f'{np.nansum(my[-1]):+.3e}, {np.nansum(mz[-1]):+.3e})\n')
    f.write('\n')
    f.write('Species totals (last snapshot, density * volume):\n')
    f.write(f'  D2:  {np.nansum(nD2[-1]):.3e} m^-3 (sum over cells)\n')
    f.write(f'  D:   {np.nansum(nD[-1]):.3e} m^-3\n')
    f.write(f'  D+:  {np.nansum(nDp[-1]):.3e} m^-3\n')
print(f'Saved {summary}')
