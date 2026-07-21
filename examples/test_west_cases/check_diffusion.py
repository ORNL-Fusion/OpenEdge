"""MSD acceptance check for fix cross_field_diffusion (in.axi_diffusion).

Usage: python3 check_diffusion.py DUMPFILE D_INPUT DT [SKIP_FRAC]

Reads a SPARTA particle dump (columns: id x y), computes the ensemble
mean-square radial displacement <(R - R0)^2>(t) relative to the first
frame, fits a line over the late portion of the run (default: skip the
first 25% while the gyro-offset transient saturates), and reports
D_measured = slope / 2 against D_INPUT. Pure axial B makes the poloidal
kick direction exactly +R, so <(R-R0)^2> = 2*D*t + O(r_g^2).
"""
import sys
import numpy as np

dump, D_in, dt = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
skip = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25

frames, ts, rows = {}, None, []
cols = None
with open(dump) as f:
    it = iter(f)
    for line in it:
        if line.startswith('ITEM: TIMESTEP'):
            if ts is not None and rows:
                frames[ts] = np.array(rows)
            ts = int(next(it)); rows = []
        elif line.startswith('ITEM: ATOMS'):
            cols = line.split()[2:]
        elif not line.startswith('ITEM:'):
            t = line.split()
            if cols and len(t) == len(cols):
                rows.append([float(x) for x in t])
if rows:
    frames[ts] = np.array(rows)

iid = cols.index('id')
iy = cols.index('y')          # slot y = R in the axi layout
steps = sorted(frames)
f0 = frames[steps[0]]
R0 = dict(zip(f0[:, iid].astype(np.int64), f0[:, iy]))

t_arr, msd = [], []
for s in steps:
    d = frames[s]
    ids = d[:, iid].astype(np.int64)
    mask = np.array([i in R0 for i in ids])
    dr = d[mask, iy] - np.array([R0[i] for i in ids[mask]])
    t_arr.append((s - steps[0]) * dt)
    msd.append((dr ** 2).mean())
t_arr, msd = np.array(t_arr), np.array(msd)

sel = t_arr >= skip * t_arr[-1]
slope, icpt = np.polyfit(t_arr[sel], msd[sel], 1)
D_meas = slope / 2.0
err = (D_meas / D_in - 1.0) * 100 if D_in > 0 else float('nan')
print(f'{dump}: frames {len(steps)}, N0 {len(f0)}')
print(f'  <dR^2> at end = {msd[-1]:.4e} m^2, gyro offset (icpt) = {icpt:.2e} m^2')
if D_in > 0:
    print(f'  D_input = {D_in:g}  D_measured = {D_meas:.4f} m^2/s  ({err:+.1f}%)')
else:
    print(f'  D_input = 0  D_measured = {D_meas:.3e} m^2/s (should be ~0; '
          f'final rms spread {np.sqrt(msd[-1])*1e3:.3f} mm ~ gyroradius)')
