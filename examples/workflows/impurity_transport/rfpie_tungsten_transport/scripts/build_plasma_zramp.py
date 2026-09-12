#!/usr/bin/env python3
"""Build a z-ramped variant of the RFPIE He background.

The baseline input/plasma_he.h5 is uniform along z (LP radial profiles only). Curt's
runs used a density that is lower at the target (LP value) and rises to the RF-column
value aloft.  This variant multiplies the electron/ion density by

    g(z) = f0 + (1 - f0) * (1 - exp(-(z - z_target) / L))      for z >= z_target
    g(z) = f0                                                   below the target face

so ne = f0 * ne_LP at the target face and -> ne_LP / f0_eff aloft.  With the
default f0 = 0.5, L = 3 mm the column value is 2x the target value (a Bohm-presheath
-like drop); Te is left unchanged.  Parameters:
    --f0    density factor at the target face relative to the value aloft (default 0.5)
    --L     ramp e-folding length in m (default 0.003)
    --scale-aloft  multiply the whole profile so that the ALOFT value equals the LP
                   value (default: the LP value is the value AT THE TARGET, aloft = LP/f0)
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, h5py
CASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--input', type=Path, default=CASE/'input/plasma_he.h5')
ap.add_argument('--output', type=Path, default=CASE/'input/plasma_he_zramp.h5')
ap.add_argument('--f0', type=float, default=0.5)
ap.add_argument('--L', type=float, default=0.003)
ap.add_argument('--scale-aloft', action='store_true', help='keep the LP value aloft (target = f0 * LP) instead of at the target')
a = ap.parse_args()
summary = json.loads((CASE/'input/geometry_summary.json').read_text())
zt = summary['target_z_m']
shutil.copy(a.input, a.output)
with h5py.File(a.output, 'a') as h5:
    z = h5['z'][...]
    g = a.f0 + (1.0 - a.f0) * (1.0 - np.exp(-np.maximum(z - zt, 0.0) / a.L))
    g = np.where(z < zt, a.f0, g)
    if not a.scale_aloft: g = g / a.f0            # LP value at the target, LP/f0 aloft
    for name in ('dens_e', 'dens_i'):
        h5[name][...] = h5[name][...] * g[:, None]
    h5['ions/dens'][...] = h5['ions/dens'][...] * g[None, :, None]
    h5.attrs['assumption_axial_profile'] = (f'z-ramp: ne(z) = ne_LP * g(z)/{"1" if a.scale_aloft else "f0"}, '
        f'g = f0 + (1-f0)(1-exp(-(z-zt)/L)), f0={a.f0}, L={a.L} m, zt={zt} m; Te unchanged (built {Path(__file__).name})')
    print('wrote', a.output, '| g(z) at target/aloft: %.3f / %.3f -> ne at r=0: %.3e (face) %.3e (z=30mm) %.3e (z=60mm)' % (
        g[np.argmin(abs(z-zt))], g[-1], h5['dens_e'][np.argmin(abs(z-zt)), 0], h5['dens_e'][np.argmin(abs(z-zt-0.03)), 0], h5['dens_e'][-1, 0]))
