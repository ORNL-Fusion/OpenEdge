#!/usr/bin/env python3
"""Build a minimal uniform plasma.h5 for the A3 CX slab test.

Shape: nz x nr = 2 x 2 constant everywhere (bilinear interpolation returns
the constant value at any interior point). Covers the SPARTA box
(x in [0, 0.6], y in [-0.01, 0.01]) with a small halo so the interpolation
is always in-domain.
"""
import os, h5py, numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'input', 'plasma_cx.h5')

Te  = 5.0    # eV  -- keeps ADAS CX / ionization competitive (cx/iz ~ 1.4 here)
Ti  = 50.0   # eV  -- high so thermalization signature is visible
ne  = 1.0e20 # m^-3 -- high for short lambda_cx in the box
ni  = ne

r = np.array([-0.1, 1.0])   # halo outside [0, 0.6]
z = np.array([-0.5, 0.5])   # halo outside [-0.01, 0.01]
nr, nz = len(r), len(z)
shape = (nz, nr)

with h5py.File(OUT, 'w') as f:
    f['r'] = r
    f['z'] = z
    f['dens_e'] = ne * np.ones(shape)
    f['dens_i'] = ni * np.ones(shape)
    f['temp_e'] = Te * np.ones(shape)
    f['temp_i'] = Ti * np.ones(shape)
    # parr_flow = 0 (no bulk drift) -- leave optional datasets absent

print(f'wrote {OUT}  Te={Te} eV  Ti={Ti} eV  ne={ne:.0e} m^-3')
