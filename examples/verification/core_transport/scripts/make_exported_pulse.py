#!/usr/bin/env python3
"""Convert fix core/transport's `export file.json` to core_scan/pulse_driver input.

Copies the fix's arrays verbatim; no equilibrium reconstruction, ADAS lookup,
profile interpolation, or deposit fitting occurs here. The existing scan driver
supports zero pinch, so nonzero exported convection is rejected explicitly.
"""
import argparse
import json
import math
from pathlib import Path


def make_case(export, output, charge=30, duration=0.005, window=1e-5):
    data = json.loads(Path(export).read_text())
    if data.get('format') not in ('openedge-core-transport-v1', 'openedge-core-transport-v2'):
        raise ValueError('unrecognized core transport export format')
    nr, nz = data['ngrid'], data['nstates']
    if data['format'].endswith('v2'):
        # the effective coefficients must be exactly the exported inputs times the factors
        for name in ('diff_input', 'conv_input', 'g1', 'g2'):
            if len(data[name]) != nr or not all(math.isfinite(x) for x in data[name]):
                raise ValueError(f'invalid exported {name}')
        if data['coefficients'] not in ('effective', 'physical') or data['solver'] not in ('substep', 'split'):
            raise ValueError('invalid coefficient convention or solver in export')
        if data['coefficients'] == 'effective' and any(g != 1.0 for g in data['g1'] + data['g2']):
            raise ValueError('effective coefficients must carry unit geometry factors')
        for z in range(nz):
            for i in range(nr):
                if data['diff'][z*nr+i] != data['g2'][i]*data['diff_input'][i] or \
                   data['conv'][z*nr+i] != data['g1'][i]*data['conv_input'][i]:
                    raise ValueError('exported effective coefficients differ from input x geometry factor')
    for name in ('rr', 'vol_per_length', 'depo', 'diff', 'conv', 'srate', 'rrate'):
        expected = nr*nz if name in ('diff', 'conv', 'srate', 'rrate') else nr
        if len(data[name]) != expected or not all(math.isfinite(x) for x in data[name]):
            raise ValueError(f'invalid exported {name}')
    if any(v != 0 for v in data['conv']):
        raise ValueError('the existing pulse_driver supports v=0 only')
    if not 0 <= charge < nz or duration <= 0 or window <= 0:
        raise ValueError('invalid charge, duration, or window')
    nwin = round(duration/window)
    if nwin < 1 or not math.isclose(nwin*window, duration, rel_tol=1e-12):
        raise ValueError('duration must be an integer number of windows')
    volumes, weights = data['vol_per_length'], data['depo']
    if any(v <= 0 for v in volumes) or any(w < 0 for w in weights):
        raise ValueError('invalid exported volumes or deposition weights')
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-12):
        raise ValueError('exported deposition weights must sum to one')
    density = [0.0]*(nr*nz)
    for i in range(nr):
        density[charge*nr+i] = weights[i]/volumes[i]
    with Path(output).open('w') as f:
        f.write(f'{nr} {nz} {nwin} {window:.17g} {data["dlen"]:.17g}\n')
        for array in (data['rr'], data['diff'], data['srate'], data['rrate'], density):
            f.write(' '.join(format(x, '.17g') for x in array)+'\n')
    return data


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('export', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--charge', type=int, default=30)
    ap.add_argument('--duration', type=float, default=0.005)
    ap.add_argument('--window', type=float, default=1e-5)
    args = ap.parse_args()
    data = make_case(args.export, args.output, args.charge, args.duration, args.window)
    print(f'W{args.charge}+ pulse: {data["ngrid"]} cells, psi_N {data["psi_n"]}, '
          f'r_vol {data["rvol_edge"]:.9g} m, depth {1e3*data["depth"]:g} mm')
