#!/usr/bin/env python3
"""Peak-normalized W I 498 nm axial emission profile: OpenEdge runs vs experiment.

usage: wi498_compare.py [--out fig.png] label=rundir [label=rundir ...]

For each run the last `rfpie_w_density.*.dump` (pweight-aware charge-state densities,
fix ave/grid) gives n_W0 per cell.  Synthetic emission per cell:
    eps = n_W0 * ne(r) * PEC(Te(r), ne(r))         (photons m^-3 s^-1, 4pi)
with ne, Te from input/plasma_he.h5 (uniform in z) and the W I PEC from the
open-ADAS adf15 table (pec40#w_ic#w0.dat, isel 49, 5000.73 A theoretical ~ W I 498 nm).
Two axial profiles are formed and peak-normalized:
  chord   : sum over cells with |y| < dy_chord of eps*vol per z bin  (spectrometer chord
            across the plume, viewing along x)
  axis    : eps averaged over R < 3 mm per z bin
The experiment curve is input/experiment_wi498_axial_digitized.csv (digitized from Curt's plot).
"""
import sys, re, json, argparse
from pathlib import Path
import numpy as np, h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

CASE = Path(__file__).resolve().parents[1]
PEC_NPZ = CASE/'input/w_i_pec_adf15_pec40_w_ic_w0_isel49_5000p73A.npz'   # open-ADAS pec40#w_ic#w0 isel 49 (5000.73 A), te_ev x ne_cm3 grid

def read_dump(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[3]); columns = lines[8].split()[2:]
    vals = np.asarray([[float(v) for v in row.split()] for row in lines[9:9+count]])
    return {name: vals[:, i] for i, name in enumerate(columns)}

def step_of(p): 
    m = re.search(r'\.(\d+)\.dump$', str(p)); return int(m.group(1)) if m else -1

def load_plasma(plasma_h5):
    with h5py.File(plasma_h5, 'r') as h5:
        r = h5['r'][...]; z = h5['z'][...]; te = h5['temp_e'][...]; ne = h5['dens_e'][...]
    return r, z, te, ne          # te, ne shape (nz, nr)

def pec_interp():
    d = np.load(PEC_NPZ)
    f = RegularGridInterpolator((np.log10(d['te_ev']), np.log10(d['ne_cm3']*1e6)), np.log10(d['pec_cm3_s'].T)*1.0,
                                bounds_error=False, fill_value=None)
    return lambda te, ne: 10.0**f(np.c_[np.log10(te), np.log10(ne)]) * 1e-6     # m^3/s

def plasma_file_for(rundir):
    """plasma file actually used by the run: from run_settings/log if recorded, else the run's input copy."""
    rundir = Path(rundir)
    for log in list(rundir.parent.glob('output_log.*')) + list(rundir.parent.glob('screen.*')):
        for line in Path(log).read_text(errors='ignore').splitlines()[:400]:
            if line.lstrip().startswith('fix') and 'background' in line and 'file' in line and '.h5' in line:
                for tok in line.replace('"',' ').split():
                    if tok.endswith('.h5') and 'plasma_he' in tok:
                        cand = rundir.parent / tok
                        if cand.exists(): return cand
    for name in ('plasma_he_zramp.h5', 'plasma_he.h5'):
        if (rundir.parent/'input'/name).exists() and name != 'plasma_he_zramp.h5': return rundir.parent/'input'/name
    return CASE/'input/plasma_he.h5'

def profiles(rundir, plasma_h5, summary, dy_chord=0.0005, axis_r=0.003, nz_bins=None, last_n=1):
    files = sorted(Path(rundir).glob('rfpie_w_density.*.dump'), key=step_of)
    files = [f for f in files if step_of(f) > 0]
    assert files, f'no density dumps in {rundir}'
    use = files[-last_n:]
    G = read_dump(use[0]); nW0 = np.zeros_like(G['xc'])
    for f in use: nW0 += read_dump(f)['f_fWdens[1]'] / len(use)
    x, y, z, vol = G['xc'], G['yc'], G['zc'], G['vol']
    r = np.hypot(x, y)
    rr, zz, te, ne = load_plasma(plasma_h5)
    # bilinear (z, r) lookup so z-ramped backgrounds are weighted correctly
    fte = RegularGridInterpolator((zz, rr), te, bounds_error=False, fill_value=None)
    fne = RegularGridInterpolator((zz, rr), ne, bounds_error=False, fill_value=None)
    pts = np.c_[np.clip(z, zz.min(), zz.max()), np.clip(r, rr.min(), rr.max())]
    te_r = fte(pts); ne_r = fne(pts)
    pec = pec_interp()(te_r, ne_r)
    eps = nW0 * ne_r * pec
    target_z = summary['target_z_m']
    zc = np.unique(np.round(z, 9)); z_edges = np.concatenate([[zc[0]-0.5*(zc[1]-zc[0])], 0.5*(zc[1:]+zc[:-1]), [zc[-1]+0.5*(zc[-1]-zc[-2])]])   # one bin per grid layer
    zmid = 0.5*(z_edges[:-1]+z_edges[1:]); dist_mm = (zmid - target_z)*1e3
    chord = np.abs(y) < dy_chord
    p_chord = np.histogram(z[chord], bins=z_edges, weights=(eps*vol)[chord])[0]
    core = r <= axis_r
    p_axis = np.histogram(z[core], bins=z_edges, weights=(eps*vol)[core])[0] / np.maximum(np.histogram(z[core], bins=z_edges, weights=vol[core])[0], 1e-300)
    p_dens = np.histogram(z[core], bins=z_edges, weights=(nW0*vol)[core])[0] / np.maximum(np.histogram(z[core], bins=z_edges, weights=vol[core])[0], 1e-300)
    norm = lambda p: p/p.max() if p.max() > 0 else p
    return dict(step=step_of(use[-1]), dist_mm=dist_mm, chord=norm(p_chord), axis=norm(p_axis), dens=norm(p_dens), nW0_peak=nW0.max())

def save_profiles(results, path):
    """results: {label: profiles(...) dict} -> one CSV (dist_mm, then <label>:chord/axis/dens columns)."""
    labels = list(results); dist = results[labels[0]]['dist_mm']
    cols = ['dist_mm'] + [f'{l}:{k}' for l in labels for k in ('chord', 'axis', 'dens')]
    data = np.column_stack([dist] + [np.interp(dist, results[l]['dist_mm'], results[l][k]) for l in labels for k in ('chord', 'axis', 'dens')])
    hdr = ','.join(cols) + '\n' + ','.join(f"{l}:step={results[l]['step']}" for l in labels)
    np.savetxt(path, data, delimiter=',', header=hdr, comments='# ')

def load_profiles(path):
    """inverse of save_profiles -> {label: dict(step, dist_mm, chord, axis, dens)}"""
    lines = Path(path).read_text().splitlines()
    cols = lines[0].lstrip('# ').split(','); steps = dict(x.split(':step=') for x in lines[1].lstrip('# ').split(','))
    data = np.loadtxt(path, delimiter=',', comments='#'); out = {}
    for i, c in enumerate(cols[1:], 1):
        label, key = c.rsplit(':', 1)
        out.setdefault(label, dict(step=int(steps[label]), dist_mm=data[:, 0]))[key] = data[:, i]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(CASE/'output/wi498_compare.png'))
    ap.add_argument('--profile', default='chord', choices=['chord','axis','dens']); ap.add_argument('--last', type=int, default=1)
    ap.add_argument('runs', nargs='+'); a = ap.parse_args()
    summary = json.loads((CASE/'input/geometry_summary.json').read_text())
    exp = np.loadtxt(CASE/'input/experiment_wi498_axial_digitized.csv', delimiter=',')
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(exp[:,0], exp[:,1], 'k-', lw=1.2, label='experiment (digitized)')
    styles = ['-', '--', '-.', ':']
    for i, spec in enumerate(a.runs):
        label, rundir = spec.split('=', 1)
        pl = plasma_file_for(rundir)
        P = profiles(rundir, pl, summary, last_n=a.last)
        print(f"{label}: plasma file {pl}")
        ax.plot(P['dist_mm'], P[a.profile], styles[i % 4], lw=1.8, label=f"{label} (step {P['step']})")
        print(f"{label}: step {P['step']}  peak at {P['dist_mm'][P[a.profile].argmax()]:.2f} mm  value at 10/20/30 mm: "
              + ' '.join('%.3f' % np.interp(v, P['dist_mm'], P[a.profile]) for v in (10, 20, 30)))
    ax.set(xlim=(-1, 31), ylim=(0, 1.05), xlabel='Distance from target (mm)', ylabel='Peak-normalized W I 498 nm')
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.minorticks_on(); ax.tick_params(direction='in', top=True, right=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print('saved', a.out)
if __name__ == '__main__': main()
