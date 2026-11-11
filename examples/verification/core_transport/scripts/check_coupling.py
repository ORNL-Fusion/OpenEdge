#!/usr/bin/env python3
"""Optional SPARC coupling checks; no notebooks are opened or executed.

python scripts/check_coupling.py --exe /path/to/spa --ranks 1 4
Requires the existing SPARC plasma/geometry/core-profile inputs and MPI.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
from make_exported_pulse import make_case

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--exe', required=True, type=Path)
ap.add_argument('--ranks', nargs='+', type=int, default=[1, 4])
ap.add_argument('--skip-smoke', action='store_true', help='run only lifecycle and first-push checks')
ap.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[3] / 'wip/sparc')
a = ap.parse_args()
exe, case = a.exe.resolve(), a.case.resolve()
root = Path(__file__).resolve().parents[1] / 'output/coupling'
root.mkdir(parents=True, exist_ok=True)
base = (case / 'in.core_smoke').read_text()


def run(name, ranks, deck):
    out = root / name
    out.mkdir(exist_ok=True)
    deck = deck.replace('variable            outdir index output_core_smoke',
                        f'variable            outdir index {out}')
    inp = out / 'in.check'
    inp.write_text(deck)
    command = [str(exe), '-in', str(inp), '-log', str(out / 'log')]
    if ranks > 1:
        command = ['mpirun', '-np', str(ranks)] + command
    with (out / 'stdout').open('w') as f:
        p = subprocess.run(command, cwd=case, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError((out / 'stdout').read_text()[-4000:])
    lines = (out / 'core_1d.dat').read_text().splitlines()
    headers = [dict(zip(s.split()[1::2], s.split()[2::2])) for s in lines if s.startswith('# step')]
    for h in headers:
        core, gin, gout = [float(h[k]) for k in ('Ncore', 'Gin_total', 'Gout_total')]
        assert abs(core - gin + gout) <= 3e-6 * max(gin, 1), (name, h)
    for row in lines:
        if row and row[0].isdigit():
            assert all(math.isfinite(float(x)) and float(x) >= 0 for x in row.split()[7:])
        if row.startswith('Z '):
            assert float(row.split()[3]) >= 0
    return out, headers


for ranks in ([] if a.skip_smoke else a.ranks):
    for mode in ('uniform', 'entry'):
        deck = base.replace('variable            pos index uniform', f'variable            pos index {mode}')
        _, h = run(f'{mode}{ranks}', ranks, deck)
        assert int(h[-1]['step']) == 3000
        print(f'PASS {mode}, {ranks} ranks: positivity, core ledger, globally created emission', flush=True)

# Install the fix off the absolute coupling cadence; flush across two runs
# with different timesteps, checking both physical time and partial duration.
pre = base.replace('fix                 fcore core/transport',
                   'timestep 2e-7\nrun 7\nfix                 fcore core/transport')
pre = pre.replace('run                 ${runsteps}', 'run 55\ntimestep 4e-7\nrun 17')
_, h = run('partial_windows', 1, pre)
assert [int(x['step']) for x in h] == [62, 79], h
assert math.isclose(float(h[0]['dt_window']), 15*2e-7, rel_tol=1e-12)
assert math.isclose(float(h[1]['dt_window']), 17*4e-7, rel_tol=1e-12)
assert math.isclose(float(h[1]['time']), 62*2e-7+17*4e-7, rel_tol=1e-6)
print('PASS off-cadence creation, partial windows, consecutive runs and changed dt', flush=True)

# Recreating the fix must initialize its output file again.
start = base.index('fix                 fcore core/transport')
end = base.index('create_particles', start)
fresh = base.replace('run                 ${runsteps}', 'run 21\nunfix fcore\n'+base[start:end]+'run 1')
_, h = run('fresh_instance', 1, fresh)
assert len(h) == 1 and int(h[0]['step']) == 22, h
print('PASS output initialization belongs to the fix instance', flush=True)

# The first move of a returned marker must respond to E. Surface-born
# PSURF markers would retain their old vz instead of receiving this kick.
push = base.replace('run                 ${runsteps}', '''dump dp particle all 1 ${outdir}/part.*.dump id type x y z vx vy vz p_transit
dump_modify dp format float %.17g
run 1
variable Ex particle "0.0"
variable Ey particle "0.0"
variable Ez particle "1000.0"
fix ef efield/particle Ex Ey Ez
global efield particle ef 0
global pusher mode boris subcycles 1
run 1''')
out, _ = run('first_push', 1, push)


def particles(step):
    txt = (out / f'part.{step}.dump').read_text().split('ITEM: ATOMS', 1)[1].splitlines()[1:]
    return {int(r[0]): r for r in (list(map(float, line.split())) for line in txt) if r[-1] == 1}

before, after = particles(1), particles(2)
count = 0
for key in before.keys() & after.keys():
    p, q = before[key], after[key]
    charge = p[1] - 1
    expected = charge*1.602176634e-19/3.077624296e-25*1000*2e-7
    if charge > 0:
        assert math.isclose(q[7]-p[7], expected, rel_tol=1e-6, abs_tol=1e-6), (p, q, expected)
        count += 1
assert count > 20, count
print(f'PASS first Boris push: {count} returned ions receive qE dt/m', flush=True)

# Exercise the actual fix exporter, then its conversion to a unit pulse.
for nr, depth in ((50, 0.005), (200, 0.010)):
    deck = base.replace('grid 50', f'grid {nr}').replace(
        'nevery 20 nlaunch', f'export ${{outdir}}/inputs.json depth {depth} nevery 20 nlaunch')
    deck = deck.replace('run                 ${runsteps}', 'run 0')
    out, _ = run(f'export_{nr}', 1, deck)
    data = make_case(out / 'inputs.json', out / 'pulse.dat', duration=1e-5)
    assert data['ngrid'] == nr and data['depth'] == depth
    assert math.isclose(data['rvol_edge'], 0.24685981598169177, rel_tol=1e-6)
    assert math.isclose(sum(data['vol_torus']), 2*math.pi*data['r_axis']*sum(data['vol_per_length']), rel_tol=1e-14)
    with (out / 'pulse.dat').open() as f:
        header = f.readline().split()
        assert list(map(int, header[:3])) == [nr, data['nstates'], 1]
        assert float(header[4]) == data['dlen']
        for key in ('rr', 'diff', 'srate', 'rrate'):
            assert list(map(float, f.readline().split())) == data[key]
        density = list(map(float, f.readline().split()))
    assert math.isclose(sum(density[30*nr+i]*data['vol_per_length'][i] for i in range(nr)), 1, rel_tol=1e-14)
    assert all(x == 0 for x in density[:30*nr]+density[31*nr:])
    data['conv'][0] = 1
    (out / 'pinch.json').write_text(json.dumps(data))
    try:
        make_case(out / 'pinch.json', out / 'unsupported.dat')
    except ValueError:
        pass
    else:
        raise AssertionError('nonzero pinch silently discarded')
print('PASS fix export: exact grid/rates, two depths, unit pulse and unsupported-pinch rejection', flush=True)
