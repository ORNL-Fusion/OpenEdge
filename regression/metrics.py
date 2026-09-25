#!/usr/bin/env python3
"""Regression metrics: extract end-of-run scalars from a run log, compare them
against a stored reference with per-metric tolerances, or (re)write the reference.

  metrics.py extract <log> <metrics.json>
  metrics.py compare <metrics.json> <reference.json>      (exit 1 on any miss)
  metrics.py update  <metrics.json> <reference.json>      (keeps existing tolerances)

Metrics: the last row of the last stats block (Step, Np and every stats column)
plus `loop_particles` (from "Loop time of ... with N particles") and
`host_fallback_calls` (from the Kokkos fallback report, informational only).
Default tolerances: 3 % on particle counts, 5 % on other columns, exact for
metrics whose reference value is 0 unless the reference gives an 'abs' bound. Edit the reference file to tighten or
loosen individual metrics; tolerances survive `update`.
"""
import sys, json, re, math
PARTICLE_KEYS = ('Np', 'loop_particles')
# per-step event counters (last step only) are Poisson noise, not regression metrics
INFO_KEYS = ('Step', 'CPU', 'host_fallback_calls', 'Nscoll', 'Nsreact', 'Nattempt', 'Ncoll', 'Nreact', 'Ntouch', 'Ncomm', 'Nbound', 'Nexit')

def extract(logpath):
    lines = open(logpath, errors='ignore').read().splitlines()
    header = None; last = None; loop_np = None; fb = 0
    for l in lines:
        t = l.split()
        if not t: continue
        if t[0] == 'Step' and len(t) > 1:
            header = t; last = None; continue
        if header and len(t) == len(header):
            try:
                vals = [float(x) for x in t]; last = vals
            except ValueError:
                pass
        m = re.match(r'Loop time of \S+ on \d+ procs for \d+ steps with (\d+) particles', l)
        if m: loop_np = int(m.group(1))
        m = re.match(r'\s+\S.*\bcalls (\d+)\s+ranks', l)
        if m: fb += int(m.group(1))
    out = {}
    if header and last:
        for k, v in zip(header, last): out[k] = v
    if loop_np is not None: out['loop_particles'] = loop_np
    out['host_fallback_calls'] = fb
    return out

def default_tol(key):
    if key in INFO_KEYS: return None
    return 0.03 if key in PARTICLE_KEYS else 0.05

def compare(metrics, ref):
    fails = []; lines = []
    for key, spec in ref.items():
        tol = spec.get('tol'); r = spec['value']
        if tol is None:
            continue
        if key not in metrics:
            fails.append(key); lines.append(f'  {key:22s} missing in run'); continue
        v = metrics[key]
        if r == 0.0:
            # a zero reference (e.g. a charge state not yet populated) may
            # legitimately become a few markers: allow |run| <= 'abs' if given
            err = abs(v); ok = err <= float(spec.get('abs', 0.0))
        else:
            err = abs(v - r) / abs(r); ok = err <= tol
        lines.append(f'  {key:22s} run {v:.6g}  ref {r:.6g}  rel {err:.2e}  tol {tol:.2g}  {"ok" if ok else "MISS"}')
        if not ok: fails.append(key)
    return fails, lines

def main():
    cmd = sys.argv[1]
    if cmd == 'extract':
        json.dump(extract(sys.argv[2]), open(sys.argv[3], 'w'), indent=1); return 0
    metrics = json.load(open(sys.argv[2]))
    if cmd == 'update':
        try: ref = json.load(open(sys.argv[3]))
        except (OSError, ValueError): ref = {}
        new = {}
        for k, v in metrics.items():
            old = ref.get(k, {})
            new[k] = dict(old)
            new[k]['value'] = v
            new[k]['tol'] = old.get('tol', default_tol(k))
        json.dump(new, open(sys.argv[3], 'w'), indent=1); print(f'reference written: {len(new)} metrics'); return 0
    if cmd == 'compare':
        ref = json.load(open(sys.argv[3]))
        fails, lines = compare(metrics, ref)
        print('\n'.join(lines)); print('METRICS', 'FAIL' if fails else 'PASS', ','.join(fails))
        return 1 if fails else 0
    print(__doc__); return 2
if __name__ == '__main__': sys.exit(main())
