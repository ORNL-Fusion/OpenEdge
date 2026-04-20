#!/usr/bin/env python3
"""Quantitative diagnosis of the W-density overshoot.

Uses the more informative dump files added to in.west:
  - output/tmp.grid.density_weighted.phaseA.west_o   id xc yc vol f_fwdenA[1..21]
  - output/state.west.final                          id type x y z vx vy vz p_pweight
  - output/pmi_diag.west_o.surf                      id c_cpmiO[1..4]

Prints, side-by-side:
  - total physical W from the particle dump    =  sum(pweight)
  - total physical W from the density dump     =  sum(n * vol)   (must match)
  - per-species breakdown
  - pweight distribution (min / median / max / 99th pct)
  - per-macro average residence time (N / S)
  - density histogram (how many cells with n > threshold?)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE      = Path(__file__).resolve().parent.parent
SURF_DUMP = HERE / 'output' / 'pmi_diag.west_o.surf'
DENS_DUMP = HERE / 'output' / 'tmp.grid.density_weighted.phaseA.west_o'
PART_DUMP = HERE / 'output' / 'state.west.final'
WALL_FILE = HERE / 'input' / 'wall.surf'


def parse_last_snapshot_generic(path: Path, item_count: str):
    """Parse the final snapshot of a SPARTA dump. item_count is e.g.
    'ITEM: NUMBER OF CELLS' or 'ITEM: NUMBER OF ATOMS'."""
    lines = path.read_text().splitlines()
    snaps = []
    cur = None
    state = None
    header = None
    for ln in lines:
        if ln.startswith('ITEM: TIMESTEP'):
            cur = {'t': None, 'hdr': None, 'rows': []}
            snaps.append(cur); state = 't'; continue
        if ln.startswith(item_count):
            state = 'n'; continue
        if ln.startswith('ITEM: BOX BOUNDS'):
            state = 'b'; continue
        if ln.startswith('ITEM: CELLS') or ln.startswith('ITEM: ATOMS') or ln.startswith('ITEM: SURFS'):
            cur['hdr'] = ln.split()[2:]
            state = 'r'; continue
        parts = ln.split()
        if not parts: continue
        if state == 't':
            cur['t'] = int(parts[0]); state = None
        elif state == 'n':
            state = None
        elif state == 'r':
            try: cur['rows'].append([float(x) for x in parts])
            except ValueError: pass
    if not snaps:
        raise RuntimeError(f'no snapshots in {path}')
    last = snaps[-1]
    return last['t'], last['hdr'], np.array(last['rows'])


def parse_wall_lengths(path: Path):
    lines = path.read_text().splitlines()
    pts = {}; segs = {}; mode = None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#'): continue
        low = s.lower()
        if low.endswith('points'): mode = 'pts'; continue
        if low.endswith('lines'):  mode = 'lines'; continue
        parts = s.split()
        if mode == 'pts' and len(parts) >= 3:
            try: pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError: pass
        elif mode == 'lines' and len(parts) >= 3:
            try: segs[int(parts[0])] = (int(parts[1]), int(parts[2]))
            except ValueError: pass
    id2len = {}; id2rmid = {}
    for sid, (p1, p2) in segs.items():
        if p1 in pts and p2 in pts:
            x1, y1 = pts[p1]; x2, y2 = pts[p2]
            id2len[sid]  = np.hypot(x2-x1, y2-y1)
            id2rmid[sid] = 0.5 * (x1 + x2)
    return id2len, id2rmid


def main():
    print('== test_west_axi density / weight diagnosis ==')

    # --- surface dump: source rate -----------------------------------------
    t_s, hdr_s, surf = parse_last_snapshot_generic(SURF_DUMP, 'ITEM: NUMBER OF SURFS')
    # col 0=id, 1=nflux(D+), 2=angle, 3=energy, 4=sputter_total
    id2len, id2rmid = parse_wall_lengths(WALL_FILE)
    surf_ids = surf[:, 0].astype(int)
    flux_tot = surf[:, 4]
    lengths  = np.array([id2len.get(i, 0.0)  for i in surf_ids])
    rmids    = np.array([id2rmid.get(i, 0.0) for i in surf_ids])
    area_planar = lengths * 0.10
    area_axi    = 2.0 * np.pi * rmids * lengths
    S_planar = (flux_tot * area_planar).sum()
    S_axi    = (flux_tot * area_axi).sum()
    print(f'\n[A] SOURCE  at step {t_s}')
    print(f'    S (planar slab) = {S_planar:.3e} W/s')
    print(f'    S (axisym)      = {S_axi:.3e} W/s   <-- physical WEST rate')

    # --- grid dump: n, vol -------------------------------------------------
    t_g, hdr_g, grid = parse_last_snapshot_generic(DENS_DUMP, 'ITEM: NUMBER OF CELLS')
    # hdr_g: id xc yc vol f_fwdenA[1]..f_fwdenA[21]
    # 0=id 1=xc 2=yc 3=vol 4..24 densities
    xc  = grid[:, 1]; yc = grid[:, 2]; vol = grid[:, 3]
    n   = grid[:, 4:25]                     # 21 species
    n_W0   = n[:, 0]
    n_all  = n.sum(axis=1)

    N_total = (n_all * vol).sum()
    print(f'\n[B] GRID   at step {t_g}')
    print(f'    N_particles in domain         = {N_total:.3e}')
    print(f'    cells                         = {len(grid)}')
    occ = n_all > 0
    print(f'    occupied cells                = {occ.sum()}')
    print(f'    peak n(W_all)                 = {n_all.max():.3e} m^-3')
    print(f'    peak n(W0)                    = {n_W0.max():.3e} m^-3')
    print(f'    mean n(W_all) | occ           = {n_all[occ].mean():.3e} m^-3')
    print(f'    median n(W_all) | occ         = {np.median(n_all[occ]):.3e} m^-3')
    # per-species breakdown
    print('    per-species physical count in domain:')
    labels = ['W'] + [f'W{z}+' for z in range(1, 21)]
    for j, lbl in enumerate(labels):
        Nsp = (n[:, j] * vol).sum()
        if Nsp > 0:
            print(f'      {lbl:>5}  : {Nsp:.3e}')

    # --- particle dump: pweight distribution -------------------------------
    try:
        t_p, hdr_p, part = parse_last_snapshot_generic(PART_DUMP, 'ITEM: NUMBER OF ATOMS')
        # hdr: id type x y z vx vy vz p_pweight
        pw_col = hdr_p.index('p_pweight')
        pw = part[:, pw_col]
        N_from_part = pw.sum()
        print(f'\n[C] PARTICLE DUMP   at step {t_p}')
        print(f'    macros in domain              = {len(part)}')
        print(f'    sum(pweight)                  = {N_from_part:.3e}  (should match B)')
        print(f'    pweight stats:')
        print(f'      min    = {pw.min():.3e}')
        print(f'      median = {np.median(pw):.3e}')
        print(f'      mean   = {pw.mean():.3e}')
        print(f'      p90    = {np.percentile(pw, 90):.3e}')
        print(f'      p99    = {np.percentile(pw, 99):.3e}')
        print(f'      max    = {pw.max():.3e}')
        # top 10 by weight
        top = np.argsort(pw)[-10:][::-1]
        print(f'    top-10 macros (pweight, species, R, Z):')
        for i in top:
            sp = int(part[i, 1])
            print(f'      w={pw[i]:.2e}  sp={sp}  R={part[i, 2]:.3f}  Z={part[i, 3]:.3f}')
        print(f'\n    ratio N(B)/N(C) = {N_total/max(N_from_part, 1e-30):.3f}'
              '  (1.0 means consistent)')
    except Exception as e:
        print(f'\n[C] PARTICLE DUMP: unable to read -- {e}')

    # --- derived confinement time ------------------------------------------
    print(f'\n[D] DERIVED')
    tau_planar = N_total / max(S_planar, 1e-30)
    tau_axi    = N_total / max(S_axi, 1e-30)
    print(f'    tau_conf (planar S) = {tau_planar:.3e} s')
    print(f'    tau_conf (axi   S)  = {tau_axi:.3e} s')
    print(f'    expected edge tau   ~ L_sol^2/D = (0.05)^2/0.1 = 2.5e-2 s')

    # --- axisymmetric correction -------------------------------------------
    # in.west uses `dimension 2` (planar slab) with box_z = 0.1 m. All cell
    # volumes carry that slab_z factor, and emit/surf/pmi uses tasks[i].area
    # = segment_length (no slab_z). To compare against a truly axisymmetric
    # WEST run, density should be multiplied by 2*pi*R_avg / slab_z.
    slab_z = 0.10
    R_avg = 0.5 * (xc.min() + xc.max())
    axi_factor = 2.0 * np.pi * R_avg / slab_z
    print(f'\n[E] AXISYMMETRIC CORRECTION  (planar slab -> ring)')
    print(f'    slab_z         = {slab_z} m')
    print(f'    <R>_domain     = {R_avg:.3f} m')
    print(f'    n_axi / n_plan = 2*pi*<R>/slab_z = {axi_factor:.1f}')
    print(f'    peak n(W_all) x axi_factor  = {n_all.max()*axi_factor:.3e} m^-3')
    print(f'    mean n(W_all) x axi_factor  = {n_all[occ].mean()*axi_factor:.3e} m^-3')

    # --- reference ---------------------------------------------------------
    print(f'\n[F] REFERENCE  (Ciraolo IAEA TH/P4-12, Fig. 5)')
    print(f'    log10(n_W) range in SOL  : 10.5 .. 13  (m^-3)')
    print(f'    peak n(W) near strike    : ~1e13 m^-3')
    ratio_peak = n_all.max() * axi_factor / 1e13
    ratio_mean = n_all[occ].mean() * axi_factor / 1e13
    print(f'    OpenEdge/Ciraolo   peak  : {ratio_peak:.2f}x   (after axi correction)')
    print(f'    OpenEdge/Ciraolo   mean  : {ratio_mean:.2f}x')


if __name__ == '__main__':
    main()
