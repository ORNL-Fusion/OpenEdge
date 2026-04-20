#!/usr/bin/env python3
"""Source/loss balance check for test_west_axi tungsten runs.

Reads:
  - output/pmi_diag.west_o.surf     per-surf sputter flux dump
  - output/tmp.grid.density_weighted.phaseA.west_o   per-cell W densities
  - input/wall.surf                 surface geometry (for area computation)
  - log.openedge                    stats log (N(t), c_csrc, ...)

Reports:
  - Total W gross erosion rate  [particles/s]    (sum flux x surface area)
  - Peak / mean sputter flux     [particles/m^2/s]
  - Total W in domain            [particles]     (sum n_W x V_cell)
  - Peak / mean W0 density       [m^-3]
  - Effective confinement time   tau = N / S_rate
  - Comparison vs. Ciraolo et al. ERO2.0 reference (Fig. 5, Table 1).

Caveat: in.west uses `dimension 2` (planar slab, z=0.1 m thick), NOT
axisymmetric. Flux x length gives [part/m/s]; we multiply by the slab
thickness to get [part/s]. Cell volumes in the dump are dR x dZ x 0.1.
This is an approximation of axisymmetric WEST -- true axisym would
multiply by 2*pi*R for each ring. Keep in mind when comparing absolute
magnitudes against a fully axisymmetric ERO2.0 run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent   # test_west_axi/
SURF_DUMP = HERE / 'output' / 'pmi_diag.west_o.surf'
DENS_DUMP = HERE / 'output' / 'tmp.grid.density_weighted.phaseA.west_o'
WALL_FILE = HERE / 'input' / 'wall.surf'
LOG_FILE  = HERE / 'log.openedge'

SLAB_Z = 0.10   # domain z-extent from create_box -0.05 .. 0.05 [m]


# ---------------------------------------------------------------------------
# wall.surf parsing: id -> (p1, p2) line segment
# ---------------------------------------------------------------------------

def parse_wall_segments(path: Path):
    lines = path.read_text().splitlines()
    pts = {}
    segs = {}       # line_id -> (p1_id, p2_id)
    mode = None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        low = s.lower()
        if low.endswith('points'):
            mode = 'pts'; continue
        if low.endswith('lines'):
            mode = 'lines'; continue
        parts = s.split()
        if mode == 'pts' and len(parts) >= 3:
            try:
                pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass
        elif mode == 'lines' and len(parts) >= 3:
            try:
                segs[int(parts[0])] = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass
    id2len = {}
    id2rmid = {}
    for sid, (p1, p2) in segs.items():
        if p1 in pts and p2 in pts:
            x1, y1 = pts[p1]; x2, y2 = pts[p2]
            id2len[sid]  = np.hypot(x2 - x1, y2 - y1)
            id2rmid[sid] = 0.5 * (x1 + x2)
    return id2len, id2rmid


# ---------------------------------------------------------------------------
# surf dump parsing: last snapshot, columns [id c_cpmiO[1..12]]
# ---------------------------------------------------------------------------

def parse_last_surf_snapshot(path: Path):
    text = path.read_text().splitlines()
    snaps = []
    cur = None
    state = None
    for ln in text:
        if ln.startswith('ITEM: TIMESTEP'):
            cur = {'t': None, 'rows': []}; snaps.append(cur); state = 't'; continue
        if ln.startswith('ITEM: NUMBER OF SURFS'):
            state = 'nsurfs'; continue
        if ln.startswith('ITEM: SURFS'):
            state = 'rows'; continue
        parts = ln.split()
        if not parts: continue
        if state == 't':
            cur['t'] = int(parts[0]); state = None
        elif state == 'nsurfs':
            state = None
        elif state == 'rows':
            try:
                cur['rows'].append([float(x) for x in parts])
            except ValueError:
                pass
    if not snaps:
        raise RuntimeError(f'no surf snapshots in {path}')
    last = snaps[-1]
    return last['t'], np.array(last['rows'])


# ---------------------------------------------------------------------------
# grid density dump parsing: last snapshot, columns [id xc yc f_fwdenA[1..21]]
# ---------------------------------------------------------------------------

def parse_last_grid_snapshot(path: Path):
    text = path.read_text().splitlines()
    snaps = []
    cur = None
    state = None
    for ln in text:
        if ln.startswith('ITEM: TIMESTEP'):
            cur = {'t': None, 'rows': []}; snaps.append(cur); state = 't'; continue
        if ln.startswith('ITEM: NUMBER OF CELLS'):
            state = 'ncells'; continue
        if ln.startswith('ITEM: CELLS'):
            state = 'rows'; continue
        parts = ln.split()
        if not parts: continue
        if state == 't':
            cur['t'] = int(parts[0]); state = None
        elif state == 'ncells':
            state = None
        elif state == 'rows':
            try:
                cur['rows'].append([float(x) for x in parts])
            except ValueError:
                pass
    if not snaps:
        raise RuntimeError(f'no grid snapshots in {path}')
    last = snaps[-1]
    return last['t'], np.array(last['rows'])


# ---------------------------------------------------------------------------
# log parsing: harvest stats columns
# ---------------------------------------------------------------------------

def parse_stats(path: Path):
    if not path.exists():
        return None
    text = path.read_text().splitlines()
    header = None
    rows = []
    for ln in text:
        s = ln.strip()
        if s.startswith('Step '):
            header = s.split()
            rows = []
            continue
        if header is None:
            continue
        if s.startswith('Loop time') or s == '' or s.startswith('---'):
            header = None
            continue
        parts = s.split()
        if len(parts) == len(header):
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                header = None
    if header is None or not rows:
        return None
    return header, np.array(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def fmt_sci(x, digits=3):
    if not np.isfinite(x) or x == 0:
        return f'{x:.{digits}g}'
    return f'{x:.{digits}e}'


def main():
    # Allow positional overrides for dump paths.
    args = sys.argv[1:]
    surf_dump = Path(args[0]) if len(args) > 0 else SURF_DUMP
    dens_dump = Path(args[1]) if len(args) > 1 else DENS_DUMP

    print(f'test_west_axi source/loss balance check')
    print(f'  surf dump : {surf_dump}')
    print(f'  dens dump : {dens_dump}')
    print(f'  wall      : {WALL_FILE}')
    print()

    # --- 1. source rate ----------------------------------------------------
    id2len, id2rmid = parse_wall_segments(WALL_FILE)
    t_surf, surf = parse_last_surf_snapshot(surf_dump)

    # dump layout: col 0 = id, then cpmiO[1..4]. `all` in the compute spec
    # expands at construction time (when FixPlasmaData.nion still reads 0),
    # so the per-species columns collapse and we get only the 4-col summary:
    #   [1] nflux   (slot 1: D+)   [part/m^2/s]
    #   [2] angle   (slot 1: D+)   [deg]
    #   [3] energy  (slot 1: D+)   [eV]
    #   [4] sputter_flux_total     [part/m^2/s]   -- still summed over all O+..O8+
    surf_ids = surf[:, 0].astype(int)
    bohm     = surf[:, 1]           # [part/m^2/s]  D+ Bohm
    angle    = surf[:, 2]           # [deg]         D+ incidence
    energy   = surf[:, 3]           # [eV]          D+ incidence
    flux_tot = surf[:, 4]           # [part/m^2/s]  total O-driven W sputter flux

    lengths = np.array([id2len.get(i, 0.0)  for i in surf_ids])
    rmids   = np.array([id2rmid.get(i, 0.0) for i in surf_ids])

    # Planar slab area [m^2] = length * slab_z
    area_planar = lengths * SLAB_Z
    src_planar  = (flux_tot * area_planar).sum()         # [part/s]

    # Axisymmetric ring area [m^2] = 2*pi*R_mid*length  (what ERO/SOLEDGE use)
    area_axi = 2.0 * np.pi * rmids * lengths
    src_axi  = (flux_tot * area_axi).sum()               # [part/s]

    active = (flux_tot > 0.0) & (lengths > 0.0)
    print(f'[1] W GROSS EROSION RATE  (dump at step {t_surf})')
    print(f'    active surfaces            : {active.sum()} / {len(surf_ids)}')
    print(f'    total wall length           : {lengths.sum():.3f} m')
    print(f'    total planar wall area      : {area_planar.sum():.3f} m^2  (slab {SLAB_Z} m)')
    print(f'    total axisym  wall area     : {area_axi.sum():.3f} m^2  (2*pi*R*dl)')
    print(f'    Sigma S_W (planar)          : {fmt_sci(src_planar)} particles/s')
    print(f'    Sigma S_W (axisym correction)         : {fmt_sci(src_axi)} particles/s')
    if active.sum():
        print(f'    peak sputter flux           : {fmt_sci(flux_tot[active].max())} part/m^2/s')
        print(f'    mean sputter flux (active)  : {fmt_sci(flux_tot[active].mean())} part/m^2/s')
        print(f'    mean Bohm inc flux (active) : {fmt_sci(bohm[active].mean())} part/m^2/s')
        print(f'    mean incident angle (deg)   : {angle[active].mean():.2f}')
        print(f'    mean incident energy (eV)   : {energy[active].mean():.2f}')
    print()

    # --- 2. W content in domain -------------------------------------------
    t_grid, grid = parse_last_grid_snapshot(dens_dump)
    # cols: id, xc, yc, f_fwdenA[1..21]
    xc = grid[:, 1]; yc = grid[:, 2]
    dens = grid[:, 3:]            # (ncell, 21)  (W0..W20+)
    n_W0  = dens[:, 0]
    n_tot = dens.sum(axis=1)

    # Cell-volume reconstruction from dump is not available directly here --
    # we only have (xc, yc). Approximate per-cell dR*dZ via the nearest
    # unique-x/y grid spacing and multiply by slab thickness.
    ux = np.unique(xc); uy = np.unique(yc)
    if ux.size > 1 and uy.size > 1:
        # assume uniform finest grid; for AMR this underestimates coarse cells
        dx = np.median(np.diff(np.sort(ux)))
        dy = np.median(np.diff(np.sort(uy)))
        vol_planar = dx * dy * SLAB_Z
        vol_axi    = 2.0 * np.pi * xc * dx * dy
    else:
        vol_planar = np.full_like(n_W0, np.nan)
        vol_axi    = np.full_like(n_W0, np.nan)

    N_tot_planar = (n_tot * vol_planar).sum()
    N_tot_axi    = (n_tot * vol_axi).sum()

    print(f'[2] W CONTENT IN DOMAIN  (dump at step {t_grid})')
    print(f'    cells                       : {len(grid)}')
    print(f'    n(W0)   peak                : {fmt_sci(n_W0.max())} m^-3')
    print(f'    n(W0)   mean (occupied)     : {fmt_sci(n_W0[n_W0>0].mean()) if (n_W0>0).any() else 0} m^-3')
    print(f'    n(W_all) peak               : {fmt_sci(n_tot.max())} m^-3')
    print(f'    n(W_all) mean (occupied)    : {fmt_sci(n_tot[n_tot>0].mean()) if (n_tot>0).any() else 0} m^-3')
    print(f'    sum(n_W * V) planar         : {fmt_sci(N_tot_planar)} particles')
    print(f'    sum(n_W * V) axisym         : {fmt_sci(N_tot_axi)} particles')
    print()

    # --- 3. derived quantities --------------------------------------------
    print(f'[3] DERIVED')
    if src_planar > 0:
        tau_planar = N_tot_planar / src_planar
        print(f'    tau_conf (planar)           : {fmt_sci(tau_planar)} s')
    if src_axi > 0:
        tau_axi = N_tot_axi / src_axi
        print(f'    tau_conf (axisym)           : {fmt_sci(tau_axi)} s')
    print()

    # --- 4. reference -------------------------------------------------------
    print('[4] REFERENCE (Ciraolo et al. IAEA TH/P4-12, WEST #54067)')
    print('    ERO2.0 log10(n_W) in SOL     : 10.5 .. 13 m^-3   (Fig. 5)')
    print('    W -> core integrated         : ~1e11 part / discharge inner tgt')
    print('                                   ~7e6  part / discharge outer tgt')
    print('                                   (attached plasma, Table 1)')
    print('    plasma: D main + 3% O,  D_perp = 0.3 m^2/s')
    print()

    log_data = parse_stats(LOG_FILE)
    if log_data is not None:
        hdr, arr = log_data
        print('[5] LIVE STATS (last row)')
        for i, name in enumerate(hdr):
            print(f'    {name:20s} {arr[-1, i]:.6g}')
    else:
        print('[5] log.openedge not parsed (run did not produce recognizable stats table)')

    print()


if __name__ == '__main__':
    main()
