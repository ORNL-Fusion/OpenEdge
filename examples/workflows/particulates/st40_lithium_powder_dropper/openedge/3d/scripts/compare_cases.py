#!/usr/bin/env python3
"""Compare the runs under a directory (each <case>/{case.json,stdout,output/}): ablated
fraction, where the Li is born, grain radial spread, landing arc. Prints a markdown table.
    python scripts/compare_cases.py runs/sens [from_step] [to_step]"""
import sys, json, glob, re
from pathlib import Path
import numpy as np, h5py, pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
base = Path(sys.argv[1]); FROM = int(sys.argv[2]) if len(sys.argv) > 2 else 100000; TO = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
deck = open(ROOT / "in.openedge").read()
NPHI = int(re.search(r"(\d+) strips", open(ROOT / "input/st40_wall_3d.surf").readline()).group(1))
drop_ids = []
for line in open(ROOT / "input/st40_groups.inc"):
    t = line.split()
    if t[:1] == ["group"]: drop_ids += list(range(int(t[5]), int(t[6]) + 1))

def plasma_sampler(path):
    f = h5py.File(path, "r"); g = f["mesh"]
    tri = g["triangles"][...]; vr = g["vtx_r"][...]; vz = g["vtx_z"][...]; cidx = g["cell_index"][...]
    ok = (cidx >= 0) & (cidx < len(g["temp_e"]))
    tree = cKDTree(np.column_stack([vr[tri].mean(1)[ok], vz[tri].mean(1)[ok]])); cid = cidx[ok]
    te, ne = g["temp_e"][...], g["dens_e"][...]
    def at(R, Z):
        d, i = tree.query(np.column_stack([R, Z])); t_, n_ = te[cid[i]].copy(), ne[cid[i]].copy()
        t_[d > 0.03] = np.nan; n_[d > 0.03] = np.nan; return t_, n_
    return at

rows = []
for cdir in sorted(p for p in base.iterdir() if (p / "case.json").exists()):
    cj = json.load(open(cdir / "case.json")); name = cj["case"]
    extra = cj.get("extra", ""); m = re.search(r"plasma (\S+)", extra)
    plasma = ROOT / m.group(1) if m else ROOT.parent / "input/plasma_13589.h5"
    feed = float(cj.get("feed", 3e21)); atoms = float(cj.get("atoms_per_grain", 2.43e16)); nw = float(cj.get("nweight", 1))
    # log: ionization in window
    cols = None; S = []
    for line in open(cdir / "stdout"):
        t = line.split()
        if t[:2] == ["Step", "CPU"]: cols = t; continue
        if cols and len(t) == len(cols) and t[0].isdigit(): S.append([float(v) for v in t])
        elif cols and t[:1] == ["Loop"]: break
    S = np.array(S); sp = S[:, cols.index("c_cSp_tot")]; st = S[:, 0]; w = (st > FROM) & (st <= TO)
    ion = sp[w].mean() if w.any() else np.nan          # mean of the instantaneous stats samples in the window
    n_ion = int(w.sum())
    # landing
    li_wall = gr_wall = phi_mean = np.nan; p5 = p95 = np.nan; n_win = 0; tor = None
    for f in sorted(glob.glob(str(cdir / "output/landing.*.vtu"))):
        s_ = int(re.findall(r"\d+", f)[-1])
        if not (FROM < s_ <= TO): continue        # file <step> = END of its 20 ms window
        mm = pv.read(f); order = np.argsort(np.asarray(mm["id"])); ids = np.asarray(mm["id"])[order]
        P = np.asarray(mm.points); T = mm.cells.reshape(-1, 4)[:, 1:][order]
        v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]; area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        ng = np.asarray(mm["f_fland[1]"])[order]; ng[np.isin(ids, drop_ids)] = 0.0
        fl = np.asarray(mm["f_fland[4]"])[order]
        j = ((ids - 1) % (2 * NPHI)) // 2
        t_ = np.zeros(NPHI); np.add.at(t_, j, fl * area)
        tor = t_ if tor is None else tor + t_
        li_wall = (0 if np.isnan(li_wall) else li_wall) + (fl * area).sum(); gr_wall = (0 if np.isnan(gr_wall) else gr_wall) + ng.sum() / 2e-6 * nw; n_win += 1
    if n_win:
        li_wall /= n_win; gr_wall /= n_win; phi_c = (np.arange(NPHI) - NPHI // 2) * 360 / NPHI
        wgt = tor / tor.sum(); phi_mean = (wgt * phi_c).sum(); p5, p95 = np.interp([0.05, 0.95], np.cumsum(wgt), phi_c)
    # last state: grains + Li birth plasma
    sf = sorted(glob.glob(str(cdir / "output/state.*.vtu")), key=lambda p: int(re.findall(r"\d+", p)[-1]))
    R90 = tegr = negr = teli = sweep = np.nan
    if sf:
        mm = pv.read(sf[-1]); P = np.asarray(mm.points); typ = np.asarray(mm["type"])
        g = P[typ == 1]; li = P[typ == 2]
        if len(g): Rg = np.hypot(g[:, 0], g[:, 1]); R90 = np.percentile(Rg, 90); sweep = np.percentile(np.degrees(np.arctan2(g[:, 1], g[:, 0])), 1)
        if plasma.exists():
            at = plasma_sampler(plasma)
            if len(g): te, ne = at(Rg, g[:, 2]); tegr, negr = np.nanmedian(te), np.nanmedian(ne)
            if len(li): te, ne = at(np.hypot(li[:, 0], li[:, 1]), li[:, 2]); teli = np.nanmedian(te)
    rows.append((name, extra, ion, 100 * ion / feed, li_wall, gr_wall, 100 * gr_wall / (feed / atoms), phi_mean, p5, p95, R90, sweep, tegr, negr, teli, n_ion))

print(f"# Sensitivity set, comparison window ({FROM}, {TO}] steps (landing files are named by the END of their 20 ms window)\n")
print("Ionized = mean of the instantaneous stats samples in the window (every nstats steps; each sample sums ~2e4 events, ~1% noise);")
print("plasma columns = median Te/ne sampled at the grain positions, and Te at the Li-atom positions, in the last saved state (not birth conditions).")
print("nocoul = `coulomb/self no`, which also sets chi = 0: uncharged-grain collection cross-section, no charge-enhanced collection, no orbital scattering.\n")
print("| case | change | ionized [1e21/s] (n) | % of feed | Li atoms to wall [1e18/s] | grains to wall [% fed] | Li lands phi mean (5-95%) | grain R 90% [m] | grain phi 1% | Te / ne at grains | Te at Li atoms |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1] or 'reference'} | {r[2]/1e21:.2f} ({r[15]}) | {r[3]:.0f} | {r[4]/1e18:.1f} | {r[6]:.0f} | {r[7]:.0f} ({r[8]:.0f}..{r[9]:.0f}) | {r[10]:.2f} | {r[11]:.0f} | {r[12]:.1f} eV / {r[13]:.1e} | {r[14]:.1f} eV |")
