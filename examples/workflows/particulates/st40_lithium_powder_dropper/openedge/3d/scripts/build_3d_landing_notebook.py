#!/usr/bin/env python3
"""Generate and execute notebooks/st40_3d_landing.ipynb.

Where the Li atoms and the grains land in the full-torus run of
3d/in.openedge: inventories from the log, the
per-triangle landing map (landing.*.vtu) unwrapped in toroidal angle and
poloidal wall coordinate, poloidal and toroidal deposition profiles, the
grain wall fraction, the toroidal sweep of the plume and the core-hole
check from the particle states.

    python scripts/build_3d_landing_notebook.py [--outdir output_prod] [--log log.prod]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]          # openedge/3d


def build(outdir: str, log: str, steady_from: int, steady_to: int, atoms_per_grain: float, feed: float) -> nbformat.NotebookNode:
    c = []
    c.append(new_markdown_cell(f"""# ST40 Li powder dropper, full 3-D torus: where does the lithium land?

Run of `3d/in.openedge` (constant feed 3.0e21 atoms/s from the dropper patch at
phi = 0, absorbing walls), read from `{outdir}` and `{log}`.
Coordinates: x = R cos phi, y = R sin phi, z = Z. Each `landing.<step>.vtu` is one
20 ms window with per-triangle fields `v_area` [m^2], `v_grain_hits`, `v_grain_rate`
[grains/s], `v_Li_hits`, `v_Li_flux` [atoms/m^2/s] and `v_Li_rate` [atoms/s]
(pweight-aware `compute surf/weighted`, emission not tallied). Comparison window = ({steady_from}, {steady_to}] (steady only if the inventories are flat there)."""))

    c.append(new_code_cell(f"""import re, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath
import pyvista as pv

CASE = Path.cwd().resolve()
if CASE.name == "notebooks":
    CASE = CASE.parent
OUT = CASE / "{outdir}"
LOG = CASE / "{log}"
STEADY_FROM = {steady_from}
STEADY_TO = {steady_to}
DT = 2.0e-6                       # s per step
deck = open(CASE / "in.openedge").read()
NEVERY = int(re.search(r"ave/surf\\s+\\S+\\s+(\\d+)", deck).group(1))      # compute surf sampling interval
FNUM = float(re.search(r"variable fnum\\s+index\\s+(\\S+)", deck).group(1))
NWEIGHT = float(re.search(r"variable nweight\\s+index\\s+(\\S+)", deck).group(1))
gsp = re.search(r"variable gspecies\\s+index\\s+(\\S+)", deck).group(1)        # grain species file of the deck
molwt = float([l.split() for l in open(CASE / gsp) if l.startswith("droplet")][0][1])
ATOMS_PER_GRAIN = {atoms_per_grain} or molwt / 6.941      # atoms per physical grain (from the species file unless overridden)
FEED = {feed:.4g}                     # atoms/s; 0 = time-dependent source, no percentages
def pct(x): return f"{{100*x/FEED:.1f}}% of feed" if FEED > 0 else "(time-dependent feed)"
import json
CASE_JSON = OUT.parent / "case.json"          # written by scripts/run_matrix.sh: the -var overrides of the deck
if CASE_JSON.exists():
    cj = json.load(open(CASE_JSON))
    NWEIGHT = float(cj.get("nweight", NWEIGHT)); ATOMS_PER_GRAIN = float(cj.get("atoms_per_grain", ATOMS_PER_GRAIN)); FEED = float(cj.get("feed", FEED))
    print("matrix case:", cj)
plt.rcParams.update({{
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 220,
}})
print("reading", OUT, LOG.name)

def read_loop(path):
    \"\"\"2-D wall loop in axi layout -> (R, Z) in file order.\"\"\"
    pts, mode = {{}}, None
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "Points": mode = "p"; continue
        if t[0] == "Lines": mode = "l"; continue
        if mode == "p" and len(t) == 3 and t[0].isdigit():
            pts[int(t[0])] = (float(t[2]), float(t[1]))
    return np.array([pts[i] for i in sorted(pts)])

WALL = read_loop(CASE.parent / "input/st40_wall_axi.surf")      # (R, Z), 141 points
CORE = read_loop(CASE.parent / "input/st40_core.surf")          # (R, Z), 90 points
NSEG = len(WALL)
hdr = open(CASE / "input/st40_wall_3d.surf").readline()
NPHI = int(re.search(r"(\\d+) strips", hdr).group(1))
DPHI = 360.0 / NPHI
drop_ids = []
for line in open(CASE / "input/st40_groups.inc"):
    t = line.split()
    if t[:1] == ["group"]: drop_ids += list(range(int(t[5]), int(t[6]) + 1))
drop_k = sorted({{(i - 1) // (2 * NPHI) for i in drop_ids}})
drop_j = sorted({{((i - 1) % (2 * NPHI)) // 2 for i in drop_ids}})
print(f"{{NSEG}} wall segments x {{NPHI}} strips; dropper segments {{drop_k}}, strips {{drop_j}}; ave/surf nevery {{NEVERY}}, fnum {{FNUM:g}}, nweight {{NWEIGHT:g}}")

# poloidal wall coordinate s [m]: arc length along the loop, 0 at the dropper
seg_len = np.hypot(*(np.roll(WALL, -1, axis=0) - WALL).T)
s_edge = np.concatenate([[0.0], np.cumsum(seg_len)])
s_mid = 0.5 * (s_edge[:-1] + s_edge[1:])
s0 = s_mid[drop_k].mean()
s_mid_d = s_mid - s0
phi_c = (np.arange(NPHI) - NPHI // 2) * DPHI          # strip centres, dropper at 0
seg_mid = 0.5 * (WALL + np.roll(WALL, -1, axis=0))      # (R, Z) of each segment"""))

    c.append(new_markdown_cell("## Inventories from the log"))
    c.append(new_code_cell("""cols = rows = None
if not LOG.exists():
    print("no log for this run: inventories unavailable (log was overwritten)")
for line in (open(LOG) if LOG.exists() else []):
    t = line.split()
    if t[:2] == ["Step", "CPU"]: cols = t; rows = []; continue
    if cols is not None and len(t) == len(cols) and t[0].isdigit():
        rows.append([float(v) for v in t])
    elif cols is not None and t[:1] == ["Loop"]: break
if cols is None: rows = []
S = {k: np.array([r[i] for r in rows]) for i, k in enumerate(cols)} if cols else {}
if S:
  t_ms = S["Step"] * DT * 1e3
  names = ["grains", "Li", "Li$^+$", "Li$^{2+}$", "Li$^{3+}$"]
  fig, ax = plt.subplots(figsize=(8, 4.5))
  for i, n in enumerate(names, 1):
      if f"c_cnt[{i}]" in S: ax.plot(t_ms, S[f"c_cnt[{i}]"], label=n)
  ax.set(xlabel="time [ms]", ylabel="macroparticles in flight", yscale="log", ylim=(1, None)); ax.legend()
  ax.axvspan(STEADY_FROM * DT * 1e3, STEADY_TO * DT * 1e3, color="0.9")
  plt.tight_layout(); plt.show()
  steady = (S["Step"] > STEADY_FROM) & (S["Step"] <= STEADY_TO)
  print("steady, in flight: " + ", ".join(f"{n.replace('$', '')} {S[f'c_cnt[{i}]'][steady].mean():.0f}"
                                         for i, n in enumerate(names, 1) if f"c_cnt[{i}]" in S))"""))

    c.append(new_markdown_cell("""## Landing map

Triangle ids follow `make_geometry.py`: 2-D segment k owns ids `2*NPHI*k+1 .. 2*NPHI*(k+1)`,
strip j the pair `2*NPHI*k + 2*j + 1, +2`, so the wall is a regular (segment, strip) lattice.
Per quad: Li deposition and grain landing rates = sum of `v_Li_rate` / `v_grain_rate`
over its two triangles."""))
    c.append(new_code_cell("""def load_landing(path):
    m = pv.read(path)
    order = np.argsort(np.asarray(m["id"]))
    ids = np.asarray(m["id"])[order]
    assert ids[0] == 1 and ids[-1] == 2 * NPHI * NSEG
    area = np.asarray(m["v_area"])[order]                          # m^2
    li_rate = np.asarray(m["v_Li_rate"])[order]                    # atoms/s
    gr_rate = np.asarray(m["v_grain_rate"])[order]                 # grains/s
    k = (ids - 1) // (2 * NPHI); j = ((ids - 1) % (2 * NPHI)) // 2
    def quad(v):
        a = np.zeros((NSEG, NPHI)); np.add.at(a, (k, j), v); return a
    ions = {k: quad(np.asarray(m[f"v_{k}_rate"])[order]) for k in ("Li1", "Li2", "Li3") if f"v_{k}_rate" in m.cell_data}
    return dict(area=quad(area), li_rate=quad(li_rate), grain_rate=quad(gr_rate), **ions)

files = sorted(glob.glob(str(OUT / "landing.*.vtu")), key=lambda p: int(re.findall(r"\\d+", p)[-1]))
steps = np.array([int(re.findall(r"\\d+", p)[-1]) for p in files])
use = [f for f, s in zip(files, steps) if STEADY_FROM < s <= STEADY_TO]        # file <step> = END of its window
print(f"{len(files)} windows, averaging {len(use)} in ({STEADY_FROM}, {STEADY_TO}]")
L = [load_landing(f) for f in use]
AREA = L[0]["area"]
LI = np.mean([d["li_rate"] for d in L], axis=0)        # atoms/s per quad
GR = np.mean([d["grain_rate"] for d in L], axis=0)     # grains/s per quad
LI_FLUX = LI / AREA
li_wall = LI.sum(); gr_wall = GR.sum()
print(f"Li to wall  {li_wall:.3e} atoms/s  {pct(li_wall)}")
print(f"grains to wall {gr_wall:.3e} grains/s")
ION = {k: np.mean([d[k] for d in L], axis=0) for k in ("Li1", "Li2", "Li3") if k in L[0]}
for k, lab in (("Li1", "Li+"), ("Li2", "Li2+"), ("Li3", "Li3+")):
    if k in ION: print(f"{lab} to wall {ION[k].sum():.3e} atoms/s  {pct(ION[k].sum())}")
if FEED > 0:
    GRAIN_FEED = FEED / ATOMS_PER_GRAIN
    print(f"fed {GRAIN_FEED:.3e} grains/s: {100*gr_wall/GRAIN_FEED:.0f}% of the grains reach the wall")"""))

    c.append(new_markdown_cell("## Poloidal and toroidal deposition profiles"))
    c.append(new_code_cell("""LION = sum(ION.values()) if ION else np.zeros_like(LI)   # Li+ + Li2+ + Li3+, atoms/s per quad
LTOT = LI + LION                               # all lithium to the wall
pol, ipol, gpol = LTOT.sum(axis=1) / seg_len, LION.sum(axis=1) / seg_len, GR.sum(axis=1) / seg_len
npol = LI.sum(axis=1) / seg_len
tor, itor, ntor = LTOT.sum(axis=0), LION.sum(axis=0), LI.sum(axis=0)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].step(s_mid_d, ipol / 1e18, where="mid", label="Li ions"); ax[0].step(s_mid_d, npol / 1e18, where="mid", label="Li atoms")
ax[0].step(s_mid_d, gpol / 1e5, where="mid", label="grains [$10^5$ s$^{-1}$ m$^{-1}$]")
ax[0].set(xlabel="wall coordinate from dropper s [m]", ylabel=r"deposition [$10^{18}$ s$^{-1}$ m$^{-1}$]"); ax[0].legend(); ax[0].axvline(0, c="k", lw=0.6)
ax[1].step(phi_c, itor / 1e18, where="mid", label="Li ions"); ax[1].step(phi_c, ntor / 1e18, where="mid", label="Li atoms")
ax[1].set(xlabel=r"toroidal angle $\\phi$ [deg]", ylabel=r"Li per 1.5$^\\circ$ [$10^{18}$ s$^{-1}$]", xlim=(-180, 180)); ax[1].axvline(0, c="k", lw=0.6); ax[1].legend()
ax[2].plot(phi_c, np.cumsum(tor) / tor.sum()); ax[2].set(xlabel=r"toroidal angle $\\phi$ [deg]", ylabel="cumulative Li fraction (all)", xlim=(-180, 180)); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
w = tor / tor.sum() if tor.sum() > 0 else np.full_like(tor, np.nan)
phi_mean = (w * phi_c).sum(); phi_sd = np.sqrt((w * (phi_c - phi_mean) ** 2).sum())
p5, p95 = np.interp([0.05, 0.95], np.cumsum(w), phi_c) if tor.sum() > 0 else (np.nan, np.nan)
print(f"all Li (ions + atoms) to wall {LTOT.sum():.3e} atoms/s {pct(LTOT.sum())}")
print(f"Li lands at phi = {phi_mean:.1f} deg (mean), spatial spread sd {phi_sd:.1f} deg, 90% between {p5:.0f} and {p95:.0f} deg")
top = np.argsort(pol)[::-1][:5]
for k in top:
    print(f"  s = {s_mid_d[k]:+.3f} m  (R={seg_mid[k,0]:.3f}, Z={seg_mid[k,1]:+.3f})  {100*LTOT[k].sum()/LTOT.sum():.1f}% of the Li")"""))

    c.append(new_code_cell("""# poloidal view: the 2-D wall coloured by its toroidally integrated Li deposition (ions + atoms)
from matplotlib.collections import LineCollection
segs = np.stack([WALL, np.roll(WALL, -1, axis=0)], axis=1)
fig, ax = plt.subplots(figsize=(6, 8))
lc = LineCollection(segs, cmap="inferno", norm=LogNorm(vmin=max(pol[pol > 0].min(), pol.max() * 1e-4), vmax=pol.max()) if (pol > 0).any() else None, linewidths=4)
lc.set_array(np.where(pol > 0, pol, np.nan)); ax.add_collection(lc)
ax.plot(*np.vstack([CORE, CORE[:1]]).T, "c--", lw=0.8, label="core contour (vacuum inside)")
ax.plot(seg_mid[drop_k, 0], seg_mid[drop_k, 1], "w*", ms=14, mec="k", label="dropper")
fig.colorbar(lc, ax=ax, label=r"Li deposition (ions + atoms), all $\\phi$ [atoms/s/m]")
ax.set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", xlim=(0.1, 1.05), ylim=(-0.95, 0.95)); ax.legend(loc="lower left")
plt.show()"""))

    c.append(new_markdown_cell("## The plume: last particle snapshot"))
    c.append(new_code_cell("""sfiles = sorted(glob.glob(str(OUT / "state.*.vtu")), key=lambda p: int(re.findall(r"\\d+", p)[-1]))
m = pv.read(sfiles[-1]); P = np.asarray(m.points); typ = np.asarray(m["type"])
g, li = P[typ == 1], P[typ == 2]
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].plot(*np.vstack([WALL, WALL[:1]]).T, "k-", lw=1); ax[0].plot(*np.vstack([CORE, CORE[:1]]).T, "c--", lw=0.8)
ax[0].scatter(np.hypot(li[:, 0], li[:, 1]), li[:, 2], s=1, c="r", alpha=0.3, label=f"Li ({len(li)})")
ax[0].scatter(np.hypot(g[:, 0], g[:, 1]), g[:, 2], s=1, c="k", alpha=0.3, label=f"grains ({len(g)})")
ax[0].set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", title=f"poloidal projection, step {steps[-1]}"); ax[0].legend(markerscale=8)
th = np.linspace(0, 2 * np.pi, 361)
for r in (WALL[:, 0].min(), WALL[:, 0].max()): ax[1].plot(r * np.cos(th), r * np.sin(th), "k-", lw=0.8)
ax[1].scatter(li[:, 0], li[:, 1], s=1, c="r", alpha=0.3); ax[1].scatter(g[:, 0], g[:, 1], s=1, c="k", alpha=0.3)
ax[1].plot([0, 1.05], [0, 0], "b:", lw=1); ax[1].text(1.0, 0.03, "dropper phi=0", color="b")
ax[1].set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="top view: the plume sweeps towards negative phi")
plt.tight_layout(); plt.show()"""))

    c.append(new_markdown_cell("""## 2-D (R, Z) distributions, toroidally averaged

Marker positions from the `state.*.vtu` snapshots in the steady window, binned on
1.5 cm (R, Z) bins: density = markers x weight / (2 pi R dR dZ) / snapshots
(fnum atoms per Li or Li-ion marker, nweight real grains per grain marker)."""))
    c.append(new_code_cell("""sfiles = sorted(glob.glob(str(OUT / "state.*.vtu")), key=lambda p: int(re.findall(r"\\d+", p)[-1]))
suse = [f for f in sfiles if STEADY_FROM < int(re.findall(r"\\d+", f)[-1]) <= STEADY_TO] or sfiles[-1:]
names = ["grains", "Li", "Li$^+$", "Li$^{2+}$", "Li$^{3+}$"]
weight = [NWEIGHT, FNUM, FNUM, FNUM, FNUM]          # real grains / atoms per marker
units = ["grains m$^{-3}$"] + ["m$^{-3}$"] * 4
dR = dZ = 0.015
rb = np.arange(0.1, 1.05 + dR, dR); zb = np.arange(-0.95, 0.95 + dZ, dZ)
ring = 2 * np.pi * 0.5 * (rb[:-1] + rb[1:])[:, None] * dR * dZ
NB = np.zeros((5, len(rb) - 1, len(zb) - 1))
for f in suse:
    m = pv.read(f); P = np.asarray(m.points); typ = np.asarray(m["type"]); R = np.hypot(P[:, 0], P[:, 1])
    for t in range(1, 6):
        NB[t - 1] += np.histogram2d(R[typ == t], P[typ == t, 2], bins=[rb, zb])[0]
NB *= np.array(weight)[:, None, None] / ring / len(suse)
show = [t for t in range(5) if NB[t].any()]
fig, ax = plt.subplots(1, len(show), figsize=(3.6 * len(show), 6.5), sharey=True)
for a, t in zip(np.atleast_1d(ax), show):
    H = np.where(NB[t] > 0, NB[t], np.nan)
    im = a.pcolormesh(rb, zb, H.T, norm=LogNorm(vmin=np.nanmax(H) * 1e-3, vmax=np.nanmax(H)), cmap="inferno")
    a.plot(*np.vstack([WALL, WALL[:1]]).T, "k-", lw=0.8); a.plot(*np.vstack([CORE, CORE[:1]]).T, "c--", lw=0.6)
    a.set(xlabel="R [m]", aspect="equal", title=names[t], xlim=(0.1, 1.05), ylim=(-0.95, 0.95))
    fig.colorbar(im, ax=a, orientation="horizontal", pad=0.08, label=units[t])
np.atleast_1d(ax)[0].set_ylabel("Z [m]")
plt.tight_layout(); plt.show()
print(f"averaged {len(suse)} snapshots, steps {int(re.findall(r'\\d+', suse[0])[-1])}..{int(re.findall(r'\\d+', suse[-1])[-1])}")"""))

    c.append(new_markdown_cell("""## Summary

- Most of the ablated lithium is ionized; the wall load is dominated by Li ions (Li+, Li2+, Li3+), with
  neutral Li atoms a small fraction. Rates are printed above as % of the feed.
- Grains reach the wall partly ablated: the grain wall fraction counts grains, not lithium mass.
- The ions fill a scrape-off-layer ring around the core; neutral Li stays close to the grain path, which
  sweeps towards negative phi as the grains pick up toroidal velocity from ion drag."""))
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output_prod", help="run output directory, relative to 3d/")
    ap.add_argument("--log", default="output/stdout.openedge", help="log file of that run")
    ap.add_argument("--steady-from", type=int, default=120000, help="first step of the steady-state average")
    ap.add_argument("--steady-to", type=int, default=200000, help="last step of the steady-state average")
    ap.add_argument("--output", type=Path, default=ROOT / "notebooks/st40_3d_landing.ipynb")
    ap.add_argument("--atoms-per-grain", type=float, default=0, help="override atoms per physical grain (default: from the deck's grain species file)")
    ap.add_argument("--feed", type=float, default=3.0e21, help="constant feed [atoms/s]; 0 for a time-dependent source")
    ap.add_argument("--no-execute", action="store_true")
    args = ap.parse_args()
    nb = build(args.outdir, args.log, args.steady_from, args.steady_to, args.atoms_per_grain, args.feed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print("wrote", args.output)
    if not args.no_execute:
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
                        str(args.output)], check=True, cwd=ROOT)
        print("executed", args.output)


if __name__ == "__main__":
    main()
