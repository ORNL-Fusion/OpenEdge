#!/usr/bin/env python3
"""Tsurf leg of the SOLPS<->OpenEdge loop: ld_*.dat -> strip CSV -> wall_strip.surf.

After SOLPS converges and WLLD writes the target heat-load profiles
(ld_tg_o.dat, ld_tg_i.dat), the liquid-metal surface temperature and the
deuterium incident flux along the divertor strips are obtained from a short
OpenEdge `surface/state/lm` run (the "strip deck"). The resulting per-strip
profile (Tsurf_C, Gamma_D_m2s) is then baked onto the refined wall geometry as
two custom per-line columns (Tsurf_lm, Gamma_D_lm) that the full coupled deck
reads via `read_surf ... custom Tsurf_lm float 0 custom Gamma_D_lm float 0`.

This module is the Python port of the `analysis.ipynb` cells (refine geometry
parse + strip-deck run + `write_case_strip_wall_surf`). It is keyed by the
SOLPS *base* case (attached / detached): Tsurf/Gamma_D depend on the SOLPS
heat load + background, not on the mist/no_mist OpenEdge option, so all four
full runs of a base case share one wall_strip_<base>.surf and one plasma_<base>.h5.

Pipeline (per base case):

    SOLPS (converged) --WLLD--> ld_tg_{o,i}.dat
        --strip deck (surface/state/lm)--> <base>.strip.{outer,inner}.csv
        --bake (KDTree map onto wall_refined.surf)--> wall_strip_<base>.surf

CLI:

    python3 build_tsurf_surf.py config.json --case attached__mist            # strip + bake
    python3 build_tsurf_surf.py config.json --case attached__mist --bake     # bake only (CSVs exist)
    python3 build_tsurf_surf.py config.json --case attached__mist --strip    # strip only
    python3 build_tsurf_surf.py config.json --case attached__mist --dry      # print plan

The bake reuses the existing (already validated, 288-line) wall_refined.surf;
it does NOT re-refine the geometry, so the full decks' group IDs stay valid.
"""
import os
import sys
import json
import shlex
import argparse
import subprocess

import numpy as np
import pandas as pd

# Divertor LM line-ID ranges in wall_refined.surf (inclusive), matching the
# full deck's `group div_front_{outer,inner} surf id <> ...`. Override per
# config["tsurf_leg"] if the refined geometry ever changes.
DEFAULT_OUTER_LIDS = (185, 282)
DEFAULT_INNER_LIDS = (110, 165)

# surface/state/lm model parameters (h0 film thickness, U0/Bs/alpha/width/Tin),
# carried verbatim from the notebook strip deck. Override via config.
DEFAULT_LM = dict(h0="5.0e-3", U0="8.0", Bs="5.0", alpha="43.0",
                  width="1.67", Tin="350.0")

STRIP_DECK_TEMPLATE = """\
seed 1234
dimension   2
boundary    o o p
global      gridcut 0.0 comm/sort no
timestep    5e-05

create_box  2 5.7 -3.8 3.8 -0.05 0.05
create_grid 100 100 1
balance_grid rcb cell

read_surf   {surf} group slab particle check
group div_front_outer surf id <> {ol} {oh}
group div_front_inner surf id <> {il} {ih}

surf_collide vac specular
surf_modify  slab collide vac

fix pd background file {plasma} static yes

fix flm_o surface/state/lm div_front_outer 1 solps_b2pl ld_tg_o.dat &
          h0 {h0} U0 {U0} Bs {Bs} alpha {alpha} width {width} Tin {Tin} static yes csv {strip_outer}
fix flm_i surface/state/lm div_front_inner 1 solps_b2pl ld_tg_i.dat &
          h0 {h0} U0 {U0} Bs {Bs} alpha {alpha} width {width} Tin {Tin} static yes csv {strip_inner}

stats 1
run   0
"""


# ----------------------------------------------------------------------
# Geometry: parse the refined wall.surf -> line list + segment midpoints
# ----------------------------------------------------------------------
def parse_refined_surf(path):
    """Parse a refined SPARTA .surf -> (lines, pts_arr, midpoints).

    lines: list of (lid, a, b) using the file's 1-based point IDs.
    pts_arr: (N,2) float array indexed by point_id-1.
    midpoints: (len(lines),2) segment midpoints, ordered by line id-1.
    """
    pts = {}
    lines = []
    section = None
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low.startswith("points"):
                section = "points"; continue
            if low.startswith("lines"):
                section = "lines"; continue
            if not s[0].isdigit():
                continue
            parts = s.split()
            if section == "points":
                pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "lines":
                lines.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not lines:
        raise ValueError(f"no Lines section parsed from {path}")
    n = max(pts)
    pts_arr = np.zeros((n, 2))
    for pid, (x, y) in pts.items():
        pts_arr[pid - 1] = (x, y)
    lines.sort()
    mids = np.zeros((len(lines), 2))
    for k, (_, a, b) in enumerate(lines):
        mids[k] = 0.5 * (pts_arr[a - 1] + pts_arr[b - 1])
    return lines, pts_arr, mids


# ----------------------------------------------------------------------
# Bake: strip CSV (Tsurf_C, Gamma_D_m2s) -> wall_strip.surf custom columns
# ----------------------------------------------------------------------
def bake_strip_to_surf(wall_refined, strip_outer_csv, strip_inner_csv, out_surf,
                       outer_lids=DEFAULT_OUTER_LIDS, inner_lids=DEFAULT_INNER_LIDS,
                       case=""):
    """Map strip CSV profiles onto wall_refined.surf -> wall_strip.surf.

    For each LM line, nearest-neighbour (in R,Z) onto the strip CSV samples
    gives Tsurf [degC] and Gamma_D [m^-2 s^-1]; non-LM lines get 0/0. Writes a
    copy of wall_refined.surf with two extra columns appended to each Lines row.
    """
    from scipy.spatial import cKDTree

    lines, _, mids = parse_refined_surf(wall_refined)
    nseg = len(lines)
    t_col = np.zeros(nseg)
    g_col = np.zeros(nseg)

    for (lo, hi), csv in ((outer_lids, strip_outer_csv),
                          (inner_lids, strip_inner_csv)):
        df = pd.read_csv(csv)
        tree = cKDTree(df[["R_m", "Z_m"]].values)
        sl = slice(lo - 1, hi)                       # lids are 1-based, contiguous
        _, idx = tree.query(mids[sl])
        t_col[sl] = df["Tsurf_C"].values[idx]
        g_col[sl] = df["Gamma_D_m2s"].values[idx]

    with open(wall_refined) as f, open(out_surf, "w") as g:
        g.write("# wall_refined.surf with Tsurf_lm [degC] and "
                "Gamma_D_lm [m^-2 s^-1] columns\n")
        g.write(f"# case={case}; LM ranges (lid): "
                f"outer {outer_lids[0]}..{outer_lids[1]}, "
                f"inner {inner_lids[0]}..{inner_lids[1]}\n\n")
        in_lines = False
        for raw in f:
            s = raw.strip()
            if s.lower().startswith("lines"):
                in_lines = True
                g.write(raw); continue
            if not in_lines or not s or not s[0].isdigit():
                g.write(raw); continue
            lid_str, a, b = s.split()[:3]
            lid = int(lid_str)
            g.write(f"{lid} {a} {b} {t_col[lid-1]:.6e} {g_col[lid-1]:.6e}\n")
    return out_surf


# ----------------------------------------------------------------------
# Strip deck: ld_tg_{o,i}.dat + plasma -> strip CSVs (short OpenEdge run)
# ----------------------------------------------------------------------
def _symlink(run_dir, src, name):
    link = os.path.join(run_dir, name)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(src), link)
    return link


def run_strip_deck(run_dir, wall_refined, plasma_h5, ld_outer, ld_inner,
                   strip_outer_csv, strip_inner_csv, binary, mpirun="mpirun",
                   np_oe=1, setvars=None, lm=DEFAULT_LM,
                   outer_lids=DEFAULT_OUTER_LIDS, inner_lids=DEFAULT_INNER_LIDS,
                   dry=False):
    """Render + run the surface/state/lm strip deck -> strip CSVs."""
    os.makedirs(run_dir, exist_ok=True)
    _symlink(run_dir, ld_outer, "ld_tg_o.dat")
    _symlink(run_dir, ld_inner, "ld_tg_i.dat")

    deck = STRIP_DECK_TEMPLATE.format(
        surf=wall_refined, plasma=plasma_h5,
        ol=outer_lids[0], oh=outer_lids[1],
        il=inner_lids[0], ih=inner_lids[1],
        strip_outer=strip_outer_csv, strip_inner=strip_inner_csv, **lm)
    deck_path = os.path.join(run_dir, "in.strip_wall")
    with open(deck_path, "w") as f:
        f.write(deck)

    inner = [mpirun, "-np", str(np_oe), binary, "-in", "in.strip_wall"]
    if setvars:
        cmd = ["bash", "-c",
               f"source {shlex.quote(setvars)} --force >/dev/null 2>&1 && "
               f"exec {' '.join(shlex.quote(a) for a in inner)}"]
    else:
        cmd = inner
    print(f"  strip deck: {deck_path}")
    print(f"  launch    : {' '.join(inner)}  (cwd={run_dir})")
    if dry:
        return deck_path
    subprocess.run(cmd, cwd=run_dir, check=True)
    return deck_path


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--case", required=True,
                    help="full case key (e.g. attached__mist); strip/bake key off the base")
    ap.add_argument("--strip", action="store_true", help="run strip deck only")
    ap.add_argument("--bake", action="store_true", help="bake CSVs -> surf only")
    ap.add_argument("--dry", action="store_true", help="print plan, launch nothing")
    args = ap.parse_args()

    do_strip = args.strip or not (args.strip or args.bake)
    do_bake = args.bake or not (args.strip or args.bake)

    with open(args.config) as f:
        cfg = json.load(f)
    coupled = cfg["coupled_dir"]

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    base = args.case.split("__")[0]                  # attached__mist -> attached
    cc = cfg["cases"][args.case]
    solps_run_dir = absp(cc["solps_run_dir"])

    tl = cfg.get("tsurf_leg", {})
    oe_dir = absp("openedge")
    surf_dir = absp(tl.get("surf_dir", "openedge/surf_with_strip"))
    strip_dir = absp(tl.get("strip_dir", "openedge/strip_csv"))
    runs_dir = absp(tl.get("strip_runs_dir", "openedge/runs"))
    plasma_dir = absp(tl.get("plasma_dir", "openedge/plasma"))
    wall_refined = absp(tl.get("wall_refined",
                               os.path.join(surf_dir, "wall_refined.surf")))
    outer_lids = tuple(tl.get("outer_lids", DEFAULT_OUTER_LIDS))
    inner_lids = tuple(tl.get("inner_lids", DEFAULT_INNER_LIDS))
    lm = {**DEFAULT_LM, **tl.get("lm", {})}

    # WLLD output: ld_tg_{o,i}.dat. Default to <solps_run_dir>/b2pl.exe.dir.
    wlld_dir = absp(tl.get("wlld_dir", os.path.join(solps_run_dir, "b2pl.exe.dir")))
    ld_outer = os.path.join(wlld_dir, "ld_tg_o.dat")
    ld_inner = os.path.join(wlld_dir, "ld_tg_i.dat")

    plasma_h5 = os.path.join(plasma_dir, f"plasma_{base}.h5")
    strip_outer = os.path.join(strip_dir, f"{base}.strip.outer.csv")
    strip_inner = os.path.join(strip_dir, f"{base}.strip.inner.csv")
    out_surf = os.path.join(surf_dir, f"wall_strip_{base}.surf")
    run_dir = os.path.join(runs_dir, base)

    binary = cfg["openedge_binary"]
    mpirun = cfg.get("mpirun", "mpirun")
    setvars = cfg.get("oneapi_setvars")
    np_oe = int(tl.get("mpi_np_strip", 1))

    print(f"== Tsurf leg  case='{args.case}' (base='{base}') ==")
    print(f"  wall_refined : {wall_refined}")
    print(f"  plasma       : {plasma_h5}")
    print(f"  WLLD ld files: {ld_outer}")
    print(f"                 {ld_inner}")
    print(f"  strip CSVs   : {strip_outer}")
    print(f"                 {strip_inner}")
    print(f"  out surf     : {out_surf}")
    print(f"  LM lids      : outer {outer_lids}, inner {inner_lids}")

    for p in (wall_refined,):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing required input: {p}")

    if do_strip:
        for p in (plasma_h5, ld_outer, ld_inner):
            if not os.path.exists(p) and not args.dry:
                sys.exit(f"ERROR: strip deck needs {p} (run plasma build / WLLD first)")
        os.makedirs(strip_dir, exist_ok=True)
        print("\n== strip deck (surface/state/lm) ==")
        run_strip_deck(run_dir, wall_refined, plasma_h5, ld_outer, ld_inner,
                       strip_outer, strip_inner, binary, mpirun=mpirun,
                       np_oe=np_oe, setvars=setvars, lm=lm,
                       outer_lids=outer_lids, inner_lids=inner_lids, dry=args.dry)

    if do_bake:
        for p in (strip_outer, strip_inner):
            if not os.path.exists(p) and not args.dry:
                sys.exit(f"ERROR: bake needs {p} (run --strip first)")
        print("\n== bake strip CSV -> wall_strip surf ==")
        if args.dry:
            print(f"  would write {out_surf}")
        else:
            bake_strip_to_surf(wall_refined, strip_outer, strip_inner, out_surf,
                               outer_lids=outer_lids, inner_lids=inner_lids,
                               case=base)
            n = sum(1 for ln in open(out_surf)
                    if ln.strip()[:1].isdigit() and len(ln.split()) >= 5)
            print(f"  wrote {out_surf}  ({n} lines with custom Tsurf_lm/Gamma_D_lm)")


if __name__ == "__main__":
    main()
