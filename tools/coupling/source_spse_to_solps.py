#!/usr/bin/env python3
"""Single-shot: OpenEdge neutral-Li source_SpSe.* -> SOLPS-ITER source files.

The new (non-droplet) workflow: OpenEdge emits neutral Li (sputter +
evaporation + adatom), tracks it with `fix volume/chem/adas mode neutral`,
and dumps the running-average per-cell ionization source Sp (f_fsrc[1],
m^-3 s^-1) and electron-energy source Qe (f_fsrc[5], W m^-3) to
output/source_SpSe.* . This script:

  1. reads the source_SpSe.* dumps for the selected case,
  2. checks the running average has plateaued (convergence),
  3. takes the converged (last) snapshot, forms per-cell totals
     Sp*vol [Li/s] and Qe*vol [W],
  4. maps them onto the SOLPS B2 grid (conservative nearest-neighbour),
  5. writes source2d.* (Sp -> sna0, Li+ species) and she2d.* (Qe -> she0),
     plus a b2.sources.profile carrying read_sna0_2d + read_she0_2d.

SOLPS then runs to termination reading these external sources. The OpenEdge
side is run ONCE; there is no back-and-forth in this mode.

Usage:
    python3 source_spse_to_solps.py config.json [--check]

--check  read + convergence + conservation report only; write no files
         (use when the SOLPS geometry is not yet staged).
"""
import sys
import os
import glob
import json
import numpy as np
import pandas as pd

SP_COL, SE_COL = "f_fsrc[1]", "f_fsrc[5]"


# ----------------------------------------------------------------------
# Read SPARTA grid dumps (same parser as the user's analysis snippet)
# ----------------------------------------------------------------------
def read_grid_dump(path):
    """Parse one SPARTA grid-dump snapshot -> (step:int, DataFrame)."""
    with open(path) as fh:
        lines = fh.readlines()
    step = i0 = cols = None
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
        elif ln.startswith("ITEM: CELLS"):
            cols = ln.split()[2:]          # id xc yc vol f_fsrc[1] f_fsrc[5]
            i0 = i + 1
            break
    data = np.loadtxt(lines[i0:])
    return step, pd.DataFrame(data, columns=cols)


def source_dump_files(out_dir):
    """source_SpSe.* sorted by integer timestep."""
    files = glob.glob(os.path.join(out_dir, "source_SpSe.*"))
    return sorted(files, key=lambda p: int(p.rsplit(".", 1)[1]))


def integrate(df):
    """Volume-integrated (int Sp dV [Li/s], int Qe dV [W])."""
    vol = df["vol"].to_numpy()
    return (float((df[SP_COL].to_numpy() * vol).sum()),
            float((df[SE_COL].to_numpy() * vol).sum()))


def check_convergence(files, rel_tol, window):
    """Has int Sp dV plateaued over the last `window` snapshots?

    Returns (converged:bool, history:list[(step, intSp, intQe)]).
    """
    history = []
    for p in files:
        st, df = read_grid_dump(p)
        sp, qe = integrate(df)
        history.append((st, sp, qe))
    if len(history) < 2:
        return False, history
    tail = [h[1] for h in history[-window:]] if window > 0 else [h[1] for h in history]
    tail = [v for v in tail if v != 0.0]
    if len(tail) < 2:
        return False, history
    ref = tail[-1]
    if ref == 0.0:
        return False, history
    spread = (max(tail) - min(tail)) / abs(ref)
    return spread <= rel_tol, history


# ----------------------------------------------------------------------
# Conservative nearest-neighbour map: OpenEdge cells -> SOLPS B2 cells
# ----------------------------------------------------------------------
def map_to_solps(xc, yc, per_cell_total, solps, max_dist):
    """Accumulate per-cell totals onto the SOLPS grid (sum-conserving).

    Args:
        xc, yc: OpenEdge cell centres (R, Z) [m]
        per_cell_total: already integrated over the OE cell (Li/s or W)
        solps: SolpsInterface with geometry loaded
        max_dist: drop OE cells farther than this from any SOLPS centre [m]

    Returns:
        (nx+2, ny+2) array, mapped: int.
    """
    from scipy.spatial import cKDTree
    nxp2, nyp2 = solps.nx + 2, solps.ny + 2
    R_cen, Z_cen = solps.get_cell_centers()
    valid = (R_cen > 0) & np.isfinite(R_cen) & np.isfinite(Z_cen)
    valid_flat = np.where(valid.ravel())[0]
    tree = cKDTree(np.column_stack([R_cen[valid].ravel(), Z_cen[valid].ravel()]))

    out = np.zeros(nxp2 * nyp2)
    mask = np.abs(per_cell_total) > 0
    dropped = 0.0
    if mask.sum():
        dist, idx = tree.query(np.column_stack([xc[mask], yc[mask]]))
        vals = per_cell_total[mask]
        for d, j, v in zip(dist, idx, vals):
            if d <= max_dist:
                out[valid_flat[j]] += v
            else:
                dropped += v
    return out.reshape(nxp2, nyp2), dropped


def set_b2mn_flags(run_dir, updates):
    """Set 'key' 'value' pairs in b2mn.dat, replacing or appending.

    SOLPS b2mn.dat lines look like:  'b2sral_inputfile'   '2'   # comment
    """
    path = os.path.join(run_dir, "b2mn.dat")
    if not os.path.exists(path):
        print(f"  WARNING: no b2mn.dat in {run_dir}; cannot set source flags")
        return
    bak = path + ".openedge.bak"
    if not os.path.exists(bak):
        import shutil
        shutil.copy2(path, bak)
    with open(path) as f:
        lines = f.readlines()
    seen = set()
    out = []
    for line in lines:
        replaced = False
        for key, val in updates.items():
            if f"'{key}'" in line:
                comment = ""
                if "#" in line:
                    comment = " # " + line.split("#", 1)[1].strip()
                out.append(f"'{key}'{' ' * max(1, 34 - len(key))}'{val}'{comment}\n")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"'{key}'{' ' * max(1, 34 - len(key))}'{val}'\n")
    with open(path, "w") as f:
        f.writelines(out)
    print(f"  b2mn.dat: set " + ", ".join(f"{k}={v}" for k, v in updates.items()))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    case_override = next((a.split("=", 1)[1] for a in sys.argv[1:]
                          if a.startswith("--case=")), None)
    if not args:
        print(__doc__)
        sys.exit(1)
    with open(args[0]) as f:
        cfg = json.load(f)

    coupled = cfg["coupled_dir"]
    case = case_override or cfg.get("case", "attached")
    cases = cfg.get("cases", {})
    if case not in cases:
        sys.exit(f"case '{case}' not in config['cases'] ({list(cases)})")
    cc = cases[case]

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    out_dir = _abs(cc["oe_output_dir"])
    solps_run_dir = _abs(cc["solps_run_dir"])

    exp = cfg.get("source_export", {})
    sna_prefix = exp.get("sna_prefix", "source2d")
    she_prefix = exp.get("she_prefix", "she2d")
    write_energy = exp.get("write_energy", True)
    max_dist = exp.get("max_map_dist_m", 0.5)

    conv = cfg.get("oe_convergence", {})
    rel_tol = conv.get("rel_tol", 0.02)
    window = conv.get("window", 3)

    # -- 1/2. read dumps + convergence -------------------------------------
    files = source_dump_files(out_dir)
    if not files:
        sys.exit(f"no source_SpSe.* dumps in {out_dir}")
    converged, hist = check_convergence(files, rel_tol, window)

    print(f"== case '{case}'  ({len(files)} source_SpSe snapshots) ==")
    print(f"{'step':>10} {'int Sp dV [Li/s]':>20} {'int Qe dV [W]':>16}")
    for st, sp, qe in hist:
        print(f"{st:>10d} {sp:>20.6e} {qe:>16.6e}")
    tag = "CONVERGED" if converged else "NOT converged"
    print(f"-> running-average plateau ({tag}, rel_tol={rel_tol}, "
          f"window={window})")
    if not converged:
        print("   WARNING: int Sp dV still drifting -- run OpenEdge longer "
              "or relax rel_tol. Using the last snapshot anyway.")

    # -- take the converged (last) snapshot --------------------------------
    step, df = read_grid_dump(files[-1])
    vol = df["vol"].to_numpy()
    xc = df["xc"].to_numpy()
    yc = df["yc"].to_numpy()
    Sp_tot = df[SP_COL].to_numpy() * vol      # [Li/s] per cell
    Qe_tot = df[SE_COL].to_numpy() * vol      # [W]    per cell
    intSp_oe, intQe_oe = float(Sp_tot.sum()), float(Qe_tot.sum())
    print(f"\nusing snapshot step {step}: "
          f"int Sp = {intSp_oe:.6e} Li/s, int Qe = {intQe_oe:.6e} W")

    if check_only:
        print("\n--check: no SOLPS files written.")
        return

    # -- 3. load SOLPS geometry (baserun) + state (case dir) ---------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solps_interface import SolpsInterface
    base_dir = cfg.get("solps_base_dir")
    base_dir = _abs(base_dir) if base_dir else None
    solps = SolpsInterface(solps_run_dir, baserun_dir=base_dir)
    solps.load_geometry()        # shared mesh from baserun/b2fgmtry
    solps.load_plasma_state()    # case b2fstate -> ns + Li+ species index
    print(f"SOLPS grid: nx={solps.nx} ny={solps.ny} ns={solps.ns}")

    # -- 4. map both channels onto the B2 grid -----------------------------
    sna0, sp_drop = map_to_solps(xc, yc, Sp_tot, solps, max_dist)
    she0, qe_drop = map_to_solps(xc, yc, Qe_tot, solps, max_dist)

    # -- 5. write source files ---------------------------------------------
    sna_file = f"{sna_prefix}.{1:05d}"
    she_file = f"{she_prefix}.{1:05d}"
    solps.write_source2d(sna0, sna_file)               # Sp -> sna0 (Li+)
    she_filenames = None
    if write_energy:
        solps.write_she0_2d(she0, she_file)            # Qe -> she0
        she_filenames = [she_file]
    solps.write_sources_profile_chain(
        n_windows=1, dt_windows=[1.0e20], t_start=0.0,
        source_filenames=[sna_file], she_filenames=she_filenames)

    # activate external-source reading in b2mn.dat (+ optional ntim cap)
    b2mn_updates = {"b2sral_inputfile": "2"}
    if "solps_ntim" in cfg:
        b2mn_updates["b2mndr_ntim"] = str(cfg["solps_ntim"])
    set_b2mn_flags(solps_run_dir, b2mn_updates)

    # -- conservation report ----------------------------------------------
    print("\n== conservation (OpenEdge int  ->  SOLPS-grid sum) ==")
    print(f"  Sp [Li/s]: {intSp_oe:.6e}  ->  {sna0.sum():.6e}"
          f"   (dropped >{max_dist} m: {sp_drop:.3e})")
    print(f"  Qe [W]   : {intQe_oe:.6e}  ->  {she0.sum():.6e}"
          f"   (dropped >{max_dist} m: {qe_drop:.3e})")
    print(f"\nwrote in {solps_run_dir}:")
    print(f"  {sna_file}            (read_sna0_2d, Li+ particle source)")
    if write_energy:
        print(f"  {she_file}              (read_she0_2d, electron energy source)")
    print(f"  b2.sources.profile   (read_sna0_2d + read_she0_2d)")
    print(f"  b2mn.dat             (b2sral_inputfile='2'; backup .openedge.bak)")
    print("\nNow run SOLPS to termination in that directory.")


if __name__ == "__main__":
    main()
