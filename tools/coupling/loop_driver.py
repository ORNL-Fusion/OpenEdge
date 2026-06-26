#!/usr/bin/env python3
"""Iterative OpenEdge <-> SOLPS-ITER coupling driver (back-and-forth to plateau).

One iteration = one OpenEdge run + one SOLPS run (the validated one-shot leg),
followed by rebuilding the SOLPS-derived inputs OpenEdge reads next:

    iter N:  OpenEdge (plasma_<base>.h5 + wall_strip_<base>.surf)
               -> source_SpSe -> source2d/she2d -> SOLPS            [oneshot_driver]
             archive iter_N (OE source + SOLPS diagnostics); record int Sp
             converged?  (|d int Sp| / int Sp <= rel_tol, K iters)  -> stop
             else rebuild for iter N+1:
               plasma_<base>.h5  <- convert_solps_plasma --heatflux total (new b2fstate)
               ld_tg_{o,i}.dat   <- run_b2plot_wlld.sh  (WLLD)
               wall_strip_<base>.surf <- build_tsurf_surf  (strip deck + bake)

The OpenEdge deck points at plasma_<base>.h5 and wall_strip_<base>.surf by
fixed path; rebuilding those files in place means the next OE run picks up the
new background + surface state with no deck edits.

Convergence is on the volume-integrated Li ionization source int Sp dV (the
coupling-relevant scalar), tracked across iterations in convergence.csv.

    python3 loop_driver.py config.json [--case attached__mist] [--dry]

--dry  resolve everything + print the per-iteration plan; launch nothing.
"""
import os
import sys
import json
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from source_spse_to_solps import source_dump_files, read_grid_dump, integrate  # noqa: E402


# ----------------------------------------------------------------------
def run(cmd, cwd=None, dry=False):
    print("+ " + " ".join(str(c) for c in cmd) + (f"   (cwd={cwd})" if cwd else ""))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=cwd, check=True).returncode


def copy_if_exists(src, dst_dir, dry=False):
    if not os.path.exists(src):
        return False
    if not dry:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
    return True


def _nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def solps_run_status(run_dir):
    """(finished:bool, reason:str) for a SOLPS run directory.

    A finished run wrote a plasma solution (b2fplasma/b2fplasmf) AND a usable
    state (non-empty b2fstate, or a non-empty b2fstati fallback for the cosmetic
    0-byte b2fstate cleanup bug). A *missing* b2fplasma means the run never
    completed (e.g. interrupted mid-transient) -- distinct from the cosmetic
    empty-b2fstate case, and must NOT silently fall back to the initial state.
    """
    has_plasma = _nonempty(os.path.join(run_dir, "b2fplasma")) or \
        _nonempty(os.path.join(run_dir, "b2fplasmf"))
    if not has_plasma:
        return False, ("no b2fplasma/b2fplasmf -> SOLPS run did not complete "
                       "(interrupted before writing the converged solution)")
    if not (_nonempty(os.path.join(run_dir, "b2fstate")) or
            _nonempty(os.path.join(run_dir, "b2fstati"))):
        return False, "no usable b2fstate/b2fstati"
    return True, "ok"


def pick_b2fstate(run_dir):
    """Prefer b2fstate; fall back to b2fstati only if b2fstate is empty.

    Assumes solps_run_status() already confirmed the run finished, so an empty
    b2fstate here is the cosmetic touch-to-0 cleanup bug, not an unfinished run.
    """
    st = os.path.join(run_dir, "b2fstate")
    if _nonempty(st):
        return st
    sti = os.path.join(run_dir, "b2fstati")
    if _nonempty(sti):
        print(f"  note: {st} empty (cosmetic cleanup bug) -> using {sti}")
        return sti
    sys.exit(f"ERROR: no usable b2fstate/b2fstati in {run_dir}")


def compute_int_source(out_dir):
    """(int Sp dV [Li/s], int Qe dV [W], step) from the last source_SpSe dump."""
    files = source_dump_files(out_dir)
    if not files:
        return None, None, None
    step, df = read_grid_dump(files[-1])
    sp, qe = integrate(df)
    return sp, qe, step


def converged(history, rel_tol, k):
    """True when the last k consecutive int-Sp changes are each <= rel_tol."""
    sp = [h["int_Sp"] for h in history if h["int_Sp"]]
    if len(sp) < k + 1:
        return False
    for a, b in zip(sp[-(k + 1):-1], sp[-k:]):
        if a == 0 or abs(b - a) / abs(b) > rel_tol:
            return False
    return True


# ----------------------------------------------------------------------
def build_plasma(cfg, coupled, base, run_dir, plasma_out, dry=False):
    pb = cfg.get("plasma_build", {})

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    converter = pb.get("converter",
                       os.path.join(HERE, "..", "converters", "convert_solps_plasma.py"))
    converter = os.path.abspath(converter)
    b2fgmtry = absp(pb.get("b2fgmtry", "solps/baserun/b2fgmtry"))
    equ_file = absp(pb.get("equ_file", "solps/baserun/dg.equ"))
    b2fstate = pick_b2fstate(run_dir)

    cmd = [sys.executable, converter, run_dir,
           "--b2fgmtry", b2fgmtry,
           "--b2fstate", b2fstate,
           "--equ-file", equ_file,
           "--plasma-out", plasma_out,
           "--geometry", pb.get("geometry", "cart"),
           "--heatflux", pb.get("heatflux", "total")]
    wall_in = pb.get("wall_in")
    if wall_in:
        wall_in = absp(wall_in)
        if os.path.exists(wall_in):
            cmd += ["--wall-in", wall_in]
        else:
            print(f"  warning: wall_in {wall_in} missing -> omitting --wall-in")
    print("\n== rebuild plasma (--heatflux total) ==")
    run(cmd, cwd=coupled, dry=dry)


def run_wlld(cfg, coupled, run_dir, dry=False):
    script = cfg.get("wlld_script",
                     os.path.join(coupled, "run_b2plot_wlld.sh"))
    print("\n== WLLD (b2plot wlld) ==")
    run([script, run_dir], cwd=coupled, dry=dry)


def build_tsurf(cfg_path, coupled, case, dry=False):
    print("\n== rebuild Tsurf surf (strip deck + bake) ==")
    cmd = [sys.executable, os.path.join(HERE, "build_tsurf_surf.py"),
           cfg_path, "--case", case]
    if dry:
        cmd.append("--dry")
    run(cmd, cwd=coupled, dry=False)        # build_tsurf honours --dry itself


# scratch / exe / redundant-input dirs and files skipped by the full archive
# (b2pl.exe.dir etc. are regenerable; *.i are the iteration's input re-copies)
_ARCH_SKIP = shutil.ignore_patterns("b2pl.exe.dir", "*.exe.dir", "*.i")


def _copytree_full(src, dst, dry=False):
    """Snapshot an entire run dir (symlinks kept as links, scratch dirs skipped)."""
    if not os.path.isdir(src):
        return False
    if not dry:
        shutil.copytree(src, dst, symlinks=True, ignore=_ARCH_SKIP,
                        dirs_exist_ok=True)
    return True


def archive_iteration(arch_dir, out_dir, solps_run_dir, strip_dir, base,
                      archive_state=False, archive_full=False, dry=False):
    """Snapshot this iteration's OE + SOLPS run into arch_dir.

    archive_full=True  -> copy the WHOLE OE deck dir (input script + output)
                          and the WHOLE SOLPS run dir (b2f*, fort.*, *.nc,
                          parameters, WLLD, ...), minus scratch/exe dirs.
    archive_full=False -> light snapshot (last OE source + log + strip CSVs;
                          SOLPS b2time/b2tallies/b2fmovie, state on request).
    """
    oe_dst = os.path.join(arch_dir, "openedge")
    solps_dst = os.path.join(arch_dir, "solps")

    if archive_full:
        # OpenEdge: whole deck dir (in.oedge_wall, *.species, log, output/) but
        # DROP the per-100-step source_SpSe.* intermediates (~8 MB each x ~100 =
        # ~0.8 GB/iter) — keep only the final source dump the loop consumes.
        oe_src = os.path.dirname(out_dir)
        if not dry and os.path.isdir(oe_src):
            oe_ignore = shutil.ignore_patterns("b2pl.exe.dir", "*.exe.dir",
                                               "*.i", "source_SpSe.*")
            shutil.copytree(oe_src, oe_dst, symlinks=True, ignore=oe_ignore,
                            dirs_exist_ok=True)
            files = source_dump_files(out_dir)
            if files:
                outsub = os.path.join(oe_dst, os.path.basename(out_dir))
                os.makedirs(outsub, exist_ok=True)
                shutil.copy2(files[-1], outsub)
        # SOLPS: the entire run directory
        _copytree_full(solps_run_dir, solps_dst, dry=dry)
        print(f"  archived (full) -> {arch_dir}")
        return

    # OpenEdge: last source dump + log + surf-flux dumps (light)
    files = source_dump_files(out_dir)
    if files:
        copy_if_exists(files[-1], oe_dst, dry=dry)
    copy_if_exists(os.path.join(os.path.dirname(out_dir), "log.openedge"), oe_dst, dry=dry)
    # strip CSVs that fed this iteration's surface state
    for side in ("outer", "inner"):
        copy_if_exists(os.path.join(strip_dir, f"{base}.strip.{side}.csv"), oe_dst, dry=dry)
    # SOLPS: light time-series diagnostics always; heavy state on request
    for name in ("b2time.nc", "b2tallies.nc", "b2fmovie"):
        copy_if_exists(os.path.join(solps_run_dir, name), solps_dst, dry=dry)
    if archive_state:
        for name in ("b2fstate", "b2fplasma", "b2fplasmf"):
            copy_if_exists(os.path.join(solps_run_dir, name), solps_dst, dry=dry)
    print(f"  archived -> {arch_dir}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--case", help="full case key (default: config['loop']['case'] or config['case'])")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    coupled = cfg["coupled_dir"]
    loop = cfg.get("loop", {})
    case = args.case or loop.get("case") or cfg.get("case", "attached__mist")
    base = case.split("__")[0]

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    cc = cfg["cases"][case]
    out_dir = absp(cc["oe_output_dir"])
    solps_run_dir = absp(cc["solps_run_dir"])
    plasma_out = absp(os.path.join("openedge", "plasma", f"plasma_{base}.h5"))
    strip_dir = absp(cfg.get("tsurf_leg", {}).get("strip_dir", "openedge/strip_csv"))

    max_iters = int(loop.get("max_iters", 8))
    rel_tol = float(loop.get("rel_tol", 0.05))
    kconsec = int(loop.get("consecutive", 2))
    archive_state = bool(loop.get("archive_state", False))
    archive_full = bool(loop.get("archive_full", False))
    check_conv = bool(loop.get("check_convergence", False))

    oneshot = os.path.join(HERE, "oneshot_driver.py")
    work = absp(os.path.join(f"loop_{case}"))
    conv_csv = os.path.join(work, "convergence.csv")

    print(f"== loop driver  case='{case}' (base='{base}') ==")
    print(f"  OE output dir : {out_dir}")
    print(f"  SOLPS run dir : {solps_run_dir}")
    print(f"  plasma (rebuilt each iter): {plasma_out}")
    print(f"  work / archive: {work}")
    if check_conv:
        print(f"  loop converge : rel_tol={rel_tol} consecutive={kconsec} max_iters={max_iters}")
    else:
        print(f"  loop converge : DISABLED (run all {max_iters} iters; int Sp recorded only)")
    print(f"  archive_state : {archive_state}  archive_full : {archive_full}")

    if not args.dry:
        os.makedirs(work, exist_ok=True)
        if not os.path.exists(conv_csv):
            with open(conv_csv, "w") as f:
                f.write("iter,step,int_Sp_Li_per_s,int_Qe_W,rel_change\n")

    def current_max_iters(default):
        # live re-read so the config file is the single source of truth:
        # editing loop.max_iters takes effect even while the loop is running.
        try:
            with open(cfg_path) as f:
                return int(json.load(f).get("loop", {}).get("max_iters", default))
        except (OSError, ValueError, TypeError, KeyError):
            return default

    history = []
    it = 0
    while True:
        max_iters = current_max_iters(max_iters)
        if it >= max_iters:
            print(f"\n*** reached max_iters={max_iters}. Stopping. ***")
            break
        print("\n" + "=" * 64)
        print(f"## ITERATION {it} / {max_iters}  (case={case})")
        print("=" * 64)

        # --- one full OE -> SOLPS leg (validated one-shot path) -----------
        run([sys.executable, "-u", oneshot, cfg_path, f"--case={case}"],
            cwd=coupled, dry=args.dry)

        # --- record int Sp from this iteration's OpenEdge source ----------
        if args.dry:
            sp, qe, step = 1.0, 1.0, 0
        else:
            sp, qe, step = compute_int_source(out_dir)
            if sp is None:
                sys.exit(f"ERROR: no source_SpSe.* in {out_dir} after iteration {it}")
        prev = history[-1]["int_Sp"] if history else None
        rel = abs(sp - prev) / abs(sp) if (prev and sp) else float("nan")
        history.append({"iter": it, "step": step, "int_Sp": sp, "int_Qe": qe, "rel": rel})
        print(f"\n  >> iter {it}: int Sp = {sp:.6e} Li/s   int Qe = {qe:.6e} W"
              f"   rel d = {rel:.4f}")
        if not args.dry:
            with open(conv_csv, "a") as f:
                f.write(f"{it},{step},{sp:.6e},{qe:.6e},{rel:.6e}\n")

        # --- archive both sides -------------------------------------------
        archive_iteration(os.path.join(work, f"iter_{it:02d}"), out_dir,
                          solps_run_dir, strip_dir, base,
                          archive_state=archive_state,
                          archive_full=archive_full, dry=args.dry)

        # --- guard: did this iteration's SOLPS run actually finish? --------
        # A missing b2fplasma means SOLPS was interrupted before converging;
        # rebuilding the plasma from b2fstati would silently feed the loop the
        # initial condition (wrong physics). Hard-stop instead.
        if not args.dry:
            ok, reason = solps_run_status(solps_run_dir)
            if not ok:
                sys.exit(f"\n*** ABORT at iteration {it}: SOLPS did not finish "
                         f"-- {reason}.\n    dir: {solps_run_dir}\n"
                         f"    int Sp for this iter is recorded; inspect "
                         f"b2time.nc / b2mn.prt and rerun SOLPS to completion "
                         f"before continuing. ***")

        # --- converged? (disabled by default; int Sp recorded regardless) --
        if check_conv and not args.dry and converged(history, rel_tol, kconsec):
            print(f"\n*** CONVERGED: int Sp within {rel_tol} for {kconsec} "
                  f"consecutive iterations (iter {it}). Stopping. ***")
            break
        # stop before the (expensive) rebuild if no further iteration will run
        if it + 1 >= current_max_iters(max_iters):
            print(f"\n*** reached max_iters without convergence. ***")
            break

        # --- rebuild SOLPS-derived inputs for the next OE run -------------
        build_plasma(cfg, coupled, base, solps_run_dir, plasma_out, dry=args.dry)
        run_wlld(cfg, coupled, solps_run_dir, dry=args.dry)
        build_tsurf(cfg_path, coupled, case, dry=args.dry)
        it += 1

    print("\n== loop summary ==")
    print(f"{'iter':>4} {'step':>9} {'int Sp [Li/s]':>16} {'int Qe [W]':>14} {'rel d':>8}")
    for h in history:
        print(f"{h['iter']:>4} {h['step']:>9} {h['int_Sp']:>16.6e} "
              f"{h['int_Qe']:>14.6e} {h['rel']:>8.4f}")
    if not args.dry:
        print(f"\nconvergence trace: {conv_csv}")
        print(f"per-iteration archive: {work}/iter_*/")


if __name__ == "__main__":
    main()
