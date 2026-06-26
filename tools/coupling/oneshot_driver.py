#!/usr/bin/env python3
"""One-shot OpenEdge -> SOLPS-ITER driver (mode B, log-driven).

Run OpenEdge once until the volume-integrated ionization source plateaus
(read live from log.openedge c_Sp_int), stop it, post-process the final
source_SpSe.* into SOLPS external source files (sna0 + she0) and activate
the b2mn.dat flag, then launch SOLPS to termination. No back-and-forth.

    python3 oneshot_driver.py config.json [--dry]

--dry  resolve paths + print the plan; do not launch anything.

Convergence is judged from the log scalar (cheap, finely sampled), NOT by
re-reading the big grid dumps; the grid dumps are used only for the final
spatial field handed to SOLPS.
"""
import sys
import os
import glob
import json
import time
import signal
import shutil
import subprocess


# ----------------------------------------------------------------------
# Droplet emission rate -> per-leg per-step count
# ----------------------------------------------------------------------
def droplet_n_per_leg(de, dt):
    """Per-leg droplet count per emit step for the deck's `n ${ndrop}`.

    Each macro-droplet = 1 physical droplet (fix_droplet_evaporate.cpp:302),
    so physical rate R = 2*n_per_leg / (nevery*dt). Precedence:
      enabled=false                 -> 0   (no_mist cases)
      explicit n_per_leg            -> that integer (feasibility cap escape hatch)
      rate_per_s (Table 4)          -> round(rate * nevery * dt / 2)
    """
    if not de.get("enabled", True):
        return 0
    if de.get("n_per_leg") is not None:
        return int(de["n_per_leg"])
    rate = float(de.get("rate_per_s", 0.0))
    nevery = int(de.get("nevery", 1))
    return max(0, round(rate * nevery * dt / 2.0))


# ----------------------------------------------------------------------
# Log parsing + plateau test  (reused by tests; no side effects)
# ----------------------------------------------------------------------
def parse_log_series(log_path):
    """Extract (steps, Sp, Qe) from SPARTA stats table.

    Finds the header row starting with 'Step' and auto-detects the Sp and
    Se/Qe columns by name (matches c_Sp_int/c_cSp_tot and c_Qe_int/c_cSe_tot
    alike), so it works regardless of the deck's exact compute/reduce names.
    """
    steps, sp, qe = [], [], []
    if not os.path.exists(log_path):
        return steps, sp, qe
    cols = sp_key = qe_key = None
    with open(log_path) as f:
        for line in f:
            # skip a partially-flushed final line (log is parsed live while
            # SPARTA is mid-write); else a truncated number like "-3.5500"
            # from "-3.5506596e+09" parses with the right token count and
            # corrupts the last Sp/Qe sample.
            if not line.endswith("\n"):
                continue
            toks = line.split()
            if toks[:1] == ["Step"] and any("Sp" in t for t in toks):
                cols = toks
                sp_key = next((t for t in toks if "Sp" in t), None)
                qe_key = next((t for t in toks if "Qe" in t or "Se" in t), None)
                continue
            if cols is None:
                continue
            if len(toks) != len(cols):
                cols = None       # table ended
                continue
            try:
                rec = dict(zip(cols, [float(x) for x in toks]))
            except ValueError:
                cols = None
                continue
            steps.append(int(rec["Step"]))
            sp.append(rec[sp_key] if sp_key else 0.0)
            qe.append(rec[qe_key] if qe_key else 0.0)
    return steps, sp, qe


def plateaued(values, rel_tol, window):
    """True when the last `window` nonzero samples agree within rel_tol."""
    tail = [v for v in values[-window:] if v != 0.0]
    if len(tail) < window:
        return False
    ref = tail[-1]
    if ref == 0.0:
        return False
    return (max(tail) - min(tail)) / abs(ref) <= rel_tol


# ----------------------------------------------------------------------
def main():
    dry = "--dry" in sys.argv
    no_solps = "--no-solps" in sys.argv
    case_override = next((a.split("=", 1)[1] for a in sys.argv[1:]
                          if a.startswith("--case=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    cfg = json.load(open(args[0]))

    coupled = cfg["coupled_dir"]
    case = case_override or cfg.get("case", "attached")
    cc = cfg["cases"][case]

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(coupled, p)

    deck = absp(cc["oe_deck"])
    deck_dir = os.path.dirname(deck)
    out_dir = absp(cc["oe_output_dir"])
    solps_run_dir = absp(cc["solps_run_dir"])
    binary = cfg["openedge_binary"]
    np_oe = int(cfg.get("mpi_np_openedge", 4))

    conv = cfg.get("oe_convergence", {})
    rel_tol = conv.get("rel_tol", 0.02)
    window = conv.get("window", 3)
    poll = conv.get("poll_seconds", 15)
    max_steps = conv.get("max_steps")     # None -> use deck's Nrun default

    log_path = os.path.join(deck_dir, "log.openedge")
    here = os.path.dirname(os.path.abspath(__file__))
    postproc = os.path.join(here, "source_spse_to_solps.py")

    import shlex
    mpirun = cfg.get("mpirun", "mpirun")
    setvars = cfg.get("oneapi_setvars")   # source so mpirun matches the linked MPI
    inner = [mpirun, "-np", str(np_oe), binary, "-in", os.path.basename(deck),
             "-log", log_path]
    if max_steps:
        inner += ["-var", "Nrun", str(max_steps)]

    # droplet emission rate -> per-leg per-step count (deck uses `n ${ndrop}`).
    # Each emitted macro-droplet = 1 physical droplet, so the physical rate is
    #   R = n_total / (nevery * dt) = 2*n_per_leg / (nevery*dt)   [droplets/s]
    # => n_per_leg = round(rate_per_s * nevery * dt / 2).  enabled=false -> 0.
    de = cfg.get("droplet_emit")
    if de is not None:
        ndrop = droplet_n_per_leg(de, cfg.get("dt_oe", 5e-6))
        inner += ["-var", "ndrop", str(ndrop)]

    if setvars:
        inner_str = " ".join(shlex.quote(a) for a in inner)
        cmd = ["bash", "-c",
               f"source {shlex.quote(setvars)} --force >/dev/null 2>&1 && "
               f"exec {inner_str}"]
    else:
        cmd = inner

    print(f"== one-shot driver  case='{case}' ==")
    print(f"  OpenEdge deck : {deck}")
    print(f"  OE output     : {out_dir}")
    print(f"  SOLPS run dir : {solps_run_dir}")
    print(f"  convergence   : rel_tol={rel_tol} window={window} "
          f"poll={poll}s max_steps={max_steps or 'deck default'}")
    if setvars:
        print(f"  oneAPI        : source {setvars}")
    print(f"  launch        : {' '.join(inner)}")
    if dry:
        print("\n--dry: nothing launched.")
        return

    # fresh outputs/log
    for p in glob.glob(os.path.join(out_dir, "source_SpSe.*")):
        os.remove(p)
    if os.path.exists(log_path):
        os.remove(log_path)

    # 1. launch OpenEdge (own process group so we can stop the whole job)
    proc = subprocess.Popen(cmd, cwd=deck_dir, start_new_session=True)

    # 2. watch the log for the plateau
    converged = False
    try:
        while proc.poll() is None:
            time.sleep(poll)
            steps, sp, qe = parse_log_series(log_path)
            if steps:
                print(f"  step {steps[-1]:>7d}: int Sp={sp[-1]:.4e} Li/s  "
                      f"int Qe={qe[-1]:.4e} W")
                if plateaued(sp, rel_tol, window):
                    print(f"  -> plateau (last {window} within {rel_tol}); "
                          f"stopping OpenEdge")
                    converged = True
                    break
        # 3. stop OpenEdge if still running
        stopped_by_us = False
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            stopped_by_us = True
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise

    # An OpenEdge that exited on its own with a nonzero code crashed (segfault,
    # MPI abort, bad deck). Its last source_SpSe dump is partial/garbage, so
    # abort instead of post-processing it -- exiting nonzero stops the coupling
    # via loop_driver's check=True. A code we caused (SIGTERM on plateau) is
    # negative and flagged by stopped_by_us, so it is not treated as a crash.
    rc = proc.returncode
    if not stopped_by_us and rc not in (0, None):
        sys.exit(f"\n*** ABORT: OpenEdge exited with code {rc} before a plateau "
                 f"-- treating as a crash; not post-processing a partial source.\n"
                 f"    inspect {log_path} ***")

    if not converged:
        print("  WARNING: OpenEdge ran to max_steps without a plateau "
              "(clean exit). Post-processing the last dump anyway.")

    if no_solps:
        print("\n--no-solps: OpenEdge done; skipping post-process + SOLPS.")
        return

    # 4. post-process -> SOLPS source files + b2mn flags
    print("\n== post-processing source_SpSe -> SOLPS ==")
    subprocess.run([sys.executable, postproc, args[0], f"--case={case}"],
                   check=True)

    # 5. launch SOLPS to termination (script takes <run_dir> [nprocs])
    script = cfg.get("solps_run_script")
    np_solps = int(cfg.get("mpi_np_solps", 1))
    if script and os.path.exists(script):
        # restart from the latest converged state, not the flat/stale b2fstati:
        # SOLPS reads b2fstati and writes b2fstate, so seed b2fstati <- b2fstate
        # (iter0: staged standalone state; iter>=1: previous iter's output).
        b2state = os.path.join(solps_run_dir, "b2fstate")
        b2stati = os.path.join(solps_run_dir, "b2fstati")
        if os.path.exists(b2state) and os.path.getsize(b2state) > 10000:
            shutil.copy2(b2state, b2stati)
            print(f"\n== RESTART: copying converged b2fstate -> b2fstati ==")
            print(f"   {b2state}")
            print(f"   -> {b2stati}  ({os.path.getsize(b2state)} bytes)")
        else:
            print(f"\n== RESTART: no usable b2fstate in {solps_run_dir}; "
                  f"SOLPS will start from existing b2fstati (flat/initial) ==")
        print(f"\n== launching SOLPS in {solps_run_dir} (np={np_solps}) ==")
        sp_rc = subprocess.run([script, solps_run_dir, str(np_solps)],
                               check=False).returncode
        # SOLPS's csh/Makefile wrapper swallows exit codes ("Error 1 (ignored)"),
        # so rc alone is unreliable -- confirm a plasma solution was written.
        # loop_driver.solps_run_status() does the hard stop (after archiving);
        # here we just warn loudly so a standalone one-shot run is not silent.
        wrote_plasma = any(
            os.path.exists(os.path.join(solps_run_dir, n)) and
            os.path.getsize(os.path.join(solps_run_dir, n)) > 0
            for n in ("b2fplasma", "b2fplasmf"))
        if not wrote_plasma:
            print(f"  WARNING: SOLPS wrote no b2fplasma/b2fplasmf (rc={sp_rc}) "
                  f"-- run likely did not complete; inspect b2mn.prt.")
    else:
        print(f"\nSOLPS source staged in {solps_run_dir}; launch SOLPS there "
              f"(no solps_run_script set).")


if __name__ == "__main__":
    main()
