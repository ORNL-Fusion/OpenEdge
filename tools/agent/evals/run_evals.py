#!/usr/bin/env python3
"""Robustness suite for openedge-agent's deterministic layer.

  A. seeded faults: a small base case is built in a temporary directory, each fault
     is applied to a copy, and preflight must report the expected (check, level);
     the clean base must pass;
  B. clean decks: every tracked verification deck must give no error, except the
     listed ones (with the reason);
  C. scrub: shipped files must not contain private strings (patterns in
     $OPENEDGE_AGENT_SCRUB, one regex per line, or the defaults below).

  python3 evals/run_evals.py [--binary PATH] [--root OPENEDGE_ROOT]
Exit 0 when every item passes.  Static: no simulation is run.
"""
import argparse
import contextlib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from openedge_agent import core, paths  # noqa: E402

BASE = """\
variable    dt      index 1.0e-8
variable    nsteps  index 2000
variable    nmon    index 1000
seed        12345
dimension   3
boundary    r r o
global      gridcut 0.0 comm/sort no
global      fnum 1.0
timestep    ${dt}
create_box  0 1 0 1 0 0.5
create_grid 4 4 8
balance_grid rcb cell
species     input/o.species O O+ O2+
mixture     src O frac 1.0 nrho 1.0e19 temp 300.0 vstream 0.0 0.0 0.0
read_surf   input/patch.surf group source particle check
surf_collide vac vanish
surf_react  pwi surface/pwi input/o.recycle twall 773.15 adens_nevery 100
surf_modify source collide vac react pwi
fix pw      particle/weight
fix bg      background constant te 12.0 ne 5e19 bcart 0.1147 0.0 0.0
global      pusher plasma bg mode boris subcycles 1
fix femit   surface/emit/source src source constant 4.0e10 perspecies no normal yes nlaunch_total 20 model thompson 2.0 emax 50
fix fchem   volume/chem/adas 1 O input/o.reactions
fix dtm     dt/monitor ${nmon}
stats       ${nsteps}
run         ${nsteps}
"""
RECYCLE = "O --> O\nT o_on_w\nO --> O\nS O_on_W\n"
INPUTS = {"o.species": "examples/verification/surface_emission/newborn_exposure/input/o.species",
          "patch.surf": "examples/verification/surface_emission/prompt_redeposition/input/patch.surf",
          "o.reactions": "examples/verification/timestep_monitor/input/o.reactions"}


def sub(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


# name: (deck transform, recycle transform, preflight kwargs, env, setup, expected [(check, level)])
FAULTS = {
    "missing_reflection": (None, lambda r: sub(r, "T o_on_w", "T o_on_zz"), {}, {}, None,
                           [("database", "error")]),
    "sputter_analytic_fallback": (None, lambda r: sub(r, "S O_on_W", "S Kr_on_B"), {}, {}, None,
                                  [("database", "warn")]),
    "sputter_unknown_pair": (None, lambda r: sub(r, "S O_on_W", "S O_on_Zz"), {}, {}, None,
                             [("database", "error")]),
    "runtime_variable": (lambda d: sub(d, "run         ${nsteps}", "variable nr equal step+100\nrun         ${nr}"),
                         None, {}, {}, None, [("parse", "unknown")]),
    "monitor_before_chem": (lambda d: sub(sub(d, "fix dtm     dt/monitor ${nmon}\n", ""),
                                          "fix fchem", "fix dtm     dt/monitor ${nmon}\nfix fchem"),
                            None, {}, {}, None, [("dt_monitor", "error")]),
    "unrescaled_steps": (lambda d: sub(d, "emax 50", "emax 50 nevery 10"), None,
                         {"var": ["dt=5e-9"], "reference_var": ["dt=1e-8"]}, {}, None,
                         [("step_settings", "warn")]),
    "missing_input": (lambda d: sub(d, "input/o.species", "input/o_typo.species"), None, {}, {}, None,
                      [("files", "error")]),
    "style_typo": (lambda d: sub(d, "volume/chem/adas", "volume/chem/adass"), None, {}, {}, None,
                   [("styles", "error")]),
    "wrong_database": (None, None, {}, {"OPENEDGE_ROOT": "/nonexistent/openedge"}, None,
                       [("database", "error")]),
    "output_overwrite": (lambda d: sub(d, "stats ", "dump d1 particle all 100 output/part.*.dump id x y z\nstats "),
                         None, {}, {}, "output/part.100.dump", [("outputs", "warn")]),
    "no_run": (lambda d: sub(d, "run         ${nsteps}\n", ""), None, {}, {}, None, [("runs", "error")]),
    "unresolved_if": (lambda d: sub(d, "stats ", 'if "${undefined_var} > 0" then "fix pr print 10 hi"\nstats '),
                      None, {}, {}, None, [("parse", "unknown")]),
    "adens_default_mpi": (lambda d: sub(d, " adens_nevery 100", ""), None, {"np": 8}, {}, None,
                          [("adens_nevery", "warn")]),
    "long_run_no_monitor": (lambda d: sub(sub(d, "fix dtm     dt/monitor ${nmon}\n", ""),
                                          "nsteps  index 2000", "nsteps  index 2000000"),
                            None, {}, {}, None, [("dt_monitor", "warn"), ("dt_pair", "warn")]),
    "missing_reactions": (lambda d: sub(d, "volume/chem/adas 1 O input/o.reactions",
                                        "volume/chem/adas 1 Zz auto"), None, {}, {}, None,
                          [("database", "error")]),
}

# tracked decks that are expected to fail preflight on their own
EXPECTED_FAIL = {
    "examples/verification/kokkos_parity/in.parity": "reads state.restart written by in.make_state",
    "examples/verification/kokkos_parity/in.parity_efield": "reads state.restart written by in.make_state",
    "examples/verification/kokkos_parity/in.parity_pop": "reads state.restart written by in.make_state",
    "examples/verification/kokkos_parity/in.parity_write": "reads state.restart written by in.make_state",
    "examples/verification/kokkos_parity/in.parity_surf": "reads state_surf.restart written by in.make_state_surf",
}

# tracked decks whose inputs a script generates (gitignored): error only before it has run
GENERATED = {
    "examples/verification/particulates/dustt2005/uniform_plasma_benchmark/in.grain": "run.sh makes input/",
}

SCRUB_DEFAULT = [r"/Users/", r"/home/\w+", r"/scratch/", r"\bpscratch\b", r"@\w+\.(gov|edu|com)"]


@contextlib.contextmanager
def env(extra):
    old = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_base(root, where):
    os.makedirs(os.path.join(where, "input"))
    for name, src in INPUTS.items():
        shutil.copy(os.path.join(root, src), os.path.join(where, "input", name))
    with open(os.path.join(where, "in.base"), "w") as f:
        f.write(BASE)
    with open(os.path.join(where, "input", "o.recycle"), "w") as f:
        f.write(RECYCLE)


def suite_faults(root, binary, tmp):
    rows, ok = [], True
    base = os.path.join(tmp, "base")
    build_base(root, base)
    pf = core.preflight(os.path.join(base, "in.base"), binary=binary, root=root)
    good = pf["verdict"] == "pass"
    rows.append(("A", "clean base", "pass", pf["verdict"], good))
    ok &= good
    for name, (dt, rt, kw, ev, touch, expect) in FAULTS.items():
        case = os.path.join(tmp, name)
        shutil.copytree(base, case)
        if dt:
            with open(os.path.join(case, "in.base")) as f:
                d = dt(f.read())
            with open(os.path.join(case, "in.base"), "w") as f:
                f.write(d)
        if rt:
            p = os.path.join(case, "input", "o.recycle")
            with open(p) as f:
                r = rt(f.read())
            with open(p, "w") as f:
                f.write(r)
        if touch:
            os.makedirs(os.path.dirname(os.path.join(case, touch)), exist_ok=True)
            open(os.path.join(case, touch), "w").close()
        with env(ev):
            pf = core.preflight(os.path.join(case, "in.base"), binary=binary, root=root, **kw)
        got = {(r["check"], r["level"]) for r in pf["results"]}
        miss = [e for e in expect if e not in got]
        rows.append(("A", name, " ".join("%s:%s" % e for e in expect),
                     "caught" if not miss else "MISSED %s (verdict %s)" % (miss, pf["verdict"]), not miss))
        ok &= not miss
    return rows, ok


def suite_clean(root, binary):
    rows, ok = [], True
    files = subprocess.run(["git", "-C", root, "ls-files", "examples/verification"],
                           capture_output=True, text=True).stdout.split()
    decks = [f for f in files if os.path.basename(f).startswith("in.")
             and not f.endswith((".inc", ".py", ".sh"))]
    for d in sorted(decks):
        pf = core.preflight(os.path.join(root, d), binary=binary, root=root)
        errs = [r for r in pf["results"] if r["level"] == "error"]
        if d in GENERATED:
            rows.append(("B", d, "any (%s)" % GENERATED[d], pf["verdict"], True))
            continue
        if d in EXPECTED_FAIL:
            good = bool(errs)
            rows.append(("B", d, "error (%s)" % EXPECTED_FAIL[d], pf["verdict"], good))
        else:
            good = not errs
            rows.append(("B", d, "no error", "ok" if good else
                         "ERROR: " + "; ".join(r["message"][:100] for r in errs[:2]), good))
        ok &= good
    return rows, ok


def suite_scrub():
    pats = SCRUB_DEFAULT
    extra = os.environ.get("OPENEDGE_AGENT_SCRUB")
    if extra and os.path.isfile(extra):
        with open(extra) as f:
            pats = pats + [ln.strip() for ln in f if ln.strip()]
    agent = paths.AGENT_DIR
    rows, ok = [], True
    for fp in sorted(glob.glob(os.path.join(agent, "**", "*"), recursive=True)):
        if not os.path.isfile(fp) or "__pycache__" in fp or "/.venv/" in fp:
            continue
        with open(fp, errors="replace") as f:
            text = f.read()
        hits = [p for p in pats if re.search(p, text) and not (fp.endswith("run_evals.py") and p in SCRUB_DEFAULT)]
        if hits:
            rows.append(("C", os.path.relpath(fp, agent), "no private strings", "FOUND %s" % hits, False))
            ok = False
    if ok:
        rows.append(("C", "shipped files", "no private strings", "clean", True))
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", help="binary, to locate its database")
    ap.add_argument("--root")
    ap.add_argument("--only", choices=["A", "B", "C"])
    a = ap.parse_args()
    root = paths.openedge_root(a.root)
    if not root:
        sys.exit("OpenEdge source tree not found (--root)")
    all_rows, all_ok = [], True
    with tempfile.TemporaryDirectory(prefix="openedge-agent-evals-") as tmp:
        os.environ.setdefault("OPENEDGE_AGENT_HOME", os.path.join(tmp, "home"))
        for key, fn in (("A", lambda: suite_faults(root, a.binary, tmp)),
                        ("B", lambda: suite_clean(root, a.binary)), ("C", suite_scrub)):
            if a.only and a.only != key:
                continue
            rows, ok = fn()
            all_rows += rows
            all_ok &= ok
    for s, name, exp, got, good in all_rows:
        print("%s %-4s %-58s %-34s %s" % (s, "ok" if good else "FAIL", name[:58], exp[:34], got))
    n_bad = sum(not r[4] for r in all_rows)
    print("\n%d items, %d failed" % (len(all_rows), n_bad))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
