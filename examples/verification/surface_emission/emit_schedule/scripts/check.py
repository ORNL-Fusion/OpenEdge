#!/usr/bin/env python3
"""Source accounting of fix surface/emit/source with the release schedule.
  A  emitted weight vs flux*area*time: launch modes, window, ranks, rebalance, restart
  B  weight emitted on each step of the window (elements with a fractional number of markers per window)
  C  source changing every step: one run vs split runs vs rebalance, both launch modes
SPA=<binary> (default ~/build_oe/src/spa_mac_mpi); KK=<kokkos binary> adds device runs.
LAUNCH=<launcher>, default "mpirun -np {np}"; a fixed launcher (no {np}) runs every case
on its rank count.  NP_LIST=<rank counts>, default "1 2 4" (the last is the base count)."""
import hashlib, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, ".."))
SPA = os.environ.get("SPA", os.path.expanduser("~/build_oe/src/spa_mac_mpi")).split()
KK = os.environ.get("KK", "").split()
LAUNCH = os.environ.get("LAUNCH", "mpirun -np {np}")
NPL = [int(x) for x in os.environ.get("NP_LIST", "1 2 4").split()]
NPB = NPL[-1]
ROOT = os.environ.get("OPENEDGE_ROOT", os.path.join(HERE, "../../../../.."))
os.makedirs("output", exist_ok=True)
bad = 0

def sha(path):
    try: return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError: return "unreadable"

def launch(deck, tag, np_=None, exe=None, timeout=None, **var):
    """run a deck; returns the log path, or None (output kept in output/<tag>.out)"""
    log, out = "output/log." + tag, "output/" + tag + ".out"
    cmd = LAUNCH.format(np=np_ or NPB).split() + (exe or SPA) + ["-in", deck, "-log", log]
    for k, v in var.items(): cmd += ["-var", k, str(v)]
    with open(out, "w") as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print("  FAIL %-26s timed out after %s s, see %s" % (tag, timeout, out))
            return None
    if rc != 0 or "ERROR" in open(out).read():
        print("  FAIL %-26s run failed (exit %d), see %s" % (tag, rc, out)); return None
    return log

def last(log, step):
    r = None
    for l in open(log):
        w = l.split()
        if len(w) == 4 and w[0] == str(step): r = (int(w[1]), float(w[3]))
    return r

def check(tag, expect, tol, deck="in.emit_schedule", step=700, **kw):
    global bad
    log = launch(deck, tag, **kw)
    r = last(log, step) if log else None
    ok = r is not None and abs(r[1] / expect - 1) <= tol
    bad += not ok
    if log: print("  %s %-26s markers %-7s weight/expected %s" %
                  ("ok  " if ok else "FAIL", tag, r and r[0], "none" if r is None else "%.8f" % (r[1] / expect)))
    return r[0] if r else None

print("binary %s sha256 %s" % (SPA[0], sha(SPA[0])))
print("database/processes.h5 sha256 %s" % sha(os.path.join(ROOT, "database/processes.h5")))

print("A  constant source, 700 steps")
E = 1.0e20 * 1.0e-9 * 700
for mode, tol in (("total", 1e-7), ("per", 1e-7), ("flux", 5e-3)):
    for ne in (1, 7):
        check("%s_nevery%d" % (mode, ne), E, tol, mode=mode, nemit=ne)
for mode in ("total", "per"):
    n = [check("%s_np%d" % (mode, k), E, 1e-7, np_=k, mode=mode) for k in NPL[:-1]]
    n.append(check("%s_rebalance" % mode, E, 1e-7, mode=mode, bal=1))
    if len(set(n)) != 1: bad += 1; print("  FAIL marker count depends on ranks/rebalance:", n)
    # restart at a window boundary is exact; inside a window the source is evaluated
    # again and the schedule redrawn: one marker more or less per element
    for nrs, tol in ((350, 1e-7), (353, 2e-3)):
        if launch("in.emit_schedule", "%s_rs%d_write" % (mode, nrs), mode=mode, rs=1, nrs=nrs):
            check("%s_restart%d" % (mode, nrs), E, tol, mode=mode, rs=2)
        else: bad += 1

print("B  weight by step within the window (must be 1/3 each)")
def phase(tag, exe=None, np_=NPL[0]):
    global bad
    log = launch("in.phase", tag, np_=np_, exe=exe)
    if not log: bad += 1; return
    a = np.array([l.split() for l in open(log) if re.match(r"^\s+\d+\s+\d+\s+\d+\s+\S+\s*$", l)], float)
    w, j = np.diff(a[:, 3]), (a[1:, 0].astype(int) - 1) % 3
    f = [w[j == k].sum() / w.sum() * 3 for k in range(3)]
    ok = max(abs(x - 1) for x in f) < 0.05 and abs(w.sum() / (1e20 * 1e-9 * (a[-1, 0] - a[0, 0])) - 1) < 0.01
    bad += not ok
    print("  %s %-26s %.3f %.3f %.3f" % ("ok  " if ok else "FAIL", tag, *f))
phase("phase_cpu")
if KK: phase("phase_kk", exe=KK + ["-k", "on", "-sf", "kk"])

print("C  source changing every step, windows of 7 steps, 21 steps")
E = 1.0e20 * 1.0e-9 * 7 * (1 + 8 + 15)     # value at the first step of each window
for mode in ("total", "per"):
    for src in ("flux", "conc"):
        n = [check("ev_%s_%s%s" % (mode, src, t), E, 1e-7, deck="in.evolve", step=21, mode=mode, src=src, **kw)
             for t, kw in (("", {}), ("_split", {"split": 1}), ("_rebal", {"bal": 1}), ("_np1", {"np_": NPL[0]}))]
        if len(set(n)) != 1: bad += 1; print("  FAIL marker count differs:", n)
        # the grid repartitioned between run segments, replicated and distributed surfs
        for t, kw in (("_repart", {"split": 2}), ("_dist", {"dist": 1}), ("_dist_split", {"dist": 1, "split": 1}),
                      ("_dist_repart", {"dist": 1, "split": 2})):
            check("ev_%s_%s%s" % (mode, src, t), E, 1e-7, deck="in.evolve", step=21, mode=mode, src=src, **kw)
if KK:
    check("kk_total_nevery7", 1.0e20 * 1.0e-9 * 700, 1e-7, exe=KK + ["-k", "on", "-sf", "kk"], mode="total", nemit=7)

print("D  distributed surfaces: empty local/ghost surface ranks")
if NPB < 2:
    print("  SKIP empty-rank coverage requires at least two MPI ranks")
else:
    for mode in ("total", "per"):
        for src in ("flux", "conc"):
            for repart in (0, 1):
                tag = "empty_%s_%s%s" % (mode, src, "_repart" if repart else "")
                check(tag, E, 1e-7, deck="in.empty_rank", step=21, mode=mode,
                      src=src, ranks=NPB, repart=repart, timeout=180)

print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
