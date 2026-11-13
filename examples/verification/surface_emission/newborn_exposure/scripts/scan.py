#!/usr/bin/env python3
"""Neutral transmission through a slab vs timestep and emission cadence.
Exact: T = 2*E3(nu*d/v).  nu comes from in.decay (log.decay)."""
import hashlib, math, os, re, subprocess, sys
import numpy as np
from scipy.special import expn

SPA = os.environ.get("SPA", os.path.expanduser("~/build_oe/src/spa_mac_mpi"))
KK = os.environ.get("KK", "")          # Kokkos binary: adds device emission + chemistry rows
QUICK = "--quick" in sys.argv          # two timesteps, one seed (regression)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def sha(path):
    try: return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError: return "unreadable"

def mpirun(args, out):
    """run; a failed run stops the scan with its output kept in <out>"""
    with open(out, "w") as f: rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or "ERROR" in open(out).read():
        print("FAIL: run failed (exit %d), see %s" % (rc, out)); sys.exit(1)
NP = os.environ.get("NP", "4")
LAUNCH = os.environ.get("LAUNCH", "mpirun -np {np}").format(np=NP).split()   # or a fixed launcher
D, EV, MASS = 0.011275, 5.0, 2.67619504e-26
SEEDS = (12345, 2222, 98765)

def rate():
    rows = [l.split() for l in open("log.decay")
            if re.match(r"^\s+\d+\s+\d+\s+\d+\s*$", l)]
    a = np.array(rows, float)
    dt = 1e-8   # in.decay default
    return -np.polyfit(a[:, 0] * dt, np.log(a[:, 1]), 1)[0]

def run(dt, nemit, seed, d=D, exe=None, extra=()):
    ntrans = int(math.ceil(8.0 * max(1.0, d / D) / (NU * dt) / nemit)) * nemit
    log = "output/log.%g_%d_%d_%g%s" % (dt, nemit, seed, d, "_kk" if exe else "")
    mpirun(LAUNCH + (exe or SPA.split()) + ["-in", "in.newborn", "-log", log,
            "-var", "d", "%g" % d, "-var", "dt", "%g" % dt, "-var", "nemit", str(nemit),
            "-var", "ntrans", str(ntrans), "-var", "seed", str(seed)] + list(extra),
           log + ".out")
    ntot = 6 * ntrans
    for l in open(log):
        w = l.split()
        if len(w) == 5 and w[0] == str(ntot):
            return float(w[3]) * ntot / float(w[2])   # neutral exits / emitted
    raise RuntimeError("no result in " + log)

os.makedirs("output", exist_ok=True)
if QUICK: SEEDS = SEEDS[:1]
mpirun(LAUNCH + SPA.split() + ["-in", "in.decay", "-log", "log.decay", "-var", "dt", "1e-8"],
       "output/decay.out")
NU = rate()
try:    # independent reference: log-bilinear scd(O0) at Te = 12 eV, ne = 5e19 m^-3
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    root = os.environ.get("OPENEDGE_ROOT", os.path.join(os.path.dirname(__file__), "../../../../.."))
    h = h5py.File(os.path.join(root, "database/processes.h5"), "r")["/volume/rates/scd/o"]
    k = RegularGridInterpolator((h["temperature"][:], h["density"][:]), h["coefficient"][0])(
        [[math.log10(12.0), math.log10(5e13)]])[0]
    nutab = 10 ** k * 1e-6 * 5e19
    print("nu from the table = %.4g /s (decay fit / table = %.4f)" % (nutab, NU / nutab))
    if abs(NU / nutab - 1) > 0.01: raise SystemExit("FAIL: decay rate differs from the table")
except ImportError:
    print("h5py not available: table cross-check skipped")
v = math.sqrt(2 * EV * 1.602176634e-19 / MASS)
x = NU * D / v
exact = 2 * expn(3, x)
ROOT = os.environ.get("OPENEDGE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
print("binary %s sha256 %s  ranks %s" % (SPA, sha(SPA.split()[0]), NP))
if KK: print("kokkos binary %s sha256 %s" % (KK, sha(KK.split()[0])))
print("database/processes.h5 sha256 %s" % sha(os.path.join(ROOT, "database/processes.h5")))
print("nu = %.4g /s   v = %.5g m/s   nu*d/v = %.4f   exact T = %.5f" % (NU, v, x, exact))
print("%8s %8s %6s %9s %9s %8s" % ("nu*dt", "nu*Temit", "nemit", "T", "+-", "T/exact"))
bad = 0
def err(t): return t.std(ddof=1) / math.sqrt(len(t)) if len(t) > 1 else float("nan")
for nudt in ((0.2, 0.05) if QUICK else (0.2, 0.1, 0.05, 0.02, 0.01)):
    dt = nudt / NU
    for nut in (None, 0.4, 1.0):
        nemit = 1 if nut is None else max(1, round(nut / nudt))
        if nut is not None and nemit == 1: continue
        if QUICK and nut == 0.4: continue
        t = np.array([run(dt, nemit, s) for s in SEEDS])
        print("%8.3f %8.3f %6d %9.5f %9.5f %8.4f" %
              (nudt, nemit * nudt, nemit, t.mean(), err(t), t.mean() / exact), flush=True)
        if abs(t.mean() / exact - 1) > 0.01: bad += 1

# un-ionized fraction at other depths = distribution of first-ionization positions.
# Reported for convergence with the timestep; gated at the finer step only.
print("%8s %8s %6s %9s %9s %8s" % ("nu*dt", "depth/mfp", "nemit", "T", "+-", "T/exact"))
for nudt in (() if QUICK else (0.2, 0.05)):
    for f in (0.25, 0.5, 2.0):
        nemit = max(1, round(1.0 / nudt))
        ex = 2 * expn(3, f * x)
        t = np.array([run(nudt / NU, nemit, s, f * D) for s in SEEDS])
        print("%8.3f %8.2f %6d %9.5f %9.5f %8.4f" %
              (nudt, f, nemit, t.mean(), err(t), t.mean() / ex), flush=True)
        if nudt == 0.05 and abs(t.mean() / ex - 1) > 0.02: bad += 1

if KK:   # device emission (nlaunch_total) with device chemistry; the log must show it active
    print("Kokkos: device emission + chemistry, nu*dt = 0.1")
    kk = KK.split() + ["-k", "on", "-sf", "kk"]
    for nemit, nlt in ((1, 3750), (10, 37500)):
        t = np.array([run(0.1 / NU, nemit, s, exe=kk, extra=["-var", "launch", "total", "-var", "nlt", str(nlt),
                                                             "-var", "nxy", "1"]) for s in SEEDS])
        out = open("output/log.%g_%d_%d_%g_kk.out" % (0.1 / NU, nemit, SEEDS[-1], D)).read()
        dev = "device emission ACTIVE" in out
        print("%8.3f %8.3f %6d %9.5f %9.5f %8.4f  %s" % (0.1, nemit * 0.1, nemit, t.mean(), err(t), t.mean() / exact,
              "device" if dev else "HOST emission"), flush=True)
        if abs(t.mean() / exact - 1) > 0.01 or not dev: bad += 1
# gate: every row (emission every step and batched) within 1 % of the exact value
print("PASS" if bad == 0 else "FAIL")
sys.exit(1 if bad else 0)
