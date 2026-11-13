#!/usr/bin/env python3
"""Prompt redeposition of normally launched O in a uniform B parallel to the wall:
exact redeposited fraction f = 1 - exp(-nu/Omega). Scans the timestep (nu*dt =
Omega*dt) with emission every step and batched; f is measured from the growth of
the O+ inventory per emitted atom. The coarse steps show the splitting error
(ionization at the end of a step, Boris phase); the gate is on the fine steps.
Both launch modes are run: fnum markers and weighted nlaunch_total markers; f uses
the weighted O+ inventory and the emitted weight. With a Kokkos binary (-sf kk in
SPA) the weighted rows must run device emission.
SPA=<binary> NP=<ranks> LAUNCH=<launcher, default "mpirun -np {np}">; --quick = one
seed, two timesteps."""
import hashlib, math, os, subprocess, sys
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPA = os.environ.get("SPA", os.path.expanduser("~/build_oe/src/spa_mac_mpi")).split()
NP = os.environ.get("NP", "4")
LAUNCH = os.environ.get("LAUNCH", "mpirun -np {np}").format(np=NP).split()
FLUX, AREA = 4.0e10, 0.36
ROOT = os.environ.get("OPENEDGE_ROOT", "../../../..")
QUICK = "--quick" in sys.argv
SEEDS = (12345,) if QUICK else (12345, 2222, 98765)
B, MASS, Q = 0.1147, 2.67619504e-26, 1.602176634e-19
os.makedirs("output", exist_ok=True)

def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError: return "unreadable"

def table_rate():   # log-bilinear scd(O0) at Te = 12 eV, ne = 5e19 m^-3
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    h = h5py.File(os.path.join(ROOT, "database/processes.h5"), "r")["/volume/rates/scd/o"]
    k = RegularGridInterpolator((h["temperature"][:], h["density"][:]), h["coefficient"][0])(
        [[math.log10(12.0), math.log10(5e13)]])[0]
    return 10 ** k * 1e-6 * 5e19

def run(dt, nemit, seed, launch="flux"):
    ntrans = int(math.ceil(20.0 / (NU * dt) / nemit)) * nemit
    nlt = max(2, round(FLUX * AREA * dt * nemit / 40))      # 1/40 of the fnum marker count
    tag = "%g_%d_%d_%s" % (dt, nemit, seed, launch)
    out = "output/%s.out" % tag
    with open(out, "w") as f:
        rc = subprocess.run(LAUNCH + SPA + ["-in", "in.redep", "-log", "output/log." + tag,
                             "-var", "dt", "%g" % dt, "-var", "nemit", str(nemit), "-var", "ntrans", str(ntrans),
                             "-var", "seed", str(seed), "-var", "launch", launch, "-var", "nlt", str(nlt)],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or "ERROR" in open(out).read():
        print("FAIL: run failed (exit %d), see %s" % (rc, out)); sys.exit(1)
    rows = [l.split() for l in open("output/log." + tag) if len(l.split()) == 5 and l.split()[0].isdigit()]
    ion0 = float([r for r in rows if int(r[0]) == ntrans][-1][4])
    last = [r for r in rows if int(r[0]) == 3 * ntrans][-1]
    # emitted weight in the tally window: exact for weighted markers, counted for fnum markers
    wemit = FLUX * AREA * dt * 2 * ntrans if launch == "total" else float(last[2])
    if launch == "total" and "-sf" in SPA and "device emission ACTIVE" not in open(out).read():
        print("FAIL: weighted run did not use device emission, see %s" % out); sys.exit(1)
    return 1.0 - (float(last[4]) - ion0) / wemit

NU = table_rate()
OM = Q * B / MASS
exact = 1.0 - math.exp(-NU / OM)
print("binary %s sha256 %s  ranks %s" % (SPA[0], sha(SPA[0]), NP))
print("database/processes.h5 sha256 %s" % sha(os.path.join(ROOT, "database/processes.h5")))
print("nu = %.4g /s (table)  Omega = %.4g rad/s  exact f = %.5f" % (NU, OM, exact))
print("%8s %8s %6s %6s %9s %9s %8s" % ("nu*dt", "nu*Temit", "nemit", "launch", "f", "+-", "f/exact"))
bad = 0
for nudt in ((0.2, 0.02) if QUICK else (0.2, 0.1, 0.05, 0.02, 0.01)):
    for nut, launch in ((None, "flux"), (1.0, "flux"), (None, "total"), (1.0, "total")):
        nemit = 1 if nut is None else max(1, round(nut / nudt))
        f = np.array([run(nudt / NU, nemit, s, launch) for s in SEEDS])
        e = f.std(ddof=1) / math.sqrt(len(f)) if len(f) > 1 else float("nan")
        print("%8.3f %8.3f %6d %6s %9.5f %9.5f %8.4f" % (nudt, nemit * nudt, nemit, launch, f.mean(), e, f.mean() / exact), flush=True)
        if nudt <= 0.02 and abs(f.mean() / exact - 1) > (0.01 if launch == "flux" else 0.02): bad += 1
print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
