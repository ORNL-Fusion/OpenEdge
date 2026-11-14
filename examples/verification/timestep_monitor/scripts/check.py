#!/usr/bin/env python3
"""fix dt/monitor against known values (uniform B and plasma, O -> O+ -> O2+).
SPA=<binary incl. Kokkos args>, LAUNCH (default "mpirun -np {np}"), NP_LIST (default "1 2 4")."""
import hashlib, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, ".."))
SPA = os.environ.get("SPA", os.path.expanduser("~/build_oe/src/spa_mac_mpi")).split()
LAUNCH = os.environ.get("LAUNCH", "mpirun -np {np}")
NPL = [int(x) for x in os.environ.get("NP_LIST", "1 2 4").split()]
ROOT = os.environ.get("OPENEDGE_ROOT", os.path.join(HERE, "../../.."))
os.makedirs("output", exist_ok=True)
Q, MASS, B, DT = 1.602176634e-19, 2.67619504e-26, 0.1147, 1.45e-8
OM = Q * B / MASS
bad = 0

def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError: return "unreadable"

def rate(sp):   # log-bilinear ADAS scd of charge state sp at Te = 12 eV, ne = 5e19 m^-3
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    h = h5py.File(os.path.join(ROOT, "database/processes.h5"), "r")["/volume/rates/scd/o"]
    k = RegularGridInterpolator((h["temperature"][:], h["density"][:]), h["coefficient"][sp])(
        [[math.log10(12.0), math.log10(5e13)]])[0]
    return 10 ** k * 1e-6 * 5e19

def run(tag, np_=None, deck="in.monitor", fail=None, **var):
    """returns (stats row, table rows {section: {species: [numbers]}}, text) or None;
    fail = error text the run must stop with (returns True/False)"""
    pre = "" if os.path.basename(os.getcwd()) == "output" else "output/"
    log, out = pre + "log." + tag, pre + tag + ".out"
    cmd = LAUNCH.format(np=np_ or NPL[-1]).split() + SPA + ["-in", deck, "-log", log]
    for k, v in var.items(): cmd += ["-var", k, str(v)]
    with open(out, "w") as f: rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(out).read()
    if fail: return fail in txt
    if rc != 0 or "ERROR" in txt:
        print("  FAIL %-22s run failed (exit %d), see %s" % (tag, rc, out)); return None
    L = open(log).read().split("\n")
    def num(l):
        try: return [float(x) for x in l.split()]
        except ValueError: return []
    row = [r for r in map(num, L) if len(r) >= 2][-1]      # last stats line
    tab, sec = {"gyro": {}, "chem": {}, "status": {}, "range": ""}, None
    for l in L:
        if l.startswith("Timestep monitor"):       # the last table of the log is kept
            tab = {"gyro": {}, "chem": {}, "status": {}, "range": l.split("steps")[1].strip(" :")}
        elif l.startswith("  Gyromotion"): sec = "gyro"
        elif l.startswith("  Reactions"): sec = "chem"
        elif sec and l.startswith("    status:"): tab["status"][sec] = l.split("status:")[1].strip()
        elif sec and re.match(r"^    O\S*\s+[-0-9]", l):
            w = l.split(); tab[sec][w[0]] = [float(x) if x != "-" else None for x in w[1:]]
    return row, tab, txt

def ok(cond, msg):
    global bad
    bad += not cond
    print("  %s %s" % ("ok  " if cond else "FAIL", msg))

def close(a, b, tol): return a is not None and abs(a / b - 1) <= tol

print("binary %s sha256 %s" % (SPA[0], sha(SPA[0])))
print("database/processes.h5 sha256 %s" % sha(os.path.join(ROOT, "database/processes.h5")))
NU0, NU1 = rate(0), rate(1)
print("Omega(O+) dt = %.6g   nu(O) dt = %.6g   nu(O+) dt = %.6g" % (OM * DT, NU0 * DT, NU1 * DT))

print("A  known maxima; exposure and updated weight against the averaged inventories")
for tag, var, sub, nchem in (("base", {}, 1, 1), ("sub2", {"sub": 2}, 2, 1), ("nchem3", {"nchem": 3, "nsteps": 3000}, 1, 3)):
    r = run(tag, **var)
    if not r: bad += 1; continue
    row, t, _ = r
    n = row[0]
    g, c = t["gyro"], t["chem"]
    ok(close(g["O+"][0], OM * DT / sub, 1e-3) and close(g["O2+"][0], 2 * OM * DT / sub, 1e-3),
       "%-7s max |q/m|B dt_sub: O+ %.5g, O2+ %.5g" % (tag, g["O+"][0], g["O2+"][0]))
    ok(close(c["O"][0], NU0 * DT * nchem, 5e-3) and close(c["O+"][0], NU1 * DT * nchem, 5e-3),
       "%-7s max rate x interval: O %.5g, O+ %.5g" % (tag, c["O"][0], c["O+"][0]))
    ok(g["O+"][1] == 0 and c["O"][1] == 0 and "below" in t["status"]["gyro"] and "below" in t["status"]["chem"],
       "%-7s nothing above the levels, status says so" % tag)
    # every update assesses the inventory present before it: exact for nchem = 1
    if nchem == 1:
        ok(close(c["O"][2], row[3] * n, 1e-6) and close(c["O+"][2], row[4] * n, 1e-6),
           "%-7s updated weight = sum of the inventories (O %.5g, O+ %.5g atoms)" % (tag, c["O"][2], c["O+"][2]))
    ok(close(g["O+"][2], row[4] * n * DT, 0.02) and close(g["O2+"][2], row[5] * n * DT, 0.02),
       "%-7s Boris exposure = inventory x time within 2 %% (O+ %.4g, O2+ %.4g atom s)" % (tag, g["O+"][2], g["O2+"][2]))
    ok(close(row[6], 2 * OM * DT / sub, 1e-3) and row[7] == 0 and close(row[8], NU0 * DT * nchem, 5e-3) and row[9] == 0,
       "%-7s stats vector of the last window" % tag)

print("B  level between the charge states: weighted fractions, markers of unequal weight")
r = run("gyrofrac", limit=0.015, nmon=2000)      # one window = the run
if r:
    row, t, txt = r
    g = t["gyro"]
    f = (g["O2+"][1] * g["O2+"][2]) / (g["O+"][2] + g["O2+"][2])
    ok(g["O+"][1] == 0 and g["O2+"][1] == 1, "O+ all below, O2+ all above the level 0.015")
    ok(close(f, row[5] / (row[4] + row[5]), 0.02), "exposure fraction above %.4f = O2+ share of the ion inventory %.4f" % (f, row[5] / (row[4] + row[5])))
    ok(close(row[7], f, 1e-3) and "EXCEEDED" in t["status"]["gyro"], "window fraction %.4f equals the run fraction, status EXCEEDED" % row[7])
    ok("orbit resolution limit exceeded" in txt and "advisory limit 0.015" in txt and "Increase global pusher subcycles" in txt and "gyromotion only" in txt,
       "warning at the first failing report: what exceeded, value and limit, remedy and its scope")
else: bad += 1
r = run("chemfrac", dt=12 * DT, nsteps=1000, nmon=1000)
if r:
    row, t, txt = r
    c = t["chem"]
    ok(close(c["O"][0], NU0 * 12 * DT, 5e-3) and c["O"][1] == 1 and c["O+"][1] == 0, "rate x interval %.4g for O above 0.1, O+ below" % c["O"][0])
    f = c["O"][2] / (c["O"][2] + c["O+"][2])
    ok(close(f, row[3] / (row[3] + row[4]), 1e-6), "updated-weight fraction above %.5f = O share of the inventory" % f)
    ok(close(row[9], f, 1e-6) and "EXCEEDED" in t["status"]["chem"], "window fraction %.5f equals the run fraction, status EXCEEDED" % row[9])
    ok("reaction hazard limit exceeded" in txt and "advisory limit 0.1" in txt and "Boris subcycles do not change it" in txt,
       "reaction warning: value and limit, its own remedy")
else: bad += 1

print("C  reporting windows, coverage, rank counts")
for tag, var, rng in (("partial", {"nsteps": 1500, "mon": 2}, "1-1500"), ("short", {"nsteps": 300, "mon": 2}, "1-300")):
    r = run(tag, **var)
    if not r: bad += 1; continue
    row, t, _ = r
    ok(t["range"] == rng and close(t["chem"]["O"][2], row[3] * row[0], 1e-6),
       "%-7s table covers steps %s including the partial window" % (tag, t["range"]))
r = run("nochem", chem=0)
if r:
    row, t, _ = r
    ok(row[6] == -1 and row[8] == -1 and "not assessed" in t["status"]["gyro"] and "not assessed" in t["status"]["chem"],
       "no chemistry, no ions: both criteria not assessed (vector -1), never a satisfied zero")
else: bad += 1
ref = None
for k in NPL:
    r = run("np%d" % k, np_=k)
    if not r: bad += 1; continue
    row, t, _ = r
    mx = (t["gyro"]["O+"][0], t["gyro"]["O2+"][0], t["chem"]["O"][0], t["chem"]["O+"][0])
    ref = ref or mx
    ok(mx == ref and close(t["chem"]["O"][2], row[3] * row[0], 1e-6), "%d rank(s): same maxima, updated weight consistent" % k)
r = run("emptyrank", np_=NPL[-1], nsteps=20, nmon=10)     # ions exist on few ranks this early
ok(r is not None, "short run with ranks that hold no ions completes")

print("D  the monitor consumes no random numbers")
a, b = run("rng_off", np_=NPL[0], mon=0), run("rng_on", np_=NPL[0], mon=2)
ok(a is not None and b is not None and a[0][:6] == b[0][:6], "same particles and inventories with and without the fix (%d rank)" % NPL[0])

print("E  one ion into the wall: exposure = weight x time travelled")
for tag, var, exp in (("wall_n100", {}, 2.5e-6), ("wall_n1", {"nmon": 1}, 2.5e-6),
                      ("wall_react", {"rx": 1}, 2.5e-6),      # the wall changes the species: incident ion counted
                      ("cross", {"z0": 0.255, "vz": -300000, "nsteps": 1}, 0.255 / 300000)):
    for k in sorted(set((NPL[0], NPL[-1]))):
        r = run("%s_np%d" % (tag, k), np_=k, deck="in.single", **var)
        e = r[1]["gyro"]["O+"][2] if r else None
        ok(r is not None and close(e, exp, 1e-6), "%-9s %d rank(s): %s atom s (exact %.6g)" % (tag, k, e, exp))

print("F  input order and instances")
open("output/in.before", "w").write(open("in.monitor").read().replace(
    'fix pw      particle/weight', 'fix dtm0 dt/monitor 10\nfix pw      particle/weight').replace(
    "../surface_emission", "../../surface_emission").replace("input/o.reactions", "../input/o.reactions"))
open("output/in.two", "w").write(open("in.single").read().replace(
    'fix dtm     dt/monitor ${nmon}', 'fix dtm     dt/monitor ${nmon}\nfix dtm2 dt/monitor 5').replace(
    "../surface_emission", "../../surface_emission"))
os.chdir("output")
ok(run("before", np_=NPL[0], deck="in.before", fail="must be defined after fix volume/chem/adas", mon=0) is True,
   "a monitor defined before the chemistry fix is rejected")
ok(run("two", np_=NPL[0], deck="in.two", fail="Only one fix dt/monitor") is True, "a second monitor is rejected")
os.chdir("..")
r = run("unfix", np_=NPL[0], deck="in.unfix")
if r:
    w = sum(v[2] for v in r[1]["chem"].values())
    ok(r[1]["range"] == "4-4" and close(w, 1.0, 1e-9),
       "monitor removed and defined again: last table covers step 4 with %g update (exact 1)" % w)
else: bad += 1

print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
