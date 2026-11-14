#!/usr/bin/env python3
"""Failure messages of the input readers: what failed -> which file -> what to do.
Each case must stop (non-zero exit, no hang) and print the listed text.
SPA=<binary incl. Kokkos args>, LAUNCH (default "mpirun -np {np}"), NP_LIST (default "1 2")."""
import os, shutil, stat, subprocess, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "..")
SPA = os.environ.get("SPA", os.path.expanduser("~/build_oe/src/spa_mac_mpi")).split()
LAUNCH = os.environ.get("LAUNCH", "mpirun -np {np}")
NPL = [int(x) for x in os.environ.get("NP_LIST", "1 2").split()]
ROOT = os.path.abspath(os.environ.get("OPENEDGE_ROOT", os.path.join(HERE, "../../../..")))
W = os.path.join(CASE, "output")
if os.path.isdir(os.path.join(W, "root_denied/database")): os.chmod(os.path.join(W, "root_denied/database"), stat.S_IRWXU)
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
bad = 0

def plasma(name, swap=False, rank2=False, zones=False, userblock=0):
    r, z = np.linspace(0.5, 1.5, 6), np.linspace(-1.0, 1.0, 9)
    with h5py.File(os.path.join(W, name), "w", userblock_size=userblock) as f:
        if zones: f.attrs["format"] = "openedge_background3d"
        f["r"] = np.tile(r, (2, 1)) if rank2 else r
        f["z"] = z
        for k, v in (("dens_e", 5e19), ("temp_e", 12.0), ("dens_i", 5e19), ("temp_i", 12.0)):
            f[k] = np.full((6, 9) if swap else (9, 6), v)

def equ(name, npsi=12, drop_jm=False):
    with open(os.path.join(W, name), "w") as f:
        if not drop_jm: f.write("jm = 4\n")
        f.write("km = 3\npsib = 0.0\nbtf = 1.0\nrtf = 1.0\nr(1:jm);\n")
        f.write("0.5 0.8 1.1 1.4\nz(1:km);\n-1 0 1\npsi(j,k)\n")
        f.write(" ".join("0.1" for _ in range(npsi)) + "\n")

def run(tag, np_, env=None, cwd=None, deck="in.fail", **var):
    out = os.path.join(W, "%s_np%d.out" % (tag, np_))
    cmd = LAUNCH.format(np=np_).split() + SPA + ["-in", os.path.join(CASE, deck) if cwd else deck,
                                                  "-log", os.path.join(W, "log." + tag)]
    if deck == "in.fail": var["mode"] = 2 if "equ" in var else (1 if "plasma" in var else 0)
    for k, v in var.items(): cmd += ["-var", k] + str(v).split()
    e = dict(os.environ); e["OPENEDGE_ROOT"] = ROOT
    if env: e.update(env)
    try:
        with open(out, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd or CASE, env=e, timeout=120).returncode
    except subprocess.TimeoutExpired:
        return None, open(out).read()
    return rc, open(out).read()

def case(tag, want, ok_run=False, forbid=(), **kw):
    """the run stops (or completes, ok_run), its output holds every text of want and none of forbid"""
    global bad
    for np_ in NPL:
        rc, txt = run(tag, np_, **kw)
        miss = [w for w in want if w not in txt]
        seen = [w for w in forbid if w in txt]
        good = rc is not None and (rc == 0) == ok_run and not miss and not seen
        bad += not good
        why = "HANG (120 s)" if rc is None else ("exit %d" % rc) + ("" if not miss else ", missing text: %r" % miss[0]) + \
              ("" if not seen else ", unwanted text: %r" % seen[0])
        print("  %s %-22s %d rank(s)%s" % ("ok  " if good else "FAIL", tag, np_, "" if good else "  " + why))

plasma("userblock.h5", userblock=512)
plasma("good.h5"); plasma("swap.h5", swap=True); plasma("rank.h5", rank2=True); plasma("zones.h5", zones=True)
open(os.path.join(W, "text.h5"), "w").write("not an HDF5 file\n")
shutil.copy(os.path.join(W, "good.h5"), os.path.join(W, "locked.h5")); os.chmod(os.path.join(W, "locked.h5"), 0)
equ("good.equ"); equ("short.equ", npsi=7); equ("nojm.equ", drop_jm=True)
P = lambda n: "output/" + n

print("A  plasma file of fix background")
case("load_summary", ["[background] bg loaded: plasma file output/good.h5 (resolved to", "(format legacy)", "magnetic field: none", "database: "], ok_run=True, plasma=P("good.h5"))
case("plasma_userblock", ["[background] bg loaded: plasma file output/userblock.h5"], ok_run=True, plasma=P("userblock.h5"))   # valid HDF5, signature after a user block
case("plasma_missing", ["fix background bg: Background plasma file not found: output/nope.h5 (resolved to", "working directory", "Correct fix background file PATH, or copy the file"], plasma=P("nope.h5"))
# launched from another directory: the relative path is resolved there, and the message says so
case("plasma_wrong_cwd", ["Background plasma file not found: output/good.h5 (resolved to %s/output/good.h5; working directory %s)" % (os.path.realpath(W), os.path.realpath(W))],
     plasma=P("good.h5"), cwd=os.path.realpath(W), sp="../../surface_emission/newborn_exposure/input/o.species")
case("plasma_locked", ["Background plasma file cannot be read (permission denied)", "Make it readable"], plasma=P("locked.h5"))
case("plasma_not_hdf5", ["Background plasma file is not a valid HDF5 file: output/text.h5", "Regenerate it"], plasma=P("text.h5"))
case("plasma_dims", ["dataset /dens_e is rank 2 (6 x 9), expected 9 x 6 (the lengths of /z and /r)", "Reconvert the input"], plasma=P("swap.h5"))
case("plasma_rank", ["dataset /r is rank 2 (2 x 6), expected rank 1"], plasma=P("rank.h5"))
case("plasma_is_zones3d", ["is a zones3d background", "Add 'format zones3d'"], plasma=P("zones.h5"))
case("legacy_as_zones3d", ["not a zones3d background", "Remove 'format zones3d'", "[plasma file output/good.h5"],
     plasma=P("good.h5"), zones=1)

print("B  equilibrium file")
case("equ_missing", ["fix background bg: Equilibrium file not found: output/nope.equ", "Correct fix background equilibrium PATH"], plasma=P("good.h5"), equ=P("nope.equ"))
case("equ_short", ["equilibrium file output/short.equ", "field psi(j,k) (jm x km): expected 12 value(s), found 7", "regenerate the file in the supported format"], plasma=P("good.h5"), equ=P("short.equ"))
case("equ_no_jm", ["grid size 'jm = N' (at least 2): expected 1 value(s), found 0"], plasma=P("good.h5"), equ=P("nojm.equ"))

print("C  database/processes.h5")
for d in ("empty", "locked", "text"):
    os.makedirs(os.path.join(W, "root_" + d, "database"))
open(os.path.join(W, "root_text/database/processes.h5"), "w").write("not an HDF5 file\n")
shutil.copy(os.path.join(W, "text.h5"), os.path.join(W, "root_locked/database/processes.h5")); os.chmod(os.path.join(W, "root_locked/database/processes.h5"), 0)
R = lambda d: {"OPENEDGE_ROOT": os.path.join(W, "root_" + d)}
case("db_missing", ["OpenEdge database file not found:", "root_empty/database/processes.h5 (database directory from OPENEDGE_ROOT)", "Set OPENEDGE_ROOT to the OpenEdge directory"], chem=1, env=R("empty"))
case("db_locked", ["OpenEdge database file cannot be read (permission denied)", "Make it readable"], chem=1, env=R("locked"))
case("db_not_hdf5", ["OpenEdge database file is not a valid HDF5 file", "Restore database/processes.h5 from the repository"], chem=1, env=R("text"))
case("db_no_element", ["No reactions for element 'Ta'", "/volume/reactions/<element>/catalog in", "Use an element the database holds"], chem=1, elem="Ta")

print("D  tables of surf_react surface/pwi: only an absent table may fall back")
def table_db(name, yshape=(2, 2), axes=("E", "theta"), eshape=(2,)):
    os.makedirs(os.path.join(W, "root_" + name, "database"))
    with h5py.File(os.path.join(W, "root_" + name, "database/processes.h5"), "w") as f:
        g = f.create_group("surface/sputter/w_on_w")
        for a in axes: g[a] = np.array([10.0, 100.0]).reshape(eshape) if a == "E" else np.array([0.0, 60.0])
        g["Y"] = np.full(yshape, 0.1)
table_db("sput_good"); table_db("sput_shape", yshape=(2, 3)); table_db("sput_axis", axes=("theta",))
table_db("sput_erank", eshape=(2, 1))
os.makedirs(os.path.join(W, "root_dir/database/processes.h5"))          # a directory of that name
os.makedirs(os.path.join(W, "root_denied/database"))
shutil.copy(os.path.join(W, "root_sput_good/database/processes.h5"), os.path.join(W, "root_denied/database/processes.h5"))
os.chmod(os.path.join(W, "root_denied/database"), 0)                      # its directory denies access
FB = ["ANALYTIC"]      # the fallback warning must not appear
case("sput_good", ["loaded sputter yield table 'w_on_w' (2 x 2)"], ok_run=True, forbid=FB, deck="in.pwi", env=R("sput_good"))
case("sput_db_not_hdf5", ["OpenEdge database file is not a valid HDF5 file", "root_text/database/processes.h5"], forbid=FB, deck="in.pwi", env=R("text"))
case("sput_db_locked", ["OpenEdge database file cannot be read (permission denied)"], forbid=FB, deck="in.pwi", env=R("locked"))
case("sput_dims", ["surf_react surface/pwi pwi: sputter table 'w_on_w'", "incompatible table dimensions: dataset /surface/sputter/w_on_w/Y is 2 x 3, expected 2 x 2", "Regenerate the table"], forbid=FB, deck="in.pwi", env=R("sput_shape"))
case("sput_no_axis", ["an incomplete table: dataset /surface/sputter/w_on_w/E is missing"], forbid=FB, deck="in.pwi", env=R("sput_axis"))
case("sput_db_is_dir", ["OpenEdge database path is not a file", "root_dir/database/processes.h5"], forbid=FB, deck="in.pwi", env=R("dir"))
case("sput_db_denied", ["OpenEdge database file cannot be read (permission denied on its directory)", "Make it readable"], forbid=FB, deck="in.pwi", env=R("denied"))
case("sput_axis_rank", ["incompatible table dimensions: dataset /surface/sputter/w_on_w/E is rank 2 (2 x 1), expected rank 1 (a vector)"], forbid=FB, deck="in.pwi", env=R("sput_erank"))
case("refl_missing", ["surf_react surface/pwi pwi: reflection table", "not found: dataset group /surface/reflection/w_on_w in", "Use a projectile/target pair the database holds"], deck="in.pwi", env=R("sput_good"), rc="input/reflection.recycle")

os.chmod(os.path.join(W, "root_denied/database"), stat.S_IRWXU)
os.chmod(os.path.join(W, "locked.h5"), stat.S_IRUSR | stat.S_IWUSR)
os.chmod(os.path.join(W, "root_locked/database/processes.h5"), stat.S_IRUSR | stat.S_IWUSR)
print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
