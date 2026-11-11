#!/usr/bin/env python3
"""fix core/transport opt-in options: coefficient convention and solver selection.

Uses the SPARC smoke deck and inputs in examples/wip/sparc (not part of the regression).
    python scripts/check_fix_options.py --exe /path/to/spa [--np 4]
Checks: defaults unchanged, unit-factor equivalence, physical = input x factor (exactly once),
rejection of missing/invalid/short geometry, export round trip through make_exported_pulse.py,
solver dispatch and partial-window step count, and returned inventory / charge with each solver.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
case = here.parents[2] / "wip" / "sparc"
ap = argparse.ArgumentParser(); ap.add_argument("--exe", required=True); ap.add_argument("--np", type=int, default=4)
a = ap.parse_args()
out = here.parent / "output" / "options"; out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(here)); import make_exported_pulse as mep
fails = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg); fails.append(not ok)

base = (case / "in.core_smoke").read_text()
assert "output ${outdir}/core_1d.dat 1000" in base
def deck(name, extra, steps, nseed=100):
    d = base.replace("output ${outdir}/core_1d.dat 1000", "output ${outdir}/core_1d.dat 1000 export ${outdir}/core_inputs.json " + extra)
    d = re.sub(r"create_particles\s+seedW n \d+", f"create_particles    seedW n {nseed}", d)
    d = re.sub(r"^run .*$", f"run {steps}", d, flags=re.M)
    p = case / f"in.options_{name}"; p.write_text(d); return p
def run(name, extra, steps=0, nseed=100, np_=1):
    p = deck(name, extra, steps, nseed); o = out / name
    r = subprocess.run(["mpirun", "-np", str(np_), a.exe, "-in", p.name, "-var", "outdir", str(o), "-log", str(out / f"{name}.log")],
                       cwd=case, capture_output=True, text=True)
    p.unlink()
    log = (out / f"{name}.log").read_text() if (out / f"{name}.log").exists() else ""
    return r.returncode, log, o
def load(o): return json.loads((o / "core_inputs.json").read_text())

# geometry files: real, unit factors, and three bad ones
g = np.loadtxt(case / "input/core_geometry.dat")
hdr = "axis_extrapolated_below_rho_N 0.282843\nrho_N r_vol g1 g2"
unit = g.copy(); unit[:, 2:] = 1.0; np.savetxt(out / "geo_unit.dat", unit, header=hdr)
short = g[g[:, 0] < 0.3]; np.savetxt(out / "geo_short.dat", short, header=hdr)
neg = g.copy(); neg[3, 3] = -1.0; np.savetxt(out / "geo_neg.dat", neg, header=hdr)
other = g.copy(); other[:, 1] *= 1.2; np.savetxt(out / "geo_other.dat", other, header=hdr)

rc, log, o0 = run("default", "")
d0 = load(o0)
check(rc == 0 and d0["coefficients"] == "effective" and d0["solver"] == "substep", "defaults: effective coefficients, substep solver")
check(all(x == 1.0 for x in d0["g1"] + d0["g2"]) and d0["diff"][:d0["ngrid"]] == d0["diff_input"], "defaults: unit factors, coefficients used as given")
rc, log, o1 = run("unit", f"coefficients physical geometry {out/'geo_unit.dat'}")
d1 = load(o1)
check(rc == 0 and d1["diff"] == d0["diff"] and d1["conv"] == d0["conv"] and d1["srate"] == d0["srate"], "physical with unit factors == effective (identical arrays)")
rc, log, o2 = run("physical", "coefficients physical geometry input/core_geometry.dat")
d2 = load(o2); nr = d2["ngrid"]
rho = np.sqrt(np.array(d2["psiN"])); g2x = np.interp(rho, g[:, 0], g[:, 3]); g1x = np.interp(rho, g[:, 0], g[:, 2])
check(rc == 0 and np.allclose(d2["g2"], g2x, rtol=1e-12) and np.allclose(d2["g1"], g1x, rtol=1e-12), "physical: per-cell factors interpolated from the file")
ok = all(d2["diff"][z*nr+i] == d2["g2"][i]*d2["diff_input"][i] for z in range(d2["nstates"]) for i in range(nr))
check(ok and d2["diff_input"] == d0["diff_input"] and max(d2["g2"]) < 1.2 and min(d2["g2"]) > 1.0,
      f"physical: D_eff = g2 x D_perp exactly once (g2 {min(d2['g2']):.4f}..{max(d2['g2']):.4f})")
check(abs(d2["geometry_axis_extrapolated_below_rho"] - 0.282843) < 1e-6, "export carries the axis-extrapolation range of the geometry file")
for name, f, why in (("missing", "nofile.dat", "cannot open"), ("short", out / "geo_short.dat", "does not reach"),
                     ("neg", out / "geo_neg.dat", "invalid geometry"), ("other", out / "geo_other.dat", "differs from")):
    rc, log, _ = run("bad_" + name, f"coefficients physical geometry {f}")
    check(rc != 0 and why in log, f"rejected: {name} geometry ({why})")
# export round trip
for name, o in (("default", o0), ("physical", o2)):
    try: mep.make_case(o / "core_inputs.json", out / f"case_{name}.dat"); ok = True
    except Exception as e: ok = False; print("   ", e)
    check(ok, f"export round trip through make_exported_pulse.py ({name})")
bad = dict(d2); bad["diff"] = list(d2["diff"]); bad["diff"][5] *= 1.0001; (out / "tampered.json").write_text(json.dumps(bad))
try: mep.make_case(out / "tampered.json", out / "case_t.dat"); ok = False
except ValueError: ok = True
check(ok, "converter rejects effective coefficients that are not input x factor")

# solver dispatch, partial windows, returned inventory and charge with each solver
def summary(o):
    blk = (o / "core_1d.dat").read_text().split("# step")[-1].splitlines()
    h = blk[0].split(); d = dict(zip(h[1::2], map(float, h[2::2])))
    Z = np.array([[float(x) for x in l.split()[1:]] for l in blk if l.startswith("Z ")])
    rows = np.array([[float(x) for x in l.split()] for l in blk[2:] if l and l[0].isdigit()])
    return d, Z, rows
res = {}
for name, extra in (("substep", "solver substep"), ("split", "solver split max_dt 2.5e-7")):
    rc, log, o = run("solve_" + name, extra, steps=1010, nseed=200000, np_=a.np)   # 1010: last window is 10 steps
    check(rc == 0 and "ERROR" not in log, f"{name}: runs ({a.np} ranks)")
    res[name] = summary(o); res[name + "_json"] = load(o)
ds, Zs, rs = res["substep"]; dp, Zp, rp = res["split"]
dt = 2e-7
check(res["split_json"]["solver"] == "split" and res["split_json"]["split_max_dt"] == 2.5e-7, "export records solver and max_dt")
check(abs(dp["dt_window"] - 10*dt) < 1e-15 and int(dp["substeps"]) == 8, f"partial window {dp['dt_window']:.2e} s -> ceil(2e-6/2.5e-7) = 8 split steps (got {int(dp['substeps'])})")
check(int(ds["substeps"]) > 8, f"substep solver unchanged ({int(ds['substeps'])} substeps in the same window)")
check(rs[:, 7:].min() >= 0 and rp[:, 7:].min() >= 0, "non-negative densities with both solvers")
for d, nm in ((ds, "substep"), (dp, "split")):
    led = abs(d["Ncore"] - (d["Gin_total"] - d["Gout_total"])) / d["Gin_total"]
    check(led < 1e-6, f"{nm}: N_core = Gin - Gout to print precision ({led:.1e})")
e_ret = abs(dp["Gout_total"]/dp["Gin_total"] - ds["Gout_total"]/ds["Gin_total"])
tot_s, tot_p = rs[:, 7:].sum(0), rp[:, 7:].sum(0)
zc = lambda t: (t*np.arange(t.size)).sum()/t.sum()
check(e_ret < 0.01, f"returned fraction: substep {ds['Gout_total']/ds['Gin_total']:.4f}, split {dp['Gout_total']/dp['Gin_total']:.4f} (diff {e_ret:.4f})")
check(abs(zc(tot_s) - zc(tot_p)) < 0.1, f"core mean charge: substep {zc(tot_s):.3f}, split {zc(tot_p):.3f}")
print("PASS" if not any(fails) else "FAIL"); sys.exit(1 if any(fails) else 0)
