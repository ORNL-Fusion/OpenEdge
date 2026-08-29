#!/usr/bin/env bash
# surf_react surface/pwi deposit tagging + seeding verification.
#   (a) deposit_as off vs on: identical net ledger and strata thickness
#   (b) adens_init_file x scale + adens_init_group: exact seed, strata stack,
#       and zero runtime net/deposition/erosion ledger before and after an
#       otherwise inert synchronization
#   (c) yscale on the deposit material halves its erosion
#   (d) B-on-W compound deck still runs; target_like/yield_scale = 0.5 x cpmi
# Exits 0 on PASS, 1 on FAIL.

set -e
cd "$(dirname "$0")"
mkdir -p output

SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}
PYTHON=${PYTHON:-python3}   # needs numpy
run() {  # name deck [extra -var args]
  local name=$1 deck=$2; shift 2
  echo "=== $name"
  mpirun -np "$NP" "$SPA" -in "$deck" "$@" -log "output/log.$name" > "output/screen.$name" 2>&1 \
    || { echo "run $name failed:"; tail -5 "output/screen.$name"; exit 1; }
}

run untagged in.tag -var case untagged -var tag 0
run tagged   in.tag -var case tagged   -var tag 1
run split    in.tag -var case split    -var tag 1 -var recycle input/w_split.recycle
run seed     in.seed -var seedfile output/tagged.2000.dump -var scale 1000.0
if [ "${SKIP_BW:-0}" != 1 ]; then
  run bw in.bw_smoke
fi

${PYTHON:-python3} - <<'EOF'
import glob, os, sys
import numpy as np

N_W = 6.306e28
failed = False

def dump(path):
    lines = open(path).read().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("ITEM: SURFS"))
    cols = lines[h].split()[2:]
    n = int(lines[lines.index("ITEM: NUMBER OF SURFS") + 1])
    data = np.loadtxt(lines[h+1:h+1+n], ndmin=2)
    order = np.argsort(data[:, cols.index("id")])
    return cols, data[order]

def col(cols, data, name):
    return data[:, cols.index(name)]

def strata(cols, data, nmat):
    """per surf: list of (thickness, density, comp[nmat]) surface->bulk"""
    out = []
    for row in data:
        v = [row[cols.index(f"s_adens_strata[{k}]")] for k in range(1, len(cols)) if f"s_adens_strata[{k}]" in cols]
        nl = int(v[3]); m = 4; layers = []
        for _ in range(nl):
            layers.append((v[m], v[m+1], v[m+2:m+2+nmat])); m += 2 + nmat
        out.append(layers)
    return out

def check(ok, msg):
    global failed
    print(("  ok   " if ok else "  FAIL ") + msg)
    failed |= not ok

# ---- (a) untagged vs tagged -------------------------------------------
cu, du = dump("output/untagged.2000.dump")
ct, dt = dump("output/tagged.2000.dump")
net_u, net_t = col(cu, du, "s_adens_net"), col(ct, dt, "s_adens_net")
print(f"(a) net untagged {net_u}  tagged {net_t}  atoms/m^2")
check(np.allclose(net_u, net_t, rtol=1e-9, atol=0), "net ledger identical with deposit_as")
check(np.allclose(col(cu, du, "s_adens_ero"), col(ct, dt, "s_adens_ero"), rtol=1e-9),
      "gross erosion identical with deposit_as")
w_t, wd_t = col(ct, dt, "s_adens[1]"), col(ct, dt, "s_adens[2]")
# bulk W is eroded only while the reaction zone (rzone) is still filling
# with Wd (first ~35 steps at rzone 1e17); after that the debit goes to Wd
check(np.all(wd_t > 0) and np.all(np.abs(w_t) < 0.10 * wd_t),
      f"tagged: deposit in Wd column ({wd_t[0]:.3e}), bulk W debit only transient ({w_t[0]:.3e})")
check(np.allclose(col(ct, dt, "s_adens_conc[2]"), 1.0), "tagged: exposed concentration is Wd at the end")
th_u = [sum(l[0] for l in s) for s in strata(cu, du, 2)]
th_t = [sum(l[0] for l in s) for s in strata(ct, dt, 2)]
check(np.allclose(th_u, th_t, rtol=1e-9), "strata total thickness identical")
top = strata(ct, dt, 2)[0][0]
check(top[2][1] > 0.999, f"tagged: top stratum is pure Wd (c_Wd={top[2][1]:.4f})")

# ---- (c) yscale 0.5 on the deposit material ----------------------------
cs, ds = dump("output/split.2000.dump")
ratio = col(cs, ds, "s_adens_ero") / col(ct, dt, "s_adens_ero")
print(f"(c) gross erosion ratio split/tagged = {ratio}")
check(np.all((ratio > 0.45) & (ratio < 0.62)), "deposit yscale 0.5 halves erosion (0.45..0.62 with early bulk phase)")

# ---- (b) seeded layer --------------------------------------------------
cS, dS = dump("output/seed.0.dump")
cS1, dS1 = dump("output/seed.1.dump")
seed = np.maximum(net_t, 0.0) * 1000.0
wd_s, w_s = col(cS, dS, "s_adens[2]"), col(cS, dS, "s_adens[1]")
print(f"(b) seed expected {seed}  got Wd {wd_s}  W {w_s}")
check(np.isclose(wd_s[0], seed[0], rtol=1e-12) and wd_s[1] == 0.0,
      "adens_init_file x scale applied to surf 1 only (adens_init_group)")
check(np.isclose(w_s[0], 5e17) and w_s[1] == 0.0, "uniform adens_init restricted to the group")
for name in ("s_adens_net", "s_adens_dep", "s_adens_ero"):
    check(np.all(col(cS, dS, name) == 0.0) and np.all(col(cS1, dS1, name) == 0.0),
          f"initial coating excluded from runtime {name} ledger across first sync")
check(np.allclose(col(cS1, dS1, "s_adens[1]"), w_s, rtol=0, atol=0) and
      np.allclose(col(cS1, dS1, "s_adens[2]"), wd_s, rtol=0, atol=0),
      "inert first sync preserves seeded material inventory")
st = strata(cS, dS, 2)
# the uniform W layer merges into the W substrate (same material); the Wd
# seed stays a separate top stratum
check(len(st[0]) == 2 and len(st[1]) == 1, f"strata layers: surf1 {len(st[0])} (Wd on bulk), surf2 {len(st[1])}")
if len(st[0]) >= 2:
    top, sub = st[0][0], st[0][1]
    check(np.isclose(top[0] * top[1], seed[0], rtol=2e-5) and top[2][1] > 0.999,   # dump prints 6 digits
          f"top stratum = seed/n_W ({top[0]:.3e} m, c_Wd={top[2][1]:.3f}, dens {top[1]:.3e})")
    check(np.isclose(sub[0], 1.0e-3 + 5e17 / N_W, rtol=1e-6) and sub[2][0] > 0.999,
          "substrate stratum absorbed the uniform W layer")

# ---- (d) compound B-on-W with target_like / yield_scale -----------------
if os.path.exists("output/bw.500.dump"):
    cb, db = dump("output/bw.500.dump")
    cp, cwd, cpb = col(cb, db, "c_cpmi"), col(cb, db, "c_cpmiWd"), col(cb, db, "c_cpmiB")
    sel = cp > 0
    check(sel.sum() > 0 and (cpb > 0).sum() > 0 and np.allclose(cwd, 0.5 * cp, rtol=2e-5),
          f"target Wd target_like W yield_scale 0.5 == 0.5 x cpmi ({sel.sum()} surfs with W erosion, {(cpb > 0).sum()} with B)")
    check(np.isfinite(col(cb, db, "s_adens_net")).all(), "B-on-W compound run completed with finite ledger")
else:
    print("  skip (d): SKIP_BW=1")

print("PASS" if not failed else "FAIL")
sys.exit(1 if failed else 0)
EOF
