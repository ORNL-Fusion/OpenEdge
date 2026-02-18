"""
validate_drag.py — Regression test for fix_drag B-field coordinate mapping
                   and cylindrical geometric terms (DUSTT-style axisymmetric).

Tests 4 cases against an exact Python replication of the SPARTA integrator:

  Case 1: Ni=0 → nuE=0 → pure gravity           (reference)
  Case 2: Bz_cyl=1T → drag in Z toward Vpar     (physics check)
  Case 3: Bt=1T (toroidal) → no RZ drag          (KEY regression for the swap bug)
  Case 4: Bt=1T + cylindrical yes → centrifugal  (validates new geometric terms)

Key assertions:
  Case 3 vZ at T=0.001 s must be ≈ −0.01 m/s (gravity dominated),
          NOT ≈ +83 m/s (what the old bug produced by applying Bt to the Z-velocity slot).
  Case 4 vR at T=0.001 s must be > 0 (centrifugal pushes outward).

Usage:
    # run the simulation first:
    spa_mpi -in in.drag_unit
    # then validate:
    python validate_drag.py
"""
import numpy as np
import sys
from pathlib import Path

# ── Physical constants (match SPARTA/OpenEdge values) ─────────────────────────
PROTON_MASS = 1.6726218e-27   # kg
ECHARGE     = 1.6021766e-19   # C
PI          = np.pi

# ── Droplet / plasma parameters (must match in.drag_unit) ─────────────────────
A_BG        = 2.0             # deuterium background ion (amu)
RHO_D       = 534.0           # kg/m³ lithium
ALPHA_E     = 1.26            # Epstein accommodation
RD          = 1.0e-6          # m   droplet radius
MD          = 2.237e-15       # kg  droplet mass

TI_EV       = 100.0           # eV  ion temperature  (Cases 2, 3, 4)
NI          = 1.0e20          # m⁻³ ion density       (Cases 2, 3, 4)
VPAR        = 1000.0          # m/s parallel flow     (Cases 2, 3, 4)

G  = np.array([0.0, -9.8, 0.0])   # gravity vector (SPARTA x,y,z)
DT = 1.0e-6                        # timestep [s]
N  = 1000                          # number of steps
X0 = np.array([3.5, 0.0, 0.0])    # initial position (R=3.5, Z=0, phi=0)
V0 = np.array([0.0, 0.0, 0.0])    # initial velocity

# ── Analytical nuE (Epstein drag frequency) ────────────────────────────────────
def epstein_nu(Ni, Ti_eV, rd=RD, A_bg=A_BG, rho_d=RHO_D, alpha_E=ALPHA_E):
    if Ni <= 0.0 or Ti_eV <= 0.0 or rd <= 0.0:
        return 0.0
    mi      = A_bg * PROTON_MASS
    vth_i   = np.sqrt(8.0 * Ti_eV * ECHARGE / (PI * mi))
    rho_gas = Ni * mi
    return alpha_E * rho_gas * vth_i / (rho_d * rd)

# ── Exact half-kick integrator (replicates fix_drag::kick_half drag part) ─────
def half_kick_drag(v, upar, g, nu, dt_half):
    """Exact integration of dv/dt = -nu*(v-upar) + g over dt_half."""
    if nu > 0.0:
        s  = nu * dt_half
        ex = (1.0 - s + 0.5*s*s) if abs(s) < 1.0e-8 else np.exp(-s)
        return upar + (v - upar - g/nu)*ex + g/nu
    else:
        return v + g * dt_half

# ── Cylindrical geometric half-kick (replicates the new centrifugal correction) ─
def half_kick_cylindrical(v, x, dt_half):
    """
    Explicit half-step for cylindrical geometric terms:
      dv_R/dt  += v_φ² / R
      dv_φ/dt  += -v_φ * v_R / R
    Uses beginning-of-step velocities (explicit Euler).
    """
    R   = x[0]   # radial position
    vR  = v[0]   # v_R
    vph = v[2]   # v_φ (toroidal)
    v_new = v.copy()
    if R > 1.0e-10:
        v_new[0] += vph * vph / R * dt_half   # centrifugal
        v_new[2] -= vph * vR  / R * dt_half   # angular-momentum
    return v_new

# ── SPARTA leapfrog step: start_of_step → advect → end_of_step ───────────────
def sparta_step(x, v, upar, g, nu, dt, cylindrical=False):
    # Pre-advect half-kick
    v_half = half_kick_drag(v,      upar, g, nu, dt/2)
    if cylindrical:
        v_half = half_kick_cylindrical(v_half, x, dt/2)
    # Advect
    x_new  = x + v_half * dt
    # Post-advect half-kick
    v_new  = half_kick_drag(v_half, upar, g, nu, dt/2)
    if cylindrical:
        v_new = half_kick_cylindrical(v_new, x_new, dt/2)
    return x_new, v_new

# ── Reference trajectory (Python integrator) ──────────────────────────────────
def ref_trajectory(upar, nu, n=N, dt=DT, x0=X0, v0=V0, cylindrical=False):
    """Run n steps; return arrays xs[n+1, 3], vs[n+1, 3]."""
    x, v = x0.copy(), v0.copy()
    xs = np.empty((n+1, 3)); xs[0] = x
    vs = np.empty((n+1, 3)); vs[0] = v
    for i in range(n):
        x, v = sparta_step(x, v, upar, G, nu, dt, cylindrical)
        xs[i+1] = x
        vs[i+1] = v
    return xs, vs

# ── SPARTA dump reader ────────────────────────────────────────────────────────
def read_dump(path):
    """
    Parse a SPARTA particle dump.
    Returns dict {timestep: DataFrame-like dict of arrays}.
    """
    data = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    ts = None
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            ts = int(lines[i+1].strip()); i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i+1].strip()); i += 2
        elif s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]   # column names after "ITEM: ATOMS"
            i += 1
            rows = []
            for _ in range(n_atoms):
                rows.append(list(map(float, lines[i].split()))); i += 1
            if rows:
                arr = np.array(rows)
                data[ts] = {c: arr[:, j] for j, c in enumerate(cols)}
        else:
            i += 1
    return data

# ── Test case definitions ─────────────────────────────────────────────────────
#
# upar in SPARTA (vx, vy, vz) after the bug fix:
#   vx = SPARTA x = R  → Br component
#   vy = SPARTA y = Z  → Bz_cylindrical component   (was Bt in the bug!)
#   vz = SPARTA z = tor → Bt_toroidal component
#
# Case 2: Bz_cyl=1T, Bt=0 → upar=(0, Vpar, 0): drag in Z
# Case 3: Bz_cyl=0, Bt=1T → upar=(0, 0, Vpar): drag toroidal only, no Z drag
# Case 4: same as Case 3, plus cylindrical yes → centrifugal v_φ²/R acts on v_R

NU1 = epstein_nu(0.0,   TI_EV)   # Case 1: Ni=0
NU2 = epstein_nu(NI,    TI_EV)   # Cases 2, 3, 4
NU3 = NU2
NU4 = NU2

CASES = {
    1: dict(nu=NU1, upar=np.array([0.0,    0.0,  0.0]), cylindrical=False,
            label="Case 1: Ni=0 (pure gravity)"),
    2: dict(nu=NU2, upar=np.array([0.0, VPAR,    0.0]), cylindrical=False,
            label="Case 2: Bz_cyl=1T (drag in Z)"),
    3: dict(nu=NU3, upar=np.array([0.0,    0.0, VPAR]), cylindrical=False,
            label="Case 3: Bt=1T (toroidal only, no cylindrical)"),
    4: dict(nu=NU4, upar=np.array([0.0,    0.0, VPAR]), cylindrical=True,
            label="Case 4: Bt=1T + cylindrical yes (centrifugal pushes R)"),
}

# ── Run validation ────────────────────────────────────────────────────────────
print("=" * 70)
print("fix_drag regression test — B-field coordinate mapping + cylindrical terms")
print("=" * 70)
print(f"  nuE (Cases 2-4) = {NU2:.4f} s⁻¹  (1/nuE = {1/NU2:.4f} s)")
print(f"  Run: N={N} steps, dt={DT:.0e} s,  T={N*DT:.4f} s")
print()

all_pass = True
tol_pos = 1.0e-5    # m   RZ absolute position match (ASCII dump has ~7 sig figs)
tol_vel = 1.0e-5    # relative velocity match in RZ (ASCII precision ~1e-6 rel)

for cid, cfg in CASES.items():
    dump_file = f"case.unit.{cid}"
    if not Path(dump_file).exists():
        print(f"  [{cid}] SKIP — {dump_file} not found (run sim first)")
        all_pass = False
        continue

    # Reference trajectory (Python)
    xs_ref, vs_ref = ref_trajectory(cfg["upar"], cfg["nu"],
                                    cylindrical=cfg["cylindrical"])

    # SPARTA trajectory
    dump = read_dump(dump_file)
    steps_sorted = sorted(dump.keys())
    ok = True
    max_dpos, max_dvel = 0.0, 0.0

    for k, ts in enumerate(steps_sorted):
        d = dump[ts]
        # Compare only R (x) and Z (y) coordinates.
        # z is the toroidal direction — periodic in SPARTA, not wrapped in Python,
        # so it diverges after the first wrap and cannot be compared directly.
        xp_rz = np.array([d["x"][0], d["y"][0]])
        vp_rz = np.array([d["vx"][0], d["vy"][0]])
        xr_rz = xs_ref[ts, :2]
        vr_rz = vs_ref[ts, :2]

        # Absolute position error in RZ plane
        dp = np.linalg.norm(xp_rz - xr_rz)
        # Relative velocity error in RZ plane (guard against |v|=0 at t=0)
        scale = max(np.linalg.norm(vr_rz), 1.0e-6)
        dv_rel = np.linalg.norm(vp_rz - vr_rz) / scale
        max_dpos = max(max_dpos, dp)
        max_dvel = max(max_dvel, dv_rel)

        if dp > tol_pos or dv_rel > tol_vel:
            ok = False

    # Final velocity for physics check
    last_ts = steps_sorted[-1]
    vZ_final = dump[last_ts]["vy"][0]    # SPARTA vy = Z velocity
    vR_final = dump[last_ts]["vx"][0]    # SPARTA vx = R velocity

    print(f"  {cfg['label']}")
    print(f"    nuE = {cfg['nu']:.4f} s⁻¹")
    print(f"    Final vZ (SPARTA) = {vZ_final:+.6f} m/s")
    print(f"    Final vZ (Python) = {vs_ref[-1, 1]:+.6f} m/s")
    print(f"    Final vR (SPARTA) = {vR_final:+.6f} m/s")
    print(f"    Final vR (Python) = {vs_ref[-1, 0]:+.6f} m/s")
    print(f"    max |ΔposRZ|       = {max_dpos:.2e} m   (tol {tol_pos:.0e})")
    print(f"    max |ΔvelRZ|/|v|   = {max_dvel:.2e}     (tol {tol_vel:.0e})")
    print(f"    Integrator match: {'PASS' if ok else 'FAIL'}")
    print()

    all_pass = all_pass and ok

# ── Key regression assertion: Case 3 swap-bug ────────────────────────────────
print("── Regression 1: Case 3 must NOT show drag in Z (swap-bug check) ────")
dump3 = read_dump("case.unit.3") if Path("case.unit.3").exists() else None
dump1 = read_dump("case.unit.1") if Path("case.unit.1").exists() else None

if dump3 and dump1:
    last = sorted(dump3.keys())[-1]
    vZ3 = dump3[last]["vy"][0]
    vZ1 = dump1[last]["vy"][0]
    vZ_bug = VPAR * (1 - np.exp(-NU2 * N * DT))   # what the old bug would give

    print(f"  Case 1 vZ = {vZ1:+.6f} m/s  (pure gravity reference)")
    print(f"  Case 3 vZ = {vZ3:+.6f} m/s  (toroidal B, should ≈ Case 1 gravity)")
    print(f"  Bug value = {vZ_bug:+.4f} m/s  (what old swap bug would produce)")

    terminal_gravity = G[1] / NU2   # ≈ -0.11 m/s
    diff_vs_gravity  = abs(vZ3 - vZ1)
    diff_vs_bug      = abs(vZ3 - vZ_bug)

    regression1_pass = (diff_vs_gravity < 0.01) and (diff_vs_bug > 10.0)
    all_pass = all_pass and regression1_pass

    print(f"  |Case3_vZ - Case1_vZ| = {diff_vs_gravity:.4f} m/s  (expect < 0.01)")
    print(f"  |Case3_vZ - bug_vZ|   = {diff_vs_bug:.2f}   m/s  (expect > 10.0)")
    print(f"  Regression 1: {'PASS ✓' if regression1_pass else 'FAIL ✗  ← B-component swap is back!'}")
    print()

# ── Key physics assertion: Case 4 centrifugal pushes R outward ───────────────
print("── Regression 2: Case 4 centrifugal must push vR outward ────────────")
dump4 = read_dump("case.unit.4") if Path("case.unit.4").exists() else None

if dump4 and dump1:
    last4 = sorted(dump4.keys())[-1]
    vR4 = dump4[last4]["vx"][0]
    vR1 = dump1[last4]["vx"][0]

    # Python reference for Case 4
    _, vs4_ref = ref_trajectory(CASES[4]["upar"], NU4, cylindrical=True)
    vR4_ref = vs4_ref[-1, 0]

    print(f"  Case 1 vR = {vR1:+.6f} m/s  (gravity only, no outward push)")
    print(f"  Case 4 vR = {vR4:+.6f} m/s  (centrifugal from toroidal drag)")
    print(f"  Case 4 vR (Python ref) = {vR4_ref:+.6f} m/s")
    print()

    # Case 4 vR must be significantly positive (centrifugal outward push):
    #   - vR4 > 0.1 m/s  (centrifugal is real, not just numerical noise)
    #   - vR4 matches the Python reference to within 1%
    rel_err4 = abs(vR4 - vR4_ref) / max(abs(vR4_ref), 1.0e-6)
    regression2_pass = (vR4 > 0.1) and (rel_err4 < 0.01)
    all_pass = all_pass and regression2_pass

    print(f"  Case 4 vR > 0.1 m/s (centrifugal): {'PASS ✓' if vR4 > 0.1 else 'FAIL ✗  ← centrifugal term missing!'}")
    print(f"  Case 4 |SPARTA - Python| / |Python| < 1%: {'PASS ✓' if rel_err4 < 0.01 else 'FAIL ✗'}")
    print(f"  Regression 2: {'PASS ✓' if regression2_pass else 'FAIL ✗  ← centrifugal or angular-momentum bug!'}")
    print()

print("=" * 70)
print(f"Overall: {'ALL PASS ✓' if all_pass else 'FAILED — see above'}")
print("=" * 70)
sys.exit(0 if all_pass else 1)
