#!/usr/bin/env python3
"""One-to-one gates: OpenEdge grain chain vs an independent integration
of DUSTT [Pigarov et al., PoP 12, 122508 (2005)].

  drag   : OML charging Eq.(3-5) + ion friction Eq.(16), full chain
  efield : F_E = Z_d e E (Eq. 18) under the exact drag+accel integrator
  free   : toroidal kinematics Eq.(11-14) — phi(t) = atan(vphi t / R0),
           R(t) = sqrt(R0^2 + (vphi t)^2) (axi_remap geometry, exact)

The reference replicates the solver's operator order (charge, half-kick,
drift, charge, half-kick) with DUSTT expressions transcribed from the
paper, not from the C++.
"""

import math
import os

# SPARTA SI constants (update.cpp) — must match the binary exactly
QE = 1.60217646e-19
ME = 9.10938215e-31
MP = 1.6726219e-27
EPS0 = 8.8541878128e-12
SQPI = math.sqrt(math.pi)

# case setup (in.grain + make_input.py)
TE, TI, NE, NI = 10.0, 10.0, 1.0e19, 1.0e19
MI = 2.0 * MP
RD = 1.0e-6
RHOD = 534.0
MASS = 4.0 / 3.0 * math.pi * RD**3 * RHOD
R0, VPHI = 5.13, 1000.0
DT = 1e-6
VTI = math.sqrt(2.0 * TI * QE / MI)
U_SMALL = 1.0e-3

BASE = os.path.dirname(os.path.abspath(__file__))


def f_gamma(u, a):
    if u < U_SMALL:
        return 2.0 * (1.0 + a) / SQPI
    return (u + 0.5 / u + a / u) * math.erf(u) + math.exp(-u * u) / SQPI


def solve_phi(u):
    """DUSTT Eq. (3)-(5): Ce e^{phi/Te} = Ci_f max(F_G(u,-phi/Ti),0)"""
    ce = QE * math.pi * RD**2 * NE * math.sqrt(8.0 * TE * QE / (math.pi * ME))
    cif = QE * math.pi * RD**2 * NI * VTI

    def bal(phi):
        return (-ce * math.exp(phi / TE)
                + cif * max(f_gamma(u, -phi / TI), 0.0))
    lo, hi = -80.0 * max(TE, TI), 20.0 * max(TE, TI)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bal(lo) * bal(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def xi_dustt(u, chi):
    """Eq. (16) friction multiplier, self-consistent lnLambda pieces."""
    a = chi / (TI / TE)
    if u < U_SMALL:
        lnl = lnlam(u, chi)
        return (5.0 + 4.0 * a) / (3.0 * SQPI) + 2.0 * a * a * lnl * 2.0 / (3.0 * SQPI)
    e2, er = math.exp(-u * u), math.erf(u)
    coll = (u * (2 * u * u + 1 + 2 * a) * e2
            + 0.5 * SQPI * (4 * u**4 + 2 * u * u - 1 - 2 * (1 - 2 * u * u) * a) * er
            ) / (2 * u**3 * SQPI)
    Y = (er - 2 * u * e2 / SQPI) / (2 * u * u)
    return coll + 2 * a * a * lnlam(u, chi) * Y / u


def lnlam(u, chi):
    if chi <= 0.0:
        return 0.0
    tij, tej = TI * QE, TE * QE
    mve2 = tij * (3.0 + 2.0 * u * u)
    b90 = RD * chi * tej / mve2
    lamd = math.sqrt(EPS0 * tej / (NE * QE * QE))
    lams = lamd / math.sqrt(1.0 + 3.0 * tej / mve2)
    eta = 1.0 + (RD / lams) * (1.0 + math.sqrt(TE / (6.0 * TI)))
    v = 0.5 * math.log((b90**2 + (eta * lams)**2) / (b90**2 + RD**2))
    return max(v, 0.0)


def xi_dustt_manual(u, chi, delta, lnl):
    """Eq. 16 multiplier with FIXED lnLambda (golden-pin form)."""
    a = chi / delta
    if u < U_SMALL:
        return (5.0 + 4.0*a)/(3.0*SQPI) + 2.0*a*a*lnl*2.0/(3.0*SQPI)
    e2, er = math.exp(-u*u), math.erf(u)
    coll = (u*(2*u*u + 1 + 2*a)*e2
            + 0.5*SQPI*(4*u**4 + 2*u*u - 1 - 2*(1 - 2*u*u)*a)*er
            ) / (2*u**3*SQPI)
    Y = (er - 2*u*e2/SQPI) / (2*u*u)
    return coll + 2*a*a*lnl*Y/u


NU0 = 0.75 * (NI * MI * VTI) / (RHOD * RD)


def halfkick(v, up, gext, dth):
    """exact exponential relaxation toward flow + const accel, per axis"""
    u = abs(v[0] - up[0]) / VTI  # flow along slot x only in these runs
    umag = math.sqrt(sum((v[k] - up[k])**2 for k in range(3))) / VTI
    phi = solve_phi(umag)
    chi = max(-phi / TE, 0.0)
    nu = NU0 * xi_dustt(umag, chi)
    s = nu * dth
    ex = math.exp(-s)
    zd = 4.0 * math.pi * EPS0 * RD * phi / QE
    out = [0.0] * 3
    for k in range(3):
        gi = gext[k] / nu if nu > 0 else 0.0
        out[k] = up[k] + (v[k] - up[k] - gi) * ex + gi
    return out, zd


def reference(nsteps, upar, aE):
    """charge, halfkick, drift, charge, halfkick — the fix schedule"""
    up = [upar, 0.0, 0.0]
    v = [0.0, 0.0, 0.0]
    z = 0.5
    frames = {0: (z, list(v), zd0())}
    zd = zd0()
    for n in range(1, nsteps + 1):
        g = [zd * QE * aE / MASS, 0.0, 0.0]
        v, zd = halfkick(v, up, g, DT / 2)
        z += v[0] * DT
        g = [zd * QE * aE / MASS, 0.0, 0.0]
        v, zd = halfkick(v, up, g, DT / 2)
        frames[n] = (z, list(v), zd)
    return frames


def zd0():
    return 4.0 * math.pi * EPS0 * RD * solve_phi(0.0) / QE


def read_traj(tag):
    frames = {}
    with open(os.path.join(BASE, "output", f"traj.{tag}")) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            ts = int(lines[i + 1])
            n = int(lines[i + 3])
            row = lines[i + 9].split()
            frames[ts] = [float(q) for q in row[2:]]
            i += 9 + n
        else:
            i += 1
    return frames


def main():
    checks = []

    def gate(name, ok, info=""):
        checks.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))

    # --- drag: full charging+friction chain --------------------------------
    sim = read_traj("drag")
    ref = reference(1000, 2.0e4, 0.0)
    ts = sorted(k for k in sim if k > 0)
    ev, ez = 0.0, 0.0
    for t in ts:
        z_s, v_s, zd_s = sim[t][0], sim[t][3], sim[t][6]
        z_r, v_r, zd_r = ref[t]
        ev = max(ev, abs(v_s - v_r[0]) / max(abs(v_r[0]), 1e-30))
        ez = max(ez, abs(zd_s - zd_r) / abs(zd_r))
    vend = ref[ts[-1]][1][0]
    gate("drag: v_par(t) matches DUSTT chain (rtol 2e-3)", ev < 2e-3,
         f"max rel dev {ev:.2e}, v_end {vend:.1f} m/s")
    gate("drag: Z_d(t) matches DUSTT flow-dependent OML (rtol 1e-3)",
         ez < 1e-3, f"max rel dev {ez:.2e}")
    gate("drag: signal is real (grain accelerated > 10 m/s)",
         abs(sim[ts[-1]][3]) > 10.0, f"v_end sim {sim[ts[-1]][3]:.1f} m/s")

    # --- efield: F_E = Z_d e E --------------------------------------------
    sim = read_traj("efield")
    ref = reference(1000, 0.0, -100.0)
    ev = max(abs(sim[t][3] - ref[t][1][0]) / max(abs(ref[t][1][0]), 1e-30)
             for t in ts)
    gate("efield: v_par(t) matches Z_d e E / m (rtol 2e-3)", ev < 2e-3,
         f"max rel dev {ev:.2e}, v_end {ref[ts[-1]][1][0]:.4f} m/s")
    zd_sim = sim[ts[-1]][6]
    gate("efield: charge sign negative (electron-dominated OML)",
         zd_sim < 0.0, f"Z_d {zd_sim:.3e}")

    # --- free: toroidal kinematics (Eq. 11-14 geometry) -------------------
    sim = read_traj("free")
    er_, ep_ = 0.0, 0.0
    for t in ts:
        tt = t * DT
        R_s, phi_s = sim[t][1], sim[t][7]
        R_a = math.sqrt(R0**2 + (VPHI * tt)**2)
        phi_a = math.atan(VPHI * tt / R0)
        er_ = max(er_, abs(R_s - R_a) / R_a)
        ep_ = max(ep_, abs(phi_s - phi_a) / max(phi_a, 1e-30))
    gate("free: R(t) exact centrifugal geometry (rtol 1e-9)", er_ < 1e-9,
         f"max rel dev {er_:.2e}")
    gate("free: phi_unwrap(t) = atan(vphi t / R0) (rtol 1e-9)", ep_ < 1e-9,
         f"max rel dev {ep_:.2e}")

    # --- neut: neutral friction decay (Eq. 17, zeta_fric,n) ---------------
    # nn 1e19, tn 2 eV, plasma ne=ni=1e6 (ion channel negligible); grain
    # launched at 100 m/s decays toward rest. Reference: same half-kick
    # schedule with nu_n = zeta_n(s) * 0.75 rho_n v_Tn / (rho_d R_d).
    NN, TN = 1.0e19, 2.0
    VTN = math.sqrt(2.0 * TN * QE / MI)
    ZN0 = 8.0 / (3.0 * SQPI)

    def zeta_n(s):
        if s < 1.0e-3:
            return ZN0
        e2, er = math.exp(-s * s), math.erf(s)
        return ((1.0 + s*s - 0.25/(s*s)) * er + (s + 0.5/s) * e2 / SQPI) / s

    def ref_neut(nsteps, v0):
        nu0 = 0.75 * (NN * MI * VTN) / (RHOD * RD)
        v = v0
        out = {0: v}
        for n in range(1, nsteps + 1):
            for _ in range(2):
                s = abs(v) / VTN
                nu = nu0 * zeta_n(s)
                v = v * math.exp(-nu * DT / 2)
            out[n] = v
        return out

    sim = read_traj("neut")
    refn = ref_neut(1000, 100.0)
    evn = max(abs(sim[t][3] - refn[t]) / abs(refn[t]) for t in ts)
    dv = 100.0 - refn[ts[-1]]
    gate("neut: v(t) decay matches DUSTT zeta_fric,n (rtol 1e-4)",
         evn < 1e-4, f"max rel dev {evn:.2e}, decay {dv*1e3:.2f} mm/s per ms")
    gate("neut: decay is a real signal", abs(sim[0][3] - sim[ts[-1]][3]) > 1e-4,
         f"dv {sim[0][3]-sim[ts[-1]][3]:.2e} m/s")

    # --- see: secondary emission shifts the floating potential -----------
    # Te 30 eV, Sternglass delta_m 0.5 / E_m 85 eV (Li), flux-averaged.
    def delta_eff(te, dm=0.5, em=85.0):
        nq = 256
        emax, num = 30.0 * te, 0.0
        de = emax / nq
        for k in range(nq):
            E = (k + 0.5) * de
            num += 7.4 * dm * (E/em) * math.exp(-2.0*math.sqrt(E/em)) \
                   * E * math.exp(-E/te) * de
        return min(num / (te * te), 1.0)

    def solve_phi_see(te, see):
        ce = QE * math.pi * RD**2 * NE * math.sqrt(8.0*te*QE/(math.pi*ME))
        cif = QE * math.pi * RD**2 * NI * VTI
        dse = delta_eff(te) if see else 0.0

        def bal(phi):
            return (-ce * (1.0 - dse) * math.exp(phi / te)
                    + cif * max(f_gamma(0.0, -phi / TI), 0.0))
        lo, hi = -80.0 * max(te, TI), 20.0 * max(te, TI)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if bal(lo) * bal(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    sim = read_traj("see")
    zd_sim = sim[ts[-1]][6]
    zd_ref = 4.0 * math.pi * EPS0 * RD * solve_phi_see(30.0, True) / QE
    zd_off = 4.0 * math.pi * EPS0 * RD * solve_phi_see(30.0, False) / QE
    gate("see: Z_d matches (1-delta_eff) OML balance (rtol 1e-3)",
         abs(zd_sim - zd_ref) / abs(zd_ref) < 1e-3,
         f"sim {zd_sim:.4e} vs ref {zd_ref:.4e}")
    gate("see: emission reduces |Z_d| vs no-SEE",
         abs(zd_sim) < abs(zd_off),
         f"|Zd| {abs(zd_sim):.3e} < {abs(zd_off):.3e} "
         f"(delta_eff {delta_eff(30.0):.3f})")

    # --- pinned golden values (guard against coupled drift of the C++
    # and this transcription: these constants are frozen, not recomputed) --
    golden = [
        ("xi(u=0.5,chi=3,delta=1,lnL=2)", xi_dustt_manual(0.5, 3.0, 1.0, 2.0),
         14.887256857957702),
        ("F_G(u=0.5,a=3)", f_gamma(0.5, 3.0), 4.343140373065571),
        ("zeta_n(s=0.5)", zeta_n(0.5), 1.5784238073096906),
        ("delta_eff(Te=30,Li)", delta_eff(30.0), 0.42488041717066205),
    ]
    for name, got, want in golden:
        gate(f"golden: {name} pinned", abs(got/want - 1.0) < 1e-12,
             f"{got:.15g} vs {want:.15g}")

    # small-u analytic seam continuity (both xi and F_G)
    for chi, d, l in [(0.0, 1.0, 0.0), (3.0, 1.0, 2.0)]:
        lo = xi_dustt_manual(0.999e-3, chi, d, l)
        hi = xi_dustt_manual(1.001e-3, chi, d, l)
        gate(f"seam: xi continuous at U_SMALL (chi={chi})",
             abs(hi/lo - 1.0) < 1e-5, f"{lo:.9f} vs {hi:.9f}")
    lo, hi = f_gamma(0.999e-3, 2.0), f_gamma(1.001e-3, 2.0)
    gate("seam: F_G continuous at U_SMALL", abs(hi/lo - 1.0) < 1e-5,
         f"{lo:.9f} vs {hi:.9f}")

    # --- vac: pure electric kick in vacuum (no-drag fallback branch) ------
    # fixed Z_d = -2e4, E_Z = -100 V/m, zero plasma/neutrals, no gravity:
    # v(t) = Zd e E t / m exactly (half-kick pairs sum to full kicks)
    sim = read_traj("vac")
    aZ = (-2.0e4) * QE * (-100.0) / MASS
    evv = max(abs(sim[t][3] - aZ * t * DT) / abs(aZ * t * DT) for t in ts)
    gate("vac: v(t) = Zd e E t / m in vacuum (rtol 1e-9)", evv < 1e-9,
         f"max rel dev {evv:.2e}, a {aZ:.3f} m/s^2")

    # --- asymptotics + quadrature convergence (transcription-level) -------
    gate("asym: xi -> u ram limit (u=30, chi=0)",
         abs(xi_dustt_manual(30.0, 0.0, 1.0, 0.0)/30.0 - 1.0) < 2e-2,
         f"{xi_dustt_manual(30.0, 0.0, 1.0, 0.0):.3f}")
    gate("asym: F_G -> u ram limit (u=30)",
         abs(f_gamma(30.0, 0.0)/30.0 - 1.0) < 2e-2,
         f"{f_gamma(30.0, 0.0):.3f}")

    def delta_eff_n(te, nq):
        emax, num = 30.0 * te, 0.0
        de = emax / nq
        for k in range(nq):
            E = (k + 0.5) * de
            num += 7.4 * 0.5 * (E/85.0) * math.exp(-2.0*math.sqrt(E/85.0)) \
                   * E * math.exp(-E/te) * de
        return min(num / (te * te), 1.0)
    qc = max(abs(delta_eff_n(te, 256)/delta_eff_n(te, 2048) - 1.0)
             for te in (5.0, 30.0, 100.0))
    gate("see: 256-pt quadrature converged over Te 5-100 eV (rtol 2e-3)",
         qc < 2e-3, f"max dev vs 2048-pt {qc:.2e}")

    # MPI invariance: 2-rank drag run bitwise-comparable to 1-rank
    sim2 = read_traj("drag2")
    sim1 = read_traj("drag")
    dev = max(max(abs(a - b) for a, b in zip(sim1[t], sim2[t]))
              for t in ts if t in sim2)
    gate("mpi: np2 reproduces np1 drag trajectory", dev < 1e-12,
         f"max abs dev {dev:.2e}")

    ok = all(checks)
    print(("PASS" if ok else "FAIL") + ": verification/particulates/dustt vs DUSTT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
