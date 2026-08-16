"""Standalone MC replica of OpenEdge's Nanbu kernel, thermalization setup.

Scheme A: OpenEdge mixed pairing (shuffle all, pair sequentially, s ~ n_total)
Scheme B: per-species-pair (every D pairs a C, s ~ n_C) a la Smilei/WarpX
Both compared to the NRL two-temperature ODE.
"""
import numpy as np

e = 1.602176634e-19
eps0 = 8.8541878128e-12
kB = 1.380649e-23
amu = 1.66053906660e-27

mD, mC = 2.014 * amu, 12.011 * amu
qD, qC = 1 * e, 3 * e
TD0, TC0 = 10.0, 5.0        # eV
n_each = 5.0e16             # m^-3 per species
lnL = 13.10
dt = 1.0e-7                 # 10x deck dt, still s << 1
nsteps = 4000               # 400 us
Np = 2000                   # per species
rng = np.random.default_rng(7)


def maxwellian(N, T_eV, m):
    return rng.normal(0.0, np.sqrt(T_eV * e / m), size=(N, 3))


def temp_eV(v, m):
    return m * np.mean(np.sum(v * v, axis=1)) / (3.0 * kB) * kB / e


def scatter(vA, vB, mA, mB, qA, qB, n_partner):
    """OpenEdge kernel, vectorized over pairs. Returns updated vA, vB."""
    g = vB - vA
    gmag = np.linalg.norm(g, axis=1)
    gmag = np.where(gmag == 0.0, 1e-300, gmag)
    mu = mA * mB / (mA + mB)
    s = (qA * qB) ** 2 * n_partner * lnL * dt / (4 * np.pi * eps0**2 * mu**2 * gmag**3)
    U = rng.uniform(1e-30, 1.0, size=len(s))
    cos_chi = np.where(s < 0.01, 1.0 + s * np.log(U),
                       np.where(s > 6.0, 2 * U - 1.0, 1.0 + s * np.log(U)))  # s stays tiny here
    cos_chi = np.clip(cos_chi, -1.0, 1.0)
    sin_chi = np.sqrt(1.0 - cos_chi**2)
    epsang = rng.uniform(0.0, 2 * np.pi, size=len(s))
    ce, se = np.cos(epsang), np.sin(epsang)
    gp = np.sqrt(g[:, 1]**2 + g[:, 2]**2)
    h = np.empty_like(g)
    ok = gp > 1e-12 * gmag
    h[ok, 0] = gp[ok] * ce[ok]
    h[ok, 1] = -(g[ok, 0] * g[ok, 1] * ce[ok] + gmag[ok] * g[ok, 2] * se[ok]) / gp[ok]
    h[ok, 2] = -(g[ok, 0] * g[ok, 2] * ce[ok] - gmag[ok] * g[ok, 1] * se[ok]) / gp[ok]
    h[~ok, 0] = 0.0
    h[~ok, 1] = -gmag[~ok] * ce[~ok]
    h[~ok, 2] = -gmag[~ok] * se[~ok]
    dg = sin_chi[:, None] * h - (1.0 - cos_chi)[:, None] * g
    MA = mA / (mA + mB)
    MB = mB / (mA + mB)
    if np.isscalar(MB):
        vA2 = vA - MB * dg
        vB2 = vB + MA * dg
    else:
        vA2 = vA - MB[:, None] * dg
        vB2 = vB + MA[:, None] * dg
    return vA2, vB2


def run_openedge_style():
    v = np.vstack([maxwellian(Np, TD0, mD), maxwellian(Np, TC0, mC)])
    m = np.concatenate([np.full(Np, mD), np.full(Np, mC)])
    q = np.concatenate([np.full(Np, qD), np.full(Np, qC)])
    n_tot = 2 * n_each
    out = []
    for step in range(nsteps + 1):
        if step % 20 == 0:
            isD = m == mD
            out.append((step * dt, temp_eV(v[isD], mD), temp_eV(v[~isD], mC)))
        perm = rng.permutation(2 * Np)
        a, b = perm[0::2], perm[1::2]
        vA, vB = scatter(v[a], v[b], m[a], m[b], q[a], q[b], n_tot)
        v[a], v[b] = vA, vB
    return np.array(out)


def run_species_pair_style():
    vD = maxwellian(Np, TD0, mD)
    vC = maxwellian(Np, TC0, mC)
    out = []
    for step in range(nsteps + 1):
        if step % 20 == 0:
            out.append((step * dt, temp_eV(vD, mD), temp_eV(vC, mC)))
        pC = rng.permutation(Np)
        vD, vC[pC] = scatter(vD, vC[pC], mD, mC, qD, qC, n_each)
    return np.array(out)


def nrl_tau(Ta, Tb, ma, mb, Za, Zb, nb):
    ma_g, mb_g = ma * 1e3, mb * 1e3
    nu = (1.8e-19 * np.sqrt(ma_g * mb_g) * Za**2 * Zb**2 * nb * 1e-6 * lnL
          / (ma_g * Tb + mb_g * Ta) ** 1.5)
    return 1.0 / nu


def nrl_ode(t):
    TD, TC = np.empty_like(t), np.empty_like(t)
    TD[0], TC[0] = TD0, TC0
    for i in range(1, len(t)):
        tau = nrl_tau(TD[i - 1], TC[i - 1], mD, mC, 1, 3, n_each)
        d = (t[i] - t[i - 1]) * (TD[i - 1] - TC[i - 1]) / tau
        TD[i] = TD[i - 1] - d
        TC[i] = TC[i - 1] + d
    return TD, TC


def efold(t, gap):
    mask = gap > 0.1 * gap[0]
    return -1.0 / np.polyfit(t[mask], np.log(gap[mask]), 1)[0]


A = run_openedge_style()
B = run_species_pair_style()
tf = np.linspace(0, nsteps * dt, 4000)
TDo, TCo = nrl_ode(tf)

print(f"NRL tau_eq(t=0) = {nrl_tau(TD0, TC0, mD, mC, 1, 3, n_each)*1e6:.0f} us")
print(f"gap e-fold  NRL ODE          : {efold(tf, TDo - TCo)*1e6:.0f} us")
print(f"gap e-fold  A (openedge mix) : {efold(A[:,0], A[:,1]-A[:,2])*1e6:.0f} us")
print(f"gap e-fold  B (species pair) : {efold(B[:,0], B[:,1]-B[:,2])*1e6:.0f} us")
print(f"final A: TD={A[-1,1]:.2f} TC={A[-1,2]:.2f}")
print(f"final B: TD={B[-1,1]:.2f} TC={B[-1,2]:.2f}")
