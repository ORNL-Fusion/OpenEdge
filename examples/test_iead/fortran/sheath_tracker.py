#!/usr/bin/env python3
"""
sheath_tracker.py  — v3  (Tantalum multi-charge-state Chodura sheath)
=====================================================================
Track a mixture of Ta2+/Ta3+/Ta4+ ions through a magnetised sheath
with oblique B-field.  Collect per-species and total energy & angle
distributions at the wall.

Species mixture
---------------
    Ta2+  8 %      Ta3+  62 %      Ta4+  30 %

Geometry
--------
    Wall normal = z       (wall at z = z_wall)
    B = Bmag * (sin α, 0, cos α)   tilted in x-z plane

Sheath model  (prescribed, two-region)
--------------------------------------
    (i)   Magnetic pre-sheath (Chodura) :  ~5 ρ_s_max ,  drops ~0.5 Te
    (ii)  Debye sheath                  :  ~20 λ_D    ,  drops to ϕ_wall
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from time import perf_counter

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════
QE   = 1.602176634e-19
MP   = 1.672621898e-27
AMU  = 1.660539067e-27      # atomic mass unit
EPS0 = 8.854187817e-12

# ═══════════════════════════════════════════════════════════════
#  Plasma parameters
# ═══════════════════════════════════════════════════════════════
Te_eV  = 5.0                # electron temperature [eV]
Ti_eV  = 5.0                # ion temperature      [eV]
ni0    = 1.0e19             # plasma density       [m^-3]
Bmag   = 0.5                # |B|  [T]
phi_w_Te = -3.0             # wall potential in units of Te

# ── Species:  (name, charge_number Z, mass_amu, fraction) ─────
SPECIES = [
    ('Ta2+', 2, 180.948, 0.08),
    ('Ta3+', 3, 180.948, 0.62),
    ('Ta4+', 4, 180.948, 0.30),
]

# ── Tilt angles to scan (degrees from wall normal) ────────────
ANGLES = [0, 45, 85]

# ── Total particles per species per angle ─────────────────────
NP_TOTAL = 10000             # distributed among species by fraction

# ═══════════════════════════════════════════════════════════════
#  Derived scales
# ═══════════════════════════════════════════════════════════════
Te   = Te_eV * QE
Ti   = Ti_eV * QE
lD   = np.sqrt(EPS0 * Te / (ni0 * QE**2))
phi_w = phi_w_Te * Te_eV      # wall potential [V]

# Per-species derived quantities
species_info = []
for name, Z, A, frac in SPECIES:
    m  = A * AMU
    q  = Z * QE
    cs = np.sqrt(Z * Te / m)          # Bohm speed  sqrt(Z Te / m)
    wc = q * Bmag / m                 # cyclotron frequency
    rho = cs / wc                     # sound gyroradius
    species_info.append(dict(
        name=name, Z=Z, A=A, frac=frac,
        m=m, q=q, cs=cs, wci=wc, rho_s=rho
    ))

# Domain sized by the LARGEST gyroradius species
rho_max = max(s['rho_s'] for s in species_info)
cs_min  = min(s['cs']    for s in species_info)
wci_min = min(s['wci']   for s in species_info)

N_rho_pre = 5
N_lam_deb = 20
L_pre = N_rho_pre * rho_max
L_deb = N_lam_deb * lD
Lz    = L_pre + L_deb

PHI_PRE = -0.5 * Te_eV       # presheath potential drop [V]
PHI_DEB = phi_w - PHI_PRE    # Debye sheath drop [V]

# Time step: resolve fastest gyration AND Debye-sheath transit
wci_max = max(s['wci'] for s in species_info)
dt = 0.1 * min(2.0 * np.pi / wci_max, lD / cs_min)


# ═══════════════════════════════════════════════════════════════
#  Sheath potential & E-field
# ═══════════════════════════════════════════════════════════════
def sheath_phi(z):
    z = np.asarray(z, dtype=float)
    zc = np.clip(z, 0.0, Lz)
    in_pre = zc <= L_pre
    phi_pre = PHI_PRE * (zc / L_pre)
    frac = np.clip((zc - L_pre) / L_deb, 0.0, 1.0)
    phi_deb = PHI_PRE + PHI_DEB * frac**(4.0 / 3.0)
    return np.where(in_pre, phi_pre, phi_deb)


def sheath_Ez(z):
    dz = 0.005 * lD
    return -(sheath_phi(z + dz) - sheath_phi(z - dz)) / (2.0 * dz)


# ═══════════════════════════════════════════════════════════════
#  Boris pusher  (takes q, m as arguments for multi-species)
# ═══════════════════════════════════════════════════════════════
def boris(pos, vel, B_vec, q, m):
    N = pos.shape[0]
    E = np.zeros((N, 3))
    E[:, 2] = sheath_Ez(pos[:, 2])

    hqm = 0.5 * dt * q / m

    vm = vel + hqm * E
    t_vec = hqm * B_vec
    t2    = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t2)
    vp = vm + np.cross(vm, t_vec)
    vr = vm + np.cross(vp, s_vec)
    vel_new = vr + hqm * E
    pos_new = pos + dt * vel_new
    return pos_new, vel_new


# ═══════════════════════════════════════════════════════════════
#  Particle initialisation  (per species)
# ═══════════════════════════════════════════════════════════════
def init_particles(Np, sp, B_vec):
    """
    Ions enter at z=0 along B at species Bohm speed:
        v = cs(Z) * b_hat  +  thermal(Ti)
    """
    b_hat = B_vec / np.linalg.norm(B_vec)
    pos = np.zeros((Np, 3))
    vel = np.zeros((Np, 3))

    vth = np.sqrt(Ti / sp['m'])
    v_drift = sp['cs'] * b_hat

    vel[:, 0] = v_drift[0] + vth * np.random.randn(Np)
    vel[:, 1] = v_drift[1] + vth * np.random.randn(Np)
    vel[:, 2] = v_drift[2] + 0.2 * vth * np.abs(np.random.randn(Np))
    return pos, vel


# ═══════════════════════════════════════════════════════════════
#  Run one species at one angle
# ═══════════════════════════════════════════════════════════════
def run_species(sp, alpha_deg, Np, max_steps=2_000_000, verbose=True):
    alpha = np.radians(alpha_deg)
    B_vec = Bmag * np.array([np.sin(alpha), 0.0, np.cos(alpha)])

    pos, vel = init_particles(Np, sp, B_vec)
    alive = np.ones(Np, dtype=bool)
    hit_KE  = []
    hit_ang = []

    # At large tilt (e.g. 85°) gyromotion can push z slightly < 0.
    # Only count as "lost" if ion goes more than one gyroradius upstream.
    z_lost_threshold = -2.0 * sp['rho_s']

    if verbose:
        print(f"    {sp['name']:5s}  Np={Np:>5d}  "
              f"cs={sp['cs']:.0f} m/s  ρ_s={sp['rho_s']*1e3:.2f} mm  "
              f"ω_ci={sp['wci']:.2e} rad/s")

    for step in range(max_steps):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break

        pos[idx], vel[idx] = boris(pos[idx], vel[idx], B_vec,
                                   sp['q'], sp['m'])

        # wall hits  (z >= Lz)
        hit = idx[pos[idx, 2] >= Lz]
        if len(hit):
            alive[hit] = False
            # Boris updates v then x (x_new = x_old + dt*v_new),
            # so vel[hit] is the velocity AT the wall-crossing step.
            v = vel[hit]
            KE = 0.5 * sp['m'] * np.sum(v**2, axis=1) / QE   # eV
            vt = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
            ang = np.degrees(np.arctan2(vt, np.abs(v[:, 2])))
            hit_KE.extend(KE.tolist())
            hit_ang.extend(ang.tolist())

        # lost upstream  (allow small gyro-excursion past z=0)
        lost = idx[pos[idx, 2] < z_lost_threshold]
        if len(lost):
            alive[lost] = False

    n_hit = len(hit_KE)
    n_lost = Np - n_hit - int(alive.sum())
    if verbose:
        print(f"           hits={n_hit}  lost={n_lost}  flying={int(alive.sum())}")

    return np.asarray(hit_KE), np.asarray(hit_ang)


# ═══════════════════════════════════════════════════════════════
#  Run full mixture at one angle
# ═══════════════════════════════════════════════════════════════
def run_mixture(alpha_deg, Np_total=NP_TOTAL, verbose=True):
    """Return dict  { species_name: (energy, angle) }  and combined arrays."""
    if verbose:
        print("══════════════════════════════════════════════════════")
        print(f"  Ta mixture — Chodura sheath   α = {alpha_deg}°")
        print("══════════════════════════════════════════════════════")
        print(f"  Te = Ti = {Te_eV} eV    ni = {ni0:.0e} m⁻³    |B| = {Bmag} T")
        print(f"  λ_D     = {lD*1e6:.2f} µm")
        print(f"  L_pre   = {L_pre*1e3:.2f} mm  ({N_rho_pre} ρ_s_max)")
        print(f"  L_deb   = {L_deb*1e6:.0f} µm  ({N_lam_deb} λ_D)")
        print(f"  L_z     = {Lz*1e3:.3f} mm")
        print(f"  ϕ_wall  = {phi_w:.1f} V")
        print(f"  dt      = {dt:.3e} s")
        print("──────────────────────────────────────────────────────")

    per_species = {}
    all_en, all_ang, all_labels = [], [], []

    t0 = perf_counter()
    for sp in species_info:
        Np_sp = max(100, int(round(Np_total * sp['frac'])))
        en, ang = run_species(sp, alpha_deg, Np_sp, verbose=verbose)
        per_species[sp['name']] = (en, ang)
        if len(en):
            all_en.append(en)
            all_ang.append(ang)
            all_labels.extend([sp['name']] * len(en))

    elapsed = perf_counter() - t0
    all_en  = np.concatenate(all_en)  if all_en  else np.array([])
    all_ang = np.concatenate(all_ang) if all_ang else np.array([])

    if verbose:
        print(f"──────────────────────────────────────────────────────")
        print(f"  Total: {len(all_en)} wall hits in {elapsed:.1f} s")
        print("══════════════════════════════════════════════════════\n")

    return per_species, all_en, all_ang, np.array(all_labels)


# ═══════════════════════════════════════════════════════════════
#  Plotting style
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 16,
    "axes.linewidth": 1.3,
})

SPECIES_COLORS = {'Ta2+': 'C0', 'Ta3+': 'C1', 'Ta4+': 'C2'}

# shared bins for IEAD 2-D histograms
ANGLE_BINS  = np.linspace(0.0, 90.0, 46)       # 2° bins
ENERGY_BINS = np.linspace(0.0, 100.0, 51)      # 2 eV bins


def _iead_pcolormesh(ax, energy, angle, title='', vmin=1.0):
    """Plot a 2-D IEAD histogram on *ax* with viridis + LogNorm."""
    H, xedges, yedges = np.histogram2d(
        energy, angle, bins=[ENERGY_BINS, ANGLE_BINS])
    # mask zeros so LogNorm doesn't complain
    H_plot = np.ma.masked_where(H == 0, H)
    im = ax.pcolormesh(
        xedges, yedges, H_plot.T,
        norm=LogNorm(vmin=vmin, vmax=max(H.max(), vmin * 10)),
        cmap='plasma', shading='auto')
    ax.set_xlabel('Energy  [eV]')
    ax.set_ylabel('Angle  [deg]')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 90)
    ax.tick_params(which='both', direction='in', length=5)
    if title:
        ax.set_title(title, pad=8)
    return im


# ═══════════════════════════════════════════════════════════════
#  Plotting — 3-panel IEAD  (total mixture, one panel per angle)
# ═══════════════════════════════════════════════════════════════
def plot_iead_comparison(angle_results):
    """Single figure with 1×3 subplots: total-mixture IEAD for each angle."""
    sorted_angles = sorted(angle_results.keys())
    n = len(sorted_angles)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    im = None
    for col, adeg in enumerate(sorted_angles):
        _, all_en, all_ang, _ = angle_results[adeg]
        if len(all_en) == 0:
            continue
        im = _iead_pcolormesh(
            axes[col], all_en, all_ang,
            title=f'α = {adeg}°\n'
                  f'<E> = {np.mean(all_en):.1f} eV    '
                  f'<θ> = {np.mean(all_ang):.1f}°')
        if col > 0:
            axes[col].set_ylabel('')

    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), pad=0.02,
                            label='Counts per bin')

    fig.suptitle('Ta mixture IEAD at wall  '
                 '(Ta²⁺ 8%  /  Ta³⁺ 62%  /  Ta⁴⁺ 30%)',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('ta_iead_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved → ta_iead_comparison.png")


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── print species table ──
    print("\n  Species mix:")
    print("  ┌───────┬────────┬─────────────┬───────────────┬─────────────┐")
    print("  │ Name  │ frac   │ cs (m/s)    │ ρ_s (mm)      │ ω_ci (rad/s)│")
    print("  ├───────┼────────┼─────────────┼───────────────┼─────────────┤")
    for sp in species_info:
        print(f"  │ {sp['name']:5s} │ {sp['frac']:5.2f}  │ {sp['cs']:>10.0f}  "
              f"│ {sp['rho_s']*1e3:>12.3f}  │ {sp['wci']:>10.2e}  │")
    print("  └───────┴────────┴─────────────┴───────────────┴─────────────┘\n")

    # ── scan over tilt angles ──
    angle_results = {}
    for adeg in ANGLES:
        per_sp, all_en, all_ang, labels = run_mixture(adeg, Np_total=NP_TOTAL)
        angle_results[adeg] = (per_sp, all_en, all_ang, labels)

    # ── single 3-panel IEAD figure (total mixture) ──
    plot_iead_comparison(angle_results)

    # ── final summary table ──
    print("\n  ╔═══════╦═════════╦══════════════════╦══════════════════╦══════════════════╦══════════════════╗")
    print("  ║  α °  ║ species ║  <E> eV          ║  <θ> °           ║  σ_E eV          ║  σ_θ °           ║")
    print("  ╠═══════╬═════════╬══════════════════╬══════════════════╬══════════════════╬══════════════════╣")
    for adeg in ANGLES:
        per_sp = angle_results[adeg][0]
        for name in [s['name'] for s in species_info]:
            en, ang = per_sp.get(name, (np.array([]), np.array([])))
            if len(en):
                print(f"  ║  {adeg:>3d}  ║ {name:>5s}   ║  {np.mean(en):>14.1f}  "
                      f"║  {np.mean(ang):>14.1f}  ║  {np.std(en):>14.1f}  "
                      f"║  {np.std(ang):>14.1f}  ║")
        # total
        all_en = angle_results[adeg][1]
        all_ang = angle_results[adeg][2]
        if len(all_en):
            print(f"  ║  {adeg:>3d}  ║ TOTAL   ║  {np.mean(all_en):>14.1f}  "
                  f"║  {np.mean(all_ang):>14.1f}  ║  {np.std(all_en):>14.1f}  "
                  f"║  {np.std(all_ang):>14.1f}  ║")
        print("  ╠═══════╬═════════╬══════════════════╬══════════════════╬══════════════════╬══════════════════╣")
    print("  ╚═══════╩═════════╩══════════════════╩══════════════════╩══════════════════╩══════════════════╝")
