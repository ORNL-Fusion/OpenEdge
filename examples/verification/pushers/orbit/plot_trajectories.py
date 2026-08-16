#!/usr/bin/env python3
"""Compare Boris vs GCA trajectories for a single H+ in the Khan
analytical tokamak field — paper / journal-ready figures.

3D Cartesian dump:  position (x, y, z), velocity (vx, vy, vz).
For analysis:
    R     = sqrt(x^2 + y^2)
    Z     = z
    phi   = atan2(y, x)
    v_R   =  vx cos(phi) + vy sin(phi)
    v_phi = -vx sin(phi) + vy cos(phi)
    v_Z   =  vz

Equilibrium B is reconstructed from `khan_plasma.h5` /equilibrium/{r,z,psi,btf,rtf}
in cylindrical components (B_R, B_phi, B_Z); cross products and inner products are
done in cylindrical (R, phi, Z).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paper / journal-ready style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":       18,
    "axes.labelsize":  18,
    "axes.titlesize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 1.8,
    "axes.linewidth":  1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":       True,
    "ytick.right":     True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


def save(fig, name, outdir):
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_sparta_dump(path):
    timesteps, x, y, z, vx, vy, vz, extra = [], [], [], [], [], [], [], []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                ts = int(f.readline().strip())
                f.readline()
                natoms = int(f.readline().strip())
                f.readline()
                for _ in range(3):
                    f.readline()
                f.readline()
                if natoms == 0:
                    continue
                dline = f.readline()
                if not dline or not dline.strip():
                    continue
                a = dline.split()
                timesteps.append(ts)
                x.append(float(a[2]));  y.append(float(a[3]));  z.append(float(a[4]))
                vx.append(float(a[5])); vy.append(float(a[6])); vz.append(float(a[7]))
                extra.append([float(q) for q in a[8:13]])
                for _ in range(natoms - 1):
                    f.readline()
    out = {
        "t": np.asarray(timesteps),
        "x": np.asarray(x),  "y": np.asarray(y),  "z": np.asarray(z),
        "vx": np.asarray(vx), "vy": np.asarray(vy), "vz": np.asarray(vz),
    }
    # optional trailing columns (GCA dumps):
    # p_gca_x p_gca_y p_gca_z p_gca_vpar p_gca_mu
    if extra and len(extra[0]) == 5:
        e = np.asarray(extra)
        out["gcx"], out["gcy"], out["gcz"] = e[:, 0], e[:, 1], e[:, 2]
        out["vpar"], out["mu"] = e[:, 3], e[:, 4]
    return out


def read_equilibrium_bfield(path):
    """Read /equilibrium from plasma.h5; build B_R, B_phi, B_Z + d|B|/dR, d|B|/dZ."""
    import h5py
    with h5py.File(path, "r") as f:
        r   = np.asarray(f["equilibrium/r"][...],   dtype=float)
        z   = np.asarray(f["equilibrium/z"][...],   dtype=float)
        psi = np.asarray(f["equilibrium/psi"][...], dtype=float)
        btf = float(np.asarray(f["equilibrium/btf"][...]).item())
        rtf = float(np.asarray(f["equilibrium/rtf"][...]).item())

    # make_khan_plasma_h5.py stores psi as (nz, nr); transpose only a
    # genuinely mismatched non-square array (a square grid always
    # "matches" both orders, and blind-transposing corrupts B by ~3%)
    if psi.shape != (len(z), len(r)):
        if psi.shape == (len(r), len(z)):
            psi = psi.T
        else:
            raise RuntimeError(f"Unexpected psi shape {psi.shape}")

    dpsi_dz, dpsi_dr = np.gradient(psi, z[1] - z[0], r[1] - r[0])
    rr = np.meshgrid(r, z)[0]
    safe_r = np.where(np.abs(rr) > 1e-12, rr, 1e-12)

    bR    = -dpsi_dz / safe_r
    bZ    =  dpsi_dr / safe_r
    bphi  = (btf * rtf) / safe_r
    bmag  = np.sqrt(bR * bR + bphi * bphi + bZ * bZ)
    dBdz, dBdr = np.gradient(bmag, z[1] - z[0], r[1] - r[0])
    return r, z, bR, bphi, bZ, dBdr, dBdz


def bilinear_sample(rg, zg, field, r_t, z_t):
    ir = np.clip(np.searchsorted(rg, r_t) - 1, 0, len(rg) - 2)
    iz = np.clip(np.searchsorted(zg, z_t) - 1, 0, len(zg) - 2)
    r0, r1 = rg[ir], rg[ir + 1]
    z0, z1 = zg[iz], zg[iz + 1]
    tr = np.where(r1 > r0, (r_t - r0) / (r1 - r0), 0.0)
    tz = np.where(z1 > z0, (z_t - z0) / (z1 - z0), 0.0)
    f00, f10 = field[iz, ir],     field[iz, ir + 1]
    f01, f11 = field[iz + 1, ir], field[iz + 1, ir + 1]
    return (1 - tr) * (1 - tz) * f00 + tr * (1 - tz) * f10 \
         + (1 - tr) * tz       * f01 + tr * tz       * f11


# ---------------------------------------------------------------------------
# Cylindrical decomposition + GC diagnostics
# ---------------------------------------------------------------------------
def cyl_from_cart(traj, mode="3d"):
    x, y, z = traj["x"], traj["y"], traj["z"]
    vx, vy, vz = traj["vx"], traj["vy"], traj["vz"]
    if mode == "2dcart":       # slots: x=R, y=Z, z=toroidal
        return x, y, np.zeros_like(x), vx, vz, vy
    if mode == "axi":          # slots: x=Z, y=R, z=toroidal
        return y, x, np.zeros_like(x), vy, vz, vx
    R   = np.sqrt(x * x + y * y)
    safe_R = np.where(R > 1e-12, R, 1e-12)
    cosp = x / safe_R
    sinp = y / safe_R
    vR   =  vx * cosp + vy * sinp
    vphi = -vx * sinp + vy * cosp
    return R, z, np.arctan2(y, x), vR, vphi, vz


def gc_diagnostics(traj, eq_data, mass_kg, charge_C):
    rg, zg, bRg, bphig, bZg, dBdr_g, dBdz_g = eq_data
    R, Z, _, vR, vphi, vZ = cyl_from_cart(traj)

    bR   = bilinear_sample(rg, zg, bRg,   R, Z)
    bphi = bilinear_sample(rg, zg, bphig, R, Z)
    bZ   = bilinear_sample(rg, zg, bZg,   R, Z)
    dBdr = bilinear_sample(rg, zg, dBdr_g, R, Z)
    dBdz = bilinear_sample(rg, zg, dBdz_g, R, Z)

    b2 = bR * bR + bphi * bphi + bZ * bZ
    bmag    = np.sqrt(b2)
    safe_b  = np.where(bmag > 1e-30, bmag, 1e-30)
    safe_b2 = np.where(b2   > 1e-30, b2,   1e-30)

    cR   = vphi * bZ   - vZ * bphi
    cZ   = vR   * bphi - vphi * bR
    alpha = mass_kg / (charge_C * safe_b2)
    Rgc = R + alpha * cR
    Zgc = Z + alpha * cZ

    vpar   = (vR * bR + vphi * bphi + vZ * bZ) / safe_b
    v2     = vR * vR + vphi * vphi + vZ * vZ
    vperp2 = np.clip(v2 - vpar * vpar, 0.0, None)
    mu     = mass_kg * vperp2 / (2.0 * safe_b)
    speed  = np.sqrt(v2)

    vperp = np.sqrt(vperp2)
    qm    = abs(charge_C) / mass_kg
    rho_l = vperp / np.where(qm * safe_b > 1e-30, qm * safe_b, 1e-30)
    gradb = np.sqrt(dBdr * dBdr + dBdz * dBdz)
    lb    = safe_b / np.where(gradb > 1e-30, gradb, np.inf)
    ratio = rho_l / np.where(lb > 1e-30, lb, np.inf)
    return Rgc, Zgc, mu, speed, ratio


# ---------------------------------------------------------------------------
def fmt_range(arr):
    return "[empty]" if arr.size == 0 else f"[{arr.min():.4f}, {arr.max():.4f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Compare a Boris trajectory with one selectable GCA dump")
    parser.add_argument(
        "--gca-dump", default="traj.gca.rk2",
        help="GCA dump basename in output/ (default: traj.gca.rk2)")
    parser.add_argument(
        "--tag", default="rk2",
        help="suffix for generated figure names (default: rk2)")
    parser.add_argument(
        "--mode", default="3d", choices=["3d", "2dcart", "axi"],
        help="slot layout of the GCA dump (Boris reference is always 3d)")
    args = parser.parse_args()

    base   = os.path.dirname(__file__)
    outdir = os.path.join(base, "output")
    bpath  = os.path.join(outdir, "traj.boris")
    gpath  = os.path.join(outdir, args.gca_dump)
    for p in (bpath, gpath):
        if not os.path.exists(p):
            raise SystemExit(f"FAIL: missing dump {p} — rerun the decks first")
    import time
    for p in (bpath, gpath):
        age = (time.time() - os.path.getmtime(p)) / 60.0
        print(f"  using {p}  (modified {age:.0f} min ago)")
    boris  = read_sparta_dump(bpath)
    gca    = read_sparta_dump(gpath)
    if len(boris["t"]) == 0 or len(gca["t"]) == 0:
        raise SystemExit("FAIL: empty dump")

    # drop frames dumped before the pusher populated the GC state
    if "vpar" in gca:
        live = (gca["mu"] > 0.0) | (gca["vpar"] != 0.0)
        gca = {k: arr[live] for k, arr in gca.items()}
        # 2D dumps carry the GC in (slot) x/y with gcz = toroidal drift;
        # a 3D dump has gcz = Z. Catch a dump/mode mismatch early.
        planar = np.all(gca["z"] == 0.0)
        if args.mode == "3d" and planar:
            raise SystemExit(f"FAIL: {args.gca_dump} looks like a 2D dump "
                             "— pass --mode 2dcart or --mode axi")
        if args.mode != "3d" and not planar:
            raise SystemExit(f"FAIL: {args.gca_dump} looks like a 3D dump "
                             "— drop the --mode flag")
        print(f"  GC diagnostics: stored p_gca_* state ({args.mode})")
    else:
        print("  GC diagnostics: reconstructed from x,v (legacy dump)")

    R_b, Z_b, _, vR_b, vphi_b, vZ_b = cyl_from_cart(boris)
    R_g, Z_g, _, vR_g, vphi_g, vZ_g = cyl_from_cart(gca, args.mode)
    v_b = np.sqrt(boris["vx"]**2 + boris["vy"]**2 + boris["vz"]**2)
    v_g = np.sqrt(gca["vx"]**2   + gca["vy"]**2   + gca["vz"]**2)

    mH = 1.6726219236951e-27
    qH = 1.602176634e-19
    eq_data = read_equilibrium_bfield(os.path.join(base, "khan_plasma.h5"))
    Rgc_b, Zgc_b, mu_b, sp_b, rat_b = gc_diagnostics(boris, eq_data, mH, qH)
    if "vpar" in gca:
        # Stored GC state: the gated invariant is H = 0.5 m vpar^2 + mu B(X_gc)
        # (mu is invariant by construction, so speed traces H exactly).
        rg, zg, bRg, bphig, bZg, dBdr_g, dBdz_g = eq_data
        if args.mode == "2dcart":
            Rgc_g, Zgc_g = gca["gcx"], gca["gcy"]
        elif args.mode == "axi":
            Rgc_g, Zgc_g = gca["gcy"], gca["gcx"]
        else:
            Rgc_g = np.sqrt(gca["gcx"]**2 + gca["gcy"]**2)
            Zgc_g = gca["gcz"]
        bR   = bilinear_sample(rg, zg, bRg,   Rgc_g, Zgc_g)
        bphi = bilinear_sample(rg, zg, bphig, Rgc_g, Zgc_g)
        bZ   = bilinear_sample(rg, zg, bZg,   Rgc_g, Zgc_g)
        bmag = np.sqrt(bR*bR + bphi*bphi + bZ*bZ)
        dBdr = bilinear_sample(rg, zg, dBdr_g, Rgc_g, Zgc_g)
        dBdz = bilinear_sample(rg, zg, dBdz_g, Rgc_g, Zgc_g)
        mu_g   = gca["mu"]
        vperp2 = 2.0 * mu_g * bmag / mH
        sp_g   = np.sqrt(gca["vpar"]**2 + vperp2)  # sqrt(2H/m)
        if args.mode != "3d":
            v_g = sp_g                             # 2D dumps carry the GC chord
        rho_l  = np.sqrt(vperp2) / ((qH / mH) * bmag)
        gradb  = np.sqrt(dBdr*dBdr + dBdz*dBdz)
        rat_g  = rho_l * gradb / bmag
    else:
        Rgc_g, Zgc_g, mu_g, sp_g, rat_g = gc_diagnostics(gca, eq_data, mH, qH)

    dt   = 5e-10                         # match in.boris / in.gca timestep
    t_b  = boris["t"] * dt * 1e6         # μs
    t_g  = gca["t"]   * dt * 1e6

    # ---------------------------------------------------------------- Figure 1: trajectory
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(R_b, Z_b, "b-",  label="Boris",      alpha=0.85)
    ax.plot(R_g, Z_g, "r--", label="GCA hybrid", alpha=0.85)
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(r"$R$--$Z$ trajectory")
    ax.legend(frameon=False)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.plot(t_b, R_b, "b-",  label="Boris")
    ax.plot(t_g, R_g, "r--", label="GCA")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$R$ (m)")
    ax.set_title(r"$R(t)$")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(t_b, Z_b, "b-",  label="Boris")
    ax.plot(t_g, Z_g, "r--", label="GCA")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(r"$Z(t)$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(t_b, v_b / 1e3, "b-",  label="Boris")
    ax.plot(t_g, v_g / 1e3, "r--", label="GCA")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$|v|$ (km/s)")
    ax.set_title(r"$|v|(t)$")
    ax.legend(frameon=False)

    fig.suptitle(r"Boris vs.\ GCA hybrid pusher — single H$^+$ in Khan tokamak field",
                 fontsize=18)
    fig.tight_layout()
    figure1 = f"boris_vs_gca_{args.tag}"
    save(fig, figure1, outdir)
    print(f"Wrote {os.path.join(outdir, figure1 + '.{png,pdf}')}")
    print(f"  Boris: {len(boris['t'])} frames, R range {fmt_range(R_b)}")
    print(f"  GCA:   {len(gca['t'])} frames, R range {fmt_range(R_g)}")

    # ---------------------------------------------------------------- Figure 2: GC diagnostics
    switch_factor = 2.5
    ratio_limit   = 1.0 / switch_factor

    fig2, ax2 = plt.subplots(2, 2, figsize=(13, 10))

    ax = ax2[0, 0]
    ax.plot(Rgc_b, Zgc_b, "b-",  label="Boris GC")
    ax.plot(Rgc_g, Zgc_g, "r--", label="GCA GC")
    ax.set_xlabel(r"$R_{\rm gc}$ (m)")
    ax.set_ylabel(r"$Z_{\rm gc}$ (m)")
    ax.set_title("Guiding-center $R$--$Z$")
    ax.legend(frameon=False)
    ax.set_aspect("equal")

    ax = ax2[0, 1]
    ax.plot(t_b, Rgc_b, "b-",  label="Boris GC")
    ax.plot(t_g, Rgc_g, "r--", label="GCA GC")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$R_{\rm gc}$ (m)")
    ax.set_title(r"$R_{\rm gc}(t)$")
    ax.legend(frameon=False)

    mu_b_rel = (mu_b - mu_b[0]) / max(abs(mu_b[0]), 1e-40)
    mu_g_rel = (mu_g - mu_g[0]) / max(abs(mu_g[0]), 1e-40)
    ax = ax2[1, 0]
    ax.plot(t_b, 1e6 * mu_b_rel, "b-",  label="Boris")
    ax.plot(t_g, 1e6 * mu_g_rel, "r--", label="GCA")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$\Delta\mu/\mu_0$ (ppm)")
    ax.set_title("magnetic-moment drift")
    ax.legend(frameon=False)

    sp_b_rel = (sp_b - sp_b[0]) / max(abs(sp_b[0]), 1e-40)
    sp_g_rel = (sp_g - sp_g[0]) / max(abs(sp_g[0]), 1e-40)
    ax = ax2[1, 1]
    ax.plot(t_b, 1e6 * sp_b_rel, "b-",  label="Boris")
    ax.plot(t_g, 1e6 * sp_g_rel, "r--", label="GCA")
    ax.set_xlabel(r"$t$ (μs)")
    ax.set_ylabel(r"$\Delta|v|/|v_0|$ (ppm)")
    ax.set_title("speed drift")
    ax.legend(frameon=False)

    fig2.suptitle("Guiding-center diagnostics", fontsize=18)
    fig2.tight_layout()
    figure2 = f"boris_vs_gca_gc_{args.tag}"
    save(fig2, figure2, outdir)
    print(f"Wrote {os.path.join(outdir, figure2 + '.{png,pdf}')}")

    # ---------------------------------------------------------------- Figure 3: rho_L / L_B
    fig3, ax3 = plt.subplots(1, 1, figsize=(11, 5))
    ax3.plot(t_b, rat_b, "b-",  label=r"Boris  $\rho_L / L_B$")
    ax3.plot(t_g, rat_g, "r--", label=r"GCA  $\rho_L / L_B$")
    ax3.axhline(ratio_limit, color="k", ls=":",
                label=f"switch threshold = 1/{switch_factor:g}")
    ax3.set_xlabel(r"$t$ (μs)")
    ax3.set_ylabel(r"$\rho_L / L_B$")
    ax3.set_title("GCA validity diagnostic")
    ax3.legend(frameon=False)
    fig3.tight_layout()
    figure3 = f"boris_vs_gca_rhoL_over_LB_{args.tag}"
    save(fig3, figure3, outdir)
    print(f"Wrote {os.path.join(outdir, figure3 + '.{png,pdf}')}")

    common, ib, ig = np.intersect1d(boris["t"], gca["t"], return_indices=True)
    rms_gc = np.sqrt(np.mean((Rgc_b[ib] - Rgc_g[ig])**2 +
                              (Zgc_b[ib] - Zgc_g[ig])**2))
    print(f"  Boris frames: {len(Rgc_b)}   GCA frames: {len(Rgc_g)}   "
          f"common: {len(common)}")
    print(f"  GC RMS distance (m):   {rms_gc:.3e}")
    print(f"  Final Δμ/μ₀ (ppm):     Boris {1e6*mu_b_rel[-1]:+.3f}, "
          f"GCA {1e6*mu_g_rel[-1]:+.3f}")
    print(f"  Final Δ|v|/|v₀| (ppm): Boris {1e6*sp_b_rel[-1]:+.3f}, "
          f"GCA {1e6*sp_g_rel[-1]:+.3f}")
    print(f"  ρ_L/L_B threshold:     {ratio_limit:.4f}  (switch_factor={switch_factor:g})")
    print(f"  fraction below thresh: Boris {np.mean(rat_b < ratio_limit):.3f}, "
          f"GCA {np.mean(rat_g < ratio_limit):.3f}")
    print(f"  max ρ_L/L_B:           Boris {np.max(rat_b):.4f}, GCA {np.max(rat_g):.4f}")

    # ---- PASS/FAIL gate ----
    # secular = fitted linear slope of the energy trace over the whole run
    # (a final-value check can hide drift under a bounce-phase oscillation)
    e_rel = (sp_g / sp_g[0])**2 - 1.0     # ΔH/H of the gated invariant
    secular_ppm = 1e6 * np.polyfit(t_g, e_rel, 1)[0] * (t_g[-1] - t_g[0])
    print(f"  ΔH/H secular (fit):    {secular_ppm:+.3f} ppm over the run")
    checks = [
        ("GCA secular energy drift: fitted |ΔH/H| < 20 ppm",
         abs(secular_ppm) < 20.0),
        ("GCA bounded energy excursion: max |ΔH/H| < 500 ppm",
         1e6 * np.max(np.abs(e_rel)) < 500.0),
        ("GC tracks Boris: RMS distance < 1e-2 m", rms_gc < 1e-2),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    allok = all(ok for _, ok in checks)
    print(f"{'PASS' if allok else 'FAIL'}: verification/pushers/orbit ({args.tag})")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
