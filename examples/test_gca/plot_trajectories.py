#!/usr/bin/env python3
"""Compare Boris vs GCA trajectories and guiding-center diagnostics."""

import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def read_sparta_dump(path):
    """Read SPARTA particle dump file, return dict of arrays."""
    timesteps, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                ts = int(f.readline().strip())
                f.readline()  # NUMBER OF ATOMS
                natoms = int(f.readline().strip())
                f.readline()  # ITEM: BOX BOUNDS ...
                for _ in range(3):  # x/y/z bounds lines
                    f.readline()
                f.readline()  # ITEM: ATOMS header
                if natoms == 0:
                    continue
                # Read first (only) particle
                dline = f.readline()
                if not dline or not dline.strip():
                    continue
                parts = dline.split()
                try:
                    timesteps.append(ts)
                    x.append(float(parts[2]))
                    y.append(float(parts[3]))
                    z.append(float(parts[4]))
                    vx.append(float(parts[5]))
                    vy.append(float(parts[6]))
                    vz.append(float(parts[7]))
                except (ValueError, IndexError):
                    timesteps.pop() if len(timesteps) > len(x) else None
                    continue
                # Skip remaining atoms
                for _ in range(natoms - 1):
                    f.readline()
    return {
        "t": np.array(timesteps),
        "x": np.array(x), "y": np.array(y), "z": np.array(z),
        "vx": np.array(vx), "vy": np.array(vy), "vz": np.array(vz),
    }


def fmt_range(arr):
    """Format [min, max] for non-empty arrays and mark empty arrays clearly."""
    if arr.size == 0:
        return "[empty]"
    return f"[{arr.min():.4f}, {arr.max():.4f}]"


def read_equilibrium_bfield(path):
    """Read .equ and build Br/Bt/Bz and grad|B| arrays on (Z,R) grid."""
    jm = km = None
    btf = rtf = None
    r_vals, z_vals, psi_vals = [], [], []
    mode = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if len(tok) >= 3 and tok[0] == "jm" and tok[1] == "=":
                jm = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "km" and tok[1] == "=":
                km = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "btf" and tok[1] == "=":
                btf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "rtf" and tok[1] == "=":
                rtf = float(tok[2]); continue

            if tok[0] == "r(1:jm);":
                mode = "r"; continue
            if tok[0] == "z(1:km);":
                mode = "z"; continue
            if tok[0].startswith("((psi(j,k)-psib,j=1,jm),k=1,km)"):
                mode = "psi"; continue

            for t in tok:
                try:
                    v = float(t)
                except ValueError:
                    continue
                if mode == "r":
                    r_vals.append(v)
                elif mode == "z":
                    z_vals.append(v)
                elif mode == "psi":
                    psi_vals.append(v)

    if jm is None or km is None or btf is None or rtf is None:
        raise RuntimeError(f"Failed to parse equilibrium header: {path}")

    r = np.asarray(r_vals[:jm], dtype=float)
    z = np.asarray(z_vals[:km], dtype=float)
    psi = np.asarray(psi_vals[: jm * km], dtype=float).reshape((km, jm))

    dpsi_dz, dpsi_dr = np.gradient(psi, z[1] - z[0], r[1] - r[0])
    rr = np.meshgrid(r, z)[0]
    safe_r = np.where(np.abs(rr) > 1e-12, rr, 1e-12)

    br = -dpsi_dz / safe_r
    bz = dpsi_dr / safe_r
    bt = (btf * rtf) / safe_r
    bmag = np.sqrt(br * br + bt * bt + bz * bz)
    dbmag_dz, dbmag_dr = np.gradient(bmag, z[1] - z[0], r[1] - r[0])
    return r, z, br, bt, bz, dbmag_dr, dbmag_dz


def bilinear_sample(rg, zg, field, r_t, z_t):
    """Bilinear sample field[iz,ir] at arrays (r_t,z_t)."""
    ir = np.searchsorted(rg, r_t) - 1
    iz = np.searchsorted(zg, z_t) - 1
    ir = np.clip(ir, 0, len(rg) - 2)
    iz = np.clip(iz, 0, len(zg) - 2)

    r0 = rg[ir]; r1 = rg[ir + 1]
    z0 = zg[iz]; z1 = zg[iz + 1]
    tr = np.where(r1 > r0, (r_t - r0) / (r1 - r0), 0.0)
    tz = np.where(z1 > z0, (z_t - z0) / (z1 - z0), 0.0)

    f00 = field[iz, ir]
    f10 = field[iz, ir + 1]
    f01 = field[iz + 1, ir]
    f11 = field[iz + 1, ir + 1]
    return (1 - tr) * (1 - tz) * f00 + tr * (1 - tz) * f10 + (1 - tr) * tz * f01 + tr * tz * f11


def sample_b_cart(eq_data, x, y, z):
    """Sample B and grad|B| from equilibrium and return Cartesian components."""
    rg, zg, brg, btg, bzg, dbdr_g, dbdz_g = eq_data
    r = np.sqrt(x * x + y * y)
    rr = np.where(r > 1e-20, r, 1e-20)
    br = bilinear_sample(rg, zg, brg, r, z)
    bt = bilinear_sample(rg, zg, btg, r, z)
    bz = bilinear_sample(rg, zg, bzg, r, z)
    dbdr = bilinear_sample(rg, zg, dbdr_g, r, z)
    dbdz = bilinear_sample(rg, zg, dbdz_g, r, z)
    cphi = x / rr
    sphi = y / rr
    bx = br * cphi - bt * sphi
    by = br * sphi + bt * cphi
    gbx = dbdr * cphi
    gby = dbdr * sphi
    gbz = dbdz
    return bx, by, bz, gbx, gby, gbz


def gc_diagnostics(traj, eq_data):
    """Compute guiding-center position, mu, and rho_L/L_B diagnostics."""
    mp = 1.67262192369e-27
    qe = 1.602176634e-19

    x, y, z = traj["x"], traj["y"], traj["z"]
    vx, vy, vz = traj["vx"], traj["vy"], traj["vz"]
    bx, by, bz, gbx, gby, gbz = sample_b_cart(eq_data, x, y, z)

    b2 = bx * bx + by * by + bz * bz
    bmag = np.sqrt(b2)
    safe_b2 = np.where(b2 > 1e-30, b2, 1e-30)
    safe_b = np.where(bmag > 1e-30, bmag, 1e-30)

    cx = vy * bz - vz * by
    cy = vz * bx - vx * bz
    cz = vx * by - vy * bx

    alpha = mp / (qe * safe_b2)
    xgc = x + alpha * cx
    ygc = y + alpha * cy
    zgc = z + alpha * cz

    vpar = (vx * bx + vy * by + vz * bz) / safe_b
    v2 = vx * vx + vy * vy + vz * vz
    vperp2 = np.clip(v2 - vpar * vpar, 0.0, None)
    mu = mp * vperp2 / (2.0 * safe_b)
    speed = np.sqrt(v2)

    qm_abs = qe / mp
    vperp = np.sqrt(vperp2)
    rho_l = vperp / np.where(qm_abs * safe_b > 1e-30, qm_abs * safe_b, 1e-30)
    gradb_mag = np.sqrt(gbx * gbx + gby * gby + gbz * gbz)
    lb = safe_b / np.where(gradb_mag > 1e-30, gradb_mag, np.inf)
    ratio = rho_l / np.where(lb > 1e-30, lb, np.inf)

    return xgc, ygc, zgc, mu, speed, ratio


def main():
    outdir = os.path.join(os.path.dirname(__file__), "output")
    boris_file = os.path.join(outdir, "traj.boris")
    gca_file = os.path.join(outdir, "traj.gca")

    boris = read_sparta_dump(boris_file)
    gca = read_sparta_dump(gca_file)

    # R = sqrt(x^2 + y^2), Z = z
    R_boris = np.sqrt(boris["x"]**2 + boris["y"]**2)
    Z_boris = boris["z"]
    R_gca = np.sqrt(gca["x"]**2 + gca["y"]**2)
    Z_gca = gca["z"]

    # Speed
    v_boris = np.sqrt(boris["vx"]**2 + boris["vy"]**2 + boris["vz"]**2)
    v_gca = np.sqrt(gca["vx"]**2 + gca["vy"]**2 + gca["vz"]**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- R-Z trajectory ---
    ax = axes[0, 0]
    ax.plot(R_boris, Z_boris, "b-", lw=0.5, alpha=0.7, label="Boris (500 sub)")
    ax.plot(R_gca, Z_gca, "r--", lw=1.0, alpha=0.8, label="GCA hybrid")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("R-Z Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # --- R vs time ---
    ax = axes[0, 1]
    dt = 1e-9  # from input
    t_boris = boris["t"] * dt * 1e6  # microseconds
    t_gca = gca["t"] * dt * 1e6
    ax.plot(t_boris, R_boris, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, R_gca, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("R [m]")
    ax.set_title("R vs Time")
    ax.legend()

    # --- Z vs time ---
    ax = axes[1, 0]
    ax.plot(t_boris, Z_boris, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, Z_gca, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Z vs Time")
    ax.legend()

    # --- Speed vs time ---
    ax = axes[1, 1]
    ax.plot(t_boris, v_boris / 1e3, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, v_gca / 1e3, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("|v| [km/s]")
    ax.set_title("Speed vs Time")
    ax.legend()

    fig.suptitle("Boris vs GCA Hybrid Pusher Comparison", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(outdir, "boris_vs_gca.png")
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")
    print(f"  Boris: {len(boris['t'])} frames, R range {fmt_range(R_boris)}")
    print(f"  GCA:   {len(gca['t'])} frames, R range {fmt_range(R_gca)}")

    # --- Guiding-center diagnostics ---
    equ_file = os.path.join(os.path.dirname(__file__), "g000001.00001_symm.X4.equ")
    eq_data = read_equilibrium_bfield(equ_file)
    bxgc, bygc, bzgc, mu_b, sp_b, ratio_b = gc_diagnostics(boris, eq_data)
    gxgc, gygc, gzgc, mu_g, sp_g, ratio_g = gc_diagnostics(gca, eq_data)
    Rgc_b = np.sqrt(bxgc * bxgc + bygc * bygc)
    Rgc_g = np.sqrt(gxgc * gxgc + gygc * gygc)
    switch_factor = 2.5
    ratio_limit = 1.0 / switch_factor

    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax2[0, 0]
    ax.plot(Rgc_b, bzgc, "b-", lw=0.8, alpha=0.8, label="Boris GC")
    ax.plot(Rgc_g, gzgc, "r--", lw=1.0, alpha=0.9, label="GCA GC")
    ax.set_xlabel("R_gc [m]")
    ax.set_ylabel("Z_gc [m]")
    ax.set_title("Guiding-Center R-Z")
    ax.legend()
    ax.set_aspect("equal")

    ax = ax2[0, 1]
    ax.plot(t_boris, Rgc_b, "b-", lw=0.8, alpha=0.8, label="Boris GC")
    ax.plot(t_gca, Rgc_g, "r--", lw=1.0, alpha=0.9, label="GCA GC")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("R_gc [m]")
    ax.set_title("R_gc vs Time")
    ax.legend()

    mu_b_rel = (mu_b - mu_b[0]) / max(abs(mu_b[0]), 1e-40)
    mu_g_rel = (mu_g - mu_g[0]) / max(abs(mu_g[0]), 1e-40)
    ax = ax2[1, 0]
    ax.plot(t_boris, 1e6 * mu_b_rel, "b-", lw=0.8, alpha=0.8, label="Boris")
    ax.plot(t_gca, 1e6 * mu_g_rel, "r--", lw=1.0, alpha=0.9, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Δmu/mu0 [ppm]")
    ax.set_title("Magnetic Moment Drift")
    ax.legend()

    sp_b_rel = (sp_b - sp_b[0]) / max(abs(sp_b[0]), 1e-40)
    sp_g_rel = (sp_g - sp_g[0]) / max(abs(sp_g[0]), 1e-40)
    ax = ax2[1, 1]
    ax.plot(t_boris, 1e6 * sp_b_rel, "b-", lw=0.8, alpha=0.8, label="Boris")
    ax.plot(t_gca, 1e6 * sp_g_rel, "r--", lw=1.0, alpha=0.9, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Δ|v|/|v0| [ppm]")
    ax.set_title("Speed Drift")
    ax.legend()

    fig2.suptitle("Boris vs GCA Guiding-Center Diagnostics", fontsize=14)
    fig2.tight_layout()
    out_gc_png = os.path.join(outdir, "boris_vs_gca_gc.png")
    fig2.savefig(out_gc_png, dpi=180)
    print(f"Wrote {out_gc_png}")

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4.5))
    ax3.plot(t_boris, ratio_b, "b-", lw=0.8, alpha=0.8, label="Boris rho_L/L_B")
    ax3.plot(t_gca, ratio_g, "r--", lw=1.0, alpha=0.9, label="GCA rho_L/L_B")
    ax3.axhline(ratio_limit, color="k", ls=":", lw=1.2,
                label=f"switch threshold = 1/{switch_factor:g}")
    ax3.set_xlabel("Time [us]")
    ax3.set_ylabel("rho_L / L_B")
    ax3.set_title("GCA Validity Diagnostic")
    ax3.legend()
    fig3.tight_layout()
    out_ratio_png = os.path.join(outdir, "boris_vs_gca_rhoL_over_LB.png")
    fig3.savefig(out_ratio_png, dpi=180)
    print(f"Wrote {out_ratio_png}")

    rms_gc = np.sqrt(np.mean((Rgc_b - Rgc_g) ** 2 + (bzgc - gzgc) ** 2))
    print(f"  GC RMS distance [m]: {rms_gc:.6e}")
    print(f"  Final mu drift [ppm]: Boris {1e6*mu_b_rel[-1]:.3f}, GCA {1e6*mu_g_rel[-1]:.3f}")
    print(f"  Final |v| drift [ppm]: Boris {1e6*sp_b_rel[-1]:.3f}, GCA {1e6*sp_g_rel[-1]:.3f}")
    print(f"  rho_L/L_B threshold: {ratio_limit:.6f} (switch_factor={switch_factor:g})")
    print(f"  Fraction below threshold: Boris {np.mean(ratio_b < ratio_limit):.3f}, "
          f"GCA {np.mean(ratio_g < ratio_limit):.3f}")
    print(f"  Max rho_L/L_B: Boris {np.max(ratio_b):.6f}, GCA {np.max(ratio_g):.6f}")
    plt.show()

if __name__ == "__main__":
    main()
