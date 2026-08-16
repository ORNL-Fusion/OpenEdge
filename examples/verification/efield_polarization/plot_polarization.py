"""Polarization-drift verification: compare Boris vs. guiding-center theory.

    B = B0 * zhat
    Ex(t) = E0 * sin(omega * t)

Theory (gyro-averaged):
    v_E(t)   = -Ex(t) / B0
    v_pol(t) = -(m/q) * (1/B0^2) * dEx/dt
    v_y(t)   = v_E(t) + v_pol(t)

Reads `state` (produced by running in.input), writes three figures
into `figs_polarization/`.
"""

import math
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pass/fail thresholds (fraction of E0/B0 drift amplitude)
RMS_TOL    = 0.05   # rms of gyro-averaged vy - theory
OFFSET_TOL = 0.01   # mean systematic offset
DISP_TOL   = 0.05   # max |y - y_theory| / max|y_theory|


def parse_dump(path):
    timesteps, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            t = int(lines[i+1]); i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i+1]); i += 2
        elif s.startswith("ITEM: ATOMS"):
            timesteps.append(t)
            for j in range(n):
                a = lines[i+1+j].split()
                x.append(float(a[2])); y.append(float(a[3])); z.append(float(a[4]))
                vx.append(float(a[5])); vy.append(float(a[6])); vz.append(float(a[7]))
            i += n + 1
        else:
            i += 1
    return (np.asarray(timesteps, dtype=int),
            np.asarray(x), np.asarray(y), np.asarray(z),
            np.asarray(vx), np.asarray(vy), np.asarray(vz))


def moving_average(a, window):
    if window <= 1:
        return a
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="same")


def save(fig, name, outdir):
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")


def main():
    base = Path(__file__).resolve().parent
    outdir = base / "figs_polarization"
    outdir.mkdir(parents=True, exist_ok=True)

    # SI constants — must match in.input
    qe = 1.602176634e-19
    mi = 1.67262192369e-27
    B0 = 1.0
    E0 = 50.0
    omegac = qe * B0 / mi
    omega  = 0.05 * omegac
    dt_sim = 0.02 / omegac
    m_over_q = mi / qe

    tstep, _, y, _, _, vy, _ = parse_dump(base / "state")
    dump_stride = (tstep[1] - tstep[0]) if len(tstep) > 1 else 1
    dt_out = dt_sim * dump_stride
    t = tstep * dt_sim

    Ex     = E0 * np.sin(omega * t)
    dEx_dt = E0 * omega * np.cos(omega * t)
    v_theory = -Ex / B0 - m_over_q * dEx_dt / (B0 * B0)

    Tgyro = 2.0 * math.pi * m_over_q / B0
    Ngyro = max(1, int(round(Tgyro / dt_out)))
    vy_smooth  = moving_average(vy, Ngyro)
    vth_smooth = moving_average(v_theory, Ngyro)
    y_gc_theory = np.cumsum(v_theory) * dt_out
    print(f"gyro-averaging window Ngyro = {Ngyro} outputs")

    plt.rcParams.update({
        "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14,
    })

    def new_ax(title, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        return fig, ax

    fig, ax = new_ax(r"Raw $v_y(t)$: sim vs theory", r"$v_y$ [m/s]")
    ax.plot(t, vy, label="sim (raw)", alpha=0.4)
    ax.plot(t, v_theory, label="theory (guiding center)", lw=1.5)
    ax.legend(frameon=False); fig.tight_layout()
    save(fig, "a_vy_raw_vs_theory", outdir)

    fig, ax = new_ax(r"Gyro-averaged $v_y(t)$", r"$\langle v_y \rangle_{\rm gyro}$ [m/s]")
    ax.plot(t, vy_smooth,  label="sim",    lw=1.5)
    ax.plot(t, vth_smooth, label="theory", lw=1.5, ls="--")
    ax.legend(frameon=False); fig.tight_layout()
    save(fig, "b_vy_gyroavg_vs_theory", outdir)

    fig, ax = new_ax(r"Guiding-center $y(t)$", "y [m]")
    ax.plot(t, y,           label="sim",    alpha=0.6)
    ax.plot(t, y_gc_theory, label="theory", lw=1.5)
    ax.legend(frameon=False); fig.tight_layout()
    save(fig, "c_y_displacement_vs_theory", outdir)

    print(f"wrote figures to {outdir}")

    # pass/fail: compare gyro-averaged drift and displacement to theory
    amp = E0 / B0
    sl = slice(2 * Ngyro, -2 * Ngyro)
    rms    = float(np.sqrt(np.mean((vy_smooth[sl] - vth_smooth[sl]) ** 2))) / amp
    offset = abs(float(np.mean(vy_smooth[sl] - vth_smooth[sl]))) / amp
    disp   = float(np.max(np.abs(y - y_gc_theory))) / float(np.max(np.abs(y_gc_theory)))

    checks = [
        ("vy rms vs theory",    rms,    RMS_TOL),
        ("vy mean offset",      offset, OFFSET_TOL),
        ("y displacement error", disp,  DISP_TOL),
    ]
    failed = False
    for name, val, tol in checks:
        ok = val < tol
        failed |= not ok
        print(f"  {name}: {val:.4f} (tol {tol}) {'ok' if ok else 'FAIL'}")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
