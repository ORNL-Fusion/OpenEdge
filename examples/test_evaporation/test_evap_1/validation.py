from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Mirror model in src/fix_evaporation.cpp
AM = 1.53e-26
RHO = 534.0
CP = 4200.0
DHM = 3.158e3
AN = 6.022e23
A1 = 5.055
B1 = -8023.0
XM1 = 6.939


def parse_case_file(case_path: Path) -> dict:
    text = case_path.read_text()

    def get_float(pattern: str, default: float) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    def get_dump(pattern: str, default: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else default

    return {
        "dt": get_float(r"variable\s+dt\s+equal\s+([0-9eE+\-.]+)", 1.0e-5),
        "rd0": get_float(r"variable\s+rd\s+equal\s+([0-9eE+\-.]+)", 2.5e-3),
        "T0": get_float(r"variable\s+temp\s+equal\s+([0-9eE+\-.]+)", 773.15),
        "Qs": get_float(r"heatflux/constant\s+([0-9eE+\-.]+)", 50e6),
        "dump": get_dump(r"dump\s+\d+\s+particle\s+all\s+\$\{N1\}\s+(\S+)", "case.1"),
    }


def load_openedge_dump(path_dump: Path, dt: float):
    tstep, rd, te = [], [], []
    tracked_id = None

    with path_dump.open("r") as f:
        lines = f.readlines()

    i = 0
    cur_t = None
    natoms = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            cur_t = int(lines[i + 1].strip())
            i += 2
            continue
        if line == "ITEM: NUMBER OF ATOMS":
            natoms = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: ATOMS"):
            cols = line.split()[2:]
            col = {name: idx for idx, name in enumerate(cols)}
            req = ["id", "temp", "radius"]
            if not all(k in col for k in req):
                raise RuntimeError(f"Missing required dump columns in {path_dump}: {req}")

            for k in range(natoms):
                parts = lines[i + 1 + k].split()
                pid = int(parts[col["id"]])
                if tracked_id is None:
                    tracked_id = pid
                if pid == tracked_id:
                    tstep.append(cur_t)
                    te.append(float(parts[col["temp"]]))
                    rd.append(float(parts[col["radius"]]))
            i += 1 + natoms
            continue
        i += 1

    if len(tstep) < 2:
        raise RuntimeError(f"Need at least 2 samples in {path_dump}")

    t = np.asarray(tstep, dtype=float) * dt
    rd = np.asarray(rd, dtype=float)
    te = np.asarray(te, dtype=float)

    order = np.argsort(t)
    return t[order], rd[order], te[order]


def evap_rhs(_t: float, y: np.ndarray, Qs: float) -> np.ndarray:
    R, TK = y
    R_safe = max(R, 1.0e-20)
    TK_safe = max(TK, 1.0)

    vpres1 = 760.0 * (10.0 ** (A1 + B1 / TK_safe))
    gevap_atoms = 1.0e4 * 3.513e22 * vpres1 / np.sqrt(XM1 * TK_safe)

    dRdt = -AM * gevap_atoms / RHO
    HF = Qs - gevap_atoms * (DHM / AN)
    dTdt = (3.0 / (RHO * CP * R_safe)) * HF
    return np.array([dRdt, dTdt], dtype=float)


def solve_rk45(t_eval: np.ndarray, R0: float, T0: float, Qs: float):
    sol = solve_ivp(
        fun=lambda t, y: evap_rhs(t, y, Qs),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.array([R0, T0], dtype=float),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=[1e-14, 1e-9],
    )
    if not sol.success:
        raise RuntimeError(f"RK45 failed: {sol.message}")
    return sol.y[0], sol.y[1]


def metrics(sim: np.ndarray, ref: np.ndarray) -> dict:
    err = sim - ref
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def main():
    here = Path(__file__).resolve().parent
    cases = [
        ("in.case_1", " $r_d$ = 1.5 mm"),
        ("in.case_2", " $r_d$ = 2.5 mm"),
        ("in.case_3", " $r_d$ = 3.5 mm"),
    ]


    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })


    fig, axes = plt.subplots(len(cases), 1, figsize=(6,3.5*len(cases)), dpi=400, sharex=False)

    if len(cases) == 1:
        axes = [axes]

    for ax, (in_name, label) in zip(axes, cases):
        cfg = parse_case_file(here / in_name)
        t_oe, rd_oe, te_oe = load_openedge_dump(here / cfg["dump"], cfg["dt"])

        if not np.isfinite(rd_oe[0]) or rd_oe[0] <= 0.0:
            rd_oe[0] = cfg["rd0"]
        if not np.isfinite(te_oe[0]) or te_oe[0] <= 0.0:
            te_oe[0] = cfg["T0"]

        rd_rk, te_rk = solve_rk45(t_oe, cfg["rd0"], cfg["T0"], cfg["Qs"])

        m_r = metrics(rd_oe, rd_rk)
        m_t = metrics(te_oe, te_rk)

        print(f"{label}:")
        print(f"  radius rmse={m_r['rmse']:.6e}, max_abs={m_r['max_abs']:.6e}")
        print(f"  temp   rmse={m_t['rmse']:.6e}, max_abs={m_t['max_abs']:.6e}")

        ax_r = ax.twinx()
        ax.plot(t_oe, rd_oe / cfg["rd0"], "o", ms=2.6, color="#1f77b4", label="OpenEdge")
        ax.plot(t_oe, rd_rk / cfg["rd0"], "-", lw=1.5, color="#1f77b4", alpha=0.9, label="RK45")

        ax_r.plot(t_oe, te_oe, "o", ms=2.4, mfc="none", color="#ff7f0e") #, label="OpenEdge $T_d$")
        ax_r.plot(t_oe, te_rk, "-", lw=1.5, color="#ff7f0e", alpha=0.9) #, label="RK45 $T_d$")

        ax.set_ylabel(r"$r_d/r_{d0}$")
        ax_r.set_ylabel(r"$T_d$ (K)")
        

        # Only put xlabel on the last subplot
        if ax is axes[-1]:
            ax.set_xlabel("time (s)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)  # optional: hide tick labels too
            
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        
        ax.text(0.02, 0.5, label, transform=ax.transAxes, ha="left", va="top", fontsize=14)


        # Small inset snippet: relative errors on log scale.
        err_r_rel = np.abs(rd_oe - rd_rk) / max(cfg["rd0"], 1.0e-30)
        err_t_rel = np.abs(te_oe - te_rk) / max(cfg["T0"], 1.0e-30)
        ax_in = inset_axes(
              ax, width=1.1, height=0.8,  # inches
              loc="lower right",
              bbox_to_anchor=(0.86, 0.1),
              bbox_transform=ax.transAxes,
              borderpad=0.0
          )
        ax_in.semilogy(t_oe, np.maximum(err_r_rel, 1.0e-18), color="#1f77b4", lw=1.0)
        ax_in.semilogy(t_oe, np.maximum(err_t_rel, 1.0e-18), color="#ff7f0e", lw=1.0)
        ax_in.grid(True, linestyle=":", alpha=0.35)
        ax_in.set_title("rel. error", fontsize=14)
        ax_in.tick_params(labelsize=14)
        ax_in.set_xlim(float(t_oe[0]), float(t_oe[-1]))
        ax_in.set_ylim(1e-10, 1e-1)


        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=14, frameon=False)

    fig.tight_layout()
    out = here / "Figs" / "test_evaporation_rk45_multi.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
