#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
import numpy as np

def read_series(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    out = []
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1].strip())
        i += 2
        if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Bad dump format: NUMBER OF ATOMS")
        nat = int(lines[i + 1].strip())
        i += 2
        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 4
        if not lines[i].startswith("ITEM: ATOMS"):
            raise RuntimeError("Bad dump format: ATOMS")
        cols = lines[i].split()[2:]
        i += 1
        for _ in range(nat):
            parts = lines[i].split()
            row = {cols[k]: parts[k] for k in range(len(cols))}
            out.append((
                step,
                float(row["x"]), float(row["y"]), float(row["z"]),
                float(row["vx"]), float(row["vy"]), float(row["vz"]),
            ))
            i += 1
    return out


def parse_input_dt(path: Path):
    dt = None
    var_equal = {}

    def eval_expr(expr: str):
        # Resolve ${name} and v_name using locally defined "variable ... equal ..." values.
        expr = re.sub(r"\$\{([A-Za-z_]\w*)\}", lambda m: str(var_equal.get(m.group(1), m.group(0))), expr)
        expr = re.sub(r"\bv_([A-Za-z_]\w*)\b", lambda m: str(var_equal.get(m.group(1), m.group(0))), expr)
        allowed = {"floor": math.floor, "sqrt": math.sqrt, "abs": abs}
        return float(eval(expr, {"__builtins__": {}}, allowed))

    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "variable" and parts[2] == "equal":
                name = parts[1]
                expr = " ".join(parts[3:])
                try:
                    var_equal[name] = eval_expr(expr)
                except Exception:
                    pass
            if parts[0] == "timestep" and len(parts) >= 2:
                try:
                    dt = eval_expr(parts[1])
                except Exception:
                    dt = float(parts[1])
    if dt is None:
        raise RuntimeError(f"Could not parse timestep from {path.name}")
    return dt


def main():
    p = argparse.ArgumentParser(description="Plot test_boris_grid trajectory and KE(t)")
    p.add_argument("--case", choices=["file", "constant"], default="constant")
    p.add_argument(
        "--mass",
        type=float,
        default=1.6726219e-27,
        help="Particle mass in kg for KE computation (default: proton mass)",
    )

    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required") from exc

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    base = Path(__file__).resolve().parent
    dumpf = base / "output" / f"state.{args.case}"
    series = read_series(dumpf)
    if not series:
        raise RuntimeError("No trajectory data found")

    dt = parse_input_dt(base / f"in.{args.case}")
    t = [s[0] * dt for s in series]
    x = [s[1] for s in series]
    y = [s[2] for s in series]
    z = [s[3] for s in series]
    vx = [s[4] for s in series]
    vy = [s[5] for s in series]
    vz = [s[6] for s in series]
    r = [math.sqrt(xx * xx + yy * yy) for xx, yy in zip(x, y)]
    ke = [0.5 * args.mass * (vxi * vxi + vyi * vyi + vzi * vzi) for vxi, vyi, vzi in zip(vx, vy, vz)]
    ke0 = ke[0] if ke else 1.0
    dke_rel = [((k - ke0) / ke0) if ke0 != 0.0 else 0.0 for k in ke]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=220, constrained_layout=True)

    ax[0].plot(
        r, z,
        linestyle="-",
        linewidth=1.8,
        marker="o",
        markersize=2.0,
        markerfacecolor="none",
        markeredgewidth=0.8,
        markevery=max(1, len(r) // 200),
        label="simulation",
    )
    ax[0].set_xlabel("R [m]")
    ax[0].set_ylabel("Z [m]")
    ax[0].set_title(f"Trajectory (R-Z), case={args.case}")
    ax[0].grid(True, linestyle="--", alpha=0.3)
    ax[0].minorticks_on()
    ax[0].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax[0].legend(frameon=True, framealpha=0.9)

    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Energy Conservation")
    ax[1].grid(True, linestyle="--", alpha=0.3)
    ax[1].minorticks_on()
    ax[1].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax[1].plot(t, dke_rel, "--", lw=1.3, color="tab:red", label=r"$\Delta KE/KE_0$")
    ax[1].set_ylabel(r"$\Delta KE/KE_0$")
    ax[1].tick_params(axis="y", which="both", direction="in", right=True)
    ax[1].legend(frameon=True, framealpha=0.9, loc="best")

    max_abs_dke = max(abs(vv) for vv in dke_rel) if dke_rel else 0.0
    ax[1].text(
        0.03,
        0.96,
        rf"max $|\Delta KE/KE_0|$ = {max_abs_dke:.3e}",
        transform=ax[1].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    out = base / "output" / f"trajectory_ke.{args.case}.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out}")
    print(f"KE drift range: min={min(dke_rel):.3e}, max={max(dke_rel):.3e}, max_abs={max_abs_dke:.3e}")
#    plt.show()
if __name__ == "__main__":
    main()
