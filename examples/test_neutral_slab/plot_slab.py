"""Plot 1-D neutral-slab grid dumps.

    python3 plot_slab.py iz
    python3 plot_slab.py recycle
    python3 plot_slab.py cx
    python3 plot_slab.py diss

Reads `output/slab_<case>.grid` (SPARTA grid dump) and writes
`<case>.png` in the current directory.

Column layout per deck (dump ordering in in.slab_*):

  iz / recycle : id xc yc n_D iz_rate rec_rate cx_rate diss_rate
  cx           : id xc yc n_D T_D     iz_rate cx_rate
  diss         : id xc yc n_D2 n_D n_Dp iz_rate rec_rate cx_rate diss_rate
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CASES = ("iz", "recycle", "cx", "diss")


def parse_grid_dump(path):
    """Return (timesteps, headers, data[ts] -> ndarray[ncells, ncols])."""
    headers = []
    blocks = {}
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    ts = None
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            ts = int(lines[i+1]); i += 2
        elif s == "ITEM: NUMBER OF CELLS":
            n = int(lines[i+1]); i += 2
        elif s.startswith("ITEM: CELLS"):
            headers = s.split()[2:]
            rows = np.array([
                list(map(float, lines[i+1+k].split()))
                for k in range(n)
            ])
            blocks[ts] = rows
            i += 1 + n
        else:
            i += 1
    return sorted(blocks), headers, blocks


def last_frame(path):
    ts, headers, blocks = parse_grid_dump(path)
    return headers, blocks[ts[-1]]


def col(headers, name):
    return headers.index(name)


def plot_density_decay(case, ax, headers, data, label, color="C0"):
    xc = data[:, col(headers, "xc")]
    nD = data[:, col(headers, "f_fden[1]") if "f_fden[1]" in headers
              else col(headers, "f_fmom[1]")]
    order = np.argsort(xc)
    ax.plot(xc[order], nD[order], color=color, lw=1.6, label=label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("n(x) [m^-3]")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_iz(path, outpng):
    headers, d = last_frame(path)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    plot_density_decay("iz", ax, headers, d, "simulation")
    ax.set_title("1-D slab: ionization-limited neutral density")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpng); print(f"wrote {outpng}")


def plot_recycle(path, outpng):
    headers, d = last_frame(path)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    plot_density_decay("recycle", ax, headers, d, "simulation")
    ax.set_title("1-D slab: ionization + wall recycling")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpng); print(f"wrote {outpng}")


def plot_cx(path, outpng):
    headers, d = last_frame(path)
    xc = d[:, col(headers, "xc")]
    nD = d[:, col(headers, "f_fmom[1]")]
    # compute grid ... species temp returns Kelvin (m <v^2> / (3 kB)).
    TD_eV = d[:, col(headers, "f_fmom[2]")] / 11604.525
    order = np.argsort(xc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax1.plot(xc[order], nD[order], lw=1.6)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("n_D(x) [m^-3]")
    ax1.set_yscale("log"); ax1.grid(True, alpha=0.3)
    ax1.set_title("D density (ionization-depleted)")

    ax2.plot(xc[order], TD_eV[order], lw=1.6)
    ax2.axhline(50.0, color="0.5", ls="--", lw=0.9, label="Ti = 50 eV (asymptote)")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("T_D(x) [eV]")
    ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_title("D temperature (CX thermalization)")

    fig.tight_layout(); fig.savefig(outpng); print(f"wrote {outpng}")


def plot_diss(path, outpng):
    headers, d = last_frame(path)
    xc = d[:, col(headers, "xc")]
    nD2 = d[:, col(headers, "f_fden[1]")]
    nD  = d[:, col(headers, "f_fden[2]")]
    order = np.argsort(xc)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(xc[order], nD2[order], lw=1.6, label=r"$n_{D_2}$ (reactant)")
    ax.plot(xc[order], nD[order],  lw=1.6, label=r"$n_D$ (product)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("n(x) [m^-3]")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(r"D$_2$ dissociation: reactant decay + product build-up")
    fig.tight_layout(); fig.savefig(outpng); print(f"wrote {outpng}")


PLOTTERS = {"iz": plot_iz, "recycle": plot_recycle,
            "cx": plot_cx, "diss": plot_diss}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        print(f"usage: python3 plot_slab.py {'|'.join(CASES)}")
        sys.exit(1)
    case = sys.argv[1]
    base = Path(__file__).resolve().parent
    src = base / "output" / f"slab_{case}.grid"
    if not src.exists():
        print(f"missing: {src} — run the deck first.")
        sys.exit(1)
    PLOTTERS[case](src, base / f"{case}.png")


if __name__ == "__main__":
    main()
