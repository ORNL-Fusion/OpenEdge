"""Plot 1-D neutral-slab grid dumps.

    python3 plot_slab.py iz
    python3 plot_slab.py recycle
    python3 plot_slab.py cx
    python3 plot_slab.py diss

Reads `output/slab_<case>.grid` (SPARTA grid dump) and writes
`<case>.png` in the current directory.

Column layout per deck (dump ordering in in.slab_*):

  iz / recycle : id xc yc f_fden[1]=n_D                f_frate[1..4]
  cx           : id xc yc f_fmom[1]=n_D f_fmom[3]=T_D  f_frate[1] f_frate[3]
  diss         : id xc yc f_fden[1]=n_D2 f_fden[2]=n_D f_fden[3]=n_Dp f_frate[1..4]
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Style: matplotlib mathtext (built-in LaTeX-style $...$) on a serif font.
# Set OPENEDGE_USETEX=1 to invoke a full system LaTeX install instead;
# only enable that if `latex` + `type1ec` (cm-super) are present.
# -----------------------------------------------------------------------
import os
USE_TEX = os.environ.get("OPENEDGE_USETEX", "0") == "1" and \
          shutil.which("latex") is not None
mpl.rcParams.update({
    "text.usetex":         USE_TEX,
    "mathtext.fontset":    "cm",      # Computer-Modern math glyphs
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif", "Computer Modern Roman"],
    "font.size":           12,
    "axes.titlesize":      13,
    "axes.labelsize":      13,
    "legend.fontsize":     11,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "lines.linewidth":     1.8,
    "figure.dpi":          150,
    "savefig.bbox":        "tight",
})
if USE_TEX:
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


CASES = ("iz", "recycle", "cx", "diss")


# -----------------------------------------------------------------------
# Dump parser
# -----------------------------------------------------------------------
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
            ts = int(lines[i + 1]); i += 2
        elif s == "ITEM: NUMBER OF CELLS":
            n = int(lines[i + 1]); i += 2
        elif s.startswith("ITEM: CELLS"):
            headers = s.split()[2:]
            rows = np.array([
                list(map(float, lines[i + 1 + k].split()))
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


def first_col(headers, *names):
    for name in names:
        if name in headers:
            return col(headers, name)
    raise ValueError(f"none of the expected columns were found: {names}")


# -----------------------------------------------------------------------
# Common labels
# -----------------------------------------------------------------------
LABEL_X     = r"$x \; (\mathrm{m})$"
LABEL_N     = r"$n(x) \; (\mathrm{m}^{-3})$"
LABEL_ND    = r"$n_D(x) \; (\mathrm{m}^{-3})$"
LABEL_TD_EV = r"$T_D(x) \; (\mathrm{eV})$"


def style_density_axis(ax):
    ax.set_xlabel(LABEL_X)
    ax.set_ylabel(LABEL_N)
    ax.set_yscale("log")


# -----------------------------------------------------------------------
# Plotters
# -----------------------------------------------------------------------
def plot_density_decay(ax, headers, data, color="C0"):
    xc = data[:, col(headers, "xc")]
    nD = data[:, first_col(headers, "f_fden[1]", "f_fmom[1]", "c_cden[1]")]
    order = np.argsort(xc)
    ax.plot(xc[order], nD[order], color=color)
    style_density_axis(ax)


def plot_iz(path, outpng):
    headers, d = last_frame(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_density_decay(ax, headers, d)
    ax.set_title(r"1-D slab: ionization-limited neutral density")
    fig.savefig(outpng); print(f"wrote {outpng}")


def plot_recycle(path, outpng):
    headers, d = last_frame(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_density_decay(ax, headers, d)
    ax.set_title(r"1-D slab: ionization $+$ wall recycling")
    fig.savefig(outpng); print(f"wrote {outpng}")


def plot_cx(path, outpng):
    headers, d = last_frame(path)
    xc = d[:, col(headers, "xc")]
    nD = d[:, first_col(headers, "f_fmom[1]", "c_cden[1]")]
    # compute grid ... species temp returns Kelvin (m <v^2> / (3 kB)).
    # f_fmom layout: [1]=n_D [2]=n_D+ [3]=T_D [4]=T_D+ (cden then ctemp).
    TD_eV = d[:, first_col(headers, "f_fmom[3]", "c_ctemp[1]")] / 11604.525
    order = np.argsort(xc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(xc[order], nD[order])
    ax1.set_xlabel(LABEL_X)
    ax1.set_ylabel(LABEL_ND)
    ax1.set_yscale("log")
    ax1.set_title(r"$D$ density (ionization-depleted)")

    ax2.plot(xc[order], TD_eV[order])
    ax2.axhline(50.0, color="0.5", ls="--", lw=0.9,
                label=r"$T_i = 50 \; \mathrm{eV}$ (asymptote)")
    ax2.set_xlabel(LABEL_X)
    ax2.set_ylabel(LABEL_TD_EV)
    ax2.legend()
    ax2.set_title(r"$D$ temperature (CX thermalization)")

    fig.savefig(outpng); print(f"wrote {outpng}")


def plot_diss(path, outpng):
    headers, d = last_frame(path)
    xc = d[:, col(headers, "xc")]
    nD2 = d[:, first_col(headers, "f_fden[1]", "c_cden[1]")]
    nD  = d[:, first_col(headers, "f_fden[2]", "c_cden[2]")]
    order = np.argsort(xc)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xc[order], nD2[order], label=r"$n_{D_2}$ (reactant)")
    ax.plot(xc[order], nD[order],  label=r"$n_{D}$ (product)")
    style_density_axis(ax)
    ax.legend()
    ax.set_title(r"$D_2$ dissociation: reactant decay $+$ product build-up")
    fig.savefig(outpng); print(f"wrote {outpng}")


PLOTTERS = {"iz": plot_iz, "recycle": plot_recycle,
            "cx": plot_cx, "diss": plot_diss}


# -----------------------------------------------------------------------
# CX thermalization: T_D(t) parsed from log.cx stats output
# -----------------------------------------------------------------------
def parse_log_stats(log_path, columns):
    """Parse SPARTA-style stats blocks from a log file.

    Concatenates rows from every block whose header contains all of
    `columns`. Multi-block runs (e.g. `run N` followed by `run 0` for
    a dump-trigger) yield one combined time series.
    """
    with open(log_path) as f:
        lines = f.readlines()
    rows = []
    seen_steps = set()
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        if all(c in toks for c in columns):
            header = toks
            idx = [header.index(c) for c in columns]
            i += 1
            while i < len(lines):
                row_toks = lines[i].split()
                if not row_toks:
                    break
                try:
                    row = [float(row_toks[k]) for k in idx]
                except (ValueError, IndexError):
                    break
                # skip duplicate step entries from `run 0` echo blocks
                step_key = row[0]
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    rows.append(row)
                i += 1
        else:
            i += 1
    return np.array(rows)


def plot_thermalization(base, outpng, dt=2e-8, Ti_eV=50.0):
    """T_D(t) from the 0-D CX deck (log.cx_0d).

    The closed-box deck initializes 5000 D at 3 eV against a Ti=50 eV
    D+ background; <T_D>(t) relaxes exponentially to Ti via CX. The
    streaming slab (log.cx) plateaus at a steady-state spatial average
    instead, so we deliberately read the 0-D log here.
    """
    log = base / "log.cx_0d"
    if not log.exists():
        print(f"missing: {log} — run in.slab_cx_0d first.")
        sys.exit(1)
    arr = parse_log_stats(log, ["Step", "c_TavgD"])
    if arr.size == 0:
        print(f"no rows parsed from {log}")
        sys.exit(1)
    t  = arr[:, 0] * dt * 1e6      # us
    TD = arr[:, 1] / 11604.525     # K -> eV

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(t, TD, color="C0")
    ax.axhline(Ti_eV, color="0.5", ls="--", lw=0.9,
               label=fr"$T_i = {Ti_eV:g} \; \mathrm{{eV}}$")
    ax.set_xlabel(r"$t \; (\mathrm{\mu s})$")
    ax.set_ylabel(r"$\langle T_D \rangle \; (\mathrm{eV})$")
    ax.set_title(r"0-D CX thermalization: $\langle T_D \rangle(t) \to T_i$")
    ax.legend()
    fig.savefig(outpng); print(f"wrote {outpng}")


def plot_composite(base, outpng):
    """2x2 panel for the slide: iz | recycle / cx-T_D | diss."""
    paths = {c: base / "output" / f"slab_{c}.grid" for c in CASES}
    missing = [c for c, p in paths.items() if not p.exists()]
    if missing:
        print(f"missing dumps: {missing} — run those decks first.")
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    (ax_iz, ax_rec), (ax_cx, ax_diss) = axes

    # iz
    h, d = last_frame(paths["iz"])
    plot_density_decay(ax_iz, h, d)
    ax_iz.set_title(r"(a) Ionization: $\lambda_{iz} \approx 130$ mm")

    # recycle
    h, d = last_frame(paths["recycle"])
    plot_density_decay(ax_rec, h, d)
    ax_rec.set_title(r"(b) Ionization $+$ wall recycling")

    # cx (temperature panel — the standout diagnostic)
    h, d = last_frame(paths["cx"])
    xc = d[:, col(h, "xc")]
    TD_eV = d[:, first_col(h, "f_fmom[3]", "c_ctemp[1]")] / 11604.525
    order = np.argsort(xc)
    ax_cx.plot(xc[order], TD_eV[order])
    ax_cx.axhline(50.0, color="0.5", ls="--", lw=0.9,
                  label=r"$T_i = 50 \; \mathrm{eV}$")
    ax_cx.set_xlabel(LABEL_X); ax_cx.set_ylabel(LABEL_TD_EV)
    ax_cx.set_title(r"(c) CX thermalization: $T_D \to T_i$")
    ax_cx.legend(loc="lower right")

    # diss
    h, d = last_frame(paths["diss"])
    xc = d[:, col(h, "xc")]
    nD2 = d[:, first_col(h, "f_fden[1]", "c_cden[1]")]
    nD  = d[:, first_col(h, "f_fden[2]", "c_cden[2]")]
    order = np.argsort(xc)
    ax_diss.plot(xc[order], nD2[order], label=r"$n_{D_2}$ (reactant)")
    ax_diss.plot(xc[order], nD[order],  label=r"$n_{D}$ (product)")
    style_density_axis(ax_diss)
    ax_diss.set_title(r"(d) $D_2$ dissociation: $\lambda_{diss} \approx 121$ mm")
    ax_diss.legend(loc="lower left")

    fig.savefig(outpng); print(f"wrote {outpng}")


def main():
    modes = list(CASES) + ["composite", "thermalization"]
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print(f"usage: python3 plot_slab.py {'|'.join(modes)}")
        sys.exit(1)
    arg = sys.argv[1]
    base = Path(__file__).resolve().parent
    if arg == "composite":
        plot_composite(base, base / "slab_validation_2x2.png")
        return
    if arg == "thermalization":
        plot_thermalization(base, base / "cx_thermalization.png")
        return
    src = base / "output" / f"slab_{arg}.grid"
    if not src.exists():
        print(f"missing: {src} — run the deck first.")
        sys.exit(1)
    PLOTTERS[arg](src, base / f"{arg}.png")


if __name__ == "__main__":
    main()
