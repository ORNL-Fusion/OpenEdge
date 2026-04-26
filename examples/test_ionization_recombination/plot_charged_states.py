"""Plot O charge-state fractional abundance vs time from log.openedge.

Parses the stats block produced by `in.ionization_recombination`:
    Step cpu np c_n0 c_n1 ... c_n8
where c_n<k> is the cell-averaged number density of O^(k+).
"""

import re
import numpy as np
import matplotlib.pyplot as plt


DT      = 1e-7         # must match the deck's `variable dt`
LOGFILE = "log.openedge"
OUTPNG  = "oxygen_balance.png"
ELEMENT = "O"
COLS    = [f"c_n{k}" for k in range(9)]   # n0 = O, n1 = O+, ... n8 = O8+


def parse_stats(path):
    """Return (step, {col: array}) scraped from the SPARTA stats block."""
    with open(path) as f:
        lines = f.readlines()

    header = None
    rows = []
    in_block = False
    for line in lines:
        s = line.strip()
        if not in_block:
            if s.startswith("Step ") and all(c in s for c in COLS):
                header = s.split()
                in_block = True
            continue
        if not s or s.startswith("Loop time") or re.match(r"^[A-Za-z]", s):
            in_block = False
            header = None
            continue
        parts = s.split()
        if len(parts) != len(header):
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue

    if not rows:
        raise RuntimeError(f"No stats rows found in {path}")

    arr = np.array(rows)
    cols = {name: arr[:, i] for i, name in enumerate(header)}
    return cols["Step"].astype(int), cols


def main():
    step, cols = parse_stats(LOGFILE)
    t_ms = step * DT * 1e3

    n = np.vstack([cols[c] for c in COLS])
    total = n.sum(axis=0)
    frac = n / np.where(total > 0, total, 1.0)

    cmap = plt.get_cmap("tab10")
    labels = [ELEMENT] + [rf"${ELEMENT}^{{{k}+}}$" for k in range(1, 9)]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    for k in range(9):
        ax.semilogx(t_ms, frac[k], "-", lw=2, color=cmap(k), label=labels[k])
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Fractional abundance", fontsize=13)
    ax.set_title(rf"{ELEMENT} ionization balance, "
                 rf"$T_e = 12$ eV, $n_e = 5\times10^{{18}}$ m$^{{-3}}$",
                 fontsize=12)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    ax.legend(fontsize=10, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(OUTPNG, bbox_inches="tight")
    print(f"wrote {OUTPNG}  ({len(step)} stats rows)")


if __name__ == "__main__":
    main()
