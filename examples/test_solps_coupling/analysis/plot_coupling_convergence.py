#!/usr/bin/env python3
"""Plot convergence and Li source evolution from an OpenEdge-SOLPS coupling run.

Reads the coupling.log produced by openedge_solps_driver.py and generates:
  - convergence.png/pdf  : |dne/ne| and |dTe/Te| vs coupling step
  - li_source_evolution.png/pdf : Li source rate and active particles vs step

Usage:
    python plot_coupling_convergence.py [--run-dir PATH] [--out-dir PATH]

Defaults:
    --run-dir  ../coupled_test
    --out-dir  ../coupled_test/figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_coupling_log(log_path):
    """Parse the coupling history table from coupling.log.

    Returns dict with arrays: step, particles, li_source, dne, dTe.
    Deduplicates by keeping the last occurrence per step number.
    """
    text = log_path.read_text()
    row_re = re.compile(
        r"│\s+(\d+)/\d+\s+│\s+(\d+)\s+│\s+([\d.eE+\-]+|\d+)\s+│"
        r"\s+\[.*?\]\s+│\s+\[.*?\]\s+│\s+([\d.eE+\-]+)\s+│\s+([\d.eE+\-]+)\s+│"
    )
    data = {}
    for m in row_re.finditer(text):
        s = int(m.group(1))
        data[s] = (int(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5)))

    steps = np.array(sorted(data.keys()))
    particles = np.array([data[s][0] for s in steps])
    li_source = np.array([data[s][1] for s in steps])
    dne = np.array([data[s][2] for s in steps])
    dTe = np.array([data[s][3] for s in steps])
    return dict(step=steps, particles=particles, li_source=li_source,
                dne=dne, dTe=dTe)


def plot_convergence(d, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(d["step"], d["dne"], "o-", color="#1f77b4", lw=2, ms=5,
                label=r"max $|\Delta n_e / n_e|$")
    ax.semilogy(d["step"], d["dTe"], "s-", color="#d62728", lw=2, ms=5,
                label=r"max $|\Delta T_e / T_e|$")
    ax.axhline(0.1, color="gray", ls="--", lw=1, alpha=0.7, label="10% threshold")
    ax.set_xlabel("Coupling step", fontsize=14)
    ax.set_ylabel("Relative change", fontsize=14)
    ax.set_title("OpenEdge\u2013SOLPS Coupling Convergence", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=12, frameon=True)
    ax.set_xlim(0, d["step"][-1] + 1)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=300)
    fig.savefig(out_dir / "convergence.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'convergence.png'}")


def plot_li_source(d, out_dir):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    c1, c2 = "#2ca02c", "#9467bd"

    ax1.plot(d["step"], d["li_source"], "o-", color=c1, lw=2, ms=5,
             label="Li source rate")
    ax1.set_xlabel("Coupling step", fontsize=14)
    ax1.set_ylabel("Li source (atoms/s)", fontsize=14, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_yscale("symlog", linthresh=1e20)
    ax1.set_title("Lithium Source & Active Particles", fontsize=15,
                  fontweight="bold")
    ax1.set_xlim(0, d["step"][-1] + 1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(d["step"], d["particles"], "s--", color=c2, lw=2, ms=5,
             label="Active particles")
    ax2.set_ylabel("Active particles", fontsize=14, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2, labelsize=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=12, frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "li_source_evolution.png", dpi=300)
    fig.savefig(out_dir / "li_source_evolution.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'li_source_evolution.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "coupled_test")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.run_dir / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.run_dir / "coupling.log"
    if not log_path.exists():
        raise FileNotFoundError(f"coupling.log not found in {args.run_dir}")

    d = parse_coupling_log(log_path)
    print(f"Parsed {len(d['step'])} coupling steps from {log_path}")

    plot_convergence(d, args.out_dir)
    plot_li_source(d, args.out_dir)


if __name__ == "__main__":
    main()
