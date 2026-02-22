#!/usr/bin/env python3
"""Plot transfer-matrix mean and uncertainty from sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEST_ORDER = [
    "lfs_wall",
    "up_outer_div",
    "crown",
    "up_inner_div",
    "hfs_wall",
    "lo_inner_div",
    "dome",
    "lo_outer_div",
    "core",
    "evaporated",
]

SRC_MAP = {
    "I": "lo_inner_div",
    "O": "lo_outer_div",
    "lo_inner_div": "lo_inner_div",
    "lo_outer_div": "lo_outer_div",
}


def normalize_source_name(s: str) -> str:
    s = str(s)
    return SRC_MAP.get(s, s)


def load_matrix(csv_path: Path, source_order: list[str], dest_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "source" not in df.columns:
        raise RuntimeError(f"Missing 'source' column in {csv_path}")

    df["source"] = df["source"].map(normalize_source_name)
    keep_cols = [c for c in dest_order if c in df.columns]
    if not keep_cols:
        raise RuntimeError(f"No destination columns found in {csv_path}")

    out = df.set_index("source")[keep_cols].copy()
    out = out.reindex(source_order)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mean", default="sweep_out/sweep_region_pct_mean.csv")
    p.add_argument("--std", default="sweep_out/sweep_region_pct_std.csv")
    p.add_argument("--source-I", default="lo_inner_div")
    p.add_argument("--source-O", default="lo_outer_div")
    p.add_argument("--outdir", default="sweep_out/Figs")
    p.add_argument("--tag", default="sweep")
    p.add_argument("--annot", choices=["yes", "no"], default="yes")
    p.add_argument("--show", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    source_order = [args.source_I, args.source_O]
    mean_path = Path(args.mean)
    std_path = Path(args.std)

    mean_m = load_matrix(mean_path, source_order, DEST_ORDER)
    std_m = load_matrix(std_path, source_order, DEST_ORDER)

    cv_m = 100.0 * std_m / mean_m.replace(0.0, np.nan)
    cv_m = cv_m.fillna(0.0)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: mean and std side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), dpi=180)

    sns.heatmap(
        mean_m,
        ax=axes[0],
        annot=(np.vectorize(lambda x: f"{x:.1f}")(mean_m.values) if args.annot == "yes" else False),
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "mean (%)"},
    )
    axes[0].set_title("Transfer Matrix Mean")
    axes[0].set_xlabel("destination region")
    axes[0].set_ylabel("source")
    axes[0].tick_params(axis="x", rotation=32)
    axes[0].tick_params(axis="y", rotation=0)

    vmax_std = max(1.0, float(np.nanmax(std_m.values)))
    sns.heatmap(
        std_m,
        ax=axes[1],
        annot=(np.vectorize(lambda x: f"{x:.1f}")(std_m.values) if args.annot == "yes" else False),
        fmt="",
        cmap="magma",
        vmin=0,
        vmax=vmax_std,
        linewidths=0.5,
        cbar_kws={"label": "std (%)"},
    )
    axes[1].set_title("Transfer Matrix Uncertainty (Std)")
    axes[1].set_xlabel("destination region")
    axes[1].set_ylabel("source")
    axes[1].tick_params(axis="x", rotation=32)
    axes[1].tick_params(axis="y", rotation=0)

    fig.tight_layout()
    out1 = outdir / f"transfer_matrix_mean_std_{args.tag}.png"
    fig.savefig(out1, dpi=220, bbox_inches="tight")

    # Figure 2: mean ± std annotation
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.4), dpi=180)
    annot_pm = np.empty(mean_m.shape, dtype=object)
    for i in range(mean_m.shape[0]):
        for j in range(mean_m.shape[1]):
            annot_pm[i, j] = f"{mean_m.values[i,j]:.1f}\u00b1{std_m.values[i,j]:.1f}"

    sns.heatmap(
        mean_m,
        ax=ax2,
        annot=annot_pm,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "mean (%)"},
    )
    ax2.set_title("Transfer Matrix (mean \u00b1 std, %)" )
    ax2.set_xlabel("destination region")
    ax2.set_ylabel("source")
    ax2.tick_params(axis="x", rotation=32)
    ax2.tick_params(axis="y", rotation=0)

    fig2.tight_layout()
    out2 = outdir / f"transfer_matrix_mean_pm_std_{args.tag}.png"
    fig2.savefig(out2, dpi=220, bbox_inches="tight")

    # Figure 3: coefficient of variation
    fig3, ax3 = plt.subplots(figsize=(8.2, 4.4), dpi=180)
    vmax_cv = max(1.0, float(np.nanpercentile(cv_m.values, 95)))
    sns.heatmap(
        cv_m,
        ax=ax3,
        annot=np.vectorize(lambda x: f"{x:.1f}")(cv_m.values),
        fmt="",
        cmap="cividis",
        vmin=0,
        vmax=vmax_cv,
        linewidths=0.5,
        cbar_kws={"label": "CV = 100*std/mean (%)"},
    )
    ax3.set_title("Transfer Matrix Relative Uncertainty (CV)")
    ax3.set_xlabel("destination region")
    ax3.set_ylabel("source")
    ax3.tick_params(axis="x", rotation=32)
    ax3.tick_params(axis="y", rotation=0)

    fig3.tight_layout()
    out3 = outdir / f"transfer_matrix_cv_{args.tag}.png"
    fig3.savefig(out3, dpi=220, bbox_inches="tight")

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")

    if args.show == "yes":
        plt.show()


if __name__ == "__main__":
    main()
