#!/usr/bin/env python3
"""Combined figure: segment + incidence + transfer matrix (mean ± std)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from segment import DEST_ORDER, incidence_df, plot_segment_panel

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
    return out.reindex(source_order)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mean", default="sweep_out/sweep_region_pct_mean.csv")
    p.add_argument("--std", default="sweep_out/sweep_region_pct_std.csv")
    p.add_argument("--source-I", default="lo_inner_div")
    p.add_argument("--source-O", default="lo_outer_div")
    p.add_argument("--inner", default="case.inner")
    p.add_argument("--outer", default="case.outer")
    p.add_argument("--wall", default="wall.surf")
    p.add_argument("--core", default="core.surf")
    p.add_argument("--out", default="sweep_out/Figs/combined_transfer_uncertainty.png")
    p.add_argument("--tag", default="")
    p.add_argument("--show", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    source_order = [args.source_I, args.source_O]
    dest_order = DEST_ORDER[:] + ["evaporated"]

    mean_m = load_matrix(Path(args.mean), source_order, dest_order)
    std_m = load_matrix(Path(args.std), source_order, dest_order)

    df_inner = incidence_df(Path(args.inner), Path(args.wall), "Inner")
    df_outer = incidence_df(Path(args.outer), Path(args.wall), "Outer")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

#    fig = plt.figure(figsize=(13.8, 8.9), dpi=170)
#    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.15], hspace=0.28)
#
#    top = gs[0].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.16)
#    ax_seg = fig.add_subplot(top[0, 0])
#    ax_i = fig.add_subplot(top[0, 1])
#    ax_o = fig.add_subplot(top[0, 2], sharey=ax_i)

    fig = plt.figure(figsize=(13.8, 8.9), dpi=170)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.15], hspace=0.28)

    # 4 columns: [left plot] [spacer] [inner] [outer]
    top = gs[0].subgridspec(1, 4, width_ratios=[1.35, 0.14, 1.0, 1.0], wspace=0.06)

    ax_seg = fig.add_subplot(top[0, 0])
    ax_i   = fig.add_subplot(top[0, 2])
    ax_o   = fig.add_subplot(top[0, 3], sharey=ax_i)

    pos_i = ax_i.get_position()
    pos_o = ax_o.get_position()
    ax_o.set_position([pos_i.x1, pos_o.y0, pos_o.width, pos_o.height])

    plot_segment_panel(ax_seg, Path(args.wall), Path(args.core), show_legend=True)

    if not df_inner.empty:
        ax_i.scatter(df_inner["angle_deg"], df_inner["vnormal_mps"], s=12, alpha=0.7, edgecolors="none")
    ax_i.set_title("Inner")
    ax_i.set_xlabel(r"Angle $\theta$ (deg)")
    ax_i.set_ylabel(r"Speed $|v \cdot n|$ (m/s)")
    ax_i.minorticks_on()
    ax_i.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_i.grid(True, linestyle="--", alpha=0.3)

    if not df_outer.empty:
        ax_o.scatter(df_outer["angle_deg"], df_outer["vnormal_mps"], s=12, alpha=0.7, edgecolors="none")
    ax_o.set_title("Outer")
    ax_o.set_xlabel(r"Angle $\theta$ (deg)")
    ax_o.tick_params(axis="y", labelleft=False)
    ax_o.minorticks_on()
    ax_o.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_o.grid(True, linestyle="--", alpha=0.3)

    ax_h = fig.add_subplot(gs[1, 0])
    annot = np.empty(mean_m.shape, dtype=object)
    for i in range(mean_m.shape[0]):
        for j in range(mean_m.shape[1]):
            annot[i, j] = f"{mean_m.values[i,j]:.2f}"
#            annot[i, j] = f"{mean_m.values[i,j]:.1f}\n±{std_m.values[i,j]:.1f}"

    sns.heatmap(
        mean_m,
        ax=ax_h,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "mean injected droplets (%)"},
    )

    cv = 100.0 * std_m / mean_m.replace(0.0, np.nan)
    cv = cv.fillna(0.0)
    cv_median = float(np.nanmedian(cv.values))
#    title = "Transfer matrix (mean ± std, %)"
#    if args.tag:
#        title += f" — {args.tag}"
#    title += f"\nmedian CV={cv_median:.1f}%"

#    ax_h.set_title(title)
    ax_h.set_xlabel("destination region")
    ax_h.set_ylabel("source")
    ax_h.tick_params(axis="x", rotation=32)
    ax_h.tick_params(axis="y", rotation=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")

    if args.show == "yes":
        plt.show()


if __name__ == "__main__":
    main()
