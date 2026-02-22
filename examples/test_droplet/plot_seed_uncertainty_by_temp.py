#!/usr/bin/env python3
"""Plot seed-only uncertainty vs temperature from sweep_region_pct_long.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--long", default="sweep_out/sweep_region_pct_long.csv", help="Long CSV from sweep_transfer_matrix.py")
    p.add_argument("--outdir", default="sweep_out/Figs", help="Output directory")
    p.add_argument("--top-n", type=int, default=5, help="Top destinations by mean contribution to plot per source")
    p.add_argument("--include", default="core,evaporated", help="Comma-separated destination columns to always include")
    p.add_argument("--show", choices=["yes", "no"], default="no")
    args = p.parse_args()

    long_path = Path(args.long)
    if not long_path.exists():
        raise SystemExit(f"Missing input CSV: {long_path}")

    df = pd.read_csv(long_path)
    required = {"temp_emit", "source"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {long_path}: {sorted(missing)}")

    value_cols = [c for c in df.columns if c not in {"run_tag", "source", "nemit", "temp_emit", "seed"}]
    if not value_cols:
        raise SystemExit("No destination columns found in long CSV")

    stats = (
        df.groupby(["temp_emit", "source"], dropna=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = [
        "temp_emit" if c == "temp_emit" else
        "source" if c == "source" else
        f"{c[0]}_{c[1]}"
        for c in stats.columns
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    include = [x.strip() for x in args.include.split(",") if x.strip()]

    for source in sorted(df["source"].unique()):
        dsrc = df[df["source"] == source]
        means = dsrc[value_cols].mean().sort_values(ascending=False)
        top = [c for c in means.index if means[c] > 0][: args.top_n]
        for c in include:
            if c in value_cols and c not in top:
                top.append(c)

        ssrc = stats[stats["source"] == source].sort_values("temp_emit")
        x = ssrc["temp_emit"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=170)
        for col in top:
            y = ssrc[f"{col}_mean"].to_numpy(dtype=float)
            yerr = ssrc[f"{col}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6, label=col)

        ax.set_xscale("log")
        ax.set_xlabel("temp_emit")
        ax.set_ylabel("destination fraction (%)")
        ax.set_title(f"Seed-only uncertainty vs temperature (source={source})")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()

        out = outdir / f"seed_uncertainty_vs_temp_source_{source}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")

    mean_by_temp = (
        df.groupby(["temp_emit", "source"], dropna=False)[value_cols]
        .mean()
        .reset_index()
    )
    std_by_temp = (
        df.groupby(["temp_emit", "source"], dropna=False)[value_cols]
        .std(ddof=1)
        .fillna(0.0)
        .reset_index()
    )
    mean_csv = outdir.parent / "sweep_region_pct_mean_by_temp_from_long.csv"
    std_csv = outdir.parent / "sweep_region_pct_std_by_temp_from_long.csv"
    mean_by_temp.to_csv(mean_csv, index=False)
    std_by_temp.to_csv(std_csv, index=False)
    print(f"Wrote: {mean_csv}")
    print(f"Wrote: {std_csv}")

    if args.show == "yes":
        plt.show()


if __name__ == "__main__":
    main()
