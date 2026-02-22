#!/usr/bin/env python3
"""Parameter sweep runner for droplet transfer-matrix uncertainty analysis.

Runs SPARTA for a grid of input parameters (passed via -var), then builds
ID-based transfer matrices from case.inner/case.outer and aggregates mean/std.
"""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_grid(spec: str, cast=float):
    vals = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    if not vals:
        raise ValueError(f"Empty grid spec: {spec}")
    return vals


def run(cmd, cwd: Path, dry_run: bool = False):
    print("$", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def rewrite_input_variables(src: Path, dst: Path, values: dict[str, str]):
    """
    Rewrite `variable <name> ...` lines in src and write to dst.
    If a variable is not found, prepend a definition.
    """
    text = src.read_text()
    found = set()
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        replaced = False
        for name, val in values.items():
            pat = rf"^variable\s+{re.escape(name)}\s+"
            if re.match(pat, stripped):
                out_lines.append(f"variable {name} equal {val}")
                found.add(name)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    missing = [k for k in values.keys() if k not in found]
    if missing:
        prefix = [f"variable {k} equal {values[k]}" for k in missing]
        out_lines = prefix + [""] + out_lines

    dst.write_text("\n".join(out_lines) + "\n")


def read_region_pct(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source" not in df.columns:
        raise RuntimeError(f"Missing source column in {path}")
    return df.set_index("source")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", default=".", help="Directory with input/dump files")
    p.add_argument("--input", default="in.droplet_emission", help="SPARTA input script")
    p.add_argument("--spa", default="../../src/spa_mpi", help="SPARTA executable")
    p.add_argument("--np", type=int, default=4, help="MPI ranks")
    p.add_argument("--builder", default="build_transfer_matrix.py", help="Matrix builder script")
    p.add_argument("--outdir", default="sweep_out", help="Output folder for runs and summaries")

    p.add_argument("--nemit", default="1000", help="Comma list, e.g. 200,500,1000")
    p.add_argument("--temp-emit", default="1e17,1e18,1e19,5e19",
                   help="Comma list for temp_emit variable values")
    p.add_argument("--seeds", default="1,2,3", help="Comma list of random seeds")

    # Mapping from sweep knobs to SPARTA -var names in your input file.
    p.add_argument("--var-nemit", default="nemit")
    p.add_argument("--var-temp-emit", default="temp_emit")
    p.add_argument("--var-seed", default="seed")

    p.add_argument("--launched-I", type=int, default=0)
    p.add_argument("--launched-O", type=int, default=0)
    p.add_argument("--inner-dump", default="case.inner")
    p.add_argument("--outer-dump", default="case.outer")
    p.add_argument(
        "--collide-log-base",
        default="collide_vanish_hits.csv",
        help="Base filename for surf_collide vanish logs to archive per run",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    outdir = (workdir / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    nemit_grid = parse_grid(args.nemit, int)
    temp_emit_grid = parse_grid(args.temp_emit, float)
    seed_grid = parse_grid(args.seeds, int)

    combos = list(itertools.product(nemit_grid, temp_emit_grid, seed_grid))
    print(f"Sweep runs: {len(combos)}")

    rows = []
    matrix_frames = []

    for irun, (nemit, temp_emit, seed) in enumerate(combos, start=1):
        run_tag = f"run{irun:04d}.nemit{nemit}.temp_emit{temp_emit:g}.s{seed}"
        run_dir = outdir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        run_input = run_dir / "in.sweep"
        rewrite_input_variables(
            workdir / args.input,
            run_input,
            {
                args.var_nemit: str(nemit),
                args.var_temp_emit: str(temp_emit),
                args.var_seed: str(seed),
            },
        )

        # Remove prior collide logs in workdir so each run archives only its own files.
        if not args.dry_run:
            for old in workdir.glob(f"{args.collide_log_base}*"):
                if old.is_file():
                    old.unlink()

        sparta_cmd = [
            "mpirun",
            "-np",
            str(args.np),
            args.spa,
            "-in",
            str(run_input),
        ]
        run(sparta_cmd, cwd=workdir, dry_run=args.dry_run)

        inner_src = workdir / args.inner_dump
        outer_src = workdir / args.outer_dump
        if not args.dry_run:
            if not inner_src.exists() or not outer_src.exists():
                raise RuntimeError("Missing case dumps after SPARTA run")
            inner_dst = run_dir / "case.inner"
            outer_dst = run_dir / "case.outer"
            shutil.copy2(inner_src, inner_dst)
            shutil.copy2(outer_src, outer_dst)

            # Archive per-rank collide vanish logs into this run folder.
            for f in sorted(workdir.glob(f"{args.collide_log_base}*")):
                if f.is_file():
                    shutil.copy2(f, run_dir / f.name)
        else:
            inner_dst = run_dir / "case.inner"
            outer_dst = run_dir / "case.outer"

        out_prefix = run_dir / "transfer"

        vanish_logs = sorted(run_dir.glob(f"{args.collide_log_base}.rank*"))
        if not vanish_logs:
            # Fallback: single-file logging without rank suffix
            single = run_dir / args.collide_log_base
            if single.exists():
                vanish_logs = [single]

        launched_I_run = int(args.launched_I) if int(args.launched_I) > 0 else int(nemit)
        launched_O_run = int(args.launched_O) if int(args.launched_O) > 0 else int(nemit)

        if vanish_logs:
            build_cmd = [
                "python3",
                args.builder,
                "--vanish-all",
                *[str(v) for v in vanish_logs],
                "--ispecies-I",
                "0",
                "--ispecies-O",
                "1",
                "--out-prefix",
                str(out_prefix),
                "--no-segment",
                "--launched-I",
                str(launched_I_run),
                "--launched-O",
                str(launched_O_run),
            ]
        else:
            # Last-resort fallback for non-vanish runs
            build_cmd = [
                "python3",
                args.builder,
                "--case-I",
                str(inner_dst),
                "--case-O",
                str(outer_dst),
                "--out-prefix",
                str(out_prefix),
                "--no-segment",
                "--launched-I",
                str(launched_I_run),
                "--launched-O",
                str(launched_O_run),
            ]
        run(build_cmd, cwd=workdir, dry_run=args.dry_run)

        if args.dry_run:
            continue

        region_pct = read_region_pct(Path(f"{out_prefix}.region.pct.csv"))
        region_pct["run_tag"] = run_tag
        region_pct["nemit"] = nemit
        region_pct["temp_emit"] = temp_emit
        region_pct["seed"] = seed
        matrix_frames.append(region_pct.reset_index())

        hit_cols = [c for c in region_pct.columns if c not in {"run_tag", "nemit", "temp_emit", "seed"}]
        for src in ["I", "O"]:
            if src not in region_pct.index:
                continue
            rec = {
                "run_tag": run_tag,
                "source": src,
                "nemit": nemit,
                "temp_emit": temp_emit,
                "seed": seed,
            }
            for c in hit_cols:
                rec[c] = float(region_pct.loc[src, c])
            rows.append(rec)

    if args.dry_run:
        print("Dry-run only; no files aggregated.")
        return

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No sweep results collected")

    long_csv = outdir / "sweep_region_pct_long.csv"
    long_df.to_csv(long_csv, index=False)

    value_cols = [c for c in long_df.columns if c not in {"run_tag", "source", "nemit", "temp_emit", "seed"}]
    grp = long_df.groupby("source", dropna=False)[value_cols]
    mean_df = grp.mean().reset_index()
    std_df = grp.std(ddof=1).fillna(0.0).reset_index()

    mean_csv = outdir / "sweep_region_pct_mean.csv"
    std_csv = outdir / "sweep_region_pct_std.csv"
    mean_df.to_csv(mean_csv, index=False)
    std_df.to_csv(std_csv, index=False)

    # Also aggregate by temperature to separate parametric spread from seed spread.
    grp_temp = long_df.groupby(["temp_emit", "source"], dropna=False)[value_cols]
    mean_by_temp_df = grp_temp.mean().reset_index()
    std_by_temp_df = grp_temp.std(ddof=1).fillna(0.0).reset_index()
    mean_by_temp_csv = outdir / "sweep_region_pct_mean_by_temp.csv"
    std_by_temp_csv = outdir / "sweep_region_pct_std_by_temp.csv"
    mean_by_temp_df.to_csv(mean_by_temp_csv, index=False)
    std_by_temp_df.to_csv(std_by_temp_csv, index=False)

    full_matrix_csv = outdir / "sweep_region_pct_all_runs.csv"
    pd.concat(matrix_frames, ignore_index=True).to_csv(full_matrix_csv, index=False)

    print(f"Wrote: {long_csv}")
    print(f"Wrote: {mean_csv}")
    print(f"Wrote: {std_csv}")
    print(f"Wrote: {mean_by_temp_csv}")
    print(f"Wrote: {std_by_temp_csv}")
    print(f"Wrote: {full_matrix_csv}")


if __name__ == "__main__":
    main()
