#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Optional


def run_verify_case(example_dir: Path, exe: str, mpi_procs: int, input_file: str) -> None:
    in_file = example_dir / input_file
    exe_path = (example_dir / exe).resolve() if not Path(exe).is_absolute() else Path(exe)
    cmd = ["mpirun", "-np", str(mpi_procs), str(exe_path)]
    with in_file.open("r") as f:
        proc = subprocess.run(cmd, stdin=f, cwd=example_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"Verification run failed with exit code {proc.returncode}")


def read_dataset_h5dump(h5file: Path, dataset: str):
    cmd = ["h5dump", "-d", dataset, str(h5file)]
    out = subprocess.check_output(cmd, text=True)

    m = re.search(r"DATA\s*\{(.*)\}\s*$", out, flags=re.S)
    if not m:
        raise RuntimeError(f"Could not parse DATA block for {dataset}")
    data_block = m.group(1)

    data_block = re.sub(r"\([^\)]*\)\s*:", "", data_block)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data_block)
    return [float(v) for v in nums]


def save_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r", "z", "ne"])
        w.writerows(rows)


def read_grid_dump_ne(dump_file: Path):
    lines = dump_file.read_text().splitlines()
    i = 0
    last_rows = None
    last_cols = None

    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            i += 2
            if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF CELLS"):
                continue
            i += 1
            ncell = int(lines[i].strip())
            i += 1

            if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
                continue
            i += 4

            if i >= len(lines) or not lines[i].startswith("ITEM: CELLS"):
                continue
            cols = lines[i].split()[2:]
            i += 1
            rows = []
            for _ in range(ncell):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) == len(cols):
                    rows.append(parts)
                i += 1

            last_rows = rows
            last_cols = cols
        else:
            i += 1

    if not last_rows or not last_cols:
        raise RuntimeError(f"Could not parse cells from dump file: {dump_file}")

    try:
        ix = last_cols.index("xc")
        iy = last_cols.index("yc")
    except ValueError as exc:
        raise RuntimeError("Dump file must include xc and yc columns") from exc

    ic = None
    for j, name in enumerate(last_cols):
        if name.startswith("c_"):
            ic = j
            break
    if ic is None:
        raise RuntimeError("Dump file has no compute column (c_*)")

    triples = [(float(r[ix]), float(r[iy]), float(r[ic])) for r in last_rows]
    r_vals = sorted({t[0] for t in triples})
    z_vals = sorted({t[1] for t in triples})
    val_map = {(rv, zv): vv for rv, zv, vv in triples}

    dens = []
    rows = []
    for zv in z_vals:
        for rv in r_vals:
            vv = val_map.get((rv, zv), float("nan"))
            dens.append(vv)
            rows.append((rv, zv, vv))
    return r_vals, z_vals, dens, rows


def read_surface_polygon(surface_file: Path):
    points = {}
    lines = []
    section = None

    with surface_file.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low == "points":
                section = "points"
                continue
            if low == "lines":
                section = "lines"
                continue

            parts = line.split()
            if section == "points" and len(parts) >= 3 and parts[0].isdigit():
                points[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "lines" and len(parts) >= 3 and parts[0].isdigit():
                lines.append((int(parts[1]), int(parts[2])))

    if not points:
        raise RuntimeError(f"No points parsed from {surface_file}")

    if not lines:
        ordered = [points[i] for i in sorted(points)]
        return ordered

    # Build polygon from line connectivity to avoid relying on point file order.
    chain = [lines[0][0], lines[0][1]]
    remaining = lines[1:]
    while remaining:
        last = chain[-1]
        next_idx = None
        for i, (a, b) in enumerate(remaining):
            if a == last:
                chain.append(b)
                next_idx = i
                break
            if b == last:
                chain.append(a)
                next_idx = i
                break
        if next_idx is None:
            break
        remaining.pop(next_idx)
        if chain[-1] == chain[0]:
            break

    if chain[0] != chain[-1]:
        chain.append(chain[0])

    polygon = [points[idx] for idx in chain if idx in points]
    if len(polygon) < 4:
        raise RuntimeError(f"Could not build closed core polygon from {surface_file}")
    return polygon


def save_plot_png(path: Path, r, z, dens, nr: int, nz: int, core_file: Optional[Path] = None) -> None:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.path import Path as MplPath
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib/Numpy are required for PNG plotting. "
            "Activate your plotting environment and rerun."
        ) from exc

    R = np.array(r).reshape(1, nr).repeat(nz, axis=0)
    Z = np.array(z).reshape(nz, 1).repeat(nr, axis=1)
    NE = np.array(dens).reshape(nz, nr)

    M = np.full_like(NE, np.nan, dtype=float)
    pos = NE > 0.0
    M[pos] = np.log10(NE[pos])

    core_pts = None
    if core_file is not None and core_file.exists():
        core_pts = np.array(read_surface_polygon(core_file), dtype=float)
        if core_pts.shape[0] > 2:
            poly = MplPath(core_pts)
            q = np.column_stack((R.ravel(), Z.ravel()))
            inside = poly.contains_points(q).reshape(R.shape)
            M[inside] = np.nan

    fig, ax = plt.subplots(figsize=(5.1, 4.8), dpi=150)
    fig.patch.set_facecolor("#d9d9d9")
    ax.set_facecolor("#d9d9d9")

    cmap = mpl.colormaps["hot"].copy()
    cmap.set_bad("#d9d9d9")

    finite = M[np.isfinite(M)]
    if finite.size == 0:
        raise RuntimeError("No positive density values to plot")

    vmin = float(np.nanpercentile(finite, 2.0))
    vmax = float(np.nanpercentile(finite, 98.0))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    im = ax.pcolormesh(R, Z, M, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    if core_pts is not None and core_pts.shape[0] > 2:
        ax.plot(core_pts[:, 0], core_pts[:, 1], color="#3ea6ff", lw=1.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(r"$\log_{10}(n_e\,[\mathrm{m}^{-3}])$", fontsize=12, weight="bold")

    ax.set_xlabel("R [m]", fontsize=13, weight="bold")
    ax.set_ylabel("Z [m]", fontsize=13, weight="bold")
    ax.set_title(r"$\log_{10}(n_e\,[\mathrm{m}^{-3}])$", fontsize=14, weight="bold")
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Verify compute plasma/fields and plot ne(r,z)")
    p.add_argument("--no-run", action="store_true", help="Skip SPARTA verification run")
    p.add_argument("--exe", default="../../src/spa_mpi", help="Path to executable relative to example dir")
    p.add_argument("--np", type=int, default=4, help="MPI ranks for verification run")
    p.add_argument(
        "--case",
        choices=["file", "constant"],
        default="file",
        help="Select data source test case",
    )
    p.add_argument("--core-file", default="core.txt", help="Core surface file for masking empty core")
    args = p.parse_args()

    example_dir = Path(__file__).resolve().parent
    out_dir = example_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case == "file":
        input_name = "in.file"
        csv_path = out_dir / "ne_rz_file.csv"
        png_path = out_dir / "ne_rz_file.png"
    else:
        input_name = "in.constant"
        csv_path = out_dir / "ne_rz_constant.csv"
        png_path = out_dir / "ne_rz_constant.png"

    if not args.no_run:
        run_verify_case(example_dir, args.exe, args.np, input_name)

    if args.case == "file":
        plasma = example_dir / "plasma.h5"
        r = read_dataset_h5dump(plasma, "/r")
        z = read_dataset_h5dump(plasma, "/z")
        dens = read_dataset_h5dump(plasma, "/dens_e")
        nr = len(r)
        nz = len(z)
        if len(dens) != nr * nz:
            raise RuntimeError(f"dens_e size mismatch: got {len(dens)} expected {nr*nz}")
        rows = []
        for iz, zz in enumerate(z):
            base = iz * nr
            for ir, rr in enumerate(r):
                rows.append((rr, zz, dens[base + ir]))
    else:
        dump_path = out_dir / "plasma_grid.constant.ne"
        if not dump_path.exists():
            raise RuntimeError(f"Expected dump file not found: {dump_path}")
        r, z, dens, rows = read_grid_dump_ne(dump_path)
        nr = len(r)
        nz = len(z)

    save_csv(csv_path, rows)
    save_plot_png(png_path, r, z, dens, nr, nz, example_dir / args.core_file)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
