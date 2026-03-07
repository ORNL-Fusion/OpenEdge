#!/usr/bin/env python3
"""
Compute areal density (and flux) on surfaces from OpenEdge/SPARTA surf dump files.

Reads a SPARTA surf dump (from `dump surf`), extracts per-surface-element
flux data from fix ave/surf columns, computes triangle areas from vertex
coordinates, and produces:
  - Time-resolved areal density (flux × time) per surface element
  - Total/per-species integrated particle counts
  - Optional VTP output for ParaView visualization

Simulation parameters (timestep, dump interval) are auto-detected from
the SPARTA input script or log file, with CLI overrides available.

Usage:
  # Auto-detect dt from input script, process all timesteps
  python surf_areal_density.py -f output/surf_react.dat -i in.cpc_gitr

  # Manual dt, write VTP series
  python surf_areal_density.py -f output/surf_react.dat --dt 1e-10 --vtp

  # Filter specific species columns, save CSV summary
  python surf_areal_density.py -f output/surf_react.dat -i in.script --csv output/areal.csv

  # Plot areal density vs time
  python surf_areal_density.py -f output/surf_react.dat -i in.script --plot
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Input script / log parser
# ---------------------------------------------------------------------------
def parse_input_script(path: str) -> dict:
    """Extract simulation parameters from a SPARTA input script.

    Handles variable substitution for simple `variable X equal <val>` lines.
    Returns dict with keys: timestep, dump_interval (if found).
    """
    variables = {}
    params = {}

    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue

            # Handle line continuations (&)
            while line.endswith("&"):
                line = line[:-1].strip()
                next_line = next(f, "").split("#")[0].strip()
                line += " " + next_line

            tokens = line.split()
            if not tokens:
                continue

            # variable X equal <expr>
            if tokens[0] == "variable" and len(tokens) >= 4 and tokens[2] == "equal":
                name = tokens[1]
                expr = " ".join(tokens[3:])
                # Substitute known variables
                for vname, vval in variables.items():
                    expr = expr.replace(f"${{{vname}}}", str(vval))
                    expr = expr.replace(f"${vname}", str(vval))
                try:
                    variables[name] = float(eval(expr))
                except Exception:
                    variables[name] = expr

            # timestep <val>
            if tokens[0] == "timestep":
                val = tokens[1]
                for vname, vval in variables.items():
                    val = val.replace(f"${{{vname}}}", str(vval))
                    val = val.replace(f"${vname}", str(vval))
                try:
                    params["timestep"] = float(val)
                except ValueError:
                    pass

            # run <val>
            if tokens[0] == "run":
                val = tokens[1]
                for vname, vval in variables.items():
                    val = val.replace(f"${{{vname}}}", str(vval))
                    val = val.replace(f"${vname}", str(vval))
                try:
                    params["nsteps"] = int(float(val))
                except ValueError:
                    pass

    return params


# ---------------------------------------------------------------------------
# Surf dump parser
# ---------------------------------------------------------------------------
def parse_surf_dump(filename: str) -> dict:
    """Parse a SPARTA surf dump file.

    Returns dict with:
      timestep: int array (one entry per surf-element per snapshot)
      id: int array
      v1, v2, v3: (N,3) float arrays (triangle vertices)
      area: float array (triangle areas, computed if not in header)
      <column_name>: float array for each extra column (e.g. f_fsurf[1])
    """
    with open(filename) as f:
        lines = f.readlines()

    result = {}
    rows = []  # list of dicts per row
    i = 0
    timestep = 0
    num_surfs = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF SURFS":
            num_surfs = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 4
        elif line.startswith("ITEM: SURFS"):
            header = line.split()[2:]
            for j in range(num_surfs):
                parts = lines[i + 1 + j].strip().split()
                row = {"timestep": timestep}
                for ci, name in enumerate(header):
                    if ci < len(parts):
                        if name == "id":
                            row[name] = int(parts[ci])
                        else:
                            row[name] = float(parts[ci])
                rows.append(row)
            i += 1 + num_surfs
        else:
            i += 1

    if not rows:
        raise RuntimeError(f"No surf data found in {filename}")

    # Pack into arrays
    n = len(rows)
    result["timestep"] = np.array([r["timestep"] for r in rows], dtype=int)
    result["id"] = np.array([r.get("id", 0) for r in rows], dtype=int)

    # Vertices
    vtx_keys = ["v1x", "v1y", "v1z", "v2x", "v2y", "v2z", "v3x", "v3y", "v3z"]
    has_vtx = all(k in rows[0] for k in vtx_keys)
    if has_vtx:
        result["v1"] = np.column_stack([
            [r["v1x"] for r in rows], [r["v1y"] for r in rows], [r["v1z"] for r in rows]])
        result["v2"] = np.column_stack([
            [r["v2x"] for r in rows], [r["v2y"] for r in rows], [r["v2z"] for r in rows]])
        result["v3"] = np.column_stack([
            [r["v3x"] for r in rows], [r["v3y"] for r in rows], [r["v3z"] for r in rows]])
        # Compute triangle areas
        cross = np.cross(result["v2"] - result["v1"], result["v3"] - result["v1"])
        result["area"] = 0.5 * np.linalg.norm(cross, axis=1)
    elif "area" in rows[0]:
        result["area"] = np.array([r["area"] for r in rows])

    # All other numeric columns
    skip = {"timestep", "id"} | set(vtx_keys)
    for key in rows[0]:
        if key in skip:
            continue
        if key in ("v1", "v2", "v3", "area") and key in result:
            continue
        result[key] = np.array([r.get(key, 0.0) for r in rows])

    return result


def get_flux_columns(data: dict) -> list[str]:
    """Find flux columns (f_fsurf[N] or f_f_all[N] patterns)."""
    flux_cols = []
    for key in sorted(data.keys()):
        if re.match(r"f_\w+\[\d+\]", key):
            flux_cols.append(key)
    return flux_cols


# ---------------------------------------------------------------------------
# Areal density computation
# ---------------------------------------------------------------------------
def compute_areal_density(data: dict, dt: float, flux_cols: list[str] | None = None):
    """Compute areal density for each timestep snapshot.

    Returns a list of dicts, one per timestep, each containing:
      step, time, surf_ids, area,
      flux_<col>: flux array per surf element,
      sigma_<col>: areal density (flux * time) per surf element,
      N_<col>: integrated count (sum of sigma * area)
    """
    if flux_cols is None:
        flux_cols = get_flux_columns(data)
    if not flux_cols:
        raise ValueError("No flux columns found in surf dump data")

    unique_ts = np.unique(data["timestep"])
    results = []

    for ts in unique_ts:
        mask = data["timestep"] == ts
        time = ts * dt
        area = data["area"][mask] if "area" in data else None
        entry = {
            "step": int(ts),
            "time": time,
            "surf_ids": data["id"][mask],
            "area": area,
            "n_surfs": int(mask.sum()),
        }

        for col in flux_cols:
            flux = data[col][mask]
            entry[f"flux_{col}"] = flux
            sigma = flux * time
            entry[f"sigma_{col}"] = sigma
            if area is not None:
                entry[f"N_{col}"] = float(np.sum(sigma * area))
            else:
                entry[f"N_{col}"] = float(np.sum(sigma))

        # Total across all species
        total_flux = np.zeros(int(mask.sum()))
        for col in flux_cols:
            total_flux += data[col][mask]
        entry["flux_total"] = total_flux
        entry["sigma_total"] = total_flux * time
        if area is not None:
            entry["N_total"] = float(np.sum(total_flux * time * area))
        else:
            entry["N_total"] = float(np.sum(total_flux * time))

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Output: summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], flux_cols: list[str]):
    """Print a summary table of integrated counts vs time."""
    header = f"{'step':>12s} {'time':>12s} {'N_total':>14s}"
    for col in flux_cols:
        header += f"  {f'N_{col}':>14s}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['step']:12d} {r['time']:12.4e} {r['N_total']:14.4e}"
        for col in flux_cols:
            line += f"  {r[f'N_{col}']:14.4e}"
        print(line)


def write_csv(results: list[dict], flux_cols: list[str], path: str):
    """Write summary CSV."""
    with open(path, "w") as f:
        cols = ["step", "time", "N_total"] + [f"N_{c}" for c in flux_cols]
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r["step"]), f"{r['time']:.6e}", f"{r['N_total']:.6e}"]
            for c in flux_cols:
                vals.append(f"{r[f'N_{c}']:.6e}")
            f.write(",".join(vals) + "\n")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Output: plot
# ---------------------------------------------------------------------------
def plot_areal_density(results: list[dict], flux_cols: list[str], outfile: str | None):
    """Plot integrated particle count vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["time"] for r in results]
    N_total = [r["N_total"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(times, N_total, "k-o", ms=4, lw=1.5, label="Total")

    for col in flux_cols:
        Ni = [r[f"N_{col}"] for r in results]
        # Extract a short label
        m = re.search(r"\[(\d+)\]", col)
        label = f"species {m.group(1)}" if m else col
        ax.plot(times, Ni, "-o", ms=3, lw=1, label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Integrated count (areal density × area)")
    ax.legend(fontsize=9)
    ax.set_title("Surface areal density vs time")
    ax.grid(True, alpha=0.3)

    if outfile:
        fig.savefig(outfile, dpi=150)
        print(f"Saved {outfile}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Output: VTP (optional, requires VTK)
# ---------------------------------------------------------------------------
def write_vtp(data: dict, results: list[dict], flux_cols: list[str], outdir: str):
    """Write per-timestep VTP files with areal density as cell data."""
    try:
        import vtk
    except ImportError:
        print("VTK not available — skipping VTP output. Install with: pip install vtk")
        return

    os.makedirs(outdir, exist_ok=True)

    for r in results:
        step = r["step"]
        ntri = r["n_surfs"]
        mask = data["timestep"] == step

        v1 = data["v1"][mask]
        v2 = data["v2"][mask]
        v3 = data["v3"][mask]

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        for i in range(ntri):
            p0 = points.InsertNextPoint(*map(float, v1[i]))
            p1 = points.InsertNextPoint(*map(float, v2[i]))
            p2 = points.InsertNextPoint(*map(float, v3[i]))
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, p0)
            tri.GetPointIds().SetId(1, p1)
            tri.GetPointIds().SetId(2, p2)
            polys.InsertNextCell(tri)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)

        # Surf ID
        arr_id = vtk.vtkIntArray()
        arr_id.SetName("surf_id")
        arr_id.SetNumberOfTuples(ntri)
        for i in range(ntri):
            arr_id.SetValue(i, int(r["surf_ids"][i]))
        poly.GetCellData().AddArray(arr_id)

        # Area
        if r["area"] is not None:
            arr_area = vtk.vtkDoubleArray()
            arr_area.SetName("area")
            arr_area.SetNumberOfTuples(ntri)
            for i in range(ntri):
                arr_area.SetValue(i, float(r["area"][i]))
            poly.GetCellData().AddArray(arr_area)

        # Total areal density
        arr_sig = vtk.vtkDoubleArray()
        arr_sig.SetName("areal_density_total")
        arr_sig.SetNumberOfTuples(ntri)
        for i in range(ntri):
            arr_sig.SetValue(i, float(r["sigma_total"][i]))
        poly.GetCellData().AddArray(arr_sig)
        poly.GetCellData().SetActiveScalars("areal_density_total")

        # Per-species areal density
        for col in flux_cols:
            arr = vtk.vtkDoubleArray()
            m = re.search(r"\[(\d+)\]", col)
            name = f"areal_density_sp{m.group(1)}" if m else f"areal_density_{col}"
            arr.SetName(name)
            arr.SetNumberOfTuples(ntri)
            for i in range(ntri):
                arr.SetValue(i, float(r[f"sigma_{col}"][i]))
            poly.GetCellData().AddArray(arr)

        outfile = os.path.join(outdir, f"surf_t{step:010d}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(outfile)
        writer.SetInputData(poly)
        writer.Write()

    print(f"Wrote {len(results)} VTP files to {outdir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute surface areal density from OpenEdge/SPARTA surf dumps")
    ap.add_argument("-f", "--file", required=True,
                    help="Path to surf dump file (e.g. output/surf_react.dat)")
    ap.add_argument("-i", "--input-script", default=None,
                    help="SPARTA input script (auto-detects timestep, nsteps)")
    ap.add_argument("--dt", type=float, default=None,
                    help="Override timestep [s] (otherwise read from input script)")
    ap.add_argument("--csv", default=None,
                    help="Write summary CSV to this path")
    ap.add_argument("--plot", default=None, nargs="?", const="areal_density.png",
                    help="Plot areal density vs time (optionally specify output path)")
    ap.add_argument("--vtp", default=None, nargs="?", const="vtp_out",
                    help="Write VTP files to this directory")
    ap.add_argument("--quiet", action="store_true", help="Suppress summary table")
    args = ap.parse_args()

    # Get dt
    dt = args.dt
    if dt is None and args.input_script:
        params = parse_input_script(args.input_script)
        dt = params.get("timestep")
        if dt:
            print(f"Auto-detected timestep = {dt:.2e} s from {args.input_script}")
    if dt is None:
        print("ERROR: Could not determine timestep. Provide --dt or -i <input_script>",
              file=sys.stderr)
        sys.exit(1)

    # Parse surf dump
    print(f"Parsing {args.file} ...")
    data = parse_surf_dump(args.file)
    unique_ts = np.unique(data["timestep"])
    print(f"  {len(unique_ts)} snapshots, "
          f"{len(np.unique(data['id']))} surface elements")

    flux_cols = get_flux_columns(data)
    if not flux_cols:
        print("ERROR: No flux columns (f_*[N]) found in surf dump", file=sys.stderr)
        sys.exit(1)
    print(f"  Flux columns: {flux_cols}")

    # Compute
    results = compute_areal_density(data, dt, flux_cols)

    # Output
    if not args.quiet:
        print()
        print_summary(results, flux_cols)

    if args.csv:
        write_csv(results, flux_cols, args.csv)

    if args.plot is not None:
        plot_areal_density(results, flux_cols, args.plot)

    if args.vtp is not None:
        if "v1" not in data:
            print("WARNING: No vertex data in surf dump — cannot write VTP")
        else:
            write_vtp(data, results, flux_cols, args.vtp)


if __name__ == "__main__":
    main()
