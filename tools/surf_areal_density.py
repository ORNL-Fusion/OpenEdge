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

  # Manual dt, write VTP series (vertices from geometry file)
  python surf_areal_density.py -f output/surf_react.dat --dt 1e-10 \
      --surf input/gitr_geometry.surf --vtp output/vtp

  # Per-species breakdown (requires reactions file)
  python surf_areal_density.py -f output/surf_react.dat -i in.script \
      --reactions reactions_pmi.surf --group species --csv output/areal.csv

  # R/S/A breakdown with plot
  python surf_areal_density.py -f output/surf_react.dat -i in.script \
      --reactions reactions_pmi.surf --group all --plot
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
# SPARTA geometry file parser (read_surf format)
# ---------------------------------------------------------------------------
def parse_surf_geometry(filename: str) -> dict:
    """Parse a SPARTA surface geometry file (Points + Triangles).

    Returns dict with:
      points: (npts, 3) float array (1-indexed ID -> coordinates)
      tris:   (ntri, 3) int array of point indices (1-indexed)
      v1, v2, v3: (ntri, 3) float arrays (triangle vertex coordinates)
      area: (ntri,) float array
    """
    with open(filename) as f:
        lines = f.readlines()

    npoints = 0
    ntris = 0
    points = {}
    tris = {}
    section = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        tokens = line.split()

        # Header lines: "642 points", "1280 triangles"
        if len(tokens) == 2 and tokens[1] == "points":
            npoints = int(tokens[0])
            i += 1
            continue
        if len(tokens) == 2 and tokens[1] in ("triangles", "lines"):
            ntris = int(tokens[0])
            i += 1
            continue

        if tokens[0] == "Points":
            section = "points"
            i += 1
            continue
        if tokens[0] == "Triangles":
            section = "triangles"
            i += 1
            continue
        if tokens[0] == "Lines":
            section = "lines"
            i += 1
            continue

        if section == "points" and len(tokens) >= 4:
            pid = int(tokens[0])
            points[pid] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            i += 1
        elif section in ("triangles", "lines") and len(tokens) >= 4:
            tid = int(tokens[0])
            tris[tid] = [int(t) for t in tokens[1:4]]
            i += 1
        else:
            i += 1

    if not points or not tris:
        raise RuntimeError(f"Could not parse geometry from {filename}")

    # Build vertex arrays indexed by triangle ID (1-based)
    max_tid = max(tris.keys())
    v1 = np.zeros((max_tid + 1, 3))
    v2 = np.zeros((max_tid + 1, 3))
    v3 = np.zeros((max_tid + 1, 3))

    for tid, (p1, p2, p3) in tris.items():
        v1[tid] = points[p1]
        v2[tid] = points[p2]
        v3[tid] = points[p3]

    cross = np.cross(v2[1:] - v1[1:], v3[1:] - v1[1:])
    area = 0.5 * np.linalg.norm(cross, axis=1)

    return {
        "v1": v1, "v2": v2, "v3": v3,
        "area": area,  # indexed 0..ntri-1 (surf ID 1 -> index 0)
        "ntris": len(tris),
    }


def merge_geometry(data: dict, geom: dict):
    """Merge vertex/area data from geometry file into surf dump data by ID.

    Adds v1, v2, v3, area arrays to data (matching dump row order).
    """
    ids = data["id"]
    n = len(ids)
    v1 = np.zeros((n, 3))
    v2 = np.zeros((n, 3))
    v3 = np.zeros((n, 3))
    area = np.zeros(n)

    for i, sid in enumerate(ids):
        if sid < len(geom["v1"]):
            v1[i] = geom["v1"][sid]
            v2[i] = geom["v2"][sid]
            v3[i] = geom["v3"][sid]
        # area is 0-indexed (surf ID 1 -> index 0)
        aidx = sid - 1
        if 0 <= aidx < len(geom["area"]):
            area[i] = geom["area"][aidx]

    data["v1"] = v1
    data["v2"] = v2
    data["v3"] = v3
    data["area"] = area


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
# Reaction file parser — maps column index to reaction type (R/S/A)
# ---------------------------------------------------------------------------
def parse_reactions_file(filename: str) -> list[dict]:
    """Parse a surf_react_pmi reactions file.

    Returns a list of dicts (one per reaction, in order), each with:
      reactant, product, type ('R', 'S', or 'A'), index (0-based)
    """
    reactions = []
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    i = 0
    idx = 0
    while i < len(lines):
        # Reaction line: "W+ --> W" or "W --> NULL"
        if "-->" in lines[i]:
            parts = lines[i].split("-->")
            reactant = parts[0].strip()
            product = parts[1].strip()
            # Next line: "R Simple 1.0" or "S Simple 1.0" or "A Simple 1.0"
            if i + 1 < len(lines):
                rtype = lines[i + 1].split()[0].upper()
                reactions.append({
                    "reactant": reactant,
                    "product": product,
                    "type": rtype,
                    "index": idx,
                })
                idx += 1
                i += 2
            else:
                i += 1
        else:
            i += 1

    return reactions


def build_column_groups(flux_cols: list[str],
                        reactions: list[dict] | None = None) -> dict:
    """Build named groups of flux columns for summation.

    Returns dict: group_name -> list of column names.
    Always includes 'total'.  If reactions are provided, also includes
    'reflect', 'sputter', 'absorb', and per-species groups.
    """
    groups = {"total": list(flux_cols)}

    if reactions:
        type_map = {"R": "reflect", "S": "sputter", "A": "absorb"}
        for gname in type_map.values():
            groups[gname] = []
        species_seen = {}

        for rxn in reactions:
            col_idx = rxn["index"]  # 0-based
            # Find matching flux column (1-based in SPARTA)
            col_name = None
            for fc in flux_cols:
                m = re.search(r"\[(\d+)\]", fc)
                if m and int(m.group(1)) == col_idx + 1:
                    col_name = fc
                    break
            if not col_name:
                continue

            gname = type_map.get(rxn["type"])
            if gname:
                groups[gname].append(col_name)

            sp = rxn["reactant"]
            if sp not in species_seen:
                species_seen[sp] = []
            species_seen[sp].append(col_name)

        for sp, cols in species_seen.items():
            groups[f"sp_{sp}"] = cols

    return groups


# ---------------------------------------------------------------------------
# Areal density computation
# ---------------------------------------------------------------------------
def compute_areal_density(data: dict, dt: float,
                          flux_cols: list[str],
                          groups: dict | None = None):
    """Compute areal density for each timestep snapshot.

    groups: dict of group_name -> list of column names to sum.
    If None, defaults to {'total': flux_cols}.

    Returns a list of dicts, one per timestep, each containing:
      step, time, surf_ids, area, n_surfs,
      sigma_<group>: areal density array per surf element,
      N_<group>: integrated count (sum of sigma * area)
    """
    if groups is None:
        groups = {"total": list(flux_cols)}

    unique_ts = np.unique(data["timestep"])
    results = []

    for ts in unique_ts:
        mask = data["timestep"] == ts
        time = ts * dt
        area = data["area"][mask] if "area" in data else None
        nsurf = int(mask.sum())
        entry = {
            "step": int(ts),
            "time": time,
            "surf_ids": data["id"][mask],
            "area": area,
            "n_surfs": nsurf,
        }

        for gname, cols in groups.items():
            flux = np.zeros(nsurf)
            for col in cols:
                flux += data[col][mask]
            sigma = flux * time
            entry[f"sigma_{gname}"] = sigma
            if area is not None:
                entry[f"N_{gname}"] = float(np.sum(sigma * area))
            else:
                entry[f"N_{gname}"] = float(np.sum(sigma))

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Output: summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[dict], groups: dict):
    """Print a summary table of integrated counts vs time."""
    gnames = list(groups.keys())
    header = f"{'step':>12s} {'time':>12s}"
    for g in gnames:
        header += f"  {f'N_{g}':>14s}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['step']:12d} {r['time']:12.4e}"
        for g in gnames:
            line += f"  {r[f'N_{g}']:14.4e}"
        print(line)


def write_csv(results: list[dict], groups: dict, path: str):
    """Write summary CSV."""
    gnames = list(groups.keys())
    with open(path, "w") as f:
        cols = ["step", "time"] + [f"N_{g}" for g in gnames]
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r["step"]), f"{r['time']:.6e}"]
            for g in gnames:
                vals.append(f"{r[f'N_{g}']:.6e}")
            f.write(",".join(vals) + "\n")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Output: plot
# ---------------------------------------------------------------------------
def plot_areal_density(results: list[dict], groups: dict, outfile: str | None):
    """Plot integrated particle count vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["time"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for gname in groups:
        key = f"N_{gname}"
        Ni = [r[key] for r in results]
        style = "k-o" if gname == "total" else "-o"
        ms = 4 if gname == "total" else 3
        lw = 1.5 if gname == "total" else 1
        ax.plot(times, Ni, style, ms=ms, lw=lw, label=gname)

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
def write_vtp(data: dict, results: list[dict], groups: dict, outdir: str):
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

        # Areal density per group
        for gname in groups:
            arr = vtk.vtkDoubleArray()
            arr.SetName(f"areal_density_{gname}")
            arr.SetNumberOfTuples(ntri)
            for i in range(ntri):
                arr.SetValue(i, float(r[f"sigma_{gname}"][i]))
            poly.GetCellData().AddArray(arr)

        if "total" in groups:
            poly.GetCellData().SetActiveScalars("areal_density_total")

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
    ap.add_argument("--surf", default=None,
                    help="SPARTA geometry file (e.g. input/gitr_geometry.surf) — "
                         "provides triangle vertices when not in dump")
    ap.add_argument("--vtp", default=None, nargs="?", const="vtp_out",
                    help="Write VTP files to this directory")
    ap.add_argument("--reactions", default=None,
                    help="Reactions file (reactions_pmi.surf) — enables per-type "
                         "and per-species grouping")
    ap.add_argument("--group", default="total",
                    choices=["total", "all", "species"],
                    help="Column grouping: 'total' (sum all), 'all' (total + "
                         "R/S/A breakdown), 'species' (per-species). "
                         "Requires --reactions for 'all' and 'species'.")
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

    # Merge geometry from surf file if dump lacks vertices
    if "v1" not in data and args.surf:
        print(f"Reading geometry from {args.surf} ...")
        geom = parse_surf_geometry(args.surf)
        merge_geometry(data, geom)
        print(f"  Merged {geom['ntris']} triangles")

    flux_cols = get_flux_columns(data)
    if not flux_cols:
        print("ERROR: No flux columns (f_*[N]) found in surf dump", file=sys.stderr)
        sys.exit(1)
    print(f"  Flux columns: {flux_cols}")

    # Build column groups
    reactions = None
    if args.reactions:
        reactions = parse_reactions_file(args.reactions)
        print(f"  Parsed {len(reactions)} reactions from {args.reactions}")

    all_groups = build_column_groups(flux_cols, reactions)

    # Select which groups to use based on --group
    if args.group == "total":
        groups = {"total": all_groups["total"]}
    elif args.group == "all":
        if not reactions:
            print("WARNING: --group all requires --reactions; falling back to total",
                  file=sys.stderr)
            groups = {"total": all_groups["total"]}
        else:
            groups = {k: v for k, v in all_groups.items()
                      if k in ("total", "reflect", "sputter", "absorb")}
    elif args.group == "species":
        if not reactions:
            print("WARNING: --group species requires --reactions; falling back to total",
                  file=sys.stderr)
            groups = {"total": all_groups["total"]}
        else:
            groups = {k: v for k, v in all_groups.items()
                      if k == "total" or k.startswith("sp_")}

    print(f"  Groups: {list(groups.keys())}")

    # Compute
    results = compute_areal_density(data, dt, flux_cols, groups)

    # Output
    if not args.quiet:
        print()
        print_summary(results, groups)

    if args.csv:
        write_csv(results, groups, args.csv)

    if args.plot is not None:
        plot_areal_density(results, groups, args.plot)

    if args.vtp is not None:
        if "v1" not in data:
            print("WARNING: No vertex data — use --surf <geometry.surf> to provide vertices")
        else:
            write_vtp(data, results, groups, args.vtp)


if __name__ == "__main__":
    main()
