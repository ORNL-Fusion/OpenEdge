#!/usr/bin/env python3
"""Convert WEST 3D impurity-density dump to a VTK unstructured point cloud (.vtu).

This writes only cells above a threshold, which is often easier to inspect
in ParaView than a full structured-grid box.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import vtk
from vtk.util import numpy_support


def parse_grid_dump(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    blocks = {}
    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue
        ts = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF CELLS":
            raise RuntimeError("Bad dump format: NUMBER OF CELLS")
        nc = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 4

        if not lines[i].startswith("ITEM: CELLS"):
            raise RuntimeError("Bad dump format: CELLS header")
        header = lines[i].split()[2:]
        i += 1

        cols = [[] for _ in header]
        for _ in range(nc):
            row = lines[i].split()
            i += 1
            for j in range(len(header)):
                cols[j].append(float(row[j]))
        blocks[ts] = (header, [np.array(c) for c in cols])

    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= np.sum(ker)
    return ker


def convolve1d_reflect(arr: np.ndarray, ker: np.ndarray, axis: int) -> np.ndarray:
    pad = len(ker) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="valid"), axis, padded)


def smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return grid
    ker = gaussian_kernel1d(sigma)
    out = convolve1d_reflect(grid, ker, axis=0)
    out = convolve1d_reflect(out, ker, axis=1)
    out = convolve1d_reflect(out, ker, axis=2)
    return out


def parse_label_tokens(tokens: list[str] | None) -> list[str] | None:
    if not tokens:
        return None
    labels: list[str] = []
    for token in tokens:
        labels.extend(part.strip() for part in token.split(",") if part.strip())
    return labels or None


def infer_labels_from_input(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or not line.startswith("species "):
            continue
        toks = line.split()
        if len(toks) >= 3:
            return toks[2:]
    return None


def sanitize_name(name: str) -> str:
    safe = name.strip()
    safe = safe.replace("+", "p").replace("-", "m")
    safe = safe.replace("[", "_").replace("]", "")
    safe = safe.replace("*", "star")
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "value"
    if safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


def unique_names(names: list[str]) -> list[str]:
    used: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = used.get(name, 0)
        used[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count+1}")
    return out


def resolve_component_names(
    value_names: list[str],
    explicit_labels: list[str] | None,
    input_deck: Path | None,
) -> tuple[list[str], str]:
    if explicit_labels is not None:
        if len(explicit_labels) != len(value_names):
            raise RuntimeError(
                f"--component-labels provided {len(explicit_labels)} labels for "
                f"{len(value_names)} value columns"
            )
        source = "explicit --component-labels"
        labels = explicit_labels
    else:
        inferred = infer_labels_from_input(input_deck) if input_deck is not None else None
        if inferred is not None and len(inferred) == len(value_names):
            source = f"species line in {input_deck}"
            labels = inferred
        else:
            source = "dump column names"
            labels = value_names

    safe = unique_names([sanitize_name(label) for label in labels])
    return safe, source


def vtk_array(values: np.ndarray, name: str, array_type=None):
    arr = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(values),
        deep=True,
        array_type=array_type,
    )
    arr.SetName(name)
    return arr


def main():
    here = Path(__file__).resolve().parent.parent
    default_dump = here / "output" / "tmp.grid.density"
    default_out = here / "output" / "grid_density_3d.west.vtu"
    default_input = here / "in.west_3d"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(default_dump), help="grid dump file")
    ap.add_argument("--out", default=str(default_out), help="output VTU file")
    ap.add_argument("--timestep", default="last", help="last or explicit integer")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing sigma in grid cells")
    ap.add_argument("--threshold", type=float, default=1.0e10, help="keep only values above this threshold")
    ap.add_argument(
        "--component-labels",
        nargs="*",
        help="Optional labels for value columns, space- or comma-separated",
    )
    ap.add_argument(
        "--input-deck",
        default=str(default_input),
        help="SPARTA input deck used to infer species/component labels; use '' to disable",
    )
    args = ap.parse_args()

    blocks = parse_grid_dump(Path(args.dump))
    keys = sorted(blocks)
    ts = keys[-1] if args.timestep == "last" else int(args.timestep)
    header, cols = blocks[ts]
    name_to_col = {name: arr for name, arr in zip(header, cols)}

    for req in ("xc", "yc", "zc"):
        if req not in name_to_col:
            raise RuntimeError(f"Missing required column '{req}'")

    value_names = [name for name in header if name not in {"id", "xc", "yc", "zc"}]
    if not value_names:
        raise RuntimeError("No density/value columns found in dump")

    explicit_labels = parse_label_tokens(args.component_labels)
    input_deck = Path(args.input_deck) if args.input_deck else None
    component_names, label_source = resolve_component_names(
        value_names, explicit_labels, input_deck
    )

    x = name_to_col["xc"]
    y = name_to_col["yc"]
    z = name_to_col["zc"]

    # Build regular grid, smooth if requested, then emit only cells above threshold.
    xu = np.sort(np.unique(np.round(x, 10)))
    yu = np.sort(np.unique(np.round(y, 10)))
    zu = np.sort(np.unique(np.round(z, 10)))
    nx, ny, nz = len(xu), len(yu), len(zu)

    ix = np.searchsorted(xu, np.round(x, 10))
    iy = np.searchsorted(yu, np.round(y, 10))
    iz = np.searchsorted(zu, np.round(z, 10))

    component_grids = []
    for name in value_names:
        grid = np.zeros((nz, ny, nx), dtype=float)
        np.add.at(grid, (iz, iy, ix), name_to_col[name])
        if args.smooth > 0.0:
            grid = smooth_grid(grid, args.smooth)
        component_grids.append(grid)
    component_grids = np.stack(component_grids, axis=0)

    total_grid = np.sum(component_grids, axis=0)
    dominant_component = np.argmax(component_grids, axis=0).astype(np.int32) + 1
    dominant_density = np.max(component_grids, axis=0)
    dominant_fraction = np.zeros_like(total_grid)
    positive = total_grid > 0.0
    dominant_fraction[positive] = dominant_density[positive] / total_grid[positive]
    dominant_component[~positive] = 0

    mask = total_grid > args.threshold
    kept = int(np.count_nonzero(mask))
    if kept == 0:
        raise RuntimeError("No cells remain above threshold")

    kk, jj, ii = np.where(mask)
    coords = np.column_stack((xu[ii], yu[jj], zu[kk])).astype(np.float64, copy=False)

    points = vtk.vtkPoints()
    points.SetData(vtk_array(coords, "Points"))
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    for pid in range(kept):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, pid)
        ugrid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    total_vals = total_grid[mask].astype(np.float64, copy=False)
    log_total = np.where(total_vals > 0.0, np.log10(total_vals), np.nan)
    ugrid.GetCellData().AddArray(vtk_array(total_vals, "impurity_density"))
    ugrid.GetCellData().AddArray(vtk_array(log_total, "log10_impurity_density"))

    for comp_idx, comp_name in enumerate(component_names):
        comp_vals = component_grids[comp_idx][mask].astype(np.float64, copy=False)
        log_comp = np.where(comp_vals > 0.0, np.log10(comp_vals), np.nan)
        ugrid.GetCellData().AddArray(vtk_array(comp_vals, f"impurity_density_{comp_name}"))
        ugrid.GetCellData().AddArray(vtk_array(log_comp, f"log10_impurity_density_{comp_name}"))

    ugrid.GetCellData().AddArray(
        vtk_array(dominant_component[mask].astype(np.int32, copy=False), "dominant_component_index", vtk.VTK_INT)
    )
    ugrid.GetCellData().AddArray(
        vtk_array(dominant_fraction[mask].astype(np.float64, copy=False), "dominant_component_fraction")
    )
    ugrid.GetCellData().SetActiveScalars("impurity_density")

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(args.out))
    writer.SetInputData(ugrid)
    writer.Write()

    print(f"Wrote {args.out}")
    print(f"  timestep: {ts}")
    print(f"  value columns: {value_names}")
    print(f"  component labels: {component_names}")
    print(f"  label source: {label_source}")
    print(f"  smoothing sigma: {args.smooth}")
    print(f"  threshold: {args.threshold}")
    print(f"  kept points: {kept}")
    print("  arrays: ['impurity_density', 'log10_impurity_density', 'impurity_density_<component>', 'log10_impurity_density_<component>', 'dominant_component_index', 'dominant_component_fraction']")


if __name__ == "__main__":
    main()
