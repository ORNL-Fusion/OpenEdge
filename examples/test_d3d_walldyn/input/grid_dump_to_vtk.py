#!/usr/bin/env python3
"""Convert SPARTA grid dump to VTK for ParaView.

Reads a SPARTA grid dump file (id xc yc zc field1 field2 ...) and writes
a VTK file with point/cell data for each field.

Supports two modes:
  --mode points  : VTK polydata (.vtp) with one point per cell (fast)
  --mode volume  : VTK structured grid (.vts) interpolated onto regular grid (smooth)

Use --threshold to filter out cells outside the plasma domain
(cells with field value below threshold are excluded in points mode,
or set to NaN in volume mode).

Usage:
    python grid_dump_to_vtk.py output/plasma_grid [--outdir vtk] [--threshold 1.0] [--mode volume]
"""

import os
import sys
import argparse
import numpy as np
import vtk


def parse_grid_dump(filename):
    """Parse SPARTA grid dump file. Returns dict of {field_name: array} plus xc,yc,zc."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    snapshots = []

    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
        elif lines[i].strip().startswith("ITEM: NUMBER OF"):
            natoms = int(lines[i + 1].strip())
            i += 2
        elif lines[i].strip().startswith("ITEM: BOX BOUNDS"):
            i += 4  # skip 3 lines of box bounds + header line
        elif lines[i].strip().startswith("ITEM: CELLS") or lines[i].strip().startswith("ITEM: ATOMS"):
            header = lines[i].strip().split()
            # Skip "ITEM:" and "CELLS"/"ATOMS"
            col_names = header[2:]
            i += 1

            data = []
            for _ in range(natoms):
                vals = lines[i].strip().split()
                data.append([float(v) for v in vals])
                i += 1

            data = np.array(data)
            snapshots.append((timestep, col_names, data))
        else:
            i += 1

    return snapshots


def snapshot_to_vtk(timestep, col_names, data, outdir, filestem,
                    mode="points", threshold=None):
    """Write one snapshot as VTK."""
    npts = len(data)
    if npts == 0:
        return

    ix = col_names.index('xc') if 'xc' in col_names else 1
    iy = col_names.index('yc') if 'yc' in col_names else 2
    iz = col_names.index('zc') if 'zc' in col_names else 3

    skip = {'id', 'xc', 'yc', 'zc'}
    field_cols = [(i, n) for i, n in enumerate(col_names) if n not in skip]

    if mode == "volume":
        _write_structured(timestep, col_names, data, outdir, filestem,
                          ix, iy, iz, field_cols, threshold)
    else:
        _write_points(timestep, col_names, data, outdir, filestem,
                      ix, iy, iz, field_cols, threshold)


def _write_points(timestep, col_names, data, outdir, filestem,
                  ix, iy, iz, field_cols, threshold):
    """Write filtered points as .vtp."""
    # Filter by threshold on first field
    if threshold is not None and field_cols:
        first_col = field_cols[0][0]
        mask = data[:, first_col] > threshold
        data = data[mask]

    npts = len(data)
    if npts == 0:
        return

    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    for ipt in range(npts):
        pid = points.InsertNextPoint(data[ipt, ix], data[ipt, iy], data[ipt, iz])
        cells.InsertNextCell(1)
        cells.InsertCellPoint(pid)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(cells)

    for icol, name in field_cols:
        arr = vtk.vtkDoubleArray()
        clean_name = name.replace('f_fpfields[', 'field_').rstrip(']')
        arr.SetName(clean_name)
        arr.SetNumberOfTuples(npts)
        for ipt in range(npts):
            arr.SetValue(ipt, data[ipt, icol])
        poly.GetPointData().AddArray(arr)

    if field_cols:
        clean = field_cols[0][1].replace('f_fpfields[', 'field_').rstrip(']')
        poly.GetPointData().SetActiveScalars(clean)

    outfile = os.path.join(outdir, f"{filestem}_t{timestep:06d}.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(poly)
    writer.Write()
    print(f"Wrote {outfile} ({npts} points)")


def _write_structured(timestep, col_names, data, outdir, filestem,
                      ix, iy, iz, field_cols, threshold):
    """Interpolate scattered cell centers onto a regular grid and write .vts."""
    from scipy.interpolate import griddata

    x = data[:, ix]
    y = data[:, iy]
    z = data[:, iz]

    # Determine grid dimensions from unique coordinates
    xu = np.sort(np.unique(np.round(x, 8)))
    yu = np.sort(np.unique(np.round(y, 8)))
    zu = np.sort(np.unique(np.round(z, 8)))
    nx, ny, nz = len(xu), len(yu), len(zu)

    print(f"  Structured grid: {nx} x {ny} x {nz}")

    # Build structured grid
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nx, ny, nz)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(nx * ny * nz)

    # VTK structured grid ordering: x varies fastest
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                pts.SetPoint(idx, xu[i], yu[j], zu[k])
                idx += 1
    sgrid.SetPoints(pts)

    # Map scattered data to structured grid
    # Build lookup: (rounded x,y,z) -> data row index
    coord_to_idx = {}
    for ipt in range(len(data)):
        key = (round(x[ipt], 8), round(y[ipt], 8), round(z[ipt], 8))
        coord_to_idx[key] = ipt

    for icol, name in field_cols:
        arr = vtk.vtkDoubleArray()
        clean_name = name.replace('f_fpfields[', 'field_').rstrip(']')
        arr.SetName(clean_name)
        arr.SetNumberOfTuples(nx * ny * nz)

        idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    key = (xu[i], yu[j], zu[k])
                    if key in coord_to_idx:
                        val = data[coord_to_idx[key], icol]
                        if threshold is not None and val <= threshold:
                            arr.SetValue(idx, float('nan'))
                        else:
                            arr.SetValue(idx, val)
                    else:
                        arr.SetValue(idx, float('nan'))
                    idx += 1
        sgrid.GetPointData().AddArray(arr)

    if field_cols:
        clean = field_cols[0][1].replace('f_fpfields[', 'field_').rstrip(']')
        sgrid.GetPointData().SetActiveScalars(clean)

    outfile = os.path.join(outdir, f"{filestem}_t{timestep:06d}.vts")
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(sgrid)
    writer.Write()
    print(f"Wrote {outfile} ({nx}x{ny}x{nz} = {nx*ny*nz} cells)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="SPARTA grid dump file")
    parser.add_argument("--outdir", default="vtk", help="Output directory")
    parser.add_argument("--mode", choices=["points", "volume"], default="points",
                        help="Output mode: points (.vtp) or volume (.vts)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Filter: exclude cells with first field <= threshold")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    filestem = os.path.splitext(os.path.basename(args.input))[0]

    snapshots = parse_grid_dump(args.input)
    for ts, cols, data in snapshots:
        snapshot_to_vtk(ts, cols, data, args.outdir, filestem,
                        mode=args.mode, threshold=args.threshold)


if __name__ == "__main__":
    main()
