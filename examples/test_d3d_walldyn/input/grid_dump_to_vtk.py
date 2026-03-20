#!/usr/bin/env python3
"""Convert SPARTA grid dump to VTK for ParaView.

Reads a SPARTA grid dump file (id xc yc zc field1 field2 ...) and writes
a VTK polydata (.vtp) with point data for each field.

Usage:
    python grid_dump_to_vtk.py output/plasma_grid [--outdir vtk]
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


def snapshot_to_vtk(timestep, col_names, data, outdir, filestem):
    """Write one snapshot as .vtp."""
    npts = len(data)
    if npts == 0:
        return

    # Find coordinate columns
    ix = col_names.index('xc') if 'xc' in col_names else 1
    iy = col_names.index('yc') if 'yc' in col_names else 2
    iz = col_names.index('zc') if 'zc' in col_names else 3

    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    for ipt in range(npts):
        pid = points.InsertNextPoint(data[ipt, ix], data[ipt, iy], data[ipt, iz])
        cells.InsertNextCell(1)
        cells.InsertCellPoint(pid)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(cells)

    # Add all non-coordinate columns as point data
    skip = {'id', 'xc', 'yc', 'zc'}
    for icol, name in enumerate(col_names):
        if name in skip:
            continue
        arr = vtk.vtkDoubleArray()
        # Clean up name for ParaView
        clean_name = name.replace('f_fpfields[', 'field_').rstrip(']')
        arr.SetName(clean_name)
        arr.SetNumberOfTuples(npts)
        for ipt in range(npts):
            arr.SetValue(ipt, data[ipt, icol])
        poly.GetPointData().AddArray(arr)

    # Set first field as active scalar
    field_names = [n for n in col_names if n not in skip]
    if field_names:
        clean = field_names[0].replace('f_fpfields[', 'field_').rstrip(']')
        poly.GetPointData().SetActiveScalars(clean)

    outfile = os.path.join(outdir, f"{filestem}_t{timestep:06d}.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    writer.SetInputData(poly)
    writer.Write()
    print(f"Wrote {outfile} ({npts} points, fields: {field_names})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="SPARTA grid dump file")
    parser.add_argument("--outdir", default="vtk", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    filestem = os.path.splitext(os.path.basename(args.input))[0]

    snapshots = parse_grid_dump(args.input)
    for ts, cols, data in snapshots:
        snapshot_to_vtk(ts, cols, data, args.outdir, filestem)


if __name__ == "__main__":
    main()
