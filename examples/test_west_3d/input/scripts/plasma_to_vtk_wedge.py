#!/usr/bin/env python3
"""Convert OpenEdge plasma.h5 (uniform R,Z grid) to a 3D VTK wedge for ParaView.

Revolves the 2D (R,Z) grid around the Z-axis to create a 3D wedge.
Works with plasma.h5 files produced by convert_s3x_plasma.py or
soledge_to_openedge.py (flat r, z, dens_e, temp_e, ... datasets).

Usage:
    python3 plasma_to_vtk_wedge.py --input input/data/plasma.h5 --phi-min 0 --phi-max 30
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to plasma.h5")
    p.add_argument("--bfield", default=None, help="Path to bfield.h5 (optional)")
    p.add_argument("--phi-min", type=float, default=0.0, help="Wedge start angle [deg]")
    p.add_argument("--phi-max", type=float, default=30.0, help="Wedge end angle [deg]")
    p.add_argument("--nphi", type=int, default=1, help="Number of toroidal slices")
    p.add_argument("--output", default=None, help="Output .vtu file")
    p.add_argument("--threshold", type=float, default=0.0,
                    help="Skip cells where dens_e < threshold")
    args = p.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_wedge.vtu")

    # Read plasma data
    with h5py.File(args.input, "r") as f:
        r = f["r"][:]
        z = f["z"][:]

        # Detect format: (nz, nr) or (nr, nz)
        # Convention: dims[0]=nz, dims[1]=nr
        fields_2d = {}
        field_names = ["dens_e", "temp_e", "dens_i", "temp_i", "parr_flow",
                       "ne", "te", "ni", "ti"]
        for name in field_names:
            if name in f:
                fields_2d[name] = f[name][:]

        # Handle alternate naming
        if "ne" in fields_2d and "dens_e" not in fields_2d:
            fields_2d["dens_e"] = fields_2d.pop("ne")
        if "te" in fields_2d and "temp_e" not in fields_2d:
            fields_2d["temp_e"] = fields_2d.pop("te")
        if "ni" in fields_2d and "dens_i" not in fields_2d:
            fields_2d["dens_i"] = fields_2d.pop("ni")
        if "ti" in fields_2d and "temp_i" not in fields_2d:
            fields_2d["temp_i"] = fields_2d.pop("ti")

    # Read B-field if available
    bfield_path = args.bfield or args.input.replace("plasma.h5", "bfield.h5")
    try:
        with h5py.File(bfield_path, "r") as bf:
            if "br" in bf:
                fields_2d["Br"] = bf["br"][:]
                fields_2d["Bt"] = bf["bt"][:]
                fields_2d["Bz"] = bf["bz"][:]
                # Compute |B|
                fields_2d["Bmag"] = np.sqrt(
                    fields_2d["Br"]**2 + fields_2d["Bt"]**2 + fields_2d["Bz"]**2)
    except Exception:
        pass

    nr = len(r)
    nz = len(z)

    # Determine array orientation
    for k, v in fields_2d.items():
        if v.shape == (nr, nz):
            fields_2d[k] = v.T  # transpose to (nz, nr)
        elif v.shape != (nz, nr):
            print(f"Warning: {k} shape {v.shape} doesn't match ({nz},{nr}) or ({nr},{nz}), skipping")
            del fields_2d[k]

    # Build VTK structured grid as wedge
    phi_min = np.radians(args.phi_min)
    phi_max = np.radians(args.phi_max)
    nphi = args.nphi
    phi_edges = np.linspace(phi_min, phi_max, nphi + 1)

    # VTK points: (nr+1) * (nz+1) * (nphi+1) for cell-centered data
    # Use cell edges
    dr = r[1] - r[0] if nr > 1 else 1.0
    dz = z[1] - z[0] if nz > 1 else 1.0
    r_edges = np.concatenate([[r[0] - dr/2], (r[:-1] + r[1:])/2, [r[-1] + dr/2]])
    z_edges = np.concatenate([[z[0] - dz/2], (z[:-1] + z[1:])/2, [z[-1] + dz/2]])

    import vtk

    points = vtk.vtkPoints()
    nr_e = len(r_edges)
    nz_e = len(z_edges)
    nphi_e = nphi + 1

    for ip in range(nphi_e):
        phi = phi_edges[ip]
        cp, sp = np.cos(phi), np.sin(phi)
        for iz in range(nz_e):
            for ir in range(nr_e):
                x = r_edges[ir] * cp
                y = r_edges[ir] * sp
                zz = z_edges[iz]
                points.InsertNextPoint(x, y, zz)

    def pt_id(ir, iz, ip):
        return ip * (nz_e * nr_e) + iz * nr_e + ir

    # Build hexahedral cells
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    # Threshold mask
    ne_data = fields_2d.get("dens_e")
    threshold = args.threshold

    ncells = 0
    cell_indices = []  # (iz, ir, iphi) for each cell added
    for ip in range(nphi):
        for iz in range(nz):
            for ir in range(nr):
                if ne_data is not None and threshold > 0:
                    if ne_data[iz, ir] < threshold:
                        continue

                hex_cell = vtk.vtkHexahedron()
                hex_cell.GetPointIds().SetId(0, pt_id(ir, iz, ip))
                hex_cell.GetPointIds().SetId(1, pt_id(ir+1, iz, ip))
                hex_cell.GetPointIds().SetId(2, pt_id(ir+1, iz+1, ip))
                hex_cell.GetPointIds().SetId(3, pt_id(ir, iz+1, ip))
                hex_cell.GetPointIds().SetId(4, pt_id(ir, iz, ip+1))
                hex_cell.GetPointIds().SetId(5, pt_id(ir+1, iz, ip+1))
                hex_cell.GetPointIds().SetId(6, pt_id(ir+1, iz+1, ip+1))
                hex_cell.GetPointIds().SetId(7, pt_id(ir, iz+1, ip+1))
                ugrid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())
                cell_indices.append((iz, ir))
                ncells += 1

    # Add field data
    for name, data in fields_2d.items():
        arr = vtk.vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfTuples(ncells)
        for idx, (iz, ir) in enumerate(cell_indices):
            arr.SetValue(idx, float(data[iz, ir]))
        ugrid.GetCellData().AddArray(arr)

    # Add log10(dens_e) for visualization
    if "dens_e" in fields_2d:
        arr = vtk.vtkDoubleArray()
        arr.SetName("log10_ne")
        arr.SetNumberOfTuples(ncells)
        for idx, (iz, ir) in enumerate(cell_indices):
            val = fields_2d["dens_e"][iz, ir]
            arr.SetValue(idx, np.log10(max(val, 1.0)))
        ugrid.GetCellData().AddArray(arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(ugrid)
    writer.Write()

    print(f"Wrote {args.output}")
    print(f"  Cells: {ncells}, R=[{r[0]:.3f},{r[-1]:.3f}], Z=[{z[0]:.3f},{z[-1]:.3f}]")
    print(f"  Phi: [{args.phi_min:.1f}, {args.phi_max:.1f}] deg, {nphi} slices")
    print(f"  Fields: {list(fields_2d.keys())}")


if __name__ == "__main__":
    main()
