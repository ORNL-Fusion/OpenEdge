#!/usr/bin/env python3
"""Convert plasma.h5 SOLPS mesh data to a 3D VTK wedge for ParaView.

Revolves the 2D triangular mesh around the Z-axis to create a 3D wedge
(toroidal section). Outputs a VTK unstructured grid (.vtu) with plasma
fields as cell data.

Usage:
    python input/scripts/plasma_to_vtk_wedge.py [--input input/data/plasma.h5] [--phi-min 0] [--phi-max 30] [--nphi 1]
"""

import argparse

import h5py
import numpy as np
import vtk


def build_cell_array(name, data, tri_cell_index, nphi):
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    ntri = len(tri_cell_index)
    arr.SetNumberOfTuples(ntri * nphi)
    for iphi in range(nphi):
        for itri in range(ntri):
            cell = tri_cell_index[itri]
            idx = iphi * ntri + itri
            if 0 <= cell < len(data):
                arr.SetValue(idx, data[cell])
            else:
                arr.SetValue(idx, float("nan"))
    return arr


def add_log10_array(ugrid, source_arr_name, target_arr_name):
    src = ugrid.GetCellData().GetArray(source_arr_name)
    if src is None:
        return

    out = vtk.vtkDoubleArray()
    out.SetName(target_arr_name)
    nt = src.GetNumberOfTuples()
    out.SetNumberOfTuples(nt)
    for i in range(nt):
        val = src.GetValue(i)
        out.SetValue(i, np.log10(val) if val > 0.0 else float("nan"))
    ugrid.GetCellData().AddArray(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input/data/plasma.h5", help="Path to plasma.h5")
    parser.add_argument("--phi-min", type=float, default=0.0, help="Start angle (deg)")
    parser.add_argument("--phi-max", type=float, default=30.0, help="End angle (deg)")
    parser.add_argument("--nphi", type=int, default=1,
                        help="Number of toroidal slices (1 = single wedge prism layer)")
    parser.add_argument("--output", default="plasma_wedge.vtu", help="Output VTK file")
    parser.add_argument(
        "--include-unmapped",
        action="store_true",
        help="Keep EIRENE triangles with cell_index < 0; by default they are omitted",
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        vtx_r = f["mesh/vtx_r"][:]
        vtx_z = f["mesh/vtx_z"][:]
        tris = f["mesh/triangles"][:]  # (ntri, 3) vertex indices
        cell_index = f["mesh/cell_index"][:]

        # Cell data on the SOLPS mesh
        cell_data = {}
        for name in ["temp_e", "temp_i", "dens_e", "dens_i", "parr_flow"]:
            key = f"mesh/{name}"
            if key in f:
                cell_data[name] = f[key][:]

        # Ion data
        if "mesh/ions/dens" in f:
            cell_data["ion_dens"] = f["mesh/ions/dens"][0, :]
        if "mesh/ions/temp" in f:
            cell_data["ion_temp"] = f["mesh/ions/temp"][0, :]

    nv = len(vtx_r)

    if args.include_unmapped:
        tri_ids = np.arange(len(tris), dtype=np.int32)
    else:
        tri_ids = np.flatnonzero(cell_index >= 0)

    if len(tri_ids) == 0:
        raise RuntimeError("No mapped triangles found in plasma.h5")

    tris = tris[tri_ids]
    tri_cell_index = cell_index[tri_ids]
    ntri = len(tris)

    phi_edges = np.linspace(np.radians(args.phi_min), np.radians(args.phi_max), args.nphi + 1)

    # Build 3D points: for each phi edge, revolve all 2D vertices
    nv_total = nv * (args.nphi + 1)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nv_total)

    for iphi, phi in enumerate(phi_edges):
        cp, sp = np.cos(phi), np.sin(phi)
        for iv in range(nv):
            r, z = vtx_r[iv], vtx_z[iv]
            idx = iphi * nv + iv
            points.SetPoint(idx, r * cp, r * sp, z)

    # Build 3D cells: each 2D triangle x toroidal slice = wedge (VTK_WEDGE, 6 nodes)
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    ugrid.Allocate(ntri * args.nphi)

    for iphi in range(args.nphi):
        off0 = iphi * nv
        off1 = (iphi + 1) * nv
        for itri in range(ntri):
            v0, v1, v2 = tris[itri]
            # VTK_WEDGE: nodes [0-2] = bottom triangle, [3-5] = top triangle
            wedge = vtk.vtkIdList()
            wedge.SetNumberOfIds(6)
            wedge.SetId(0, off0 + v0)
            wedge.SetId(1, off0 + v1)
            wedge.SetId(2, off0 + v2)
            wedge.SetId(3, off1 + v0)
            wedge.SetId(4, off1 + v1)
            wedge.SetId(5, off1 + v2)
            ugrid.InsertNextCell(vtk.VTK_WEDGE, wedge)

    # Add cell data (replicated across toroidal slices)
    for name, data in cell_data.items():
        ugrid.GetCellData().AddArray(build_cell_array(name, data, tri_cell_index, args.nphi))

    cell_idx_arr = vtk.vtkIntArray()
    cell_idx_arr.SetName("cell_index")
    cell_idx_arr.SetNumberOfTuples(ntri * args.nphi)
    tri_id_arr = vtk.vtkIntArray()
    tri_id_arr.SetName("triangle_id")
    tri_id_arr.SetNumberOfTuples(ntri * args.nphi)
    for iphi in range(args.nphi):
        for itri, src_tri in enumerate(tri_ids):
            idx = iphi * ntri + itri
            cell_idx_arr.SetValue(idx, int(tri_cell_index[itri]))
            tri_id_arr.SetValue(idx, int(src_tri))
    ugrid.GetCellData().AddArray(cell_idx_arr)
    ugrid.GetCellData().AddArray(tri_id_arr)

    add_log10_array(ugrid, "dens_e", "log10_dens_e")
    add_log10_array(ugrid, "dens_i", "log10_dens_i")
    add_log10_array(ugrid, "ion_dens", "log10_ion_dens")

    if cell_data:
        first_name = next(iter(cell_data))
        ugrid.GetCellData().SetActiveScalars(first_name)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(ugrid)
    writer.Write()

    print(f"Wrote {args.output}")
    print(f"  Vertices: {nv_total} ({nv} x {args.nphi+1} phi layers)")
    print(f"  Cells: {ntri * args.nphi} wedges ({ntri} triangles x {args.nphi} slices)")
    print(f"  Source triangles kept: {ntri} / {len(cell_index)}")
    print(f"  Include unmapped: {args.include_unmapped}")
    print(f"  Fields: {list(cell_data.keys())}")
    print("  Derived fields: ['log10_dens_e', 'log10_dens_i', 'log10_ion_dens']")


if __name__ == "__main__":
    main()
