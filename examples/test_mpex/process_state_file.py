## -*- coding: utf-8 -*-
#"""
#Routines to pre- and post-process OpenEdge simulations.
#
#Created on Fri Nov 22 13:59:59 2024
#
#@author: 1jn
#"""
#
#import os
#import vtk
#import numpy as np
#
#import os
#import numpy as np
#import vtk
#
#import os
#import numpy as np
#import vtk
#
#
#def parse_file(filename):
#    timesteps, x_coords, y_coords, z_coords = [], [], [], []
#    vx_coords, vy_coords, vz_coords = [], [], []
#    mass, temp, radius, ids = [], [], [], []
#
#    with open(filename, 'r') as f:
#        lines = f.readlines()
#    i = 0
#    while i < len(lines):
#        line = lines[i].strip()
#        if line == "ITEM: TIMESTEP":
#            timestep = int(lines[i+1].strip()); i += 2
#        elif line == "ITEM: NUMBER OF ATOMS":
#            num_atoms = int(lines[i+1].strip()); i += 2
#        elif line.startswith("ITEM: ATOMS"):
#            # Expect: id type x y z vx vy vz v_pmass temp radius
#            for _ in range(num_atoms):
#                atom_data = lines[i+1].strip().split()
#                # align timestep per row
#                timesteps.append(timestep)
#                ids.append(int(atom_data[1]))
#                x_coords.append(float(atom_data[2]))
#                y_coords.append(float(atom_data[3]))
#                z_coords.append(float(atom_data[4]))
#                vx_coords.append(float(atom_data[5]))
#                vy_coords.append(float(atom_data[6]))
#                vz_coords.append(float(atom_data[7]))
#                i += 1
#            i += 1
#        else:
#            i += 1
#
#    return (np.asarray(timesteps, float),
#            np.asarray(x_coords, float),
#            np.asarray(y_coords, float),
#            np.asarray(z_coords, float),
#            np.asarray(vx_coords, float),
#            np.asarray(vy_coords, float),
#            np.asarray(vz_coords, float),
#            np.asarray(ids, int))
#            
#            
#def state_to_vtk(statefile):
#    # Convert a state file to VTK polydata file (.vtp) for Paraview.
#
#    # Parse path names
#    filehead, filetail = os.path.split(statefile)
#    filestem, fileext = os.path.splitext(filetail)
#    newfile = os.path.join(filehead, filestem + ".vtp")
#    
#    # Parse the particle data
#    idz, x, y, z,vx,vy,vz,idz  = [], [], [], []
#    ts,x,y,z,vx,vy,vz,idz = parse_file(statefile)
#
#    npts = len(idz)
#    
#    # Initialize vtk arrays
#    vtkPts = vtk.vtkPoints()
#    cells = vtk.vtkCellArray()
#    vtkData1 = vtk.vtkFloatArray()
#    vtkData1.SetNumberOfTuples(npts)
#    vtkData1.SetName("id")
#    
#    # Fill vtk arrays
#    for ipt in range(npts):
#        xi = x[ipt]
#        yi = y[ipt]
#        zi = z[ipt]
#        idzi = idz[ipt]
#        id = vtkPts.InsertNextPoint(xi, yi, zi)
#        cells.InsertNextCell(1)
#        cells.InsertCellPoint(id)
#        vtkData1.InsertTuple(ipt, [idzi])
#    
#    polydata = vtk.vtkPolyData()
#    polydata.SetPoints(vtkPts)
#    polydata.SetVerts(cells)
#    polydata.GetCellData().AddArray(vtkData1)
#    polydata.Modified()
#    
#    writer = vtk.vtkXMLPolyDataWriter()
#    writer.SetFileName(newfile)
#    writer.SetInputData(polydata)
#    writer.Write()
#    
#    print("Created", newfile)
#    return
#
#
#
#statefile ="output/state"
#state_to_vtk(statefile)
#


# -*- coding: utf-8 -*-
"""
Routines to pre- and post-process OpenEdge/SPARTA/LAMMPS particle dumps.

Created on Fri Nov 22 13:59:59 2024

@author: 1jn, updated by Abdou + chatgpt
"""

import os
import numpy as np
import vtk


def parse_file(filename):
    """
    Parse a LAMMPS/SPARTA-style dump with blocks like:

      ITEM: TIMESTEP
      <t>
      ITEM: NUMBER OF ATOMS
      <n>
      ITEM: ATOMS id type x y z vx vy vz v_pmass temp radius

    Returns arrays with one entry per *row* in the file
    (so the same particle appears multiple times across timesteps).
    """
    timesteps, x_coords, y_coords, z_coords = [], [], [], []
    vx_coords, vy_coords, vz_coords = [], [], []
    ids = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    timestep = None
    num_atoms = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2

        elif line == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i + 1].strip())
            i += 2

        elif line.startswith("ITEM: ATOMS"):
            # Robust: figure out which column is which from the header:
            # e.g. "ITEM: ATOMS id type x y z vx vy vz v_pmass temp radius"
            header = line.split()[2:]  # skip "ITEM: ATOMS"
            col_index = {name: j for j, name in enumerate(header)}

            for j in range(num_atoms):
                atom_data = lines[i + 1 + j].strip().split()

                timesteps.append(timestep)
                ids.append(int(atom_data[col_index["id"]]))
                x_coords.append(float(atom_data[col_index["x"]]))
                y_coords.append(float(atom_data[col_index["y"]]))
                z_coords.append(float(atom_data[col_index["z"]]))

                # Optional velocity columns if present
                if "vx" in col_index:
                    vx_coords.append(float(atom_data[col_index["vx"]]))
                    vy_coords.append(float(atom_data[col_index["vy"]]))
                    vz_coords.append(float(atom_data[col_index["vz"]]))
                else:
                    vx_coords.append(0.0)
                    vy_coords.append(0.0)
                    vz_coords.append(0.0)

            i += 1 + num_atoms

        else:
            i += 1

    return (
        np.asarray(timesteps, dtype=int),
        np.asarray(x_coords, dtype=float),
        np.asarray(y_coords, dtype=float),
        np.asarray(z_coords, dtype=float),
        np.asarray(vx_coords, dtype=float),
        np.asarray(vy_coords, dtype=float),
        np.asarray(vz_coords, dtype=float),
        np.asarray(ids, dtype=int),
    )


def state_to_vtk(statefile):
    """
    Convert a state file to VTK polydata (.vtp) for ParaView.

    - Each row in the dump -> one VTK point.
    - PointData arrays:
        * "id"       : particle ID
        * "timestep" : timestep associated with that sample
    """
    # Parse path names
    filehead, filetail = os.path.split(statefile)
    filestem, _ = os.path.splitext(filetail)
    newfile = os.path.join(filehead, filestem + ".vtp")

    # Parse particle data
    ts, x, y, z, vx, vy, vz, ids = parse_file(statefile)
    npts = len(ids)

    # --- VTK containers ---
    vtkPts = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Particle ID as point data
    vtk_id = vtk.vtkIntArray()
    vtk_id.SetName("id")
    vtk_id.SetNumberOfTuples(npts)

    # Timestep as point data
    vtk_ts = vtk.vtkIntArray()
    vtk_ts.SetName("timestep")
    vtk_ts.SetNumberOfTuples(npts)

    # (Optionally you can also store vx,vy,vz as arrays if you want)

    # --- Fill VTK arrays ---
    for ipt in range(npts):
        pid = ids[ipt]
        ti = int(ts[ipt])

        xi = x[ipt]
        yi = y[ipt]
        zi = z[ipt]

        # Insert as a point + a vertex cell
        p_id = vtkPts.InsertNextPoint(xi, yi, zi)
        cells.InsertNextCell(1)
        cells.InsertCellPoint(p_id)

        vtk_id.SetValue(ipt, pid)
        vtk_ts.SetValue(ipt, ti)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtkPts)
    polydata.SetVerts(cells)

    # Attach as POINT data (not cell data)
    polydata.GetPointData().AddArray(vtk_id)
    polydata.GetPointData().AddArray(vtk_ts)
    polydata.GetPointData().SetActiveScalars("id")

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(newfile)
    writer.SetInputData(polydata)
    writer.Write()

    print("Created", newfile)


import os
import numpy as np
import vtk


def parse_file(filename):
    timesteps, x_coords, y_coords, z_coords = [], [], [], []
    vx_coords, vy_coords, vz_coords = [], [], []
    ids = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    timestep = None
    num_atoms = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2

        elif line == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i + 1].strip())
            i += 2

        elif line.startswith("ITEM: ATOMS"):
            header = line.split()[2:]  # after "ITEM: ATOMS"
            col_index = {name: j for j, name in enumerate(header)}

            for j in range(num_atoms):
                atom_data = lines[i + 1 + j].strip().split()

                timesteps.append(timestep)
                ids.append(int(atom_data[col_index["id"]]))
                x_coords.append(float(atom_data[col_index["x"]]))
                y_coords.append(float(atom_data[col_index["y"]]))
                z_coords.append(float(atom_data[col_index["z"]]))

                if "vx" in col_index:
                    vx_coords.append(float(atom_data[col_index["vx"]]))
                    vy_coords.append(float(atom_data[col_index["vy"]]))
                    vz_coords.append(float(atom_data[col_index["vz"]]))
                else:
                    vx_coords.append(0.0)
                    vy_coords.append(0.0)
                    vz_coords.append(0.0)

            i += 1 + num_atoms

        else:
            i += 1

    return (
        np.asarray(timesteps, dtype=int),
        np.asarray(x_coords, dtype=float),
        np.asarray(y_coords, dtype=float),
        np.asarray(z_coords, dtype=float),
        np.asarray(vx_coords, dtype=float),
        np.asarray(vy_coords, dtype=float),
        np.asarray(vz_coords, dtype=float),
        np.asarray(ids, dtype=int),
    )


def state_to_vtk_series(statefile, outdir=None):
    """
    Write one .vtp per timestep:
      state_t000000.vtp, state_t000010.vtp, ...
    """

    # parse dump
    ts, x, y, z, vx, vy, vz, ids = parse_file(statefile)
    unique_ts = np.unique(ts)

    # output dir
    if outdir is None:
        outdir = os.path.dirname(statefile) or "."
    os.makedirs(outdir, exist_ok=True)

    filestem = os.path.splitext(os.path.basename(statefile))[0]

    for t in unique_ts:
        mask = (ts == t)
        npts = int(mask.sum())
        if npts == 0:
            continue

        x_t  = x[mask]
        y_t  = y[mask]
        z_t  = z[mask]
        id_t = ids[mask]

        # --- VTK containers ---
        vtkPts = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        arr_id = vtk.vtkIntArray()
        arr_id.SetName("id")
        arr_id.SetNumberOfTuples(npts)

        arr_ts = vtk.vtkIntArray()
        arr_ts.SetName("timestep")
        arr_ts.SetNumberOfTuples(npts)

        for i_pt in range(npts):
            pid = id_t[i_pt]
            xi, yi, zi = float(x_t[i_pt]), float(y_t[i_pt]), float(z_t[i_pt])

            p_id = vtkPts.InsertNextPoint(xi, yi, zi)
            cells.InsertNextCell(1)
            cells.InsertCellPoint(p_id)

            arr_id.SetValue(i_pt, int(pid))
            arr_ts.SetValue(i_pt, int(t))

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtkPts)
        poly.SetVerts(cells)
        poly.GetPointData().AddArray(arr_id)
        poly.GetPointData().AddArray(arr_ts)
        poly.GetPointData().SetActiveScalars("id")

        poly.Modified()

        outfile = os.path.join(outdir, f"{filestem}_t{t:06d}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(outfile)
        writer.SetInputData(poly)
        writer.Write()

        print("Wrote", outfile)


if __name__ == "__main__":
    statefile = "state"
    statefile = "output/state"
    state_to_vtk_series(statefile)


#if __name__ == "__main__":
#    statefile = "output/state"
#    state_to_vtk(statefile)
