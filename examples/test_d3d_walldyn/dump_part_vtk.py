

# -*- coding: utf-8 -*-
"""
Routines to pre- and post-process OpenEdge/SPARTA/LAMMPS particle dumps.

Created on Fri Nov 22 13:59:59 2024

@author: 1jn, updated by Abdou + chatgpt
"""

import os
import numpy as np
import vtk


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


  

def parse_file(filename):
    timesteps = []
    x_coords = []
    y_coords = []
    z_coords = []
    vx_coords = []
    vy_coords = []
    vz_coords = []
    ids = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "ITEM: TIMESTEP":
                timestep = int(lines[i + 1].strip())
                i += 2  # Move to next line after timestep

            elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
                num_atoms = int(lines[i + 1].strip())
                i += 2  # Move to the line after the number of atoms

            elif lines[i].strip().startswith("ITEM: ATOMS"):
                if num_atoms > 0:
                    for _ in range(num_atoms):
                        atom_data = lines[i + 1].strip().split()

                        timesteps.append(timestep)   # one timestep per particle
                        ids.append(int(atom_data[0]))  # id is column 0, not 1
                        x_coords.append(float(atom_data[2]))
                        y_coords.append(float(atom_data[3]))
                        z_coords.append(float(atom_data[4]))
                        vx_coords.append(float(atom_data[5]))
                        vy_coords.append(float(atom_data[6]))
                        vz_coords.append(float(atom_data[7]))

                        i += 1
                i += 1
            else:
                i += 1  # Move to next line if no match

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, ids

        
def state_to_vtk_series(statefile, outdir=None):
    """
    Write one .vtp per timestep:
      state_t000000.vtp, state_t000010.vtp, ...
    """

    # parse dump
    tstep, x, y, z, vx, vy, vz, ids = parse_file(statefile)

    ts   = np.asarray(tstep, dtype=int) # * float(dt)
    x   = np.asarray(x, dtype=float)
    y   = np.asarray(y, dtype=float)
    z   = np.asarray(z, dtype=float)
    vx   = np.asarray(vx, dtype=float)
    vy   = np.asarray(vy, dtype=float)
    vz   = np.asarray(vz, dtype=float)
    ids = np.asarray(ids)
        
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


#if __name__ == "__main__":
#    statefile ="source"
#    state_to_vtk_series(statefile)


if __name__ == "__main__":
    statefile = "output/particles"
    outdir = "vtk"
    state_to_vtk_series(statefile, outdir=outdir)
