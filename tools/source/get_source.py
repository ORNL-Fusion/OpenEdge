# Copyright 2024, OpenEdge contributors
# Authors: Abdou Diaw
# License: GPL-2.0 license
"""
This script is part of OpenEdge, a particle transport code.

The script `interpolate_and_save_plasma_field` reads plasma and magnetic field data from SOLEDGE3X,
interpolates it to a specified grid, and converts it into an HDF5 format compatible with OpenEdge. 

The data includes electron and ion temperatures and densities, magnetic field components, 
and parallel gradients along magnetic field lines, among other plasma parameters.

Parameters:
- soledge_folder: Directory containing input SOLEDGE3X data files.
- openedge_plasma_file: Filename to store interpolated plasma data in OpenEdge format.
- openedge_bfield_file: Filename to store interpolated magnetic field data in OpenEdge format.
- nR, nZ: Number of grid points in radial (R) and axial (Z) directions for interpolation.

The function will create two HDF5 files (openedge_plasma_file and openedge_bfield_file) with 
interpolated plasma and magnetic field data that OpenEdge can use for simulations.
"""

import os
import h5py
from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path
from scipy.constants import k as kB, e
from scipy.interpolate import RegularGridInterpolator


import os, glob
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from utils import surface

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


import numpy as np
from pathlib import Path

def _sanitize_col(a):
    a = np.asarray(a, float)
    return np.nan_to_num(a, nan=0.0).reshape(-1, 1)

def _label_for(spec_idx, charge_by_spec):
    Z = charge_by_spec[spec_idx]
    return f"O{Z}+"

import numpy as np
from pathlib import Path

def _write_sputter_table(
    out_path,
    R, Z, Te, Ti,
    Flux_sput_i,        # dict: {spec_idx -> (Npts,)}
    total_sput,         # (Npts,)
    *,                  # force keywords for the rest
    species_list,
    charge_by_spec,     # dict mapping spec_idx -> charge (e.g., {2:1,3:2,...})
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- coerce & sanity ---
    R  = np.asarray(R,  float).ravel()
    Z  = np.asarray(Z,  float).ravel()
    Te = np.asarray(Te, float).ravel()
    Ti = np.asarray(Ti, float).ravel()
    total_sput = np.nan_to_num(np.asarray(total_sput, float).ravel(), nan=0.0)

    N = R.size
    if any(arr.size != N for arr in (Z, Te, Ti, total_sput)):
        raise ValueError("R, Z, Te, Ti, total_sput must all have the same length")

    def _col(a):
        return np.nan_to_num(np.asarray(a, float).ravel(), nan=0.0).reshape(N, 1)

    # --- header ---
    header = ["R_m", "Z_m", "Te_eV", "Ti_eV"]
    for s in species_list:
        Zchg = charge_by_spec[s]
        header.append(f"O{Zchg}+_sput")
    header.append("Total_sput")

    # --- matrix ---
    cols = [_col(R), _col(Z), _col(Te), _col(Ti)]
    for s in species_list:
        cols.append(_col(Flux_sput_i[s]))
    cols.append(_col(total_sput))
    M = np.hstack(cols)  # (N, ncols)

    # --- write ---
    with out_path.open("w") as f:
        f.write(" ".join(header) + "\n")
        np.savetxt(f, M, fmt="%.8e", delimiter=" ")

# ---------- small helpers ----------


def _tri_centroids_from_knots(Rk, Zk, tri_knots):
    pts = np.column_stack([Rk, Zk])  # (Nvert,2)
    # tri_knots can be (3,Ntri) or (Ntri,3); normalize to (Ntri,3)
    if tri_knots.ndim != 2 or 3 not in tri_knots.shape:
        raise ValueError(f"Bad tri_knots shape {tri_knots.shape}")
    tri = tri_knots.T if tri_knots.shape[0] == 3 else tri_knots
    # gather vertices for each triangle, shape -> (Ntri,3,2)
    tri_xyz = pts[tri]  # advanced indexing
    # centroids -> (Ntri,2)
    return tri_xyz.mean(axis=1)

def make_scalar_interp(points_2d, values_1d):
    """Piecewise-linear with nearest fallback. Accepts scalar or arrays of r,z."""
    lin  = LinearNDInterpolator(points_2d, values_1d, fill_value=np.nan)
    near = NearestNDInterpolator(points_2d, values_1d)
    def f(r, z):
        v = lin(r, z)
        m = ~np.isfinite(v)
        if np.any(m):
            v[m] = near(np.asarray(r)[m], np.asarray(z)[m])
        return v
    return f

def load_bca_table(path):
    with h5py.File(path, "r") as f:
        E = np.asarray(f["E"][:], float)
        A = np.asarray(f["A"][:], float)
        Y = np.asarray(f["spyld"][:], float)
        R = np.asarray(f["rfyld"][:], float)
    # sort axes
    ei, ai = np.argsort(E), np.argsort(A)
    E, A = E[ei], A[ai]
    Y, R = Y[np.ix_(ei, ai)], R[np.ix_(ei, ai)]
    return E, A, Y, R

def make_bca_interps(path, log_interp=True):
    E, A, Y, R = load_bca_table(path)
    if log_interp:
        eps = 1e-300
        Yf = RegularGridInterpolator((E, A), np.log(np.clip(Y, eps, None)),
                                     bounds_error=False, fill_value=np.nan)
        Rf = RegularGridInterpolator((E, A), np.log(np.clip(R, eps, None)),
                                     bounds_error=False, fill_value=np.nan)
        def yfunc(Eq, Aq):
            v = Yf(np.column_stack([Eq, Aq]))
            out = np.exp(v)
            # nearest fallback for NaNs
            bad = ~np.isfinite(out)
            if np.any(bad):
                Ei = np.argmin(np.abs(E[:,None] - Eq[bad]), axis=0)
                Ai = np.argmin(np.abs(A[:,None] - Aq[bad]), axis=0)
                out[bad] = Y[Ei, Ai]
            return out
        def rfunc(Eq, Aq):
            v = Rf(np.column_stack([Eq, Aq]))
            out = np.exp(v)
            bad = ~np.isfinite(out)
            if np.any(bad):
                Ei = np.argmin(np.abs(E[:,None] - Eq[bad]), axis=0)
                Ai = np.argmin(np.abs(A[:,None] - Aq[bad]), axis=0)
                out[bad] = R[Ei, Ai]
            return out
    else:
        Yf = RegularGridInterpolator((E, A), Y, bounds_error=False, fill_value=np.nan)
        Rf = RegularGridInterpolator((E, A), R, bounds_error=False, fill_value=np.nan)
        def yfunc(Eq, Aq):
            out = Yf(np.column_stack([Eq, Aq]))
            bad = ~np.isfinite(out)
            if np.any(bad):
                Ei = np.argmin(np.abs(E[:,None] - Eq[bad]), axis=0)
                Ai = np.argmin(np.abs(A[:,None] - Aq[bad]), axis=0)
                out[bad] = Y[Ei, Ai]
            return out
        def rfunc(Eq, Aq):
            out = Rf(np.column_stack([Eq, Aq]))
            bad = ~np.isfinite(out)
            if np.any(bad):
                Ei = np.argmin(np.abs(E[:,None] - Eq[bad]), axis=0)
                Ai = np.argmin(np.abs(A[:,None] - Aq[bad]), axis=0)
                out[bad] = R[Ei, Ai]
            return out
    return (E, A, yfunc, rfunc)

def _find_bca_file(incident_sym, target_sym, search_dirs):
    # your existing resolver; keep as-is. For clarity, here’s a trivial version:
    if isinstance(search_dirs, (list, tuple)):
        for d in search_dirs:
            p = os.path.join(d, f"{incident_sym}_on_{target_sym}.h5")
            if os.path.isfile(p): return p
    elif isinstance(search_dirs, str):
        if os.path.isfile(search_dirs):
            return search_dirs
        p = os.path.join(search_dirs, f"{incident_sym}_on_{target_sym}.h5")
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"Could not locate {incident_sym}_on_{target_sym}.h5 in {search_dirs}")



def compute_sputter_flux_on_W(R, Z, soledge_folder,
                              bca_dirs=None,
                              include_species=None,
                              theta_deg=0.0,
                              verbose=True, output_path=None):
    """
    Vectorized, robust version. Returns dict with per-species arrays and totals.
    """
    if include_species is None:
        include_species = list(range(2, 10))  # spec2..spec9 -> O+, ..., O8+

    # ---------- load scales ----------
    ref_file          = os.path.join(soledge_folder, 'refParam_raptorX.h5')
    mesh_eirene_file  = os.path.join(soledge_folder, 'meshEIRENE.h5')
    mesh_soledge_file = os.path.join(soledge_folder, 'mesh.h5')
    data_file         = os.path.join(soledge_folder, 'plasmaFinal.h5')

    with h5py.File(ref_file, 'r') as ref:
        n0  = float(ref['/n0'][...]);  T0 = float(ref['/T0'][...])
        c0  = float(ref['/c0'][...])

    # ---------- mesh & centroids ----------
    with h5py.File(mesh_eirene_file, 'r') as m:
        tri_knots = m['/triangles/tri_knots'][...].astype(int) - 1
        Rk = m['/knots/R'][...]/100.0
        Zk = m['/knots/Z'][...]/100.0
    centroids = _tri_centroids_from_knots(Rk, Zk, tri_knots)  # (Ntri,2)

    # ---------- plasma fields (triangle-centered) ----------
    with h5py.File(data_file, 'r') as d:
        temp_e = d['/triangles/spec0/T'][...].ravel() * T0
        temp_i = d['/triangles/spec1/T'][...].ravel() * T0
        # parallel fluxes Γ_i = n*c0*G (you had n0*c0 already)
        parr = {
            2: d['/triangles/spec2/G'][...].ravel() * c0 * n0,
            3: d['/triangles/spec3/G'][...].ravel() * c0 * n0,
            4: d['/triangles/spec4/G'][...].ravel() * c0 * n0,
            5: d['/triangles/spec5/G'][...].ravel() * c0 * n0,
            6: d['/triangles/spec6/G'][...].ravel() * c0 * n0,
            7: d['/triangles/spec7/G'][...].ravel() * c0 * n0,
            8: d['/triangles/spec8/G'][...].ravel() * c0 * n0,
            9: d['/triangles/spec9/G'][...].ravel() * c0 * n0,
        }

    # ---------- build field interpolators ----------
    f_Te = make_scalar_interp(centroids, temp_e)
    f_Ti = make_scalar_interp(centroids, temp_i)
    f_G  = {s: make_scalar_interp(centroids, parr[s]) for s in include_species}

    # evaluate Te, Ti, and Γ_i at the requested R,Z (vectorized)
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    Te = f_Te(R, Z)  # (Npts,)
    Ti = f_Ti(R, Z)
    Gi = {s: f_G[s](R, Z) for s in include_species}

    # ---------- BCA (build once) ----------
    # bca_path = _find_bca_file('O', 'W', bca_dirs)
    E_axis, A_axis, Y_fn, R_fn = make_bca_interps("/Users/42d/OpenEdge/tools/surface_data/data/O_on_W.h5", log_interp=True)

    # clamp angle to table range and broadcast
    theta = np.asarray(theta_deg, float)
    if theta.ndim == 0:
        theta = np.full_like(Te, theta)
    theta = np.clip(theta, A_axis.min(), A_axis.max())

    # ---------- incident energy model ----------
    # E_inc = Z_i * Ti + 3 * Te   (change to 3*Te + 5*Z*Ti if you prefer)
    Z_of = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}

    Ei = {}
    Yi = {}
    Ri = {}
    Flux_sput_i = {}

    for s in include_species:
        Zi = Z_of[s]
        E_inc = Zi*Ti + 3.0*Te
        # Clamp E to table range to reduce NaNs (nearest fallback still in Y_fn/R_fn)
        E_q = np.clip(E_inc, E_axis.min(), E_axis.max())
        # Interpolate (vectorized)
        Yv = Y_fn(E_q, theta)
        Rv = R_fn(E_q, theta)
        # clean up any tiny negatives / NaNs
        Yv = np.where(np.isfinite(Yv) & (Yv > 0.0), Yv, 0.0)
        # sputtered flux = Γ_i * Y_i
        Flux_sput = np.where(np.isfinite(Gi[s]), Gi[s], 0.0) * Yv

        Ei[s] = E_inc
        Yi[s] = Yv
        Ri[s] = Rv
        Flux_sput_i[s] = np.abs(Flux_sput)

    # ---------- totals ----------
    species_list = list(include_species)
    total_incident = np.sum([np.nan_to_num(Gi[s]) for s in species_list], axis=0)
    total_sput     = np.sum([np.nan_to_num(Flux_sput_i[s]) for s in species_list], axis=0)
    Y_eff = np.divide(total_sput, total_incident,
                      out=np.zeros_like(total_sput),
                      where=total_incident > 0.0)

    if verbose:
        rng = lambda a: (float(np.nanmin(a)), float(np.nanmax(a)))
        print(f"[Te eV] range: {rng(Te)} | [Ti eV] range: {rng(Ti)}")
        print(f"[Angle deg] clamped to: [{A_axis.min():.1f}, {A_axis.max():.1f}]")
        print(f"[E axis eV] range: [{E_axis.min():.3g}, {E_axis.max():.3g}]")
        for s in species_list:
            Zi = Z_of[s]
            print(f" spec{s} (O^{Zi}+)  E_inc[eV]~{rng(Ei[s])}  Y~{rng(Yi[s])}")

        print(f"[TOTAL] Γ_inc: {total_incident.min():.3e}..{total_incident.max():.3e}  "
              f"Γ_sput: {total_sput.min():.3e}..{total_sput.max():.3e}  "
              f"Y_eff: {Y_eff.min():.3e}..{Y_eff.max():.3e}")

    # ----- optional write -----
    if output_path is not None:
        _write_sputter_table(
            output_path,
            R, Z, Te, Ti, Flux_sput_i,
            total_sput,
            species_list=species_list,
            charge_by_spec=Z_of,
        )
        if verbose:
            print(f"[DONE] wrote {output_path}")


        # Write flattened file
    with h5py.File("flux_oe.h5", "w") as g:
        g.create_dataset("r", data=R)
        g.create_dataset("z", data=Z)
        g.create_dataset("eroded_flux", data=total_sput)


    # ----- return (now includes R,Z too) -----
    return {
        "R": np.asarray(R, float),
        "Z": np.asarray(Z, float),
        "centroids_R": centroids[:, 0],
        "centroids_Z": centroids[:, 1],
        "Te": Te,
        "Ti_used": {s: Ti for s in species_list},
        "Gi": Gi,
        "Ei": Ei,
        "Yi": Yi,
        "SputterFlux_i": Flux_sput_i,
        "SputterFlux_total": total_sput,
        "Y_eff": Y_eff,
    }





if __name__ == "__main__":
    # Set the directory for SOLEDGE3X input files
    plasma_dir = '/Users/42d/Desktop/OpenEdge/tools/plasma/soledge3x/soledge_data'  
    bca_dirs = ['/Users/42d/OpenEdge/tools/surface_data/data/O_on_W.h5']
    geometryFile = '/Users/42d/test_west/input/wall.txt'
    wall = surface(geometryFile, "2D")
    domain = wall.polygon
    R, Z = domain.exterior.xy
    # compute_sputter_flux_on_W(Rwall, Zwall, plasma_dir, bca_dirs=bca_dirs, theta_deg=0.0, verbose=True)
    compute_sputter_flux_on_W(R, Z, plasma_dir,
                                bca_dirs=bca_dirs,
                                include_species=None,
                                theta_deg=0.0,
                                verbose=False,
                                output_path="output.txt"
    )