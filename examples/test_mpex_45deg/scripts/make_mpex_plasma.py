#!/usr/bin/env python3
"""Build the MPEX challenge plasma.h5 (mesh-only schema + slab equilibrium).

Plasma prescription (from MPEX Science Impurity Transport Challenge 2025):
    B_parallel = 0.5 T, uniform axial
    n_e(r) = 1e19 * exp(-(r / 2 cm)^12)               [m^-3]
    T_e(r) = T_i(r) = 1 + 4 * exp(-(r / 2 cm)^12)     [eV]
    v_parallel = E_parallel = 0
    Background ion: D+

Slab equilibrium so the GCA pusher gets a uniform Bz = B0 with no toroidal
component:
    psi(R, Z) = -0.5 * B0 * R^2
    btf = 0,  rtf = 0,  psib = 0
    => B_R   = (1/R) d psi/dZ        = 0
       B_Z   = -(1/R) d psi/dR       = B0
       B_phi = btf * rtf / R         = 0

Coordinate convention: R = sqrt((x - x0)^2 + (y - y0)^2), Z = z. The MPEX
deck uses 'fix background ... column_axis 0.05 0.05' so the box [0, 0.1] x
[0, 0.1] x [0, 0.2] sees its center at R = 0.

Wall datasets (mesh/wall_face_area, mesh/wall_surf_cell, mesh/wall_surf_area)
are NOT written — they are only consumed by fix surface/emit/recycle, which
this case does not use.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

# Plasma parameters
B0       = 0.5             # T, uniform axial
NE0      = 1.0e19          # m^-3, on-axis
TE_HALO  = 1.0             # eV, far-from-axis floor
TE_PEAK  = 4.0             # eV, super-gaussian amplitude (so on-axis Te = 5 eV)
TI_HALO  = 1.0
TI_PEAK  = 4.0
R_NOM    = 0.02            # m, super-gaussian nominal radius (2 cm)
SG_EXP   = 12              # super-gaussian exponent
ION_NAME = "D+"
ION_ELEM = "D"
ION_Z    = 1
ION_M    = 2.014           # amu


def super_gauss(r, r_nom=R_NOM, n=SG_EXP):
    return np.exp(-((r / r_nom) ** n))


def build_mesh(r_max=0.08, z_max=0.20, nr=33, nz=41):
    """Structured (nr x nz) vertices over [0, r_max] x [0, z_max], triangulated
    as 2 tris per (i, j) cell. Returns (vtx_r, vtx_z, triangles, centroids).
    """
    r = np.linspace(0.0, r_max, nr)
    z = np.linspace(0.0, z_max, nz)
    R, Z = np.meshgrid(r, z, indexing="xy")          # Z varies along axis 0
    vtx_r = R.ravel()
    vtx_z = Z.ravel()
    nvtx  = vtx_r.size

    # Vertex indexing: vidx(i, j) = j * nr + i,  i in [0, nr), j in [0, nz)
    def vidx(i, j):
        return j * nr + i

    tris = []
    for j in range(nz - 1):
        for i in range(nr - 1):
            v00 = vidx(i,     j)
            v10 = vidx(i + 1, j)
            v11 = vidx(i + 1, j + 1)
            v01 = vidx(i,     j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    triangles = np.asarray(tris, dtype=np.int32)

    centroids_r = (vtx_r[triangles[:, 0]] + vtx_r[triangles[:, 1]] + vtx_r[triangles[:, 2]]) / 3.0
    centroids_z = (vtx_z[triangles[:, 0]] + vtx_z[triangles[:, 1]] + vtx_z[triangles[:, 2]]) / 3.0

    return vtx_r, vtx_z, triangles, centroids_r, centroids_z


def build_equilibrium(b0=B0, r_max=0.08, z_max=0.20, jm=33, km=41):
    """psi(R, Z) = -0.5 * B0 * R^2 on a (jm, km) regular grid.
    psi shape is (km, jm) per the loader: psirz[k * jm + j].
    """
    r = np.linspace(0.0, r_max, jm)
    z = np.linspace(0.0, z_max, km)
    psi = np.empty((km, jm), dtype=np.float64)
    for k in range(km):
        psi[k, :] = -0.5 * b0 * r ** 2
    return r, z, psi


def write_h5(out_path, args):
    vtx_r, vtx_z, triangles, cr, cz = build_mesh(
        r_max=args.r_max, z_max=args.z_max, nr=args.nr, nz=args.nz
    )
    ntri = triangles.shape[0]

    # Per-triangle plasma values (use centroid)
    sg = super_gauss(cr)
    dens_e = NE0 * sg
    dens_i = NE0 * sg                 # quasineutrality with single-charge D+
    temp_e = TE_HALO + TE_PEAK * sg
    temp_i = TI_HALO + TI_PEAK * sg
    parr_flow = np.zeros(ntri, dtype=np.float64)

    # Cell index: one cell per triangle (no SOLPS B2 coarsening)
    cell_index = np.arange(ntri, dtype=np.int32)

    # Equilibrium on a separate regular grid (jm x km)
    eq_r, eq_z, psi = build_equilibrium(
        b0=args.b0, r_max=args.r_max, z_max=args.z_max, jm=args.eq_jm, km=args.eq_km
    )

    # Ion species metadata (single ion: D+)
    ion_names    = np.array([ION_NAME], dtype=h5py.string_dtype(encoding="utf-8"))
    ion_elements = np.array([ION_ELEM], dtype=h5py.string_dtype(encoding="utf-8"))
    ion_charge_z = np.array([ION_Z],   dtype=np.int32)
    ion_mass     = np.array([ION_M],   dtype=np.float64)
    ion_spec_idx = np.array([0],       dtype=np.int32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        # Mesh
        m = f.create_group("mesh")
        m.create_dataset("vtx_r",      data=vtx_r.astype(np.float64))
        m.create_dataset("vtx_z",      data=vtx_z.astype(np.float64))
        m.create_dataset("triangles",  data=triangles)
        m.create_dataset("cell_index", data=cell_index)
        m.create_dataset("dens_e",     data=dens_e)
        m.create_dataset("dens_i",     data=dens_i)
        m.create_dataset("temp_e",     data=temp_e)
        m.create_dataset("temp_i",     data=temp_i)
        m.create_dataset("parr_flow",  data=parr_flow)

        # Equilibrium
        e = f.create_group("equilibrium")
        e.create_dataset("r",    data=eq_r)
        e.create_dataset("z",    data=eq_z)
        e.create_dataset("psi",  data=psi)
        e.create_dataset("btf",  data=np.float64(0.0))
        e.create_dataset("rtf",  data=np.float64(0.0))
        e.create_dataset("psib", data=np.float64(0.0))

        # Ion species
        s = f.create_group("ion_species")
        s.create_dataset("names",                data=ion_names)
        s.create_dataset("elements",             data=ion_elements)
        s.create_dataset("charge_state_z",       data=ion_charge_z)
        s.create_dataset("mass_amu",             data=ion_mass)
        s.create_dataset("spec_index",           data=ion_spec_idx)
        s.create_dataset("main_ion_spec_index", data=np.int32(0))

    print(f"[make_mpex_plasma] wrote {out_path}")
    print(f"  mesh: {vtx_r.size} vertices, {ntri} triangles")
    print(f"  domain: R in [0, {args.r_max:.4f}] m, Z in [0, {args.z_max:.4f}] m")
    print(f"  on-axis: ne = {NE0:.2e} m^-3, Te = Ti = {TE_HALO + TE_PEAK:.1f} eV")
    print(f"  axial B0 = {args.b0:.3f} T (psi = -0.5 * B0 * R^2)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="../input/plasma.h5",
                   help="Output plasma.h5 path (default: ../input/plasma.h5).")
    p.add_argument("--r-max", type=float, default=0.08,
                   help="Max R [m] (default 0.08, covers a box-corner from a 0.1 m box centered at axis).")
    p.add_argument("--z-max", type=float, default=0.20,
                   help="Max Z [m] (default 0.20).")
    p.add_argument("--nr",    type=int,   default=33,
                   help="Mesh radial vertex count (default 33).")
    p.add_argument("--nz",    type=int,   default=41,
                   help="Mesh axial vertex count (default 41).")
    p.add_argument("--b0",    type=float, default=B0,
                   help=f"Axial B [T] (default {B0}).")
    p.add_argument("--eq-jm", type=int,   default=33,
                   help="Equilibrium r-grid points (default 33).")
    p.add_argument("--eq-km", type=int,   default=41,
                   help="Equilibrium z-grid points (default 41).")
    args = p.parse_args()
    write_h5(args.out, args)


if __name__ == "__main__":
    main()
