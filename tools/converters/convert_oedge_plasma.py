#!/usr/bin/env python3
"""
OEDGE/DIVIMP NetCDF -> OpenEdge converter.

Reads an OEDGE .nc background file and an equilibrium (.equ) file,
interpolates onto a regular (R,Z) grid, and writes plasma.h5 + bfield.h5
in the format expected by OpenEdge's ``compute plasma/fields file`` command.

Usage:
    python convert_oedge_plasma.py d3d-204953-bkg-v25.nc \
        --equ-file 204953_3000.x16.equ \
        --plasma-out plasma.h5 --bfield-out bfield.h5 \
        --nr 300 --nz 300 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata


# ======================================================================
# Equilibrium (.equ) file reader  — same format as SOLPS converter
# ======================================================================

def read_equilibrium(equ_file: Path):
    """
    Read .equ equilibrium file and reconstruct (Br, Bt, Bz) on its own
    regular (R,Z) grid from the poloidal flux function psi(R,Z).

    Returns
    -------
    r_eq, z_eq : 1-D arrays (jm,), (km,)
    br, bt, bz : 2-D arrays (km, jm) — B-field components [T]
    """
    jm = km = btf = rtf = psib = None
    read_r = read_z = read_psi = False
    r_vals, z_vals, psi_vals = [], [], []

    with open(equ_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            # Header scalars
            if len(tok) >= 3 and tok[0] == "jm" and tok[1] == "=":
                jm = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "km" and tok[1] == "=":
                km = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "btf" and tok[1] == "=":
                btf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "rtf" and tok[1] == "=":
                rtf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "psib" and tok[1] == "=":
                psib = float(tok[2]); continue
            # Section markers
            if tok[0] == "r(1:jm);":
                read_r, read_z, read_psi = True, False, False; continue
            if tok[0] == "z(1:km);":
                read_r, read_z, read_psi = False, True, False; continue
            if tok[0].startswith("((psi(j,k)"):
                read_r, read_z, read_psi = False, False, True; continue
            # Data body
            if read_r:
                try: r_vals.extend(float(x) for x in tok)
                except ValueError: read_r = False
                continue
            if read_z:
                try: z_vals.extend(float(x) for x in tok)
                except ValueError: read_z = False
                continue
            if read_psi:
                try: psi_vals.extend(float(x) for x in tok)
                except ValueError: read_psi = False
                continue

    if any(v is None for v in [jm, km, btf, rtf]):
        raise RuntimeError(f"Missing headers (jm/km/btf/rtf) in {equ_file}")

    r_eq = np.asarray(r_vals[:jm], dtype=np.float64)
    z_eq = np.asarray(z_vals[:km], dtype=np.float64)
    psi = np.asarray(psi_vals[: jm * km], dtype=np.float64).reshape((km, jm))

    # Reconstruct B-field from psi:
    #   Br = -(1/R) dpsi/dZ,  Bz = (1/R) dpsi/dR,  Bt = btf*rtf/R
    dz = z_eq[1] - z_eq[0]
    dr = r_eq[1] - r_eq[0]
    grad_z, grad_r = np.gradient(psi, dz, dr)

    rr = np.meshgrid(r_eq, z_eq)[0]
    safe_r = np.where(np.abs(rr) > 1e-12, rr, 1e-12)

    br = -grad_z / safe_r
    bz = grad_r / safe_r
    bt = (btf * rtf) / safe_r

    return r_eq, z_eq, br, bt, bz


# ======================================================================
# OEDGE NetCDF reader
# ======================================================================

def read_oedge_nc(nc_file: Path):
    """
    Read OEDGE/DIVIMP .nc file and extract valid cell-centred plasma data.

    Returns
    -------
    dict with keys: r, z  (1-D flat arrays of valid cell centres)
                    te, ti, ne, v_par, e_par, bts, kbfs,
                    te_grad, ti_grad  (parallel gradients)
                    + grid metadata
    """
    ds = nc.Dataset(str(nc_file), "r")

    nrs = int(ds.variables["NRS"][:])
    nks_arr = np.array(ds.variables["NKS"][:])
    irsep = int(ds.variables["IRSEP"][:])
    irwall = int(ds.variables["IRWALL"][:])
    irtrap = int(ds.variables["IRTRAP"][:])

    rs = np.array(ds.variables["RS"][:])
    zs = np.array(ds.variables["ZS"][:])
    ktebs = np.array(ds.variables["KTEBS"][:])
    ktibs = np.array(ds.variables["KTIBS"][:])
    knbs = np.array(ds.variables["KNBS"][:])
    kvhs = np.array(ds.variables["KVHS"][:])
    kes = np.array(ds.variables["KES"][:])
    bts = np.array(ds.variables["BTS"][:])
    kbfs = np.array(ds.variables["KBFS"][:])
    tegs = np.array(ds.variables["TEGS"][:])
    tigs = np.array(ds.variables["TIGS"][:])

    # Scalar metadata
    cbphi = float(ds.variables["CBPHI"][:])
    r0 = float(ds.variables["R0"][:])
    z0 = float(ds.variables["Z0"][:])
    crmb = float(ds.variables["CRMB"][:])

    ds.close()

    # Build validity mask: ring/knot structure + physical values
    valid = np.zeros(rs.shape, dtype=bool)
    for ir in range(1, nrs):
        nk = int(nks_arr[ir])
        if nk > 0:
            valid[ir, 1 : nk + 1] = True
    valid &= (rs > 0.5) & (knbs > 1e10)

    return dict(
        # Flat arrays of valid cells
        r=rs[valid],
        z=zs[valid],
        te=ktebs[valid],
        ti=ktibs[valid],
        ne=knbs[valid],
        v_par=kvhs[valid],
        e_par=kes[valid],
        bts=bts[valid],
        kbfs=kbfs[valid],
        te_grad_par=tegs[valid],
        ti_grad_par=tigs[valid],
        # Metadata
        cbphi=cbphi,
        r0=r0,
        z0=z0,
        crmb=crmb,
        irsep=irsep,
        irwall=irwall,
        irtrap=irtrap,
        n_valid=int(valid.sum()),
    )


# ======================================================================
# Interpolation helpers
# ======================================================================

def make_regular_grid(rc, zc, nr, nz, rmin=None, rmax=None, zmin=None, zmax=None):
    """Build a regular (R,Z) grid enclosing the valid cell centres."""
    if rmin is None:
        rmin = float(rc.min())
    if rmax is None:
        rmax = float(rc.max())
    if zmin is None:
        zmin = float(zc.min())
    if zmax is None:
        zmax = float(zc.max())
    # Small padding
    dr = (rmax - rmin) * 0.01
    dz = (zmax - zmin) * 0.01
    rmin -= dr
    rmax += dr
    zmin -= dz
    zmax += dz
    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    return r, z


def interp_scatter_to_grid(src_r, src_z, values, grid_r, grid_z,
                           method="linear", fill_value=0.0):
    """
    Interpolate scattered (R,Z) data onto a regular grid.

    Parameters
    ----------
    src_r, src_z : 1-D arrays of source cell centres
    values       : 1-D array of field values at those centres
    grid_r, grid_z : 1-D arrays defining the regular output grid
    method       : 'linear', 'nearest', or 'cubic'

    Returns
    -------
    2-D array (nz, nr) on the regular grid.
    """
    rr, zz = np.meshgrid(grid_r, grid_z)
    pts = np.column_stack([src_r, src_z])
    out = griddata(pts, values, (rr, zz), method=method, fill_value=np.nan)
    # Fill remaining NaNs with nearest-neighbour
    nans = np.isnan(out)
    if nans.any():
        nn = griddata(pts, values, (rr[nans], zz[nans]),
                      method="nearest")
        out[nans] = nn
    # Replace any lingering NaN with fill_value
    out = np.nan_to_num(out, nan=fill_value)
    return out


def interp_equilibrium_to_grid(r_eq, z_eq, field_eq, grid_r, grid_z):
    """
    Interpolate a field on the equilibrium's own regular grid onto
    the output regular grid using bilinear interpolation.
    """
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (z_eq, r_eq), field_eq,
        method="linear", bounds_error=False, fill_value=None,
    )
    rr, zz = np.meshgrid(grid_r, grid_z)
    pts = np.column_stack([zz.ravel(), rr.ravel()])
    return interp(pts).reshape(zz.shape)


# ======================================================================
# Flow decomposition
# ======================================================================

def decompose_parallel_flow(v_par, br, bt, bz):
    """
    Decompose parallel flow magnitude into cylindrical components
    using the B-field unit vector:
        v_r = v_par * Br/|B|,  v_t = v_par * Bt/|B|,  v_z = v_par * Bz/|B|
    """
    bmag = np.sqrt(br**2 + bt**2 + bz**2)
    safe_b = np.where(bmag > 1e-12, bmag, 1e-12)
    vr = v_par * br / safe_b
    vt = v_par * bt / safe_b
    vz = v_par * bz / safe_b
    return v_par, vr, vt, vz


def decompose_parallel_gradient(grad_par, br, bt, bz):
    """
    Project a parallel gradient into cylindrical (R, toroidal, Z) components
    using the B-field unit vector.  Same decomposition as flow.
    """
    bmag = np.sqrt(br**2 + bt**2 + bz**2)
    safe_b = np.where(bmag > 1e-12, bmag, 1e-12)
    gr = grad_par * br / safe_b
    gt = grad_par * bt / safe_b
    gz = grad_par * bz / safe_b
    return gr, gt, gz


# ======================================================================
# Main conversion
# ======================================================================

def convert_oedge_to_openedge(
    nc_file: Path,
    equ_file: Path,
    plasma_out: Path = Path("plasma.h5"),
    bfield_out: Path = Path("bfield.h5"),
    nr: int = 300,
    nz: int = 300,
    rmin: float = None,
    rmax: float = None,
    zmin: float = None,
    zmax: float = None,
    plot: bool = False,
    plot_prefix: Path = None,
):
    print(f"Reading OEDGE file: {nc_file}")
    oedge = read_oedge_nc(nc_file)
    print(f"  {oedge['n_valid']} valid cells, "
          f"R=[{oedge['r'].min():.3f}, {oedge['r'].max():.3f}], "
          f"Z=[{oedge['z'].min():.3f}, {oedge['z'].max():.3f}]")

    print(f"Reading equilibrium: {equ_file}")
    r_eq, z_eq, br_eq, bt_eq, bz_eq = read_equilibrium(equ_file)
    print(f"  Equilibrium grid: {len(r_eq)}x{len(z_eq)}, "
          f"R=[{r_eq[0]:.3f}, {r_eq[-1]:.3f}], Z=[{z_eq[0]:.3f}, {z_eq[-1]:.3f}]")

    # ---- Build output regular grid ----
    grid_r, grid_z = make_regular_grid(
        oedge["r"], oedge["z"], nr, nz, rmin, rmax, zmin, zmax
    )
    print(f"Output grid: {nr}x{nz}, "
          f"R=[{grid_r[0]:.3f}, {grid_r[-1]:.3f}], "
          f"Z=[{grid_z[0]:.3f}, {grid_z[-1]:.3f}]")

    # ---- Interpolate B-field from equilibrium onto output grid ----
    print("Interpolating B-field from equilibrium ...")
    br_grid = interp_equilibrium_to_grid(r_eq, z_eq, br_eq, grid_r, grid_z)
    bt_grid = interp_equilibrium_to_grid(r_eq, z_eq, bt_eq, grid_r, grid_z)
    bz_grid = interp_equilibrium_to_grid(r_eq, z_eq, bz_eq, grid_r, grid_z)

    # ---- Interpolate plasma fields from scattered OEDGE cells ----
    print("Interpolating plasma fields ...")
    src_r, src_z = oedge["r"], oedge["z"]

    ne_grid = interp_scatter_to_grid(src_r, src_z, oedge["ne"], grid_r, grid_z)
    te_grid = interp_scatter_to_grid(src_r, src_z, oedge["te"], grid_r, grid_z)
    ti_grid = interp_scatter_to_grid(src_r, src_z, oedge["ti"], grid_r, grid_z)
    vpar_grid = interp_scatter_to_grid(src_r, src_z, oedge["v_par"], grid_r, grid_z)

    # ---- Decompose parallel flow into (R, toroidal, Z) ----
    print("Decomposing parallel flow and gradients ...")
    parr_flow, parr_flow_r, parr_flow_t, parr_flow_z = decompose_parallel_flow(
        vpar_grid, br_grid, bt_grid, bz_grid
    )

    # ---- Decompose parallel temperature gradients ----
    tegrad_par_grid = interp_scatter_to_grid(
        src_r, src_z, oedge["te_grad_par"], grid_r, grid_z
    )
    tigrad_par_grid = interp_scatter_to_grid(
        src_r, src_z, oedge["ti_grad_par"], grid_r, grid_z
    )

    gte_r, gte_t, gte_z = decompose_parallel_gradient(
        tegrad_par_grid, br_grid, bt_grid, bz_grid
    )
    gti_r, gti_t, gti_z = decompose_parallel_gradient(
        tigrad_par_grid, br_grid, bt_grid, bz_grid
    )

    # ---- Write plasma.h5 ----
    print(f"Writing {plasma_out} ...")
    with h5py.File(str(plasma_out), "w") as f:
        f.create_dataset("r", data=grid_r)
        f.create_dataset("z", data=grid_z)
        f.create_dataset("dens_e", data=ne_grid)
        f.create_dataset("temp_e", data=te_grid)
        f.create_dataset("dens_i", data=ne_grid)      # quasi-neutrality: ni = ne
        f.create_dataset("temp_i", data=ti_grid)
        f.create_dataset("parr_flow", data=parr_flow)
        f.create_dataset("parr_flow_r", data=parr_flow_r)
        f.create_dataset("parr_flow_t", data=parr_flow_t)
        f.create_dataset("parr_flow_z", data=parr_flow_z)
        f.create_dataset("grad_te_r", data=gte_r)
        f.create_dataset("grad_te_t", data=gte_t)
        f.create_dataset("grad_te_z", data=gte_z)
        f.create_dataset("grad_ti_r", data=gti_r)
        f.create_dataset("grad_ti_t", data=gti_t)
        f.create_dataset("grad_ti_z", data=gti_z)

        # Single-species metadata (deuterium)
        sdt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ion_species/names",
                         data=np.array(["D+"], dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index",
                         data=np.array([0], dtype=np.int32))
        f.create_dataset("ion_species/main_ion_spec_index",
                         data=np.array([0], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu",
                         data=np.array([oedge["crmb"]]))
        f.create_dataset("ion_species/charge_state_z",
                         data=np.array([1], dtype=np.int32))

        # Single-ion "ions/" arrays — shape (1, nz, nr)
        f.create_dataset("ions/dens",
                         data=ne_grid[np.newaxis, :, :])
        f.create_dataset("ions/temp",
                         data=ti_grid[np.newaxis, :, :])
        f.create_dataset("ions/parr_flow",
                         data=parr_flow[np.newaxis, :, :])
        f.create_dataset("ions/parr_flow_r",
                         data=parr_flow_r[np.newaxis, :, :])
        f.create_dataset("ions/parr_flow_t",
                         data=parr_flow_t[np.newaxis, :, :])
        f.create_dataset("ions/parr_flow_z",
                         data=parr_flow_z[np.newaxis, :, :])

        # Source metadata
        f.attrs["source"] = f"OEDGE: {nc_file.name}"
        f.attrs["r0"] = oedge["r0"]
        f.attrs["z0"] = oedge["z0"]

    # ---- Write bfield.h5 ----
    print(f"Writing {bfield_out} ...")
    with h5py.File(str(bfield_out), "w") as f:
        f.create_dataset("r", data=grid_r)
        f.create_dataset("z", data=grid_z)
        f.create_dataset("br", data=np.nan_to_num(br_grid))
        f.create_dataset("bt", data=np.nan_to_num(bt_grid))
        f.create_dataset("bz", data=np.nan_to_num(bz_grid))

    print("Done.")

    # ---- Optional diagnostic plots ----
    if plot:
        _plot_results(
            grid_r, grid_z,
            ne_grid, te_grid, ti_grid, vpar_grid,
            br_grid, bt_grid, bz_grid,
            src_r, src_z, oedge,
            plot_prefix,
        )


# ======================================================================
# Plotting
# ======================================================================

def _plot_results(grid_r, grid_z, ne, te, ti, vpar,
                  br, bt, bz, src_r, src_z, oedge, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if prefix is None:
        prefix = Path("oedge_convert")
    prefix = Path(prefix)

    rr, zz = np.meshgrid(grid_r, grid_z)
    bmag = np.sqrt(br**2 + bt**2 + bz**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("OEDGE → OpenEdge conversion", fontsize=14)

    fields = [
        (te, "Te [eV]", "plasma"),
        (ti, "Ti [eV]", "plasma"),
        (ne, "ne [m⁻³]", "hot"),
        (vpar, "v_par [m/s]", "RdBu_r"),
        (bmag, "|B| [T]", "inferno"),
        (bt, "Bt [T]", "RdBu_r"),
    ]

    for ax, (field, label, cmap) in zip(axes.ravel(), fields):
        vmin, vmax = np.nanpercentile(field[field != 0], [2, 98]) if np.any(field != 0) else (0, 1)
        if "RdBu" in cmap:
            vlim = max(abs(vmin), abs(vmax))
            im = ax.pcolormesh(rr, zz, field, cmap=cmap,
                               vmin=-vlim, vmax=vlim, shading="auto")
        else:
            im = ax.pcolormesh(rr, zz, field, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    outfile = f"{prefix}_fields.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {outfile}")
    plt.close(fig)

    # Scatter of original OEDGE cells for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Original OEDGE cell data (scatter)", fontsize=14)
    for ax, (vals, label, cmap) in zip(axes, [
        (oedge["te"], "Te [eV]", "plasma"),
        (oedge["ne"], "ne [m⁻³]", "hot"),
        (oedge["ti"], "Ti [eV]", "plasma"),
    ]):
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        sc = ax.scatter(src_r, src_z, c=vals, s=0.5, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    plt.tight_layout()
    outfile = f"{prefix}_oedge_scatter.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {outfile}")
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="Convert OEDGE/DIVIMP .nc to OpenEdge plasma.h5 + bfield.h5"
    )
    p.add_argument("nc_file", type=Path,
                   help="OEDGE NetCDF background file (.nc)")
    p.add_argument("--equ-file", type=Path, required=True,
                   help="Equilibrium .equ file for B-field reconstruction")
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    p.add_argument("--bfield-out", type=Path, default=Path("bfield.h5"))
    p.add_argument("--nr", type=int, default=300,
                   help="Number of R grid points (default: 300)")
    p.add_argument("--nz", type=int, default=300,
                   help="Number of Z grid points (default: 300)")
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--plot", action="store_true",
                   help="Generate diagnostic plots")
    p.add_argument("--plot-prefix", type=Path, default=None,
                   help="Prefix for plot filenames (default: oedge_convert)")
    return p


def main():
    args = _build_parser().parse_args()
    convert_oedge_to_openedge(
        nc_file=args.nc_file,
        equ_file=args.equ_file,
        plasma_out=args.plasma_out,
        bfield_out=args.bfield_out,
        nr=args.nr,
        nz=args.nz,
        rmin=args.rmin,
        rmax=args.rmax,
        zmin=args.zmin,
        zmax=args.zmax,
        plot=args.plot,
        plot_prefix=args.plot_prefix,
    )


if __name__ == "__main__":
    main()
