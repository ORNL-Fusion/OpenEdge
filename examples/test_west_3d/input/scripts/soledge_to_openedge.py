#!/usr/bin/env python3
"""Convert SOLEDGE3X output to OpenEdge input files (plasma.h5, bfield.h5, .equ).

Reads multi-zone SOLEDGE data and interpolates onto a uniform (R,Z) grid
for OpenEdge's compute plasma/fields.

Usage:
    python3 soledge_to_openedge.py --soledge-dir /path/to/3MW --outdir input/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soledge-dir", type=Path, required=True,
                    help="Directory with SOLEDGE3X output files")
    p.add_argument("--outdir", type=Path, default=Path("input/data"),
                    help="Output directory for OpenEdge files")
    p.add_argument("--plasma-file", default="plasma_00010.h5",
                    help="Plasma state filename")
    p.add_argument("--nr", type=int, default=300,
                    help="Number of R grid points for output")
    p.add_argument("--nz", type=int, default=300,
                    help="Number of Z grid points for output")
    return p.parse_args()


def read_soledge_zones(soledge_dir: Path, plasma_file: str):
    """Read multi-zone SOLEDGE data and return scattered (R,Z) points with fields."""

    mesh_raptor = h5py.File(soledge_dir / "mesh_raptorX.h5", "r")
    plasma = h5py.File(soledge_dir / plasma_file, "r")
    ref = h5py.File(soledge_dir / "refParam_raptorX.h5", "r")

    # Reference values for denormalization
    n0 = float(ref["n0"][0])
    T0 = float(ref["T0"][0])
    c0 = float(ref["c0"][0])
    rho0 = float(ref["rho0"][0])
    B0 = float(ref["B0"][0])

    print(f"Reference: n0={n0:.2e}, T0={T0:.1f} eV, c0={c0:.1f} m/s, "
          f"rho0={rho0:.6f} m, B0={B0:.4f} T")

    nzones = int(mesh_raptor["NZones"][0])
    print(f"Number of zones: {nzones}")

    # Determine number of ion species by scanning zone1 for all spec* groups
    # Nspecies in the file may not count all species (e.g. excludes electrons),
    # so count explicitly.
    all_spec_keys = sorted([k for k in plasma["zone1"].keys() if k.startswith("spec")])
    # spec0 = electrons, rest are ions
    ion_spec_indices = []
    ion_charges = []
    for k in all_spec_keys:
        idx = int(k.replace("spec", ""))
        charge = int(plasma[f"zone1/{k}/charge"][0])
        if charge > 0:  # skip electrons
            ion_spec_indices.append(idx)
            ion_charges.append(charge)
    nion = len(ion_spec_indices)
    print(f"  Ion species: {nion} (charges: {ion_charges})")

    # Collect all cell-center data across zones
    all_r = []
    all_z = []
    all_br = []
    all_bt = []
    all_bz = []
    all_te = []
    all_ne = []
    all_ti = []
    all_ni = []
    all_ve = []
    all_vi = []
    # Per-species ion data (for PMI sputtering)
    all_ion_n = [[] for _ in range(nion)]    # nion lists
    all_ion_t = [[] for _ in range(nion)]
    all_ion_v = [[] for _ in range(nion)]

    for iz in range(1, nzones + 1):
        zname = f"zone{iz}"
        if zname not in mesh_raptor:
            continue

        # Cell centers in rho0 units -> meters
        rc = mesh_raptor[f"{zname}/Rcells"][:, :, 0] * rho0
        zc = mesh_raptor[f"{zname}/Zcells"][:, :, 0] * rho0

        # B-field in B0 units -> Tesla
        br = mesh_raptor[f"{zname}/Br"][:, :, 0] * B0
        bt = mesh_raptor[f"{zname}/Bphi"][:, :, 0] * B0
        bz = mesh_raptor[f"{zname}/Bz"][:, :, 0] * B0

        # Plasma: spec0=electrons, spec1=D+
        te = plasma[f"{zname}/spec0/T"][:, :, -1] * T0  # eV
        ne = plasma[f"{zname}/spec0/n"][:, :, -1] * n0  # m^-3
        ti = plasma[f"{zname}/spec1/T"][:, :, -1] * T0  # eV
        ni = plasma[f"{zname}/spec1/n"][:, :, -1] * n0  # m^-3
        ve = plasma[f"{zname}/spec0/G"][:, :, -1] * c0  # m/s
        vi = plasma[f"{zname}/spec1/G"][:, :, -1] * c0  # m/s

        all_r.append(rc.ravel())
        all_z.append(zc.ravel())
        all_br.append(br.ravel())
        all_bt.append(bt.ravel())
        all_bz.append(bz.ravel())
        all_te.append(te.ravel())
        all_ne.append(ne.ravel())
        all_ti.append(ti.ravel())
        all_ni.append(ni.ravel())
        all_ve.append(ve.ravel())
        all_vi.append(vi.ravel())

        # Per-species ion data (D+, O+, O2+, ..., O8+)
        for si, spec_idx in enumerate(ion_spec_indices):
            sp = f"{zname}/spec{spec_idx}"
            all_ion_n[si].append((plasma[f"{sp}/n"][:, :, -1] * n0).ravel())
            all_ion_t[si].append((plasma[f"{sp}/T"][:, :, -1] * T0).ravel())
            all_ion_v[si].append((plasma[f"{sp}/G"][:, :, -1] * c0).ravel())

    mesh_raptor.close()
    plasma.close()
    ref.close()

    data = {
        "r": np.concatenate(all_r),
        "z": np.concatenate(all_z),
        "br": np.concatenate(all_br),
        "bt": np.concatenate(all_bt),
        "bz": np.concatenate(all_bz),
        "te": np.concatenate(all_te),
        "ne": np.concatenate(all_ne),
        "ti": np.concatenate(all_ti),
        "ni": np.concatenate(all_ni),
        "ve": np.concatenate(all_ve),
        "vi": np.concatenate(all_vi),
        "nion": nion,
        "ion_charges": ion_charges,
        "ion_n": [np.concatenate(a) for a in all_ion_n],
        "ion_t": [np.concatenate(a) for a in all_ion_t],
        "ion_v": [np.concatenate(a) for a in all_ion_v],
    }
    print(f"Total scattered points: {len(data['r'])}")
    print(f"  R range: [{data['r'].min():.4f}, {data['r'].max():.4f}] m")
    print(f"  Z range: [{data['z'].min():.4f}, {data['z'].max():.4f}] m")

    return data


def interpolate_to_uniform(data, nr, nz):
    """Interpolate scattered SOLEDGE data onto a uniform (R,Z) grid."""

    r_min, r_max = data["r"].min(), data["r"].max()
    z_min, z_max = data["z"].min(), data["z"].max()

    r_grid = np.linspace(r_min, r_max, nr)
    z_grid = np.linspace(z_min, z_max, nz)
    R, Z = np.meshgrid(r_grid, z_grid, indexing="ij")

    points = np.column_stack([data["r"], data["z"]])
    target = np.column_stack([R.ravel(), Z.ravel()])

    result = {"r": r_grid, "z": z_grid}

    fields = ["br", "bt", "bz", "te", "ne", "ti", "ni", "ve", "vi"]
    for fname in fields:
        vals = data[fname]
        # Remove NaN/inf
        mask = np.isfinite(vals)
        interp = griddata(points[mask], vals[mask], target,
                          method="linear", fill_value=0.0)
        result[fname] = interp.reshape(nr, nz)
        print(f"  {fname}: [{result[fname].min():.4e}, {result[fname].max():.4e}]")

    # Per-species ion data
    nion = data["nion"]
    result["nion"] = nion
    result["ion_charges"] = data["ion_charges"]
    result["ion_n"] = []
    result["ion_t"] = []
    result["ion_v"] = []
    for si in range(nion):
        for label, src_list, dst_list in [
            ("n", data["ion_n"], result["ion_n"]),
            ("t", data["ion_t"], result["ion_t"]),
            ("v", data["ion_v"], result["ion_v"]),
        ]:
            vals = src_list[si]
            mask = np.isfinite(vals)
            interp = griddata(points[mask], vals[mask], target,
                              method="linear", fill_value=0.0)
            dst_list.append(interp.reshape(nr, nz))
        charge = data["ion_charges"][si]
        n_max = result["ion_n"][-1].max()
        print(f"  ion spec{si+1} (Z={charge:+d}): n_max={n_max:.2e}")

    return result


def write_bfield_h5(path, grid):
    """Write OpenEdge bfield.h5 format."""
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=grid["r"])
        f.create_dataset("z", data=grid["z"])
        f.create_dataset("br", data=grid["br"])
        f.create_dataset("bt", data=grid["bt"])
        f.create_dataset("bz", data=grid["bz"])
    print(f"Wrote {path}")


def write_plasma_h5(path, grid):
    """Write OpenEdge plasma.h5 format with per-species ion data.

    Dataset names must match what compute_plasma_fields.cpp expects:
      dens_e, temp_e, dens_i, temp_i, parr_flow, parr_flow_r/t/z,
      grad_te_r/t/z, grad_ti_r/t/z
    """
    nr, nz = len(grid["r"]), len(grid["z"])
    zeros = np.zeros((nr, nz))

    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=grid["r"])
        f.create_dataset("z", data=grid["z"])
        f.create_dataset("dens_e", data=grid["ne"])
        f.create_dataset("temp_e", data=grid["te"])
        f.create_dataset("dens_i", data=grid["ni"])
        f.create_dataset("temp_i", data=grid["ti"])
        # Parallel flow: vi is the total parallel flow magnitude
        f.create_dataset("parr_flow", data=grid["vi"])
        # Flow components (R, toroidal, Z) — set to zero, computed from B-field
        f.create_dataset("parr_flow_r", data=zeros)
        f.create_dataset("parr_flow_t", data=zeros)
        f.create_dataset("parr_flow_z", data=zeros)
        # Temperature gradients — set to zero (computed by finite differences)
        f.create_dataset("grad_te_r", data=zeros)
        f.create_dataset("grad_te_t", data=zeros)
        f.create_dataset("grad_te_z", data=zeros)
        f.create_dataset("grad_ti_r", data=zeros)
        f.create_dataset("grad_ti_t", data=zeros)
        f.create_dataset("grad_ti_z", data=zeros)

        # Per-species ion data for PMI sputtering
        nion = grid["nion"]
        nr, nz = len(grid["r"]), len(grid["z"])

        # ions/dens: (nion, nr, nz)
        ions_dens = np.stack(grid["ion_n"], axis=0)
        ions_temp = np.stack(grid["ion_t"], axis=0)
        ions_flow = np.stack(grid["ion_v"], axis=0)
        f.create_dataset("ions/dens", data=ions_dens)
        f.create_dataset("ions/temp", data=ions_temp)
        f.create_dataset("ions/parr_flow", data=ions_flow)

        # Charge states for each ion species
        f.create_dataset("ion_species/charge_state_z",
                         data=np.array(grid["ion_charges"], dtype=np.int32))

    print(f"Wrote {path} ({nion} ion species)")


def write_equ_file(path, soledge_dir):
    """Write an .equ file from SOLEDGE mesh.h5 psi data.

    The .equ format expected by fix_reflect_psi / compute plasma/fields:
      jm=<nr>; km=<nz>; btf=<btf>; rtf=<rtf>; psib=<psib>;
      r(1:jm); <r values>
      z(1:km); <z values>
      ((psi(j,k)-psib,j=1,jm),k=1,km) <psi-psib values>
    """
    mesh = h5py.File(soledge_dir / "mesh.h5", "r")

    r2d = mesh["config/r"][:]  # (nr, nz), R varies along dim 0
    z2d = mesh["config/z"][:]
    psi2d = mesh["config/psi"][:]
    R0 = float(mesh["R0"][0])
    B0 = float(mesh["B0"][0])
    psisep = float(mesh["config/psisep1"][0])

    r_1d = r2d[:, 0]  # R varies along first axis
    z_1d = z2d[0, :]  # Z varies along second axis
    nr = len(r_1d)
    nz = len(z_1d)

    # btf * rtf = R0 * B_toroidal(R0) — vacuum toroidal field constant
    btf = B0
    rtf = R0

    # psib = psi at boundary (separatrix)
    psib = psisep

    mesh.close()

    with open(path, "w") as f:
        f.write(f"jm={nr}; km={nz};\n")
        f.write(f"btf={btf:.10e}; rtf={rtf:.10e}; psib={psib:.10e};\n")

        f.write("r(1:jm);\n")
        for i, r in enumerate(r_1d):
            f.write(f"  {r:.10e}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if nr % 5 != 0:
            f.write("\n")

        f.write("z(1:km);\n")
        for i, z in enumerate(z_1d):
            f.write(f"  {z:.10e}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if nz % 5 != 0:
            f.write("\n")

        # psi(j,k) - psib, where j=R index, k=Z index
        # The .equ format iterates: for k in 1..km, for j in 1..jm
        f.write("((psi(j,k)-psib,j=1,jm),k=1,km)\n")
        count = 0
        for k in range(nz):
            for j in range(nr):
                val = psi2d[j, k] - psib
                f.write(f"  {val:.10e}")
                count += 1
                if count % 5 == 0:
                    f.write("\n")
        if count % 5 != 0:
            f.write("\n")

    print(f"Wrote {path}")
    print(f"  Grid: {nr} x {nz}, R=[{r_1d[0]:.4f}, {r_1d[-1]:.4f}], "
          f"Z=[{z_1d[0]:.4f}, {z_1d[-1]:.4f}]")
    print(f"  btf={btf:.4f}, rtf={rtf:.4f}, psib={psib:.6f}")


def write_wall_txt(path, soledge_dir):
    """Write SPARTA 2D wall file from SOLEDGE mesh.h5 wall data."""
    mesh = h5py.File(soledge_dir / "mesh.h5", "r")
    Rw = mesh["wall/R"][:]
    Zw = mesh["wall/Z"][:]
    mesh.close()

    n = len(Rw)
    with open(path, "w") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i in range(n):
            f.write(f"{i+1} {Rw[i]:.12g} {Zw[i]:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            a = i + 1
            b = (i + 1) % n + 1
            f.write(f"{i+1} {a} {b}\n")

    print(f"Wrote {path} ({n} points)")


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Reading SOLEDGE zones...")
    data = read_soledge_zones(args.soledge_dir, args.plasma_file)

    print(f"\nInterpolating to {args.nr}x{args.nz} uniform grid...")
    grid = interpolate_to_uniform(data, args.nr, args.nz)

    print("\nWriting output files...")
    write_bfield_h5(args.outdir / "bfield.h5", grid)
    write_plasma_h5(args.outdir / "plasma.h5", grid)
    write_equ_file(args.outdir / "west_3mw.equ", args.soledge_dir)
    write_wall_txt(args.outdir / "wall.txt", args.soledge_dir)

    print("\nDone! Update your input script:")
    print("  compute pfields plasma/fields all file input/data/plasma.h5 &")
    print("          equilibrium input/data/west_3mw.equ &")
    print("          bx by bz ...")
    print("  global  bfield_compute pfields")


if __name__ == "__main__":
    main()
