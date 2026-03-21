#!/usr/bin/env python3
"""Convert SOLEDGE3X output to OpenEdge input files using convert_s3x_plasma.

Thin wrapper around tools/converters/convert_s3x_plasma.py for the WEST 3D case.

Usage:
    python3 soledge_to_openedge.py --soledge-dir /path/to/3MW --outdir input/data
"""

import argparse
import sys
import os
from pathlib import Path

# Add the converters and tools directories to the path
_here = Path(__file__).resolve().parent
_tools = _here.parents[3] / "tools"
sys.path.insert(0, str(_tools / "converters"))
sys.path.insert(0, str(_tools))

from convert_s3x_plasma import interpolate_and_save_plasma_field


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soledge-dir", type=Path, required=True,
                    help="Directory with SOLEDGE3X output files")
    p.add_argument("--outdir", type=Path, default=Path("input/data"),
                    help="Output directory for OpenEdge files")
    p.add_argument("--plasma-file", default="plasma_00010.h5",
                    help="Plasma state filename (default: plasma_00010.h5)")
    p.add_argument("--nr", type=int, default=300,
                    help="Number of R grid points")
    p.add_argument("--nz", type=int, default=300,
                    help="Number of Z grid points")
    p.add_argument("--equ-file", default=None,
                    help="Equilibrium .equ file for B-field (optional)")
    p.add_argument("--core-psi", type=float, default=None,
                    help="Psi level for core boundary contour (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sd = args.soledge_dir

    # SOLEDGE3X file paths
    ref_file = str(sd / "refParam_raptorX.h5")
    mesh_file = str(sd / "meshEIRENE.h5")
    bfield_file = str(sd / "mesh_raptorX.h5")
    data_file = str(sd / args.plasma_file)

    # Output paths
    out = args.outdir
    plasma_out = str(out / "plasma.h5")
    bfield_out = str(out / "bfield.h5")
    wall_out = str(out / "wall.txt")
    debug_plot = str(out / "soledge_fields.png")
    flux_total_plot = str(out / "soledge_flux_total.png")
    flux_species_plot = str(out / "soledge_flux_species.png")
    wall_flux_plot = str(out / "soledge_flux_wallcoord.png")
    wall_flux_csv = str(out / "vv_values.csv")

    # Optional core contour for inner boundary
    core_sparta = str(out / "core_boundary.surf") if args.core_psi else None

    interpolate_and_save_plasma_field(
        ref_file=ref_file,
        mesh_file=mesh_file,
        bfield_file=bfield_file,
        data_file=data_file,
        wall_file=None,
        plasma_out_file=plasma_out,
        bfield_out_file=bfield_out,
        debug_plot_file=debug_plot,
        flux_total_plot_file=flux_total_plot,
        flux_species_plot_file=flux_species_plot,
        wall_flux_plot_file=wall_flux_plot,
        wall_flux_csv_file=wall_flux_csv,
        nR=args.nr,
        nZ=args.nz,
        main_ion_spec=1,
        use_mesh_wall=True,
        wall_sparta_file=wall_out,
        core_sparta_file=core_sparta,
        core_psi_level=args.core_psi,
        equ_file=args.equ_file,
    )

    print(f"\nDone! Files written to {args.outdir}/")
    print("Update your input script:")
    print("  compute pfields plasma/fields all file input/data/plasma.h5 &")
    print("          equilibrium input/data/west_3mw.equ &")
    print("          bx by bz ...")
    print("  global  bfield_compute pfields")


if __name__ == "__main__":
    main()
