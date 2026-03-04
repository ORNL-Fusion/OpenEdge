#!/usr/bin/env python3
"""Generate a sequence of plasma HDF5 snapshots from SOLEDGE3X time outputs.

Takes a list of SOLEDGE3X output directories (one per time snapshot),
calls soledge2openedge.py conversion for each, and writes a manifest
file compatible with compute plasma/timedep.

Usage:
    python gen_plasma_sweep.py \
        --soledge-dirs run_t0/ run_t1/ run_t2/ \
        --times 0.0 0.5 1.0 \
        --ref-file run_t0/refParam_raptorX.h5 \
        --outdir plasma_sweep/ \
        --nR 200 --nZ 200

    Alternatively, for simple rigid-shift sweeps (without real S3X data),
    duplicate a single plasma snapshot with radial shifts:

    python gen_plasma_sweep.py \
        --plasma-template input/plasma.h5 \
        --delta-r 0.02 0.04 0.02 0.0 -0.02 -0.04 -0.02 0.0 \
        --times   0.0  0.5  1.0  1.5  2.0   2.5   3.0  3.5 \
        --outdir plasma_sweep/

Output:
    plasma_sweep/plasma_t0.h5 .. plasma_tN.h5
    plasma_sweep/plasma_times.txt  (manifest for compute plasma/timedep)
"""

import argparse
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Mode 1: SOLEDGE3X batch conversion
# ---------------------------------------------------------------------------

def convert_soledge_dirs(soledge_dirs, times, ref_file, outdir,
                         nR=200, nZ=200, main_ion_spec=1):
    """Convert a list of SOLEDGE3X output directories to plasma HDF5 series.

    Each directory must contain the standard SOLEDGE3X output files:
      - refParam_raptorX.h5 (or use shared ref_file)
      - meshEIRENE.h5
      - mesh_raptorX.h5 (bfield)
      - plasmaFinal.h5
    """
    # Try to import soledge2openedge
    # Look in common locations
    s2oe = None
    for search_path in [
        os.path.join(os.path.dirname(__file__), '..', 'examples', 'test_west_axi', 'input'),
        os.path.dirname(__file__),
    ]:
        sys.path.insert(0, os.path.abspath(search_path))
        try:
            from soledge2openedge import interpolate_and_save_plasma_field
            s2oe = interpolate_and_save_plasma_field
            break
        except ImportError:
            sys.path.pop(0)

    if s2oe is None:
        print("ERROR: Cannot import soledge2openedge.interpolate_and_save_plasma_field")
        print("Make sure soledge2openedge.py is in the examples/test_west_axi/input/ directory")
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)
    manifest_lines = []

    for i, (sdir, t) in enumerate(zip(soledge_dirs, times)):
        sdir = os.path.abspath(sdir)
        ref = ref_file if ref_file else os.path.join(sdir, 'refParam_raptorX.h5')
        mesh = os.path.join(sdir, 'meshEIRENE.h5')
        bfield = os.path.join(sdir, 'mesh_raptorX.h5')
        data = os.path.join(sdir, 'plasmaFinal.h5')

        for f in [ref, mesh, bfield, data]:
            if not os.path.exists(f):
                print(f"ERROR: Missing file {f}")
                sys.exit(1)

        plasma_out = os.path.join(outdir, f'plasma_t{i}.h5')
        bfield_out = os.path.join(outdir, f'bfield_t{i}.h5')

        print(f"\n--- Snapshot {i}: t={t:.3f} s from {sdir} ---")
        s2oe(
            ref, mesh, bfield, data, None,
            plasma_out, bfield_out,
            nR=nR, nZ=nZ, main_ion_spec=main_ion_spec,
            use_mesh_wall=True,
        )

        fname = f'plasma_t{i}.h5'
        manifest_lines.append(f'{t}  {fname}')
        print(f"  -> {plasma_out}")

    manifest_path = os.path.join(outdir, 'plasma_times.txt')
    with open(manifest_path, 'w') as f:
        f.write('# time_seconds  filename\n')
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nManifest written to: {manifest_path}")
    print(f"Generated {len(times)} plasma snapshots")
    return manifest_path


# ---------------------------------------------------------------------------
# Mode 2: Rigid radial shift from a template plasma file
# ---------------------------------------------------------------------------

def shift_plasma_snapshot(template_path, delta_r, out_path):
    """Create a shifted plasma snapshot by translating the R-grid.

    This is a simple approximation: shift the R coordinate array while
    keeping all field values the same on the shifted grid. This effectively
    moves the plasma profiles radially.
    """
    with h5py.File(template_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        r = fin['r'][...] + delta_r
        fout.create_dataset('r', data=r)

        # Copy all other datasets directly
        for key in fin.keys():
            if key == 'r':
                continue
            # Handle groups recursively
            if isinstance(fin[key], h5py.Group):
                fin.copy(key, fout)
            else:
                fout.create_dataset(key, data=fin[key][...])


def generate_shifted_series(template_path, delta_r_list, times, outdir, prefix='plasma_t'):
    """Generate plasma snapshot series from rigid radial shifts."""
    os.makedirs(outdir, exist_ok=True)
    manifest_lines = []

    print(f"Template plasma file: {template_path}")

    for i, (dr, t) in enumerate(zip(delta_r_list, times)):
        fname = f'{prefix}{i}.h5'
        fpath = os.path.join(outdir, fname)
        shift_plasma_snapshot(template_path, dr, fpath)
        manifest_lines.append(f'{t}  {fname}')
        print(f'  [{i}] dR={dr:+.4f} m  t={t:.3f} s  -> {fpath}')

    manifest_path = os.path.join(outdir, 'plasma_times.txt')
    with open(manifest_path, 'w') as f:
        f.write('# time_seconds  filename\n')
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nManifest written to: {manifest_path}")
    print(f"Generated {len(times)} plasma snapshots")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate plasma sweep snapshots for compute plasma/timedep'
    )
    parser.add_argument('--times', nargs='+', type=float, required=True,
                        help='Simulation times in seconds (one per snapshot)')
    parser.add_argument('--outdir', default='plasma_sweep',
                        help='Output directory')
    parser.add_argument('--prefix', default='plasma_t',
                        help='Output filename prefix')

    # Mode 1: SOLEDGE3X directories
    parser.add_argument('--soledge-dirs', nargs='+', default=None,
                        help='Paths to SOLEDGE3X output directories')
    parser.add_argument('--ref-file', default=None,
                        help='Shared reference parameter file (refParam_raptorX.h5)')
    parser.add_argument('--nR', type=int, default=200,
                        help='Number of R grid points')
    parser.add_argument('--nZ', type=int, default=200,
                        help='Number of Z grid points')
    parser.add_argument('--main-ion-spec', type=int, default=1,
                        help='Main ion species index')

    # Mode 2: rigid shift from template
    parser.add_argument('--plasma-template', default=None,
                        help='Template plasma.h5 for rigid-shift mode')
    parser.add_argument('--delta-r', nargs='+', type=float, default=None,
                        help='Radial shifts in meters (one per snapshot)')

    args = parser.parse_args()

    if args.soledge_dirs is not None:
        # Mode 1: SOLEDGE3X batch conversion
        if len(args.soledge_dirs) != len(args.times):
            print(f'Error: --soledge-dirs ({len(args.soledge_dirs)}) and '
                  f'--times ({len(args.times)}) must have same length',
                  file=sys.stderr)
            sys.exit(1)
        convert_soledge_dirs(
            args.soledge_dirs, args.times, args.ref_file, args.outdir,
            nR=args.nR, nZ=args.nZ, main_ion_spec=args.main_ion_spec
        )

    elif args.plasma_template is not None and args.delta_r is not None:
        # Mode 2: rigid shift
        if len(args.delta_r) != len(args.times):
            print(f'Error: --delta-r ({len(args.delta_r)}) and '
                  f'--times ({len(args.times)}) must have same length',
                  file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.plasma_template):
            print(f'Error: template file not found: {args.plasma_template}',
                  file=sys.stderr)
            sys.exit(1)
        generate_shifted_series(
            args.plasma_template, args.delta_r, args.times,
            args.outdir, prefix=args.prefix
        )

    else:
        print('Error: specify either --soledge-dirs or (--plasma-template + --delta-r)',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
