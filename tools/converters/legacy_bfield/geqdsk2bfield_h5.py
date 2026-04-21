#!/usr/bin/env python3
"""Convert G-EQDSK equilibrium file to OpenEdge B-field HDF5 format.

Reads a standard G-EQDSK file, computes B-field components from the
poloidal flux function psi(R,Z) and toroidal field function F(psi),
and writes the result to an HDF5 file compatible with OpenEdge
compute bfield/timedep.

B_R = -(1/R) * dpsi/dZ
B_Z =  (1/R) * dpsi/dR
B_t =  F(psi) / R

Usage:
    python geqdsk2bfield_h5.py \
        --geqdsk g000001.00000 \
        --output bfield.h5 \
        [--nR 200] [--nZ 200]

    # Generate time series from multiple geqdsk files:
    python geqdsk2bfield_h5.py \
        --geqdsk g000001.00000 g000002.00000 g000003.00000 \
        --times 0.0 0.5 1.0 \
        --outdir bfield_sweep/
"""

import argparse
import os
import sys

import h5py
import numpy as np


def read_geqdsk(path):
    """Read a G-EQDSK file and return equilibrium data.

    Returns a dict with keys:
        nw, nh: grid dimensions
        rdim, zdim, rcentr, rleft, zmid: domain parameters
        rmaxis, zmaxis, simag, sibry, bcentr, current: equilibrium params
        fpol: F(psi) profile [nw]
        pres: pressure profile [nw]
        ffprim, pprime: FF', P' profiles [nw]
        psirz: psi(R,Z) on grid [nh, nw]
        qpsi: safety factor [nw]
        r, z: coordinate arrays [nw], [nh]
        rbdry, zbdry: boundary points
        rlim, zlim: limiter points
    """
    with open(path) as f:
        # Header line
        header = f.readline()
        tokens = header.split()
        nh = int(tokens[-1])
        nw = int(tokens[-2])

        # Read all remaining numbers
        def read_numbers(f, n):
            vals = []
            while len(vals) < n:
                line = f.readline()
                if not line:
                    break
                vals.extend(float(x) for x in line.split())
            return vals

        # Read 4 lines of scalars (20 values)
        scalars = read_numbers(f, 20)

        equ = {
            'nw': nw, 'nh': nh,
            'rdim': scalars[0], 'zdim': scalars[1],
            'rcentr': scalars[2], 'rleft': scalars[3], 'zmid': scalars[4],
            'rmaxis': scalars[5], 'zmaxis': scalars[6],
            'simag': scalars[7], 'sibry': scalars[8],
            'bcentr': scalars[9], 'current': scalars[10],
        }

        # 1D profiles
        equ['fpol'] = np.array(read_numbers(f, nw))
        equ['pres'] = np.array(read_numbers(f, nw))
        equ['ffprim'] = np.array(read_numbers(f, nw))
        equ['pprime'] = np.array(read_numbers(f, nw))

        # 2D psi(R,Z)
        psirz_flat = np.array(read_numbers(f, nw * nh))
        equ['psirz'] = psirz_flat.reshape(nh, nw)

        # q profile
        equ['qpsi'] = np.array(read_numbers(f, nw))

        # Boundary and limiter
        counts = read_numbers(f, 2)
        nbdry = int(counts[0]) if len(counts) >= 1 else 0
        nlim = int(counts[1]) if len(counts) >= 2 else 0

        if nbdry > 0:
            bdry = np.array(read_numbers(f, 2 * nbdry))
            equ['rbdry'] = bdry[0::2]
            equ['zbdry'] = bdry[1::2]

        if nlim > 0:
            lim = np.array(read_numbers(f, 2 * nlim))
            equ['rlim'] = lim[0::2]
            equ['zlim'] = lim[1::2]

        # Build R, Z coordinate arrays
        dr = equ['rdim'] / (nw - 1)
        dz = equ['zdim'] / (nh - 1)
        equ['r'] = equ['rleft'] + np.arange(nw) * dr
        equ['z'] = (equ['zmid'] - equ['zdim'] / 2.0) + np.arange(nh) * dz

    return equ


def get_fpol_at_psi(equ, psi):
    """Interpolate F(psi) at arbitrary psi values."""
    nw = equ['nw']
    psi_norm = (psi - equ['simag']) / (equ['sibry'] - equ['simag'])
    psi_norm = np.clip(psi_norm, 0.0, 1.0)
    idx = psi_norm * (nw - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, nw - 2)
    frac = idx - i0
    return equ['fpol'][i0] * (1.0 - frac) + equ['fpol'][i0 + 1] * frac


def compute_bfield(equ, r_out=None, z_out=None):
    """Compute B-field components from equilibrium.

    If r_out, z_out are provided, interpolate onto that grid.
    Otherwise, compute on the native grid.

    Returns (r, z, br, bt, bz) where br/bt/bz have shape [nz, nr].
    """
    r = equ['r'] if r_out is None else r_out
    z = equ['z'] if z_out is None else z_out
    nw = len(r)
    nh = len(z)

    if r_out is not None or z_out is not None:
        # Interpolate psirz onto the new grid
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (equ['z'], equ['r']), equ['psirz'],
            method='linear', bounds_error=False,
            fill_value=equ['sibry']
        )
        zz, rr = np.meshgrid(z, r, indexing='ij')
        psirz = interp(np.column_stack([zz.ravel(), rr.ravel()])).reshape(nh, nw)
    else:
        psirz = equ['psirz']

    # dpsi/dR (central differences)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    dpsi_dr = np.zeros_like(psirz)
    dpsi_dr[:, 1:-1] = (psirz[:, 2:] - psirz[:, :-2]) / (2.0 * dr)
    dpsi_dr[:, 0] = (psirz[:, 1] - psirz[:, 0]) / dr
    dpsi_dr[:, -1] = (psirz[:, -1] - psirz[:, -2]) / dr

    dpsi_dz = np.zeros_like(psirz)
    dpsi_dz[1:-1, :] = (psirz[2:, :] - psirz[:-2, :]) / (2.0 * dz)
    dpsi_dz[0, :] = (psirz[1, :] - psirz[0, :]) / dz
    dpsi_dz[-1, :] = (psirz[-1, :] - psirz[-2, :]) / dz

    R2d = np.tile(r, (nh, 1))
    R2d = np.maximum(R2d, 1.0e-10)

    br = -dpsi_dz / R2d
    bz = dpsi_dr / R2d

    # Toroidal field: B_t = F(psi) / R
    F = get_fpol_at_psi(equ, psirz)
    bt = F / R2d

    return r, z, br, bt, bz


def write_bfield_h5(path, r, z, br, bt, bz):
    """Write B-field to OpenEdge HDF5 format."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('r', data=r)
        f.create_dataset('z', data=z)
        f.create_dataset('br', data=br)
        f.create_dataset('bt', data=bt)
        f.create_dataset('bz', data=bz)


def main():
    parser = argparse.ArgumentParser(
        description='Convert G-EQDSK to OpenEdge B-field HDF5'
    )
    parser.add_argument('--geqdsk', nargs='+', required=True,
                        help='G-EQDSK file(s)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 file (single file mode)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (multi-file mode)')
    parser.add_argument('--times', nargs='+', type=float, default=None,
                        help='Simulation times (for manifest generation)')
    parser.add_argument('--nR', type=int, default=None,
                        help='Number of R grid points (resample)')
    parser.add_argument('--nZ', type=int, default=None,
                        help='Number of Z grid points (resample)')
    parser.add_argument('--R-range', nargs=2, type=float, default=None,
                        help='R range [Rmin Rmax] for resampling')
    parser.add_argument('--Z-range', nargs=2, type=float, default=None,
                        help='Z range [Zmin Zmax] for resampling')
    args = parser.parse_args()

    n_files = len(args.geqdsk)

    if n_files == 1 and args.output:
        # Single file mode
        print(f'Reading: {args.geqdsk[0]}')
        equ = read_geqdsk(args.geqdsk[0])
        print(f'  nw={equ["nw"]} nh={equ["nh"]} '
              f'R=[{equ["r"][0]:.4f},{equ["r"][-1]:.4f}] '
              f'Z=[{equ["z"][0]:.4f},{equ["z"][-1]:.4f}]')

        r_out = z_out = None
        if args.nR and args.nZ:
            if args.R_range:
                r_out = np.linspace(args.R_range[0], args.R_range[1], args.nR)
            else:
                r_out = np.linspace(equ['r'][0], equ['r'][-1], args.nR)
            if args.Z_range:
                z_out = np.linspace(args.Z_range[0], args.Z_range[1], args.nZ)
            else:
                z_out = np.linspace(equ['z'][0], equ['z'][-1], args.nZ)

        r, z, br, bt, bz = compute_bfield(equ, r_out, z_out)
        write_bfield_h5(args.output, r, z, br, bt, bz)

        bmag = np.sqrt(br**2 + bt**2 + bz**2)
        print(f'  |B| range: [{bmag.min():.3f}, {bmag.max():.3f}] T')
        print(f'  Written: {args.output}')

    elif n_files > 1 or args.outdir:
        # Multi-file mode with manifest
        outdir = args.outdir or 'bfield_geqdsk'
        os.makedirs(outdir, exist_ok=True)

        times = args.times if args.times else [float(i) for i in range(n_files)]
        if len(times) != n_files:
            print(f'Error: {n_files} files but {len(times)} times', file=sys.stderr)
            sys.exit(1)

        manifest_lines = []

        for i, (gfile, t) in enumerate(zip(args.geqdsk, times)):
            print(f'\n--- [{i}] t={t:.3f} s: {gfile} ---')
            equ = read_geqdsk(gfile)

            r_out = z_out = None
            if args.nR and args.nZ:
                if args.R_range:
                    r_out = np.linspace(args.R_range[0], args.R_range[1], args.nR)
                else:
                    r_out = np.linspace(equ['r'][0], equ['r'][-1], args.nR)
                if args.Z_range:
                    z_out = np.linspace(args.Z_range[0], args.Z_range[1], args.nZ)
                else:
                    z_out = np.linspace(equ['z'][0], equ['z'][-1], args.nZ)

            r, z, br, bt, bz = compute_bfield(equ, r_out, z_out)
            fname = f'bfield_t{i}.h5'
            fpath = os.path.join(outdir, fname)
            write_bfield_h5(fpath, r, z, br, bt, bz)

            bmag = np.sqrt(br**2 + bt**2 + bz**2)
            print(f'  |B|: [{bmag.min():.3f}, {bmag.max():.3f}] T -> {fpath}')
            manifest_lines.append(f'{t}  {fname}')

        manifest_path = os.path.join(outdir, 'bfield_times.txt')
        with open(manifest_path, 'w') as f:
            f.write('# time_seconds  filename\n')
            for line in manifest_lines:
                f.write(line + '\n')
        print(f'\nManifest: {manifest_path}')
        print(f'Generated {n_files} B-field snapshots from G-EQDSK files')

    else:
        print('Specify --output (single file) or --outdir (multi-file)', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
