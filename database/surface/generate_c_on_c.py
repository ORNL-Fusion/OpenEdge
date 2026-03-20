#!/usr/bin/env python3
"""Generate C-on-C (6 on 6) sputtering/reflection tables using RustBCA.

Produces an HDF5 file compatible with OpenEdge surf_react pmi:
  E      [nE]        incident energy axis (eV)
  A      [nA]        incident angle axis (deg)
  RN     [nE x nA]   reflection number (probability, 0-1)
  RE     [nE x nA]   reflected energy fraction (0-1)
  spyld  [nE x nA]   sputtering yield (atoms/ion)
  E_bind [scalar]    surface binding energy (eV)

Requires: rustbca Python bindings (pip install rustbca)

Usage:
  python generate_c_on_c.py [--output 6_on_6_pmi.h5] [--nsamples 1000]
"""

import argparse
import numpy as np
import h5py

# Carbon parameters
Z_PROJ = 6        # projectile atomic number
M_PROJ = 12.011   # projectile mass (amu)
Z_TARG = 6        # target atomic number
M_TARG = 12.011   # target mass (amu)
ES_TARG = 7.37    # surface binding energy for C (eV)
EC_TARG = 3.0     # cutoff energy (eV)
EB_TARG = 3.0     # bulk binding energy (eV)


def run_rustbca_sweep(energies, angles, nsamples=1000):
    """Run RustBCA for each (E, angle) pair and collect statistics."""
    try:
        from libRustBCA import compound_bca_list_py
    except ImportError:
        raise ImportError(
            "RustBCA Python bindings not found. Install with:\n"
            "  pip install rustbca\n"
            "Or build from: https://github.com/abdoudiaw/RustBCA"
        )

    nE = len(energies)
    nA = len(angles)
    RN = np.zeros((nE, nA))
    RE = np.zeros((nE, nA))
    spyld = np.zeros((nE, nA))

    for ie, E in enumerate(energies):
        for ia, angle_deg in enumerate(angles):
            if E < EC_TARG:
                continue

            angle_rad = np.radians(angle_deg)
            # Direction cosines for incident particle
            cosx = np.cos(angle_rad)
            cosy = np.sin(angle_rad)
            cosz = 0.0

            # Lists for compound_bca_list_py
            # Run nsamples ions
            ux = [cosx] * nsamples
            uy = [cosy] * nsamples
            uz = [cosz] * nsamples
            E_list = [E] * nsamples
            Z_proj_list = [Z_PROJ] * nsamples
            M_proj_list = [M_PROJ] * nsamples
            Ec_proj_list = [1.0] * nsamples  # cutoff for projectile tracking
            Es_proj_list = [ES_TARG] * nsamples

            # Target: single-element carbon
            Z_targ_list = [float(Z_TARG)]
            M_targ_list = [float(M_TARG)]
            Es_targ_list = [ES_TARG]
            Ec_targ_list = [EC_TARG]
            Eb_targ_list = [EB_TARG]
            n_targ_list = [1.0]  # stoichiometry

            output = compound_bca_list_py(
                ux, uy, uz, E_list,
                Z_proj_list, M_proj_list, Ec_proj_list, Es_proj_list,
                Z_targ_list, M_targ_list, Es_targ_list, Ec_targ_list,
                Eb_targ_list, n_targ_list
            )

            # Parse output: each entry is [Z, m, E, x, y, z, ux, uy, uz, flag]
            # flag: 1 = reflected projectile, 2 = sputtered target atom
            n_reflected = 0
            e_reflected_sum = 0.0
            n_sputtered = 0

            for particle in output:
                Z_out = particle[0]
                E_out = particle[2]
                flag = int(particle[9]) if len(particle) > 9 else 0

                if flag == 1:  # reflected
                    n_reflected += 1
                    e_reflected_sum += E_out
                elif flag == 2:  # sputtered
                    n_sputtered += 1

            RN[ie, ia] = n_reflected / nsamples
            if n_reflected > 0:
                RE[ie, ia] = e_reflected_sum / (n_reflected * E)
            spyld[ie, ia] = n_sputtered / nsamples

        print(f"  E = {E:.1f} eV done ({ie+1}/{nE})")

    return RN, RE, spyld


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="6_on_6_pmi.h5",
                        help="Output HDF5 file (default: 6_on_6_pmi.h5)")
    parser.add_argument("--nsamples", type=int, default=1000,
                        help="Number of BCA samples per (E,angle) point")
    parser.add_argument("--ne", type=int, default=50,
                        help="Number of energy grid points")
    parser.add_argument("--na", type=int, default=45,
                        help="Number of angle grid points")
    parser.add_argument("--emin", type=float, default=1.0,
                        help="Minimum energy (eV)")
    parser.add_argument("--emax", type=float, default=1000.0,
                        help="Maximum energy (eV)")
    args = parser.parse_args()

    energies = np.geomspace(args.emin, args.emax, args.ne)
    angles = np.linspace(0.0, 89.0, args.na)

    print(f"Generating C-on-C (Z={Z_PROJ} on Z={Z_TARG}) PMI table")
    print(f"  Energies: {args.ne} points, {args.emin}–{args.emax} eV (log-spaced)")
    print(f"  Angles: {args.na} points, 0–89 deg")
    print(f"  Samples per point: {args.nsamples}")
    print(f"  Total BCA runs: {args.ne * args.na * args.nsamples:,}")
    print()

    RN, RE, spyld = run_rustbca_sweep(energies, angles, args.nsamples)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("E", data=energies)
        f.create_dataset("A", data=angles)
        f.create_dataset("RN", data=RN)
        f.create_dataset("RE", data=RE)
        f.create_dataset("spyld", data=spyld)
        f.create_dataset("E_bind", data=ES_TARG)
        # Metadata
        f.attrs["projectile_Z"] = Z_PROJ
        f.attrs["projectile_m"] = M_PROJ
        f.attrs["target_Z"] = Z_TARG
        f.attrs["target_m"] = M_TARG
        f.attrs["surface_binding_energy_eV"] = ES_TARG
        f.attrs["nsamples_per_point"] = args.nsamples

    print(f"Saved: {args.output}")
    print(f"  RN range: [{RN.min():.4f}, {RN.max():.4f}]")
    print(f"  RE range: [{RE.min():.4f}, {RE.max():.4f}]")
    print(f"  spyld range: [{spyld.min():.4f}, {spyld.max():.4f}]")


if __name__ == "__main__":
    main()
