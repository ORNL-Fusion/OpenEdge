#!/usr/bin/env python3
"""Generate D-on-C (1 on 6) sputtering/reflection tables using RustBCA.

Produces an HDF5 file compatible with OpenEdge surf_react pmi:
  E      [nE]        incident energy axis (eV)
  A      [nA]        incident angle axis (deg)
  RN     [nE x nA]   reflection number (probability, 0-1)
  RE     [nE x nA]   reflected energy fraction (0-1)
  spyld  [nE x nA]   sputtering yield (atoms/ion)
  E_bind [scalar]    surface binding energy (eV)

Requires: rustbca Python bindings (pip install rustbca)

Usage:
  python generate_d_on_c.py [--output D_on_C.h5] [--nsamples 1000]
"""

import argparse
import numpy as np
import h5py

# Deuterium projectile on Carbon target
Z_PROJ = 1         # projectile atomic number (hydrogen isotope)
M_PROJ = 2.014     # projectile mass (amu) — deuterium
Z_TARG = 6         # target atomic number
M_TARG = 12.011    # target mass (amu)
ES_TARG = 7.37     # surface binding energy for C (eV)
EC_TARG = 3.0      # cutoff energy (eV)
EB_TARG = 3.0      # bulk binding energy (eV)


def run_rustbca_sweep(energies, angles, nsamples=1000):
    """Run RustBCA for each energy, batching all angles into one call."""
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

    # Target: single-element carbon
    Z_targ_list = [float(Z_TARG)]
    M_targ_list = [float(M_TARG)]
    Es_targ_list = [ES_TARG]
    Ec_targ_list = [EC_TARG]
    Eb_targ_list = [EB_TARG]
    n_targ_list = [1.0]

    for ie, E in enumerate(energies):
        if E < EC_TARG:
            print(f"  E = {E:.1f} eV done ({ie+1}/{nE}) [skipped, below cutoff]")
            continue

        # Batch all angles x nsamples into one RustBCA call
        total = nA * nsamples
        ux = []
        uy = []
        uz = []
        E_list = [E] * total
        Z_proj_list = [Z_PROJ] * total
        M_proj_list = [M_PROJ] * total
        Ec_proj_list = [1.0] * total
        Es_proj_list = [ES_TARG] * total

        for ia, angle_deg in enumerate(angles):
            angle_rad = np.radians(angle_deg)
            cx = np.cos(angle_rad)
            cy = np.sin(angle_rad)
            ux.extend([cx] * nsamples)
            uy.extend([cy] * nsamples)
            uz.extend([0.0] * nsamples)

        output = compound_bca_list_py(
            ux, uy, uz, E_list,
            Z_proj_list, M_proj_list, Ec_proj_list, Es_proj_list,
            Z_targ_list, M_targ_list, Es_targ_list,
            Ec_targ_list, Eb_targ_list, n_targ_list
        )

        # Initialize per-angle counters
        n_refl = np.zeros(nA)
        e_refl = np.zeros(nA)
        n_sput = np.zeros(nA)

        for particle in output:
            E_out = particle[2]
            flag = int(particle[9]) if len(particle) > 9 else 0
            if len(particle) > 10:
                idx = int(particle[10])
            else:
                idx = -1

            if idx >= 0:
                ia = idx // nsamples
                if ia >= nA:
                    continue
                if flag == 1:
                    n_refl[ia] += 1
                    e_refl[ia] += E_out
                elif flag == 2:
                    n_sput[ia] += 1

        if idx < 0:
            # Fallback: run per-angle (slower but guaranteed)
            for ia, angle_deg in enumerate(angles):
                angle_rad = np.radians(angle_deg)
                cx = np.cos(angle_rad)
                cy = np.sin(angle_rad)

                out_ia = compound_bca_list_py(
                    [cx]*nsamples, [cy]*nsamples, [0.0]*nsamples,
                    [E]*nsamples,
                    [Z_PROJ]*nsamples, [M_PROJ]*nsamples,
                    [1.0]*nsamples, [ES_TARG]*nsamples,
                    Z_targ_list, M_targ_list, Es_targ_list,
                    Ec_targ_list, Eb_targ_list, n_targ_list
                )
                for p in out_ia:
                    fl = int(p[9]) if len(p) > 9 else 0
                    if fl == 1:
                        n_refl[ia] += 1
                        e_refl[ia] += p[2]
                    elif fl == 2:
                        n_sput[ia] += 1

        for ia in range(nA):
            RN[ie, ia] = n_refl[ia] / nsamples
            if n_refl[ia] > 0:
                RE[ie, ia] = e_refl[ia] / (n_refl[ia] * E)
            spyld[ie, ia] = n_sput[ia] / nsamples

        print(f"  E = {E:.1f} eV done ({ie+1}/{nE})")

    return RN, RE, spyld


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="D_on_C.h5",
                        help="Output HDF5 file (default: D_on_C.h5)")
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

    print(f"Generating D-on-C (Z={Z_PROJ}, M={M_PROJ} on Z={Z_TARG}) PMI table")
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
