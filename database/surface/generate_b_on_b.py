#!/usr/bin/env python3
"""Generate B-on-B (5 on 5) sputtering/reflection tables using RustBCA.

Produces an HDF5 file compatible with OpenEdge surf_react pmi:
  E      [nE]        incident energy axis (eV)
  A      [nA]        incident angle axis (deg)
  RN     [nE x nA]   reflection number (probability, 0-1)
  RE     [nE x nA]   reflected energy fraction (0-1)
  spyld  [nE x nA]   sputtering yield (atoms/ion)
  E_bind [scalar]    surface binding energy (eV)

Requires: rustbca Python bindings (pip install rustbca)

Usage:
  python generate_b_on_b.py [--output B_on_B.h5] [--nsamples 1000]
"""

import argparse
import numpy as np
import h5py

# Boron projectile on Boron target
Z_PROJ = 5         # projectile atomic number
M_PROJ = 10.81     # projectile mass (amu)
Z_TARG = 5         # target atomic number
M_TARG = 10.81     # target mass (amu)
ES_TARG = 5.73     # surface binding energy for B (eV)
EC_TARG = 3.0      # cutoff energy (eV)
EB_TARG = 3.0      # bulk binding energy (eV)
# Boron number density: rho=2.34 g/cm3, M=10.81 g/mol
# n = 2.34 * 6.022e23 / 10.81 = 1.303e23 /cm3 = 0.1303 /A^3
N_TARG = 0.1303    # atomic number density (1/A^3)


def run_rustbca_sweep(energies, angles, nsamples=1000):
    """Run RustBCA simple_bca for each (energy, angle) pair."""
    try:
        from libRustBCA import simple_bca_list_py
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
        if E < EC_TARG:
            print(f"  E = {E:.1f} eV done ({ie+1}/{nE}) [skipped, below cutoff]")
            continue

        for ia, angle_deg in enumerate(angles):
            angle_rad = np.radians(angle_deg)
            cx = np.cos(angle_rad)
            cy = np.sin(angle_rad)

            output = simple_bca_list_py(
                [E] * nsamples,
                [cx] * nsamples, [cy] * nsamples, [0.0] * nsamples,
                float(Z_PROJ), float(M_PROJ),
                1.0, ES_TARG,           # Ec1, Es1
                float(Z_TARG), float(M_TARG),
                EC_TARG, ES_TARG,       # Ec2, Es2
                N_TARG, EB_TARG         # n2, Eb2
            )

            # simple_bca_list_py returns a flat list of output particles.
            # Each particle: [Z, m, E, x, y, z, ux, uy, uz]
            # Particles with x > 0 left the surface (reflected or sputtered).
            # For same-species (B on B), we count:
            #   - total leaving particles with E > 0 and x > 0
            #   - first nsamples-worth that leave are "reflected" (at most
            #     nsamples can be reflected), the rest are sputtered.
            # A simpler approach: n_leaving = n_reflected + n_sputtered,
            # and n_reflected <= nsamples. Since reflected projectiles
            # typically have higher energy than sputtered atoms, we
            # separate by counting: the number of incident ions that
            # escape = nsamples - n_implanted (implanted = stopped inside).
            # n_implanted = particles with E=0 or x<0.

            n_leaving = 0
            e_leaving = 0.0
            n_total_out = len(output)

            for p in output:
                Z, m, Eout, x = p[0], p[1], p[2], p[3]
                if Eout > 0 and x > 0:
                    n_leaving += 1
                    e_leaving += Eout

            # Number of reflected = total output - nsamples gives
            # extra particles (sputtered). But it's more robust to note
            # that each incident ion produces exactly 1 entry (implanted
            # or reflected) plus 0+ sputtered atoms.
            # Reflected ions: incident particles that exit (x>0, E>0).
            # Sputtered: target recoils that exit.
            # Since Z is the same we can't distinguish, but the number
            # of output particles = nsamples (one per incident) + n_sputtered.
            n_sput = n_total_out - nsamples
            n_refl = n_leaving - max(n_sput, 0)
            n_refl = max(n_refl, 0)
            n_sput = max(n_sput, 0)

            RN[ie, ia] = n_refl / nsamples
            if n_refl > 0:
                # Approximate: reflected energy ~ e_leaving corrected for
                # sputtered contribution (sputtered atoms have ~Es energy)
                e_sput_approx = n_sput * ES_TARG
                e_refl = max(e_leaving - e_sput_approx, 0.0)
                RE[ie, ia] = e_refl / (n_refl * E)
                RE[ie, ia] = min(RE[ie, ia], 1.0)
            spyld[ie, ia] = n_sput / nsamples

        print(f"  E = {E:.1f} eV done ({ie+1}/{nE})")

    return RN, RE, spyld


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="B_on_B.h5",
                        help="Output HDF5 file (default: B_on_B.h5)")
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

    print(f"Generating B-on-B (Z={Z_PROJ} on Z={Z_TARG}) PMI table")
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
