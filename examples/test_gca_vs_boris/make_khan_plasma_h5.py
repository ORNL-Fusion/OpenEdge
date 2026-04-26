#!/usr/bin/env python3
"""Build a synthetic plasma.h5 from Khan et al. (2012) axisymmetric
tokamak field, for the Boris-vs-GCA pusher validation test.

Khan's Cartesian field (paper Sec. 4.1):
    B_x = -(2 R0 y + x z)/(2 R^2)
    B_y =  (2 R0 x - y z)/(2 R^2)
    B_z =  (R - 1)/(2 R)

decomposes axisymmetrically (R, phi, Z) into:
    B_R(R, Z) = -Z/(2 R)
    B_phi(R)  =  R0 / R
    B_Z(R)    = (R - 1)/(2 R)

so a flux function ψ(R, Z) and toroidal magnetic moment btf·rtf reproduce it:
    ψ(R, Z) = Z^2/4 + R^3/6 - R^2/4
    btf·rtf = R0 (set btf = rtf = 1, R0 = 1 to match paper)

OpenEdge's `fix background file khan_plasma.h5` reads /equilibrium/*
and reconstructs B via
    B_R = -(1/R) ∂ψ/∂Z,   B_Z = (1/R) ∂ψ/∂R,   B_phi = btf·rtf / R.
Both Boris and GCA can then run against the same analytical field.
"""

import numpy as np
import h5py


def build_khan_equilibrium(nr=257, nz=257, rmin=0.05, rmax=2.5,
                           zmin=-2.0, zmax=2.0, R0=1.0):
    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    R, Z = np.meshgrid(r, z)                                # shape (nz, nr)
    psi = Z * Z / 4.0 + R ** 3 / 6.0 - R * R / 4.0
    btf = R0
    rtf = 1.0
    psib = 0.0
    return r, z, psi, btf, rtf, psib


def write_plasma_h5(path, r, z, psi, btf, rtf, psib):
    with h5py.File(path, "w") as f:
        eq = f.create_group("equilibrium")
        eq.create_dataset("r",    data=r,   dtype="f8")
        eq.create_dataset("z",    data=z,   dtype="f8")
        eq.create_dataset("psi",  data=psi, dtype="f8")
        eq.create_dataset("btf",  data=np.float64(btf))
        eq.create_dataset("rtf",  data=np.float64(rtf))
        eq.create_dataset("psib", data=np.float64(psib))


def main():
    r, z, psi, btf, rtf, psib = build_khan_equilibrium()
    out = "khan_plasma.h5"
    write_plasma_h5(out, r, z, psi, btf, rtf, psib)
    print(f"wrote {out}")
    print(f"  R grid: {r[0]:.3f} .. {r[-1]:.3f} m  ({len(r)} pts)")
    print(f"  Z grid: {z[0]:.3f} .. {z[-1]:.3f} m  ({len(z)} pts)")
    print(f"  psi range: {psi.min():.4f} .. {psi.max():.4f}")
    print(f"  btf={btf}  rtf={rtf}  psib={psib}")
    # quick sanity: B at outer midplane R=1.09, Z=0
    R = 1.09
    BR  = 0.0
    Bphi = btf * rtf / R
    BZ  = (R - 1.0) / (2.0 * R)
    print(f"  check at (R=1.09, Z=0):  B_R={BR:.4f}  B_phi={Bphi:.4f}  B_Z={BZ:.4f} T")
    print(f"  |B| = {np.sqrt(BR*BR + Bphi*Bphi + BZ*BZ):.4f} T")


if __name__ == "__main__":
    main()
