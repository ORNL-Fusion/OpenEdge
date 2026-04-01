#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt


def read_equ_file(path):
    """
    Parse custom .equ file with sections like:
        jm := description;
        km := description;
        r  := description;
        z  := description;
        psi := description;
    followed by numeric data blocks.
    """

    fields = {}
    current_key = None
    values = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # New field header like "psi := flux per radiant ..."
            m = re.match(r"^([A-Za-z0-9_]+)\s*:=", line)
            if m:
                if current_key is not None:
                    fields[current_key] = values
                current_key = m.group(1).lower()
                values = []
                continue

            # Otherwise try to collect numeric values
            if current_key is not None:
                parts = line.split()
                for p in parts:
                    try:
                        values.append(float(p))
                    except ValueError:
                        pass

    if current_key is not None:
        fields[current_key] = values

    return fields


def build_equilibrium(fields):
    if "jm" not in fields or "km" not in fields:
        raise ValueError("Missing jm/km in .equ file")

    nr = int(fields["jm"][0])
    nz = int(fields["km"][0])

    if "r" not in fields or "z" not in fields or "psi" not in fields:
        raise ValueError("Missing one of r, z, psi in .equ file")

    r = np.array(fields["r"], dtype=float)
    z = np.array(fields["z"], dtype=float)
    psi = np.array(fields["psi"], dtype=float)

    if r.size != nr:
        raise ValueError(f"Expected {nr} r values, found {r.size}")
    if z.size != nz:
        raise ValueError(f"Expected {nz} z values, found {z.size}")
    if psi.size != nr * nz:
        raise ValueError(f"Expected {nr*nz} psi values, found {psi.size}")

    # Assume psi stored as [z, r]
    psi2d = psi.reshape(nz, nr)

    return r, z, psi2d


def compute_bfield_from_psi(r, z, psi2d):
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    dpsi_dz, dpsi_dr = np.gradient(psi2d, dz, dr)

    R2d, Z2d = np.meshgrid(r, z)
    R2d = np.maximum(R2d, 1.0e-12)

    # Standard axisymmetric convention
    Br = -dpsi_dz / R2d
    Bz =  dpsi_dr / R2d

    return R2d, Z2d, Br, Bz


def plot_equ(r, z, psi2d, Br, Bz):
    R2d, Z2d = np.meshgrid(r, z)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    c0 = axes[0].contourf(R2d, Z2d, psi2d, levels=50)
    plt.colorbar(c0, ax=axes[0])
    axes[0].set_title("psi")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal")

    c1 = axes[1].contourf(R2d, Z2d, Br, levels=50)
    plt.colorbar(c1, ax=axes[1])
    axes[1].set_title("Br")
    axes[1].set_xlabel("R [m]")
    axes[1].set_ylabel("Z [m]")
    axes[1].set_aspect("equal")

    c2 = axes[2].contourf(R2d, Z2d, Bz, levels=50)
    plt.colorbar(c2, ax=axes[2])
    axes[2].set_title("Bz")
    axes[2].set_xlabel("R [m]")
    axes[2].set_ylabel("Z [m]")
    axes[2].set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    equ_file = "g174310.03500_153.X4.equ"

    fields = read_equ_file(equ_file)
    print("Parsed keys:", sorted(fields.keys()))

    r, z, psi2d = build_equilibrium(fields)
    R2d, Z2d, Br, Bz = compute_bfield_from_psi(r, z, psi2d)

    plot_equ(r, z, psi2d, Br, Bz)
