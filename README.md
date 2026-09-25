# OpenEdge

[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](BUILD_CMAKE.md)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)](BUILD_CMAKE.md)
[![GPU](https://img.shields.io/badge/GPU-Kokkos%20%2F%20CUDA-76B900.svg)](BUILD_CMAKE.md)
[![package parity](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml/badge.svg?branch=main)](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml)

<!-- TODO(badges): add docs and DOI badges once hosted documentation
     and a Zenodo DOI exist. -->

A kinetic transport package for plasma-material and plasma-wall
interactions.

OpenEdge evolves neutrals, impurity ions, and dust/droplets in
prescribed plasma and magnetic backgrounds for edge / SOL transport
studies and PMI / PWI workflows.

## Highlights

- **Surface interactions**: sputtering, reflection, recycling, areal-density
  ledgers and strata for evolving wall composition ([RustBCA](https://github.com/lcpp-org/RustBCA)/TRIM/Eckstein data).
- **Volume chemistry**: ADAS ionisation, recombination, and charge
  exchange; dissociation.
- **Plasma backgrounds**: SOLPS / SOLEDGE3X mesh fields and equilibria,
  Boris and guiding-center/hybrid pushers, sheath models,
  cross-field diffusion, Coulomb drag, thermal forces.
- **Particulates**: dust and droplet transport.
- **Parallel**: MPI with load balancing; Kokkos backends (OpenMP, CUDA).

## Getting started

```bash
git clone https://github.com/ORNL-Fusion/OpenEdge.git
mkdir buildOpenEdge && cd buildOpenEdge
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake -DPKG_OPENEDGE=ON
make -j$(nproc)     # -> ./src/spa_mpi
```

- **Build options, GPU builds, HPC recipes:** [BUILD_CMAKE.md](BUILD_CMAKE.md)
- **Running cases and the example suite:** [examples/README.md](examples/README.md)
- **Users manual:** [doc/Manual.html](doc/Manual.html); rebuild the HTML or a
  local PDF with `make -C doc html` or `make -C doc pdf`

## Version policy

For production and reproducible simulations, use the latest tagged release.
The `main` branch is under active development and may change input behavior.

```bash
git clone --branch 26.09 --depth 1 https://github.com/ORNL-Fusion/OpenEdge.git
```

Existing checkouts can select the release with:

```bash
git fetch --tags
git switch --detach 26.09
```

OpenEdge prints its tag and commit at startup; record this information with
simulation results.

## License

[GPL-2.0](LICENSE)
