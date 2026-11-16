# OpenEdge

[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](BUILD_CMAKE.md)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)](BUILD_CMAKE.md)
[![GPU](https://img.shields.io/badge/GPU-Kokkos%20%2F%20CUDA-76B900.svg)](BUILD_CMAKE.md)
[![package parity](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml/badge.svg?branch=development)](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml)
[![Documentation Status](https://readthedocs.org/projects/openedge-ornl/badge/?version=latest)](https://openedge-ornl.readthedocs.io/en/latest/)

<!-- TODO(badges): add a DOI badge once a Zenodo DOI exists. -->

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
- **Users manual:** [openedge-ornl.readthedocs.io](https://openedge-ornl.readthedocs.io)

## Documentation

The users manual is on Read the Docs at
[openedge-ornl.readthedocs.io](https://openedge-ornl.readthedocs.io).
It covers installation, the input-script commands and the physics
models, and is rebuilt from `doc/src` on every push to `development`.
To build it locally:

```bash
pip install -r doc/utils/requirements.txt
sphinx-build -b html doc/src doc/html_sphinx   # -> doc/html_sphinx/index.html
```

## Version policy

For production and reproducible simulations, use the latest tagged release.
The `development` branch is under active development and may change input behavior.

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
