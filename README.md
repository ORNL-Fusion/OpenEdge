# OpenEdge

[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](BUILD_CMAKE.md)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)](BUILD_CMAKE.md)
[![GPU](https://img.shields.io/badge/GPU-Kokkos%20%2F%20CUDA%20(beta)-76B900.svg)](BUILD_CMAKE.md)

<!-- TODO(badges): add build-status, docs, and DOI badges once CI,
     hosted documentation, and a Zenodo DOI exist. -->

A kinetic transport package for plasma-material and plasma-wall
interactions, built on [SPARTA](https://sparta.github.io).

OpenEdge evolves neutrals, impurity ions, and dust/droplets in
prescribed plasma and magnetic backgrounds for edge / SOL transport
studies and PMI / PWI workflows.

## Highlights

- **Surface interactions**: sputtering, reflection, recycling, areal-density
  ledgers and strata for evolving wall composition (RustBCA/TRIM/Eckstein data).
- **Volume chemistry**: ADAS ionisation, recombination, and charge
  exchange; dissociation.
- **Plasma backgrounds**: SOLPS / SOLEDGE3X mesh fields and equilibria,
  Boris and guiding-center/hybrid pushers, sheath models,
  cross-field diffusion, Coulomb drag, thermal forces.
- **Particulates**: dust and droplet transport.
- **Parallel**: MPI with load balancing; Kokkos backends (OpenMP, CUDA).

> **GPU status:** the Kokkos/CUDA backend is under **active development
> and validation**. Core transport, PWI surface chemistry, and volume
> chemistry are ported and validated on reference cases; coverage is
> narrower than the CPU path and results should be checked against a
> CPU run for new problem classes. Unported features error out
> explicitly rather than running silently wrong.

## Getting started

```bash
git clone https://github.com/ORNL-Fusion/OpenEdge.git
mkdir buildOpenEdge && cd buildOpenEdge
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake -DPKG_OPENEDGE=ON
make -j$(nproc)     # -> ./src/spa_mpi
```

- **Build options, GPU builds, HPC recipes:** [BUILD_CMAKE.md](BUILD_CMAKE.md)
- **Running cases and the example suite:** [examples/README.md](examples/README.md)
- **Reference documentation** (fixes, computes, converters, performance,
  migration guides): [docs/index.md](docs/index.md)

## License

[GPL-2.0](LICENSE), following SPARTA. See [LICENSE](LICENSE) and
[LICENSE.SPARTA](LICENSE.SPARTA) for details.
