# OpenEdge

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)

A kinetic transport package for plasma-material and plasma-wall interactions.

OpenEdge evolves neutrals, impurity ions, and dust/droplets in prescribed
plasma and magnetic backgrounds, with surface and volume interactions
(sputtering, reflection, recycling, ionisation, recombination, CX,
dissociation). Used for edge / SOL transport studies, PMI / PWI workflows,
and plasma-wall.

## Build

CPU build (MPI):

```bash
git clone https://github.com/ORNL-Fusion/OpenEdge.git
mkdir buildOpenEdge && cd buildOpenEdge
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake -DPKG_OPENEDGE=ON
make -j$(nproc)
```

Produces `./src/spa_mpi`.

GPU build (Kokkos + CUDA):

```bash
mkdir buildOpenEdge_gpu && cd buildOpenEdge_gpu
cmake -C ../OpenEdge/cmake/presets/kokkos_cuda.cmake ../OpenEdge/cmake \
    -DPKG_OPENEDGE=ON -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
make -j$(nproc)
```

Set `Kokkos_ARCH_*` to your GPU (`AMPERE80`, `HOPPER90`, `VOLTA70`,
`PASCAL60`).

## Run a case

CPU (MPI):

```bash
cd OpenEdge/examples/test_west_axi
mpirun -np 8 ../../../buildOpenEdge/src/spa_mpi -in in.west
```

GPU (Kokkos + CUDA, one rank per GPU):

```bash
mpirun -np 1 ../../../buildOpenEdge_gpu/src/spa_mpi \
    -k on g 1 -sf kk -in in.west
```

Outputs (log, dumps, surface tallies) land in the case directory. See
each `examples/*/README.md` for case-specific post-processing.

### Requirements

- CMake >= 3.18, C++17 compiler (GCC, Clang, ICC)
- HDF5 with C++ bindings (`+MPI` for distributed runs)
- MPI (OpenMPI or MPICH)

## Documentation

Per-feature reference docs live under [`docs/`](docs/) — fixes, computes,
converters, performance, migration guides. Start at
[`docs/index.md`](docs/index.md).

## Examples

Validated test cases under [`examples/`](examples/):

- `test_west_axi` — WEST axisymmetric W transport (PMI-driven, SOLEDGE3X
  3 MW background).
- `test_solps_coupling` — SOLPS-ITER <-> OpenEdge Li droplet coupling.
- `test_diii_d_neutrals` — neutral-transport benchmark on a DIII-D plasma
  background.
- `test_iead` — sheath ion energy/angle distribution validation.
- `test_gca`, `test_droplet`, `test_collide`, ... — algorithmic
  verification.

In-progress benchmarks under [`examples/wip/`](examples/wip/).

## License

[GPL-3.0](LICENSE), inherited from SPARTA. See [LICENSE](LICENSE) for
details.
