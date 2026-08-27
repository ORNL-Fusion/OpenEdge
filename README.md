# OpenEdge

[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](BUILD_CMAKE.md)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)](BUILD_CMAKE.md)
[![GPU](https://img.shields.io/badge/GPU-Kokkos%20%2F%20CUDA-76B900.svg)](BUILD_CMAKE.md)

<!-- TODO(badges): once CI, hosted docs, and a DOI exist, adopt the
     WarpX-style row (build status, docs, DOI, discussions):
     https://github.com/BLAST-WarpX/warpx -->



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

Optional VTK output (enables the `grid/vtk`, `surf/vtk`, and
`particle/vtk` dump styles, for ParaView/VisIt):

```bash
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake \
    -DPKG_OPENEDGE=ON -DPKG_VTK=ON
```

Requires an external VTK installation (>= 7.1; both pre- and
post-8.90 component naming are handled). Works with any preset,
including the GPU build.

## Run a case

CPU (MPI):

```bash
cd OpenEdge/examples/workflows/west_impurity_transport
mpirun -np 8 ../../../../buildOpenEdge/src/spa_mpi -in in.axi_west_emission
```

GPU (Kokkos + CUDA, one rank per GPU):

```bash
mpirun -np 1 ../../../../buildOpenEdge_gpu/src/spa_mpi \
    -k on g 1 -sf kk -in in.axi_west_emission
```

Quick sanity check of the whole suite:

```bash
./regression/run_regression.sh --exe path/to/spa_mpi
```

Outputs (log, dumps, surface tallies) land in the case directory. See
each leaf `README.md` under `examples/` for case-specific post-processing.

### Requirements

- CMake >= 3.18, C++17 compiler (GCC, Clang, ICC)
- HDF5 with C++ bindings (`+MPI` for distributed runs)
- MPI (OpenMPI or MPICH)
- VTK >= 7.1 (optional, only for `-DPKG_VTK=ON`)

## Documentation

Per-feature reference docs live under [`docs/`](docs/) — fixes, computes,
converters, performance, migration guides. Start at
[`docs/index.md`](docs/index.md).

## License

[GPL-2.0](LICENSE), following SPARTA. See [LICENSE](LICENSE) for
details.
