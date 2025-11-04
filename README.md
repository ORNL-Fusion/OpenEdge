# OpenEdge

OpenEdge is a research code for kinetic transport of charged and neutral particles and their interactions with solid surfaces and plasmas. It targets fusion edge/PMI problems but is not limited to them.

> **Status:** Early-stage research software—APIs and physics models may change.
>
> **Contributors welcome:** Diagnostics, verification cases, and tests are especially helpful.



## Features

- **Physics**
  - Charged & neutral particle transport (trace-impurity friendly).
  - **Collisions:** Nanbu method (charged), hard-sphere (neutrals).
  - **Ionization/Recombination:** ADAS-based rate data.
  - **Surfaces & PMI:** file-driven BCA/F-TRIDYN yields/reactions; redeposition & reflection models.
  - **Fields & Pushers:** Boris pusher; external fields; optional sheath models.
- **Geometry**
  - 2D/3D watertight surface meshes; arbitrary domains.
  - Particle-passing domain decomposition for parallel runs.
- **IO & Diagnostics**
  - HDF5 input/output (state, surfaces, fields, tallies).
  - Tallies for fluxes, energy/momentum transfer, surface hits; post-processing helpers.
- **Configurability**
  - Most physics & numerics are controlled from the input file (no recompile).


## System Requirements

> - CMake ≥ **3.18**
> - C++17 compiler (**GCC**, **Clang**, or **ICC**)
> - **HDF5** (with C++ bindings; **+MPI** if running distributed)
> - **MPI** (OpenMPI or MPICH)
>

## Quick Start

```ruby
$ git clone https://github.com/ORNL-Fusion/OpenEdge.git
$ mkdir build
$ cmake -C OpenEdge/cmake/presets/mpi.cmake -B build OpenEdge/cmake
$ make -j 8
$ mpirun -np 4 ./build/src/spa_mpi -in input.in


