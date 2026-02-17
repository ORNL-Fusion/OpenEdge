# OpenEdge

OpenEdge is developed as a package for [**SPARTA**](https://github.com/sparta/sparta), using SPARTA as the particle engine (similar in spirit to domain packages in [**LAMMPS**](https://github.com/lammps/lammps)).
It is a research code for kinetic transport of charged and neutral particles and their interactions with solid surfaces and plasmas.
Primary focus is plasma-material interactions (PMI), including lithium droplet physics for the Liquid-Metal SciDAC project.

> **Status:** Early-stage research software—APIs and physics models may change.
>
> **Contributors welcome:** Diagnostics, verification cases, and tests are especially helpful.

## Current Development Focus (Feb 17, 2026)

- Integrated PMI workflow for lithium droplets near plasma-facing components
- OML-based droplet charging (`fix droplet/charge`)
- Coupled drag + evaporation + Boris trajectory pushing workflows

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

```bash
$ git clone https://github.com/ORNL-Fusion/OpenEdge.git
$ cp OpenEdge/src/openedge/* OpenEdge/src/
$ mkdir build
$ cmake -C OpenEdge/cmake/presets/mpi.cmake OpenEdge/cmake
$ make -j 4
$ mpirun -np 4 ./build/src/spa_mpi -in input.in
```


## License
OpenEdge is licensed under GPL-3.0. It is a derivative work of SPARTA (GPL-3.0).
Original SPARTA notices are preserved in file headers.
