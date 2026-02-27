# OpenEdge

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-green.svg)
![MPI](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MPICH-purple.svg)
![HDF5](https://img.shields.io/badge/HDF5-required-yellow.svg)

OpenEdge is developed as a package for [**SPARTA**](https://github.com/sparta/sparta), using SPARTA as the particle engine.
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

- **Particle Transport & Collisions**
  - Charged and neutral particle transport (trace-impurity friendly).
  - Coulomb/Nanbu collisions (charged); soft-sphere and hard-sphere/VSS (neutrals).
  - ADAS-based ionization and recombination with HDF5 rate tables.
- **Droplet Physics**
  - OML charging with thermionic emission.
  - Epstein and Coulomb drag models.
  - Evaporation with file-driven or constant heat flux.
  - Gravity (3D Cartesian and axisymmetric RZ).
- **Surfaces & PMI**
  - File-driven BCA/F-TRIDYN yield tables (energy x angle); sputtering, reflection, absorption.
  - PMI-driven surface emission and redeposition workflows.
- **Fields & Pushers**
  - Boris pusher with subcycling; grid- and particle-based E/B fields.
  - Plasma field interpolation from grid to particle position.
  - Electron and ion thermal gradient forces.
- **Sheath Physics**
  - Exact point-to-triangle nearest-wall projection per particle.
  - Sheath models: Borodkina, Stangeby, EIRENE-style multi-ion sheath drop.
  - Per-cell sheath potential and electric field diagnostics.
- **Geometry**
  - 2D/3D watertight surface meshes; arbitrary domains.
  - Particle-passing domain decomposition for parallel runs.
- **Diagnostics & IO**
  - HDF5 input/output for particles, grids, surfaces, and fields.
  - Ion energy-angle distributions (IEAD/EAD) per surface.
  - Sheath geometry (wall distance, surface normals) and field diagnostics.
  - Tallies for fluxes, energy/momentum transfer, and surface hits.
- **Configurability**
  - All physics and numerics controlled from the input file (no recompile).


## System Requirements

> - CMake ≥ **3.18**
> - C++17 compiler (**GCC**, **Clang**, or **ICC**)
> - **HDF5** (with C++ bindings; **+MPI** if running distributed)
> - **MPI** (OpenMPI or MPICH)
>

## Quick Start

```bash
$ git clone https://github.com/ORNL-Fusion/OpenEdge.git
$ mkdir build && cd build
$ cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake -DPKG_OPENEDGE=ON
$ make -j 4
$ mpirun -np 4 ./spa_mpi -in input.in
```


## License
OpenEdge is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0).
This means the code is open source — you are free to use, modify, and distribute it,
provided that any derivative work is also released under the same GPL-3.0 license.

OpenEdge builds on [SPARTA](https://github.com/sparta/sparta), which is itself GPL-3.0,
so OpenEdge inherits the same license terms. Original SPARTA copyright notices are
preserved in file headers. See [LICENSE](LICENSE) for full details.
