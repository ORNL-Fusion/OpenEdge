# OpenEdge

[![package parity](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml/badge.svg?branch=development)](https://github.com/ORNL-Fusion/OpenEdge/actions/workflows/parity.yml)
[![Documentation Status](https://readthedocs.org/projects/openedge-ornl/badge/?version=latest)](https://openedge-ornl.readthedocs.io)
[![Discussions](https://img.shields.io/badge/chat-discussions-turquoise.svg)](https://github.com/ORNL-Fusion/OpenEdge/discussions)
[![Supported Platforms](https://img.shields.io/badge/platforms-linux%20|%20osx-blue)](https://openedge-ornl.readthedocs.io/en/latest/Section_start.html)
[![GPU](https://img.shields.io/badge/GPU-Kokkos%20%2F%20CUDA-76B900.svg)](https://openedge-ornl.readthedocs.io/en/latest/Section_start.html)
[![GitHub commits since last release](https://img.shields.io/github/commits-since/ORNL-Fusion/OpenEdge/latest/development.svg)](https://github.com/ORNL-Fusion/OpenEdge/compare/development)
[![Language: C++17](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)
[![DOI (paper)](https://img.shields.io/badge/DOI%20(paper)-10.1016/j.cpc.2025.109993-blue.svg)](https://doi.org/10.1016/j.cpc.2025.109993)
[![DOI (particulates)](https://img.shields.io/badge/DOI%20(particulates)-10.1063/5.0336832-blue.svg)](https://doi.org/10.1063/5.0336832)

## Overview

OpenEdge is a **kinetic Monte Carlo code for impurity transport and plasma-material interaction** in the edge of magnetic fusion devices.
It follows neutrals, impurity ions, and dust grains or droplets through a plasma background taken from an edge code, and it evolves the wall surface that these particles erode and coat.

OpenEdge is built on the [SPARTA](https://sparta.github.io) Direct Simulation Monte Carlo code and keeps its input-script language, grid, surface handling and parallelism.
It runs on multi-core CPUs with MPI and on GPUs through Kokkos.

## Documentation

[![SOLPS-ITER](https://img.shields.io/static/v1?label="works%20with"&message="SOLPS-ITER"&color="blueviolet")](https://openedge-ornl.readthedocs.io/en/latest/fix_background.html)
[![SOLEDGE3X](https://img.shields.io/static/v1?label="works%20with"&message="SOLEDGE3X"&color="blueviolet")](https://openedge-ornl.readthedocs.io/en/latest/fix_background.html)
[![OEDGE](https://img.shields.io/static/v1?label="works%20with"&message="OEDGE"&color="blueviolet")](https://openedge-ornl.readthedocs.io/en/latest/fix_background.html)
[![EMC3-EIRENE](https://img.shields.io/static/v1?label="works%20with"&message="EMC3-EIRENE"&color="blueviolet")](https://openedge-ornl.readthedocs.io/en/latest/fix_background.html)

In order to learn how to install and run the code, please see the online documentation:
https://openedge-ornl.readthedocs.io

To contact the developers, feel free to open an issue on this repo, or visit our discussions page at https://github.com/ORNL-Fusion/OpenEdge/discussions

## Contributing

[![SPARTA](https://img.shields.io/static/v1?label="runs%20on"&message="SPARTA"&color="blueviolet")](https://sparta.github.io)
[![Kokkos](https://img.shields.io/static/v1?label="runs%20on"&message="Kokkos"&color="blueviolet")](https://kokkos.org)
[![HDF5](https://img.shields.io/static/v1?label="runs%20on"&message="HDF5"&color="blueviolet")](https://www.hdfgroup.org/)
[![OPEN-ADAS](https://img.shields.io/static/v1?label="runs%20on"&message="OPEN-ADAS"&color="blueviolet")](https://open.adas.ac.uk)
[![RustBCA](https://img.shields.io/static/v1?label="runs%20on"&message="RustBCA"&color="blueviolet")](https://github.com/lcpp-org/RustBCA)

We invite you to contribute to OpenEdge in any form, e.g., contribute to [discussions](https://github.com/ORNL-Fusion/OpenEdge/discussions), help each other in [issues](https://github.com/ORNL-Fusion/OpenEdge/issues), fix bugs, or add documentation and new functionality through pull requests!

## Citing OpenEdge

If you use OpenEdge, please cite the code paper:

A. Diaw, C. A. Johnson, E. A. Unterberg and J. Nichols, *OpenEdge: A collaborative, open-source, multi-purpose direct simulation Monte Carlo for plasma simulation in magnetic fusion environments*, Computer Physics Communications 320, 109993 (2026). https://doi.org/10.1016/j.cpc.2025.109993

Work that uses the particulate models for dust grains and liquid-metal droplets should also cite:

A. Diaw, J. D. Lore and S. Smolentsev, *Lithium droplet transport in tokamak edge plasmas*, Physics of Plasmas 33, 092501 (2026). https://doi.org/10.1063/5.0336832

## Copyright Notice

OpenEdge Copyright (C) 2023–2026 Abdourahmane Diaw, Oak Ridge National Laboratory.

OpenEdge is a derivative work of SPARTA (Copyright (C) 2014 Sandia Corporation) and is distributed under the GNU General Public License version 2, the license of SPARTA.

Please see the full license agreement in [LICENSE](LICENSE) and the SPARTA attribution in [LICENSE.SPARTA](LICENSE.SPARTA).
Please see the third-party data and code shipped with OpenEdge in [NOTICES](NOTICES).
