---
myst:
  html_meta:
    "description": "OpenEdge — kinetic plasma-edge and plasma-material interaction simulation"
---

# OpenEdge

OpenEdge is a kinetic plasma-edge and plasma-material interaction code.
It combines prescribed plasma backgrounds with charged and neutral particle
transport, sheath physics, atomic processes, plasma-wall interactions,
particulate models, liquid-metal models, and coupling utilities.

## Description

This manual is the user reference for OpenEdge. It documents both the core
input language (`species`, `create_box`, `create_particles`, `read_surf`,
`run`, `dump`, and related commands) and the OpenEdge physics commands. A
separate manual is not required to build or run a supported OpenEdge case.

OpenEdge uses the open-source SPARTA particle, mesh, surface, MPI, and Kokkos
runtime as its foundation. That implementation provenance matters for
licensing and citation, but it is not a second user interface.

## Package summary

OpenEdge currently adds the following major capability groups:

- **Plasma background and field access** for SOLPS, SOLEDGE3X, and OEDGE
  derived states.
- **Charged-particle transport** with Boris and hybrid Boris/GCA
  pushers, sheath models, and cross-field transport.
- **Surface-interaction models** for recycling, sputtering, diffuse
  reflection, source emission, and PMI/PWI reaction networks.
- **Volumetric chemistry** using ADAS and Janev data for ionization,
  recombination, charge exchange, and dissociation.
- **Liquid-metal models** for evaporation, viscous drag, droplet
  charging, and shallow-water film evolution.
- **Coupling and conversion tools** for outer-loop workflows and shared
  data formats.

## Input-script style

An OpenEdge input deck is an ordered sequence of commands. A typical setup
has the form

```text
read_grid ...
read_surf ...
species ...
mixture ...

fix ... background ...
compute ... nearest_surf/grid ...
global pusher ...
fix ... volume/chem/adas ...
fix ... surface/emit/recycle ...

run ...
```

The command pages follow the same structure:

- command names use their exact OpenEdge input form (`fix background`,
  `fix volume/chem/adas`, and `global pusher`)
- syntax blocks show the exact input form first
- sections after syntax explain behavior, restrictions, defaults, and
  examples

## Build

OpenEdge must be built out of source with `-DPKG_OPENEDGE=ON`.

::::{tab-set}

:::{tab-item} Intel MPI
```bash
mkdir -p ~/buildOpenEdge && cd ~/buildOpenEdge
source /opt/intel/oneapi/setvars.sh --force
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake/ \
      -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
      -DPKG_OPENEDGE=ON
make -j$(nproc)
```
:::

:::{tab-item} Kokkos / CUDA
```bash
mkdir -p ~/buildOpenEdgeGPU && cd ~/buildOpenEdgeGPU
cmake -C ../OpenEdge/cmake/presets/kokkos_cuda.cmake ../OpenEdge/cmake/ \
      -DPKG_OPENEDGE=ON
make -j$(nproc)
```
:::

::::

The binary is `~/buildOpenEdge/src/spa_mpi` (or `spa_kokkos_cuda`).

## Typical use cases

- **Tokamak edge impurity transport** with a plasma background loaded
  from converter-generated `plasma.h5`.
- **Wall recycling and PMI/PWI studies** with OpenEdge wall models
  attached to selected surface groups.
- **Neutral-source closure and source-term generation** for coupled
  SOLPS or Gkeyll workflows.
- **Liquid-metal plasma-facing component studies** with evaporation and
  surface transport models.

## Coordinate layouts

Two 2D layouts and one 3D layout are supported.

:::{list-table}
:header-rows: 1
:widths: 25 45 30

* - Layout
  - Description
  - Notes
* - **2D axisymmetric** *(preferred)*
  - `boundary o ao p`, `x = Z`, `y = R`
  - True cylindrical volumes + `$2\pi R L$` surface area.
* - **3D Cartesian**
  - Full domain or wedge
  - No post-processing correction needed.
:::

Use the axisymmetric layout for tokamak/divertor geometries; use 3D
Cartesian for full-vessel runs or non-axisymmetric features.

## Manual organization

The remainder of this manual is grouped by input-language and physics family:

- **Core commands** documents domain, grid, surface, particle, control-flow,
  run, and output commands used by OpenEdge decks.
- **Fixes** documents runtime models attached through `fix`.
- **Computes** documents OpenEdge field, geometry, and diagnostic
  computes.
- **Surface collide / react** documents wall-collision and wall-reaction
  models.
- **Globals** documents shared runtime controls such as `global pusher`.
- **Input data** documents converter outputs and common file formats.
- **Performance** collects notes on scalability and grid strategy.

## Citing OpenEdge

If you use OpenEdge in published work, cite the OpenEdge methods papers
*(to be added)* and the foundational runtime reference. A BibTeX snippet
will appear here once the companion papers are finalised.

```{toctree}
:caption: Core input language
:maxdepth: 2

manual_index
```

```{toctree}
:caption: Fixes
:maxdepth: 1

fixes/background
fixes/coulomb_background
fixes/coulomb_binary
fixes/cross_field_diffusion
fixes/material
particulate
fixes/particulate_charge
fixes/particulate_drag
fixes/particulate_emit
fixes/particulate_thermal
fixes/bfield_particle
fixes/efield_particle
fixes/force_gravity
fixes/force_thermal
fixes/liquid_metal
fixes/particle_weight
fixes/reflect_psi
fixes/surface_emit_puff
fixes/surface_emit_recycle
fixes/surface_emit_source
fixes/volume_chem_adas
```

```{toctree}
:caption: Computes
:maxdepth: 1

fixes/compute_grid
fixes/compute_grid_weighted
fixes/compute_nearest_surf_grid
fixes/efield_plasma
fixes/surface_physical_sputter
fixes/volume_emissivity
```

```{toctree}
:caption: Surface collide / react
:maxdepth: 1

fixes/surf_collide_diffuse
fixes/surf_collide_particulate_bounce
fixes/surf_react_surface_pwi
```

```{toctree}
:caption: Globals
:maxdepth: 1

fixes/pusher
fixes/sheath
```

```{toctree}
:caption: Input data
:maxdepth: 1

converters/plasma_h5_schema
converters/wall_geometry
database_schema
```

```{toctree}
:caption: Performance
:maxdepth: 1

performance/grid_refinement
migration/axi_cookbook
```
