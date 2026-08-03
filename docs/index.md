---
myst:
  html_meta:
    "description": "OpenEdge — a plasma–material interaction package for SPARTA"
---

# OpenEdge

OpenEdge is a plasma-material interaction package for the
[SPARTA DSMC framework](https://sparta.github.io/). It extends SPARTA
with fusion-edge plasma backgrounds, impurity transport, sheath and
surface-interaction models, neutral chemistry, liquid-metal models, and
coupling utilities for outer-loop workflows.

## Description

OpenEdge is a SPARTA *package* — it reuses SPARTA's input script,
particle / grid / surface infrastructure, and MPI + Kokkos parallelism.
This manual documents only what OpenEdge adds on top. For base commands
(`species`, `create_box`, `create_particles`, `read_surf`, `run`,
`dump`, `fix balance`, …) see the
[SPARTA manual](https://sparta.github.io/doc/Manual.html).

In practice, read this manual the same way you would read the SPARTA
manual: SPARTA owns the base grammar and simulation loop, while
OpenEdge adds new `fix`, `compute`, `global`, converter, and coupling
features on top of that base.

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

OpenEdge input decks should be read as SPARTA input decks with
OpenEdge-specific additions. A typical setup has the form

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

The documentation follows that same style where possible:

- command names are written in SPARTA grammar (`fix background`,
  `fix volume/chem/adas`, `global pusher`)
- syntax blocks show the exact input form first
- sections after syntax explain behavior, restrictions, defaults, and
  examples

## Build

OpenEdge must be built out-of-source as a SPARTA package
(`-DPKG_OPENEDGE=ON`).

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
- **Wall recycling and PMI/PWI studies** with OpenEdge wall fixes
  attached to SPARTA surface groups.
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

The remainder of this manual is grouped by command family and data path:

- **Fixes** documents runtime models attached through `fix`.
- **Computes** documents OpenEdge field, geometry, and diagnostic
  computes.
- **Surface collide / react** documents wall-collision and wall-reaction
  models.
- **Globals** documents shared runtime controls such as `global pusher`.
- **Input data** documents converter outputs and common file formats.
- **Coupling** documents the library API and external-driver workflow.
- **Performance** collects notes on scalability and grid strategy.

## Citing OpenEdge

If you use OpenEdge in published work, please cite the two methods
papers *(to be added)* and the SPARTA reference. A BibTeX snippet will
appear here once the companion papers are finalised.

```{toctree}
:caption: Fixes
:maxdepth: 1

fixes/background
fixes/bfield_grid
fixes/bfield_particle
fixes/coulomb_background
fixes/coulomb_binary
fixes/cross_field_diffusion
fixes/droplet_charge
fixes/droplet_drag
fixes/droplet_emit
fixes/droplet_evaporate
fixes/droplet_viscous
fixes/efield_grid
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
:caption: Coupling
:maxdepth: 1

coupling/library_api
```

```{toctree}
:caption: Performance
:maxdepth: 1

performance/grid_refinement
```
