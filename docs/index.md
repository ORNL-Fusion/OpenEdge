---
myst:
  html_meta:
    "description": "OpenEdge — a plasma–material interaction package for SPARTA"
---

```{raw} html
<div class="oe-hero">
  <h1>OpenEdge</h1>
  <p class="oe-tagline">
    A plasma–material interaction package for the
    <a href="https://sparta.github.io/">SPARTA DSMC framework</a>.
    Fusion-edge impurity transport, PMI/PWI physics, and tokamak-scale
    diagnostics — built on the SPARTA particle/grid/surface
    infrastructure and scalable to GPU clusters via Kokkos.
  </p>
  <div class="oe-badge-row">
    <span class="oe-badge primary">C++11 · MPI · Kokkos</span>
    <span class="oe-badge">SPARTA package</span>
    <span class="oe-badge">Axi / 2D / 3D</span>
    <span class="oe-badge">ADAS · Janev</span>
  </div>
</div>
```

::::{grid} 1 2 3 3
:gutter: 3
:margin: 0 0 4 0

:::{grid-item-card} Impurity transport
:link: fixes/force_thermal
:link-type: doc

Boris and guiding-center (GCA) pushers with Littlejohn corrections,
Braginskii thermal forces, anomalous cross-diffusion.
:::

:::{grid-item-card} Neutral chemistry
:link: fixes/volume_chem_adas
:link-type: doc

Volumetric ionization, recombination, charge exchange, and
dissociation using ADAS and Janev rate coefficients.
:::

:::{grid-item-card} Sheath & surface
:link: fixes/surface_physical_sputter
:link-type: doc

Kick and spatially-resolved sheath models, per-surface PMI tallies,
sputter yield tables, and reaction networks at the wall.
:::

:::{grid-item-card} Wall recycling
:link: fixes/surface_emit_recycle
:link-type: doc

Bohm-flux-driven recycling, hard-capped puffs, and surface reactions
with cosine re-emission.
:::

:::{grid-item-card} Liquid metals
:link: fixes/liquid_metal
:link-type: doc

MHD shallow-water Li film with Antoine evaporation, Hertz–Knudsen
flux, and Arrhenius ad-atom desorption.
:::

:::{grid-item-card} External coupling
:link: coupling/library_api
:link-type: doc

Library API for Python, Gkeyll, and SOLPS outer loops. Converters
for SOLPS, SOLEDGE3X, and OEDGE inputs.
:::

::::

## What OpenEdge is

OpenEdge is a SPARTA *package* — it reuses SPARTA's input script,
particle / grid / surface infrastructure, and MPI + Kokkos parallelism.
This manual documents only what OpenEdge adds on top. For base commands
(`species`, `create_box`, `create_particles`, `read_surf`, `run`,
`dump`, `fix balance`, …) see the
[SPARTA manual](https://sparta.github.io/doc/Manual.html).

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
  - True cylindrical volumes + `2πRL` surface area.
* - **2D Cartesian slab** *(legacy)*
  - `x = R`, `y = Z`, thin periodic slab in `z`
  - Grid/surf diagnostics need a `2π·R̄` post-multiply.
* - **3D Cartesian**
  - Full domain or wedge
  - No post-processing correction needed.
:::

New decks should use the axi layout — see the
[Axi cookbook](migration/axi_cookbook.md) for migration of legacy
Cartesian inputs.

## Citing OpenEdge

If you use OpenEdge in published work, please cite the two methods
papers *(to be added)* and the SPARTA reference. A BibTeX snippet will
appear here once the companion papers are finalised.

```{toctree}
:caption: Fixes & Computes
:maxdepth: 1
:hidden:

fixes/volume_chem_adas
fixes/cross_field_diffusion
fixes/surface_emit_puff
fixes/surface_emit_recycle
fixes/surface_emit_source
fixes/liquid_metal
fixes/force_thermal
fixes/surf_react_surface_pwi
fixes/efield_plasma
fixes/surface_physical_sputter
fixes/volume_emissivity
fixes/sheath
fixes/pusher
```

```{toctree}
:caption: Input converters
:maxdepth: 1
:hidden:

converters/plasma_h5_schema
converters/wall_geometry
database_schema
```

```{toctree}
:caption: Coupling
:maxdepth: 1
:hidden:

coupling/library_api
```

```{toctree}
:caption: Migration
:maxdepth: 1
:hidden:

migration/axi_cookbook
```

```{toctree}
:caption: Performance
:maxdepth: 1
:hidden:

performance/grid_refinement
```
