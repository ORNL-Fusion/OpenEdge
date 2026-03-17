# CLAUDE.md — OpenEdge development guide

## Project overview

OpenEdge is a plasma-edge particle transport code built as a package on top of
[SPARTA](https://sparta.github.io/) (DSMC framework). It simulates impurity
ion transport, plasma-material interactions, and dust/droplet dynamics in
magnetic fusion devices using Boris and GCA particle pushers with background
plasma and magnetic field inputs.

**Language:** C++ (C++11), with Python scripts for pre/post-processing.
**Build system:** CMake (out-of-source build required).
**Parallelism:** MPI (Intel MPI on the primary cluster).

## Repository structure

```
OpenEdge/
  cmake/presets/       CMake preset files (mpi.cmake, kokkos_cuda.cmake, ...)
  src/                 Compiled source (SPARTA base + OpenEdge overrides)
  src/OPENEDGE/        OpenEdge package reference copies (authoritative source)
  src/KOKKOS/          Kokkos GPU variants
  database/            External data (ADAS rates, PEC tables, surface models)
  examples/            Test cases and validation examples
  lib/                 External libraries (Kokkos, etc.)
```

## Build instructions

**Always build out-of-source.** Never build inside `src/`.

```bash
mkdir -p ~/buildOpenEdge && cd ~/buildOpenEdge

# On ORNL cloud (mora):
source /opt/intel/oneapi/setvars.sh --force
LD_LIBRARY_PATH= HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial cmake \
    -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake/ \
    -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
    -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial \
    -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE -DPKG_OPENEDGE=ON

LD_LIBRARY_PATH= make -j$(nproc)
```

Binary: `~/buildOpenEdge/src/spa_mpi`

## Key architecture patterns

### Three copies of override files

OpenEdge overrides some SPARTA base files. These exist in three places:

| Location | Role |
|----------|------|
| `src/update.cpp` (and .h) | **COMPILED** — what actually builds and runs |
| `src/OPENEDGE/update.cpp` | **Reference** — authoritative OpenEdge version |
| `src/src/include/update.h` | **NOT compiled** — SPARTA's original header |

**CRITICAL: When modifying ANY file that exists in both `src/` and
`src/OPENEDGE/`, ALWAYS update BOTH copies.** The build compiles from `src/`,
but `src/OPENEDGE/` is the package reference. If you only edit one, the other
goes stale and the build will silently use the old version. This applies to
all `.cpp` and `.h` files that have copies in both locations (e.g.,
`fix_chem_adas.cpp`, `update.cpp`, `sheath_models.cpp`, etc.).

New OpenEdge-only files only need to exist in `src/OPENEDGE/` (they get copied
to `src/` on install, and the cmake build handles this automatically).

### Particle properties

- `particles[i].mass` is **zero** for gas-phase particles — it is only used
  for droplets. Always use `particle->species[isp].mass` for molecular mass
  and `particle->species[isp].charge` for charge state.
- Species are defined in `.species` files and loaded via the `species` command.

### Field lookups

- All field lookups (B, plasma, gradients) should use **point queries** at
  particle position: `cp->query_plasma_at_point(x)`,
  `cp->query_bfield_at_point(x)`.
- Do not fall back to cell-center arrays for per-particle computations.
- Cylindrical-to-Cartesian conversion for B-field: use particle (x,y) to
  compute cos(phi), sin(phi) for the rotation.

### Sheath models

Two approaches for sheath electric fields:

- **Kick mode** (`global sheath ... kick yes`): applies sheath energy as
  velocity boost at wall collision. Recommended for IEADs. No per-subcycle
  E-field computation.
- **Spatial mode** (`global sheath ... model <name>`): spatially-resolved
  E-field evaluated each Boris subcycle. Models: `borodkina`,
  `coulette_manfredi`. Has overshoot guard to prevent reverse-field energy
  loss when particles cross the wall during subcycling.

### Surface collision models

- `surf_collide vanish` — absorb particle (with optional CSV logging)
- `surf_collide diffuse` — thermal re-emission
- `surf_collide toroidal` — phi-periodic boundary rotation for toroidal wedges

### Boris / GCA hybrid pusher

- Boris pusher with configurable subcycles (`global boris_subcycles N`)
- GCA (Guiding Center Approximation) pusher with RK4 integration, activated
  via `global gca ...` with Littlejohn corrections
- Automatic switching between Boris and GCA based on `gca_switch_factor`

### Synthetic diagnostics

- **`compute photon_emissivity/grid`** — per-grid volumetric photon
  emissivity: `ε = ne * nz * PEC(Te, ne)` [photons/m³/s/sr].
  Uses per-particle `pweight` for weighted density (`nz`), Te/ne from a
  `compute plasma/fields`, and a PEC table from an HDF5 file.
  ```
  compute ID photon_emissivity/grid group mix \
          pec_file PATH plasma_compute CID [pec_units cm3s|m3s]
  ```
  - PEC HDF5 layout: `te` or `te_grid` (1D), `ne` or `ne_grid` (1D),
    plus any 2D dataset (auto-detected as PEC values).
  - Default `pec_units cm3s` (ADAS convention); use `m3s` if already SI.
  - PEC files live in `database/pec/` (generated by
    [ColRadPy](https://github.com/johnson-c/ColRadPy)).
  - Output: one column per species group in the mixture. Use with
    `fix ave/grid` + `dump grid` for time-averaged emissivity maps.

## Testing

Test cases live in `examples/test_*/`. Each has a README with run instructions.
Key validated tests:

- `test_iead` — IEAD validation (sheath kick + spatial, vs Fortran reference)
- `test_sheath` — Analytical sheath profile validation (Borodkina model)
- `test_gca` — GCA pusher vs Boris, mu conservation
- `test_droplet` — Droplet transport (drag, charging, viscous forces)
- `test_collide` — Nanbu collision operator
- `test_gravity_3d` — Gravity force validation

Run a test:
```bash
cd examples/test_iead
python3 create_case.py
./run_all.sh
python3 compare_iead.py
```

## Coding conventions

- C++11 standard, no newer features
- SPARTA naming: classes use CamelCase, files use snake_case with
  prefix (`fix_`, `compute_`, `surf_collide_`, `surf_react_`)
- New commands registered via style macros (e.g., `FixStyle`, `ComputeStyle`,
  `SurfCollideStyle`) in the header file's `#ifdef` block
- Physical constants: define locally in anonymous namespace (QE, AMU, EPS0, ME)
  rather than using a global header
- Error handling: use `error->all(FLERR, "message")` for fatal errors,
  `error->warning(FLERR, "message")` for warnings
- MPI: never alias input/output buffers in MPI_Allreduce (use MPI_IN_PLACE
  or separate buffers)

## Git conventions

- Commit messages: imperative mood, concise first line, details in body
- Main branch: `main`
- Feature branches: descriptive names (e.g., `seed-timedep-multilayer`)
