# CLAUDE.md — OpenEdge development guide

OpenEdge is a plasma-edge particle transport code built as a package on top
of [SPARTA](https://sparta.github.io/) (DSMC framework). It simulates
impurity ion transport, plasma–material interactions, and dust/droplet
dynamics in magnetic fusion devices using Boris and GCA particle pushers
with background plasma and magnetic field inputs.

- **Language:** C++11, with Python scripts for pre/post-processing.
- **Build system:** CMake (out-of-source required).
- **Parallelism:** MPI (Intel MPI on the primary cluster); Kokkos CUDA for GPU.

## Repository layout

```
OpenEdge/
  cmake/presets/       CMake preset files (mpi.cmake, kokkos_cuda.cmake, ...)
  src/                 Compiled source (SPARTA base + OpenEdge overrides)
  src/OPENEDGE/        OpenEdge package reference copies (authoritative)
  src/KOKKOS/          Kokkos GPU variants
  database/            External data (ADAS rates, PEC tables, surface models)
  docs/                OpenEdge-specific reference docs (see index below)
  tools/               Converters and coupling drivers
  examples/            Test cases and validation examples
  lib/                 External libraries (Kokkos, etc.)
```

## Build

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

Binary: `~/buildOpenEdge/src/spa_mpi`.

## Architecture gotchas

### Three-copy rule for override files

OpenEdge overrides some SPARTA base files. These exist in three places:

| Location | Role |
|----------|------|
| `src/update.cpp` (`.h`) | **COMPILED** — what actually builds and runs |
| `src/OPENEDGE/update.cpp` | **Reference** — authoritative OpenEdge version |
| `src/src/include/update.h` | **NOT compiled** — SPARTA's original header |

**When modifying ANY file that exists in both `src/` and `src/OPENEDGE/`,
ALWAYS update BOTH copies.** The build compiles from `src/`, but
`src/OPENEDGE/` is the package reference. If you only edit one, the build
silently uses the old version. Applies to all `.cpp` and `.h` that exist
in both places (`fix_chem_adas.cpp`, `update.cpp`, `sheath_models.cpp`, …).

New OpenEdge-only files only need to exist in `src/OPENEDGE/` — they get
copied to `src/` on install automatically.

### Coordinate convention

Two SPARTA-level layouts are supported for tokamak / device geometries:

- **2D Cartesian (legacy)** — `boundary o o p`, `dimension 2`, `x = R`,
  `y = Z`. SPARTA treats the box as a Cartesian slab of thickness
  `dz_box` (typically 0.1 m, periodic). All `compute grid nrho`,
  `compute surf flux`, `compute boundary` quantities are per-radian-of-
  toroidal-wedge — **they need a `2π·R̄` post-multiply** to compare
  against full-3D codes (EIRENE, SOLPS, Gkeyll).

- **SPARTA-native axisymmetric (preferred)** — `boundary o ao p`,
  `dimension 2`, `x = Z` (axial), `y = R` (radial, must start at 0),
  `z = phi`. SPARTA uses true cylindrical cell volumes and
  `surf->axi_line_size()` (= `2π·R·L`) for surface area, so all
  per-volume and per-area diagnostics are full-3D out of the box. No
  post-multiply.
  - `boundary`: the `a` token must sit at `ylo`, and `boxlo[1]` must
    be 0 (`domain.cpp:174–182`, `create_box.cpp:55`). Use the 2-char
    `ao` for the y-arg so SPARTA reads `ylo='a'` (axis) and
    `yhi='o'` (outflow).
  - `create_box -1.4 1.4 0 2.4 -0.05 0.05` (xlo,xhi,ylo,yhi,zlo,zhi).

**Slot-mapping helper** — `src/OPENEDGE/openedge_geom.h` provides
`sparta_to_RZ()`, `RZphi_force_to_sparta()`, `sparta_v_to_RZphi()`. All
OpenEdge consumers that read particle `(R,Z)` or decompose cylindrical
fields onto SPARTA slots route through these, so they work in any of
2D Cart / 2D axi / 3D Cart without per-fix conditionals.

**B/E/V-field source convention**: when users pass `bx by bz` (or
`ex ey ez`, `vx vy vz`) sources to a fix or compute, the values must be
in **SPARTA slot order** (matching the velocity slots).
`compute plasma/fields` does the projection automatically — feeding
`c_cplasma[bx_col]` etc. into `fix thermal_force` or `fix efield/grid`
works in either coord layout without further user intervention.

**Edge-code converters always emit axi** — SOLPS / SOLEDGE3X / OEDGE
are axisymmetric edge codes, so `convert_solps_plasma.py`,
`convert_s3x_plasma.py`, and `create_surf_from_solps.py` unconditionally
write `wall.surf` line endpoints as `(Z, R)` to line up with SPARTA's
true axisymmetric slot mapping. There is no Cartesian-output toggle.
Legacy 2D Cartesian decks must be migrated to true axi first — see
[`docs/migration/axi_cookbook.md`](docs/migration/axi_cookbook.md).

**Tests by layout (as of 2026-04-20):**
- Axi: `test_diii_d_neutrals` (pilot)
- Cart 2D (legacy, awaiting migration): `test_west_axi`,
  `test_west_neutrals`, `test_west_timedep`, `test_d3d_walldyn`,
  `test_d3d_mateja`, `test_evaporation`, `test_solps_coupling`,
  `test_gca`, `test_neutral_transport`
- 3D Cart (unaffected): `test_west_3d`, etc.
- True 1D slab (unaffected): `test_slab_stangeby2000`

### Particle properties

- `particles[i].mass` is **zero** for gas-phase particles — it is only
  used for droplets. Always use `particle->species[isp].mass` for
  molecular mass and `particle->species[isp].charge` for charge state.
- Species are defined in `.species` files and loaded via the `species`
  command.

### Field lookups

- All field lookups (B, plasma, gradients) should use **point queries**
  at particle position: `cp->query_plasma_at_point(x)`,
  `cp->query_bfield_at_point(x)`.
- Do not fall back to cell-center arrays for per-particle computations.
- Cylindrical → Cartesian B-field: use particle `(x,y)` to compute
  `cos(phi)`, `sin(phi)` for the rotation.

### MPI trap: Allreduce in fix init

When writing a new fix whose `init()` prints a diagnostic, **every
`MPI_Allreduce` must be called on every rank**, not inside
`if (comm->me == 0)`.

```cpp
// BUG: only rank 0 calls Allreduce -> other ranks deadlock
// at the next collective (run setup, fix chem init, etc.)
if (comm->me == 0) {
  double local = ...;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  printf("diagnostic = %.3e\n", global);
}
```

Correct — run the Allreduce unconditionally, gate only the `printf`:

```cpp
double local = compute_something();
double global;
MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
if (comm->me == 0) printf("diagnostic = %.3e\n", global);
```

Symptom: run hangs with no stats line ever printed, log stops right
after the fix's init prints. Single-rank runs are fine; only shows up
under MPI.

## MPI launch on mora

The OpenEdge binary is linked against Intel MPI. The system `PATH` points
to linuxbrew's `mpirun` by default, which is a different MPI ABI —
launching with the wrong `mpirun` produces N independent singleton
processes (`Running on 1 MPI task(s)` printed N times) instead of one
N-rank job.

```bash
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 8 ~/buildOpenEdge/src/spa_mpi -in in.case
```

To verify:
```bash
which mpirun           # should be /opt/intel/oneapi/mpi/.../bin/mpirun
echo $I_MPI_ROOT       # should be non-empty
```

## Features overview

Pointers to detailed per-feature docs. Each doc covers syntax, physics,
and usage patterns.

### Pushers / sheaths

- **Boris / GCA hybrid pusher.** Boris with `global boris_subcycles N`;
  GCA (Guiding Center Approximation) RK4 with Littlejohn corrections via
  `global gca …`. Automatic switching via `gca_switch_factor`.
- **Sheath models.** Kick (`global sheath ... kick yes`) — velocity boost
  at wall collision, recommended for IEADs, no per-subcycle E-field.
  Spatial (`global sheath ... model <name>`) — per-subcycle E-field,
  models: `borodkina`, `coulette_manfredi`, with overshoot guard.
- **Surface collision.** `surf_collide vanish`, `diffuse`, `toroidal`
  (phi-periodic wedge rotation).

### Plasma I/O

- **Plasma.h5 schema (mesh-only)** — [`docs/converters/plasma_h5_schema.md`](docs/converters/plasma_h5_schema.md).
  Three top-level groups: `/equilibrium`, `/ion_species`, `/mesh`.
  Query via `fix plasma/data`.
- **Wall geometry from SOLPS** — [`docs/converters/wall_geometry.md`](docs/converters/wall_geometry.md).
  `--wall-source` options: `mesh-extra` (default), `b2`, `eirene`.
- **Axi migration cookbook** — [`docs/migration/axi_cookbook.md`](docs/migration/axi_cookbook.md).

### Transport fixes

- **`fix thermal_force`** — Braginskii ion + electron thermal forces on
  impurity ions. [`docs/fixes/thermal_force.md`](docs/fixes/thermal_force.md).
- **`fix cross_diffusion`** — anomalous perpendicular diffusion + pinch.
  [`docs/fixes/cross_diffusion.md`](docs/fixes/cross_diffusion.md).
- **Plasma-native E-field** — `compute plasma/fields` reads `E = −∇ϕ`
  from converter. [`docs/fixes/efield_plasma.md`](docs/fixes/efield_plasma.md).

### Neutral transport (EIRENE replacement)

- **`fix chem/adas`** — volumetric ionization / recombination / CX /
  dissociation with ADAS + Janev rates. 20-col per-cell source tally.
  [`docs/fixes/chem_adas.md`](docs/fixes/chem_adas.md).
- **`fix emit/surf/puff`** — hard-capped surface emission for Mode A
  puffs. [`docs/fixes/emit_surf_puff.md`](docs/fixes/emit_surf_puff.md).
- **`fix emit/surf/recycle`** — wall-recycling neutral source, Bohm
  flux × recycling coeff. [`docs/fixes/emit_surf_recycle.md`](docs/fixes/emit_surf_recycle.md).
- **`surf_react recycle`** — surface recycling with cosine re-emission.
  [`docs/fixes/surf_react_recycle.md`](docs/fixes/surf_react_recycle.md).

### Diagnostics

- **`compute photon_emissivity/grid`** — synthetic line emission
  `ε = ne · nz · PEC(Te, ne)` using ColRadPy PEC tables.
  [`docs/fixes/photon_emissivity.md`](docs/fixes/photon_emissivity.md).

### Surface / liquid-metal models

- **`fix liquid_metal`** — MHD shallow-water Li film with evaporation
  (Antoine + Hertz-Knudsen) and ad-atom (Arrhenius desorption).
  [`docs/fixes/liquid_metal.md`](docs/fixes/liquid_metal.md).

### External coupling

- Library API for Python/Gkeyll/SOLPS outer loops — see
  [`docs/coupling/library_api.md`](docs/coupling/library_api.md).

### Performance

- **Grid refinement near surface sources** — `adapt_grid` + `fix adapt` +
  `fix balance rcb part`. [`docs/performance/grid_refinement.md`](docs/performance/grid_refinement.md).

## Testing

Test cases live in `examples/test_*/`. Each has a README with run
instructions. Key validated tests:

- `test_iead` — IEAD validation (sheath kick + spatial vs. Fortran ref)
- `test_sheath` — analytical sheath profile (Borodkina model)
- `test_gca` — GCA pusher vs. Boris, mu conservation
- `test_droplet` — droplet drag, charging, viscous forces
- `test_collide` — Nanbu collision operator
- `test_gravity_3d` — gravity force validation
- `test_diii_d_neutrals` — axi pilot, EIRENE neutral-transport benchmark

Run a test:
```bash
cd examples/test_iead
python3 create_case.py
./run_all.sh
python3 compare_iead.py
```

## Coding conventions

- C++11 standard, no newer features.
- SPARTA naming: classes `CamelCase`, files `snake_case` with prefix
  (`fix_`, `compute_`, `surf_collide_`, `surf_react_`).
- Register new commands via style macros (`FixStyle`, `ComputeStyle`,
  `SurfCollideStyle`) inside the header's `#ifdef` block.
- Physical constants: define locally in an anonymous namespace
  (`QE, AMU, EPS0, ME`) rather than using a global header.
- Error handling: `error->all(FLERR, "message")` for fatal errors,
  `error->warning(FLERR, "message")` for warnings.
- MPI: never alias input/output buffers in `MPI_Allreduce` — use
  `MPI_IN_PLACE` or separate buffers.

## Git conventions

- Commit messages: imperative mood, concise first line, details in body.
- Main branch: `main`.
- Feature branches: descriptive names (e.g. `seed-timedep-multilayer`,
  `neutral`).
