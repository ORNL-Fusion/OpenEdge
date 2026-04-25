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
`c_cplasma[bx_col]` etc. into `fix force/thermal` or `fix efield/grid`
works in either coord layout without further user intervention.

**Edge-code converters always emit axi** — SOLPS / SOLEDGE3X / OEDGE
are axisymmetric edge codes, so `convert_solps_plasma.py`,
`convert_s3x_plasma.py`, and `create_surf_from_solps.py` unconditionally
write `wall.surf` line endpoints as `(Z, R)` to line up with SPARTA's
true axisymmetric slot mapping. There is no Cartesian-output toggle.
Legacy 2D Cartesian decks must be migrated to true axi first — see
[`docs/migration/axi_cookbook.md`](docs/migration/axi_cookbook.md).

**Helper-only coordinate conversion (policy)** — physics code reads
and writes physical `(R, Z, phi)`; conversion to/from SPARTA slot
order happens only through the `openedge_geom.h` helpers
(`sparta_to_RZ`, `sparta_v_to_RZphi`, `RZphi_force_to_sparta`). Never
assume `x = R, y = Z` in 2D outside that header — it breaks the axi
layout. Applies to pusher / field lookup / stencil builders / box-
bound logic alike. Grep for `dim == 2` or `xyz[0]` in new code as a
self-review tripwire.

**Unified wall-normal convention (2026-04-21)** — every surface fix in
OpenEdge now uses the SPARTA canonical convention: `normal[]` points
INTO the fluid (plasma), and emission / outgoing-reflection velocity
goes along `+normal`. Applies to `fix emit/surf`, `surface/emit/puff`,
`surface/emit/recycle`, `surface/emit/sputter`, `surf_collide diffuse`,
`surf_react surface/pwi`. The converters write `wall.surf` with walk
order giving inward normals by default, so **no `read_surf ... invert`
on wall.surf**. The old `-normal` quirk in `fix_surface_emit_recycle`
is fixed. Inner-boundary `core.surf` pushes (surface/emit/sputter from
the separatrix) still need `invert` — that surface is traversed the
other way.

**Inner core-absorb boundary (`core.surf`, 2026-04-23)** —
`convert_solps_plasma.py` and `convert_s3x_plasma.py` both accept
`--core-out <path>` and `--psi-norm-core <level>` to trace a
psi_norm = const contour around the magnetic axis from the embedded
`/equilibrium/*` and write it as a SPARTA surface file in axi (Z, R)
layout. Uses `tools/extract_psi_contour.py::write_core_surf_from_plasma_h5`
under the hood (decimates to 5 mm min segment so SPARTA cell-marking is
robust). Default `psi_norm = 0.90` keeps the contour safely inside the
wall near the X-point; **0.95 commonly dips into the private-flux
region and intersects the divertor wall**, which fails
SPARTA's grid flood-fill with `Cell type mis-match when marking on
self`. If you see that error on combined wall + core, lower
`--psi-norm-core` or verify with `ray-casting` that every core vertex
is strictly inside the wall polygon before running.

**SOLEDGE3X `config/{r, z}` axis orientation (2026-04-23)** — 3MW and
other SOLEDGE3X runs ship `mesh.h5` with 2D (r, z) meshgrid-style
arrays where `r[i, j]` varies along rows (i indexes R) and `z[i, j]`
varies along columns (j indexes Z). `convert_s3x_plasma.py` now
auto-detects which axis carries the R variation and transposes `psi`
accordingly when writing `/equilibrium/{r, z, psi}`. Older converter
output had degenerate `equilibrium/r` / `equilibrium/z` (all values
identical) for this orientation — regen any plasma.h5 written before
the 2026-04-23 fix.

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

- **Charged-particle pusher.** Single `global pusher` keyword tree:
  `mode boris|hybrid` (Boris full-orbit or Boris/GCA hybrid),
  `plasma <ID>` for the upstream provider, `subcycles N`,
  `gca_switch <factor>` for hybrid switching, `dump`/`dump_every`,
  `bad_dt_check`/`bad_dt_limit`, and a nested `sheath off|kick|spatial`
  with `geom <nearest_surf/grid-ID>` and `mD_amu <amu>`. Sheath
  internals (dmax, pot_mult, model blend) are auto.
  [`docs/fixes/pusher.md`](docs/fixes/pusher.md),
  [`docs/fixes/sheath.md`](docs/fixes/sheath.md).
- **Sheath physics.** Kick mode applies the wall potential drop as a
  velocity boost at wall collision (recommended for IEADs). Spatial
  mode integrates the sheath E-field per subcycle along Boris steps.
  Both share the auto-blended Coulette-Manfredi (close to wall) +
  Borodkina tail (s > 60 λ_D). Boltzmann `ne · exp(-φ/Te)` correction
  flows into the per-particle pcache automatically, so
  `fix volume/chem/adas` near-wall rates fall off without separate
  plumbing.
- **Surface collision.** `surf_collide vanish`, `diffuse`, `toroidal`
  (phi-periodic wedge rotation).

### Plasma I/O

- **Plasma.h5 schema (mesh-only)** — [`docs/converters/plasma_h5_schema.md`](docs/converters/plasma_h5_schema.md).
  Three top-level groups: `/equilibrium`, `/ion_species`, `/mesh`.
  `/ion_species/elements` (added 2026-04-21) carries the
  charge-state-stripped element symbol per ion. Query via
  `fix background`.
- **Wall geometry from SOLPS** — [`docs/converters/wall_geometry.md`](docs/converters/wall_geometry.md).
  `--wall-source` options: `mesh-extra` (default), `b2`, `eirene`.
- **Axi migration cookbook** — [`docs/migration/axi_cookbook.md`](docs/migration/axi_cookbook.md).

### Database path resolution

- `src/OPENEDGE/database_paths.{h,cpp}` — single source of truth for
  locating consolidated process data and per-element reaction lists.
  Lookup order: `OPENEDGE_ROOT` env var → compile-time
  `OPENEDGE_DATABASE_DIR` (set by CMake to `${repo}/database`) →
  literal `database` (cwd-relative).
- `resolve_processes_file()` → `${root}/database/processes.h5`. Single
  consolidated HDF5 carrying `/volume/adas/...` rate coefficients,
  `/volume/pec/...` photon-emission coefficients, `/surface/sputter/...`
  Eckstein yield tables, and `/surface/trim/...` reflection tables.
  Returns empty string if absent (consumers may then fall back to
  legacy text-file ADAS sources).
- `resolve_reactions_file("D", error)` →
  `${root}/database/adas/reactions/D.reactions`. Paths containing `/`
  or ending in `.reactions` pass through literal.
- Consumers: `fix volume/chem/adas`, `compute volume/emissivity/grid`,
  `surf_react surface/pwi`, `compute surface/physical/sputter`.

### Transport fixes

- **`fix force/thermal`** — Braginskii ion + electron thermal forces on
  impurity ions. [`docs/fixes/force_thermal.md`](docs/fixes/force_thermal.md).
- **`fix cross_field_diffusion`** — anomalous perpendicular diffusion + pinch.
  [`docs/fixes/cross_field_diffusion.md`](docs/fixes/cross_field_diffusion.md).
- **Plasma-native E-field** — `compute plasma/fields` reads `E = −∇ϕ`
  from converter. [`docs/fixes/efield_plasma.md`](docs/fixes/efield_plasma.md).

### Neutral transport (EIRENE replacement)

- **`fix volume/chem/adas`** — volumetric ionization / recombination / CX /
  dissociation with ADAS + Janev rates. 20-col per-cell source tally.
  Accepts element symbol or numeric Z as arg 3; reactions file arg 4
  accepts a literal path, an element symbol (resolves to
  `database/adas/reactions/<elem>.reactions`), or `auto` (same element
  as arg 3). Channel toggles: `ionization / recombination / cx /
  dissociation yes|no` disable by type without editing the file. Init
  prints per-channel active/skipped counts + missing species.
  [`docs/fixes/volume_chem_adas.md`](docs/fixes/volume_chem_adas.md).
- **`fix surface/emit/puff`** — hard-capped surface emission for Mode A
  puffs. [`docs/fixes/surface_emit_puff.md`](docs/fixes/surface_emit_puff.md).
- **`fix surface/emit/recycle`** — wall-recycling neutral source, Bohm
  flux × recycling coeff. Per-species emission temperature via
  `twall_species <sp> <T> ...` (e.g. atomic D at Franck-Condon 23 210 K,
  molecular D2 wall-thermalised at 500 K).
  [`docs/fixes/surface_emit_recycle.md`](docs/fixes/surface_emit_recycle.md).
- **`fix surface/emit/source`** — sputtered-impurity source driven by
  a `compute surface/physical/sputter` erosion-flux column. Uses `fix particle/weight`
  for spatially-variable emission rates.
  [`docs/fixes/surface_emit_source.md`](docs/fixes/surface_emit_source.md).
- **`surf_react surface/pwi`** — TRIM reflection + absorb-and-re-emit
  at walls; reads reflection tables from `database/processes.h5`.
  [`docs/fixes/surf_react_surface_pwi.md`](docs/fixes/surf_react_surface_pwi.md).

### Diagnostics

- **`compute surface/physical/sputter`** — per-surface Bohm flux, impact energy /
  angle, sputter yield, and gross erosion flux (`erosion_flux`; old
  alias `sputter_flux_total`). Post-2026-04-21 API resolves per-
  projectile yield tables automatically: `target W projectiles D,O`
  loads `D_on_W.h5` + `O_on_W.h5` and routes each plasma ion slot to
  the matching table via `slot_to_table`. Legacy
  `projectile_slots`/`mass_amu`/positional surface path still accepted.
  [`docs/fixes/surface_physical_sputter.md`](docs/fixes/surface_physical_sputter.md).
- **`compute volume/emissivity/grid`** — synthetic line emission
  `ε = ne · nz · PEC(Te, ne)` using ColRadPy PEC tables.
  [`docs/fixes/volume_emissivity.md`](docs/fixes/volume_emissivity.md).

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
