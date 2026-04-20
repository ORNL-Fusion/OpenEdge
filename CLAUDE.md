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

### Thermal forces

- **`fix thermal_force`** — Braginskii ion and electron thermal forces on
  impurity ions, applied as leapfrog half-kicks (START_OF_STEP + END_OF_STEP).
  ```
  fix ID thermal_force Nevery \
      bfield BxSRC BySRC BzSRC \
      [ion_thermal gradTiR_SRC gradTiZ_SRC [coeff VAL]] \
      [elec_thermal gradTeR_SRC gradTeZ_SRC [coeff VAL]]
  ```
  - Ion thermal force: `F = beta_i * Z^2 * e * grad_par(Ti)` (default
    `beta_i = 2.6`, Neu 1974 heavy-impurity limit).
  - Electron thermal force: `F = alpha_e * Z^2 * e * grad_par(Te)` (default
    `alpha_e = 0.71`, Braginskii Z_eff=1 limit).
  - B-field sources must be in SPARTA coordinate order (`bx`, `by`, `bz`),
    matching the velocity slot mapping. Temperature gradient sources are
    always cylindrical (`grad_ti_r`, `grad_ti_z`).
  - Both forces push impurities toward higher temperature (toward the core).

### Cross-field diffusion

- **`fix cross_diffusion`** — anomalous perpendicular diffusion and
  convective pinch for impurity ions, applied as position displacements
  at END_OF_STEP.
  ```
  fix ID cross_diffusion Nevery \
      bfield BxSRC BySRC BzSRC \
      [D_perp VAL | bohm TeSRC [scale VAL]] \
      [pinch Vr Vz]
  ```
  - Constant diffusion: `D_perp 1.0` gives D_⊥ = 1.0 m²/s.
  - Bohm diffusion: `bohm c_cplasma[Te_col] scale 0.1` gives
    D = scale × Te/(16eB). Default scale = 1.0.
  - Constant pinch: `pinch -50.0 0.0` adds a constant velocity in (R, Z).
  - Gradient-driven pinch: `gradient_pinch Cp neSRC gradNeR gradNeZ`
    gives V = Cp × D_⊥ × ∇_⊥(ne)/ne. Typical Cp = 1–3 (ITG turbulence).
  - 2D: displacement in poloidal perpendicular direction only.
    3D: two perpendicular directions via Gram-Schmidt.
  - Particles that diffuse outside the domain are reverted (no loss).
  - `compute plasma/fields` output columns `grad_ne_r`, `grad_ne_z` provide
    the electron density gradient (computed via finite differences).

### Ambipolar E-field

- `compute plasma/fields` computes the parallel ambipolar electric field
  `E_par = -(grad_pe . bhat) / (ne * e)` from electron pressure gradients.
  In FILE mode, `er`/`et`/`ez` (and `ex`/`ey`) output columns are now
  populated by decomposing `epar` into vector components via `E = epar * bhat`.
- To feed into the Boris pusher:
  ```
  compute cplasma plasma/fields all file plasma.h5 bfield.h5 ... ex ey ez
  fix fE efield/grid c_cplasma[ex_col] c_cplasma[ey_col] c_cplasma[ez_col]
  global efield grid fE 0
  ```

### Volumetric neutral reactions (EIRENE replacement)

- **`fix chem/adas`** — ADAS-based volumetric chemistry with competing
  Poisson channel selection. Supports ionization, recombination, charge
  exchange (CX), and dissociation reactions.
  ```
  fix ID chem/adas Nevery Z reactions_file \
      adas_dir PATH plasma TeSRC NeSRC \
      [mode kinetic|neutral] \
      [source_species <sp1> [sp2] ...] \
      [stop_on_exhaust yes|no] [exhaust_threshold <N>] \
      [units counts|rate|batch <N> <R_puff>|batch_fix <emit_id> <R_puff>]
  ```
  - **Rate styles:**
    - `A` (ADAS): bilinear interpolation on HDF5 rate tables
      `⟨σv⟩(Te, ne)` from ADF11 data (SCD/ACD/CCD).
    - `J` (Janev): 9-term polynomial `ln⟨σv⟩ = Σ bₙ (ln Te)ⁿ`
      from HYDHEL/Janev 1987. Used for molecular dissociation.
  - **Reaction types** in the reactions file:
    - `I`: ionization (charge state +1)
    - `R`: recombination (charge state −1)
    - `E`: charge exchange (charge state −1, CX with background H)
    - `D`: dissociation (1 reactant → 2 products, creates new particle)
  - CX rate data from ADAS CCD files (`ccd89_*.dat`), same format as
    ACD/SCD. Stored as `ChargeExchangeRateCoeff` in HDF5.
  - Dissociation uses deferred particle creation to avoid array
    invalidation during iteration.
  - After CX or dissociation, product velocity is re-sampled from a
    shifted Maxwellian at local Ti and bulk flow (EIRENE-like), when
    the per-particle plasma cache provides Ti, vpar, and B-field.
  - Per-type reaction tally printed every 10,000 steps.

  **Mode A (EIRENE semantics):** `mode neutral` deletes the neutral on
  ionization instead of relabeling it as an ion, and compresses the
  particle array via `dellist` at end_of_step. Combined with
  `source_species D D2 stop_on_exhaust yes [exhaust_threshold N]`, the
  run halts cleanly when the alive source population drops to `N`
  (default 0). `exhaust_threshold` skips the slow-converging fat tail.
  `exhaust_armed` guards against spurious exit during batch ramp-up.

  **20-column per-cell source tally (`array_grid`):** exposed as
  `f_ID[*][col]` for `dump grid` / `fix ave/grid` / library API. Layout
  is quantity-major across 4 reaction types (ion, rec, CX, dissoc):
  - cols  1– 4 : count per cell per reaction type
  - cols  5– 8 : sum of m·vx at reaction events [kg·m/s]
  - cols  9–12 : sum of m·vy
  - cols 13–16 : sum of m·vz
  - cols 17–20 : sum of ½·m·|v|² [J]

  **Tally units** (`units` keyword):
  - `counts` (default): raw cumulative totals since fix start.
  - `rate`: window-averaged SI rate (m⁻³s⁻¹, N/m³, W/m³); zeroed each
    Nevery window — use with `fix ave/grid ... ave running` for smooth
    SOLPS/Gkeyll input.
  - `batch N R_puff`: EIRENE-style MC. Each of the `N` trajectories
    carries weight `w = R_puff / N` [events/s] (divided by cell volume
    for source density). Cumulative across the run — the final value is
    the steady-state source for one puff.
  - `batch_fix <emit_id> R_puff`: same as `batch`, but `N` is pulled
    from the paired emit fix's cumulative emit count each step. Tracks
    ramp-up automatically so you don't hand-match `N` with the emit
    fix's `stop_at_np`.

  - **Data pipeline:** `database/adas/adas.py` converts ADF11 ASCII
    files to HDF5. Supports `acd` (recombination), `scd` (ionization),
    `ccd` (charge exchange). Set `ADAS_ADF11_DIR` or symlink into
    `database/adas/adf11/`.

  Reactions file example:
  ```
  D --> D+
  I A 1.0 0.0 0.0 0.0 0.0

  D+ --> D
  E A 1.0 0.0 0.0 0.0 0.0

  D2 --> D + D
  D J -2.787e+01 1.052e+01 -4.973e+00 1.451e+00 -3.063e-01 4.433e-02 -4.096e-03 2.160e-04 -4.929e-06
  ```

### Surface-source neutral puff

- **`fix emit/surf/puff`** — surface emission for Mode A puff launches.
  Like `fix emit/surf` but with a hard cap on total emitted particles so
  a one-shot EIRENE-style batch can terminate cleanly.
  ```
  fix ID emit/surf/puff mixture group \
      [n <N_per_step>] [normal yes|no] [stop_at_np <N_total>] \
      [perspecies yes|no] [region <rID>]
  ```
  - `n 200`: emit 200 particles per step (CONSTANT mode). `n 0` reverts
    to flow-based emission using the mixture vstream.
  - `normal yes`: inject along the surface normal (vs. mixture vstream).
  - `stop_at_np N`: latch emission off once `N` total particles have
    been emitted. Once latched, no further injection; combined with
    `fix chem/adas ... stop_on_exhaust yes` this gives a clean
    "puff once, track to completion, exit" cycle.
  - Pair with `units batch_fix ID R_puff` on `fix chem/adas` so the
    tally scales from the paired emit fix's actual cumulative count.

### Surface recycling

- **`surf_react recycle`** — surface recycling model for neutral transport.
  Incoming ions are neutralized and re-emitted with cosine angular
  distribution at a specified energy.
  ```
  surf_react ID recycle reactions_file
  ```
  - Reaction types: `E` (exchange, 1→1), `D` (dissociation, 1→2),
    `R` (recombination/absorption).
  - Each product has a specified return energy [eV].
  - Velocity sampled from cosine distribution relative to surface normal.

  Reactions file example:
  ```
  D+ --> D
  E 1.0 3.0

  D --> D
  E 1.0 0.025
  ```
  Format: `type probability energy1 [energy2]`
  - `1.0` = recycling probability (100%)
  - `3.0` = return energy in eV (Franck-Condon for D atoms)
  - `0.025` = thermal energy (~300 K wall temperature)

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

### Liquid metal film model

- **`fix liquid_metal`** — MHD shallow-water liquid metal film solver for
  divertor surfaces. Computes surface temperature, Li evaporation flux
  (Antoine + Hertz-Knudsen), ad-atom flux (Arrhenius desorption), and
  film thickness as per-surface custom attributes.
  ```
  fix ID liquid_metal group Nevery hf_source \
      h0 VAL U0 VAL Bs VAL alpha VAL width VAL Tin VAL \
      [dp_flux SOURCE] [Yad VAL] [Yad_Yps VAL] \
      [E_eff VAL] [A_arr VAL] \
      [Bw VAL] [sigma_w VAL] [tw VAL] [qss VAL] \
      [Nx VAL] [Ny VAL] [ncase VAL] \
      [max_iter VAL] [eps VAL] [relax VAL] [dt VAL] \
      [evap yes|no]
  ```
  - **Heat flux source** (`hf_source`): `c_compute[col]`, `f_fix[col]`,
    or a constant value [W/m²].
  - **Strip solver** (from Smolentsev): solves coupled momentum, continuity,
    free surface, and heat transfer on a 1D strip. MHD drag via Hartmann
    braking. Pseudo-time iteration to steady state.
  - **Evaporation model** (Antoine + Hertz-Knudsen):
    `P_vapor = 10^(A - B/T_K) * 101325` Pa (Antoine fit to Li vapor
    pressure data, 298–1600 K). Flux via Hertz-Knudsen:
    `Γ_evap = α * P / sqrt(2π m_Li kB T)` [atoms/m²/s].
    Evaporative cooling feedback: `Q_vapor = H_vap * Γ_evap / N_A`.
  - **Ad-atom model** (Arrhenius desorption driven by D+ flux):
    `Γ_ad = f_ad * (Yad/Yps) / (1 + A_arr * exp(E_eff / kB*T)) * Yad * Γ_D+`
    Requires D+ ion flux via `dp_flux` keyword (`c_compute[col]`,
    `f_fix[col]`, or constant). If `dp_flux` is not specified, ad-atom
    flux is zero (evaporation-only mode).
  - **Ad-atom parameters**:
    - `Yad`: ad-atom yield for D on Li (default 1e-3)
    - `Yad_Yps`: ratio Yad/Yps (default 1.0)
    - `E_eff`: effective binding energy [eV] (default 0.9)
    - `A_arr`: Arrhenius pre-factor (default 1e-7)
  - **Per-surface output columns** (accessible as `f_ID[i][col]`):
    1. `Tsurf_lm` — surface temperature [°C]
    2. `evap_lm` — evaporation flux [atoms/m²/s]
    3. `adatom_lm` — ad-atom flux [atoms/m²/s]
    4. `h_lm` — film thickness [m]
  - **Key strip parameters**:
    - `h0`: initial film thickness [m] (default 0.005)
    - `U0`: inlet velocity [m/s] (default 8.0)
    - `Bs`: streamwise magnetic field [T] (default 5.0)
    - `Bw`: wall-normal magnetic field [T] (default 0.0)
    - `alpha`: inclination angle [degrees] (default 43)
    - `width`: channel half-width [m] (default 1.67)
    - `Tin`: inlet temperature [°C] (default 350)
    - `qss`: heat flux scale [W/m²] (default 1e6)
    - `ncase`: 1 = sidewall (Hartmann via Bs), 2 = axisymmetric (via Bw)
  - **Files**: `liquid_metal_strip.h` (standalone solver, no SPARTA
    dependency), `fix_liquid_metal.{h,cpp}` (SPARTA fix wrapper).
  - Based on Fortran code by Sergey Smolentsev (UCLA).

  Example:
  ```
  # constant 2 MW/m² heat flux, no ad-atoms
  fix flm liquid_metal wall 100 2.0e6 \
      h0 0.005 U0 8.0 Bs 5.0 alpha 43 width 1.67 Tin 350

  # with D+ flux from compute for ad-atom calculation
  fix flm liquid_metal wall 100 c_hflux[2] \
      h0 0.005 U0 8.0 Bs 5.0 alpha 43 width 1.67 Tin 350 \
      dp_flux c_pflux[3] Yad 1e-3 E_eff 0.9

  # access outputs for dump
  dump dsurf surf all 1000 surf.*.dat id f_flm[1] f_flm[2] f_flm[3] f_flm[4]
  ```

## External coupling (library API)

C-callable entry points for driving OpenEdge from an outer loop (Python,
Gkeyll, SOLPS). Declared in `src/library.h`, implemented in
`src/library.cpp`. Standard SPARTA calls (`sparta_open`, `sparta_file`,
`sparta_command`, `sparta_extract_compute`, ...) are preserved; the
OpenEdge extensions are:

| Call | Purpose |
|------|---------|
| `openedge_extract_fix(ptr, id, 2, type)` | Return `double*` (`type=0`, vector_grid) or `double**` (`type≥1`, array_grid) for a per-grid fix. Used to pull source tallies from `fix chem/adas`. |
| `openedge_get_ngrid(ptr)` | Number of owned grid cells on this rank. |
| `openedge_reload_plasma(ptr, cid, path)` | Reload plasma HDF5 on a `compute plasma/fields`. `path=NULL` re-reads the existing file; otherwise swap to `path`. Used between Gkeyll/SOLPS iterations. |
| `openedge_reset_fix_tally(ptr, id)` | Zero the 20-col source tally on a `fix chem/adas`. Called between coupling iterations so each handoff sees a fresh accumulation. Leaves per-type counters and exhaust state intact. Silent no-op if the ID is wrong or not a `FixChemAdas`. |

**Coupling loop skeleton:**
```python
for k in range(n_outer):
    openedge_command(ptr, f"run {n_steps}")
    Sn_Smom_Se = ctypes_cast(openedge_extract_fix(ptr, b"fchem", 2, 1),
                             ngrid, 20)
    plasma = outer_solver.step(Sn_Smom_Se)
    plasma.write_hdf5("plasma_k+1.h5")
    openedge_reload_plasma(ptr, b"cplasma", b"plasma_k+1.h5")
    openedge_reset_fix_tally(ptr, b"fchem")
```

Existing driver: `tools/coupling/openedge_solps_driver.py` (subprocess
model); `solps_interface.py` handles SOLPS file IO.

## Performance: grid refinement near surface sources

For wall-source cases (PMI sputtering, divertor emission, evaporation), particles
spawn at the wall and cluster in a small set of cells near the strike point.
The two performance bottlenecks this creates:

1. **SurfColl checks dominate Move time** — each particle move tests against
   every surface element in its current cell. Coarse cells contain dozens of
   surface elements → dozens of intersection tests per particle per step.
2. **MPI load imbalance** — RCB partitions cells (atoms), so all particles in
   one cell go to one rank. With particles concentrated in a few cells, one
   rank does all the work and the rest sit in MPI barriers (`Other` bucket
   becomes 80–90% of loop time).

Both are solved by `adapt_grid` near the wall, which (a) reduces the number
of surface elements per cell and (b) gives RCB more granularity to subdivide
the dense region.

**Standard recipe** (place after `read_surf` and `surf_modify`, before any
`fix` commands):

```
# define refinement region(s) covering the wall area where particles cluster
region   rdiv_lo block 1.82 3.2 -0.94 -0.40 -INF INF   # lower divertor
region   rdiv_up block 1.82 3.2  0.40  0.80 -INF INF   # limiter / upper

# refine cells that overlap surfaces inside the region
# - thresh 0.001: refine if longest surf segment in cell > 1mm
# - cells 2 2 1: each parent splits into 2x2x1 children (2D)
# - maxlevel 5 + iterate 5: actually drives 5 levels of refinement
#   (iterate is the cap, NOT maxlevel — they must match or iterate must
#    be >= maxlevel)
adapt_grid all refine surf all 0.001 maxlevel 5 cells 2 2 1 region rdiv_lo all iterate 5
adapt_grid all refine surf all 0.001 maxlevel 5 cells 2 2 1 region rdiv_up all iterate 5
balance_grid rcb cell
```

**Critical ordering** for MPI runs with `gridcut 0.0`:

```
create_grid 100 100 1
balance_grid rcb cell        # FIRST balance — needed for adapt_grid in MPI
read_surf ...
surf_collide / surf_react / surf_modify ...
region ... ; adapt_grid ...  # refinement after surfaces have collision models
balance_grid rcb cell        # SECOND balance — distribute refined cells
```

If you skip the first `balance_grid`, MPI runs error out with
*"Cannot mark grid cells as inside/outside surfs because ghost cells do not
exist"*. If you run `adapt_grid` before `surf_modify`, you get
*"surface elements not assigned to a collision model"*.

**Measured impact (test_west_axi, 100×100 base grid, ~13K particles):**

| config | wall (1 rank) | wall (32 ranks) | SurfColl checks |
|---|---|---|---|
| no refinement | 8.72s | n/a (load imbalance) | 83.8M |
| ml=5 it=5 refine | 5.03s | **0.95s** | 8.5M (10× fewer) |

Total speedup: ~9× on a single rank (Move halved by fewer surface checks),
~13× on 32 ranks (refinement also makes RCB partitioning effective).

**Tuning notes:**

- `thresh` (third arg to `refine surf`) is the surf-element length below
  which refinement stops. Default `0.001` (1mm) works for typical fusion
  meshes; lower it only if your wall mesh has sub-mm features.
- `iterate N` is the cap on refinement depth, NOT `maxlevel`. Set
  `iterate >= maxlevel` or refinement stops early. Common mistake.
- **Wider regions outperform tight regions** even though they use more
  cells: the surface-check reduction across the *whole wall* is a bigger
  win than concentrated refinement at the strike. Prefer broad geometric
  bands over narrowly-tuned ones.
- `fix balance ... rcb time` (every 500 steps) outperforms `rcb part`
  for impurity transport — `time` weights cells by measured CPU cost,
  capturing per-particle work intensity (Boris subcycles, surf checks).
- **Exception for wall-sourced cases (emit/surf/pmi, emit/surf/recycle):**
  use `rcb part`, not `rcb time`. When the first balance fires, several
  ranks have zero particle work-time (no tasks emitted yet) and the RCB
  recursion in `rcb time` mode segfaults on NaN weights. `rcb part`
  partitions by particle count instead — stable with idle ranks.
  See `memory/feedback_rcb_time_segfault.md`.

### Runtime adaptive refinement: `fix adapt`

Static `adapt_grid` only sets up the grid once. For source-driven cases
where the cluster grows over time (more particles emitted than removed),
add `fix adapt` to refine cells whose particle count exceeds a threshold
*during* the run. Combined with `fix balance ... rcb time`, this gives an
extra **2–4× speedup** on top of the static refinement at moderate rank
counts (np=4–16).

**Recipe** (place in the fixes section of `in.case`):

```
# Runtime adaptive refinement: split any cell holding > 500 particles.
# Refines monotonically (no coarsen) so the grid never loses granularity.
# Capped by maxlevel + setup adapt_grid.
fix fadapt adapt 200 all refine particle 500 0 maxlevel 8 cells 2 2 1
fix fbal   balance 200 1.1 rcb time
```

- The `0` is the coarsen threshold (always required by the parser, even
  with `refine` only — leave at 0 for monotonic refinement).
- `200` (Nevery for fadapt and fbal) is the sweet spot. **More aggressive
  settings (every 100, threshold 200) actively hurt** — refine + balance
  overhead overwhelms the gain.
- `fix balance` alone every 200 steps does **not** help; the win requires
  fadapt + balance working together. Don't drop balance frequency without
  also enabling fadapt.

**Measured impact (test_west_axi, ~130K particles, nlaunch=100):**

| ranks | static refine only | + fix adapt | speedup from fadapt |
|-------|--------------------|-------------|---------------------|
| 4     | 36.8s              | 19.6s       | 1.88×               |
| 8     | 30.0s              | **9.7s**    | **3.09×**           |
| 16    | 21.2s              | 5.6s        | 3.78×               |
| 32    | 14.3s              | ~7.0s       | 2.0× (variance)     |

Total speedup vs single rank with no refinement: **~20× at np=16** for
this case. np=8 is the practical sweet spot for divertor-source cases at
this resolution — beyond that, refinement granularity becomes the new
bottleneck.

### Critical ordering for `fix adapt`

```
# ---- Diagnostics FIRST ----
compute cden grid all species nrho
fix    fden ave/grid all 1 1000 1000 c_cden[*] ave one
fix    frate ave/grid all 1 1000 1000 f_fchem[*] ave one

# ---- Runtime refinement + balance AFTER all fix ave/grid ----
fix fadapt adapt 200 all refine particle 500 0 maxlevel 5 cells 2 2 1
fix fbal   balance 500 1.1 rcb part
```

SPARTA errors out with *"Fix adapt must come after fix ave/grid"* if the
ordering is wrong. `fix adapt` invalidates any cell-to-value mapping
every time it refines, so the `ave/grid` fixes must already be set up
to hear about it.

### Measured impact: wall-recycling case (test_diii_d_neutrals)

With `fix emit/surf/recycle` driving wall emission:

| config (np=16) | ms/step @ ~50k particles |
|---|---|
| static adapt_grid maxlevel 3, no fbal/fadapt | ~110 ms |
| maxlevel 5 static + fadapt 200 + fbal rcb part | **~15 ms** |

**~7–8× speedup**, moves bottleneck off MPI load imbalance onto actual
particle work. Per-rank particle counts become balanced (min/max ratio
~1.5 instead of ~10).

### MPI traps in fix init code

When writing a new fix whose `init()` prints a diagnostic, **every
MPI_Allreduce must be called on every rank, not inside `if (comm->me == 0)`**.
A common mistake:

```cpp
// BUG: only rank 0 calls Allreduce -> other ranks deadlock
// at the next collective (run setup, fix chem init, etc.)
if (comm->me == 0) {
  double local = ...;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  printf("diagnostic = %.3e\n", global);
}
```

Correct: run the Allreduce unconditionally, then gate only the `printf`:

```cpp
double local = compute_something();
double global;
MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
if (comm->me == 0) printf("diagnostic = %.3e\n", global);
```

The symptom is a hang with no stats line ever printed — the run log
stops right after the fix's init prints. Single-rank runs are fine, so
the bug only shows up under MPI. Cost us a good chunk of a debug
session on `fix emit/surf/recycle`.

## MPI launch on mora

The OpenEdge binary is linked against Intel MPI. The system PATH points to
linuxbrew's `mpirun` by default, which is a different MPI ABI — launching
with the wrong `mpirun` produces N independent singleton processes
(`Running on 1 MPI task(s)` printed N times) instead of one N-rank job.

**Always source Intel oneAPI before running:**

```bash
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 8 ~/buildOpenEdge/src/spa_mpi -in in.case
```

To verify:
```bash
which mpirun           # should be /opt/intel/oneapi/mpi/.../bin/mpirun
echo $I_MPI_ROOT       # should be non-empty
```

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
