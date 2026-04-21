# test_diii_d_neutrals

OpenEdge-vs-EIRENE benchmark for neutral transport on the DIII-D
geometry using SOLPS-ITER case `run_lore2023_reference` (converged
D-only B2.5-EIRENE). Both codes drive wall recycling on the SAME
frozen SOLPS plasma; we compare the neutral densities and source
maps they produce.

## Layout

```
test_diii_d_neutrals/
  input/
    plasma.h5                # SOLPS plasma (ne, Te, Ti, B) + mesh +
                             #   wall_face_area + wall_surf_cell map
    bfield.h5                # B-field on regular (R,Z) grid (legacy; B
                             #   is also embedded in plasma.h5)
    wall.surf                # SPARTA wall segments (from mesh.extra)
    wall.recycle             # TRIM + thermal absorb-and-reemit spec
    dg.equ                   # DIII-D equilibrium for psi (fix reflect/psi)
    neutral.species          # SPARTA species file (D2, D, D+)
    neutral.reactions        # Mode-A reaction set (diss + iz + CX)
    eirene_truth.h5          # SOLPS-EIRENE balance.nc extract (ref truth)
  eirene/                    # standalone EIRENE driver
    fort.1.solps_ref         # baseline SOLPS-ITER-generated input deck
    Database -> ...          # symlink to EIRENE databases (AMJUEL/HYDHEL)
    run_eirene.sh
    parse_fort44.py          # extract S_iz, S_diss etc. from fort.44
  openedge/
    in.diii_d_neutrals_eirene       # Mode-A recycling run (main one)
    in.diii_d_neutrals_recycle      # legacy distributed-puff case
    in.diii_d_neutrals              # legacy single-puff case
    run_openedge.sh
  scripts/
    extract_eirene_sources.py       # rebuild eirene_truth.h5 from balance.nc
    NOTES_fnum.md                   # fnum sizing recipe
    NOTES_A5c_historical.md         # first-pass notes, kept for history
  output/                           # run outputs (logs, .grid, plots)
  compare.py                        # per-cell S_iz, S_diss comparison plot
```

## Running OpenEdge

**Regenerate plasma + wall inputs** (once, after any change to the
converter or SOLPS case):

```bash
cd /home/cloud/OpenEdge/examples/test_diii_d_neutrals
python3 ../../tools/converters/convert_solps_plasma.py \
    /home/cloud/solps-runs/diii-d/runners_d2/run_lore2023_reference \
    --b2fgmtry /home/cloud/solps-runs/diii-d/runners_d2/baserun/b2fgmtry \
    --equ-file /home/cloud/solps-runs/diii-d/runners_d2/baserun/dg.equ \
    --mesh-extra /home/cloud/solps-runs/diii-d/runners_d2/baserun/mesh.extra \
    --plasma-out input/plasma.h5 \
    --bfield-out input/bfield.h5 \
    --wall-out input/wall.surf \
    --wall-source mesh-extra
```

- `--wall-source mesh-extra` is the SOLPS-native path (recommended for
  production). Alternatives: `eirene` (exact EIRENE wall, requires
  fort.33/34/35), or `auto` (default — picks mesh-extra if available).
- The converter always writes `wall.surf` in SPARTA's true axisymmetric
  layout (column 1 = Z, column 2 = R) since SOLPS is an axisymmetric
  code. Pair with `boundary o ao p`, `create_box ... 0 R_max ...` in the
  input deck.

**Run the EIRENE-recycling case:**

```bash
cd openedge
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 16 ~/buildOpenEdge/src/spa_mpi \
    -in in.diii_d_neutrals_eirene > ../output/run_eirene.log 2>&1
```

Run length: 30000 steps (≈ 10–30 min on 16 ranks depending on cluster
load). `fnum=4e15` targets ~100 sim-particles per step and ~50k
steady-state population.

## Coordinate convention

The case runs in **SPARTA-native axisymmetric mode** (`boundary o ao p`,
`x = Z (axial)`, `y = R (radial, ≥ 0)`). All per-volume / per-area
diagnostics (`compute grid nrho`, `compute surf flux`, `fix emit/surf/recycle`
rates) are full-3D quantities — no `2*pi*R` post-multiply needed when
comparing against EIRENE / SOLPS / Gkeyll.

Pre-2026-04-20 versions of this case used a 2D Cartesian layout that
required a `2*pi*R̄ ≈ 10` post-multiply on every flux / density. See
`CLAUDE.md` § "Coordinate convention" for the migration story and the
`openedge_geom` helper that handles slot mapping.

## Physics pieces in `in.diii_d_neutrals_eirene`

- **`fix plasma/data`** loads `plasma.h5` + `dg.equ`. Provides ne, Te,
  Ti at each cell + B-field + psi map.
- **`surf_react wall_pwi`** with `D_on_C.h5` TRIM tables — incoming
  D+/D/D2 recycle via TRIM fast reflection (hot atoms) + thermal
  absorb-and-reemit at 2 eV (Franck-Condon).
- **`fix emit/surf/recycle`** — primary neutral source. Looks up the
  per-segment B2 cell via `mesh/wall_surf_cell`, reads ne/Te/Ti,
  computes Bohm flux `Γ = ne·cs·sin(α)·face_area`, emits
  `0.5 · R · Γ` D2 molecules + D atoms per step. `R = 0.99` (1% pumped).
- **`fix chem/adas`** — volumetric D ionization, D+ recombination (no
  kinetic D+ in Mode A), CX, and D2 dissociation rates from ADAS/Janev
  tables.
- **`fix fcore reflect/psi … action absorb`** at psi_norm=0.95 mimics
  EIRENE's implicit "core sink" on the innermost B2 flux surface.

## Running standalone EIRENE (for reference)

```bash
cd eirene
./run_eirene.sh fort.1.solps_ref   # ~60 s on 1 rank
python3 parse_fort44.py            # extract + plot atom/molecule densities
```

This writes `fort.44` with per-cell neutral densities, used as a
second-source of truth alongside `input/eirene_truth.h5`.

## Comparison

```bash
python3 compare.py
```

Reads the latest `output/diii_d_eirene.grid` (OpenEdge) and
`input/eirene_truth.h5` (SOLPS-EIRENE converged). Produces
`output/compare_to_eirene.png`: OE and EIRENE S_iz / S_diss maps on
the same (R, Z) grid.

## Current status (2026-04-20)

Migrated to SPARTA-native axisymmetric mode today. Diagnostics now
report full-3D quantities directly (no per-radian-wedge correction).
The `[emit/surf/recycle]` per-step diagnostic prints both a
**raw-SPARTA-segment-area** Bohm rate (from `surf->axi_line_size`)
and a **B2-aggregated-surf_area** Bohm rate (from the converter's
per-segment area aggregation); the second is what the runtime emit
code uses, and the two should agree to within geometric mismatch
between the SPARTA wall and the B2 boundary.

OpenEdge Mode A recycling infrastructure is in place and produces
peak S_iz within ~30 % of SOLPS-EIRENE on the DIII-D case. Remaining
open items:

1. **Kinetic-ion recycling (Mode B)** — Mode A doesn't reproduce the
   multi-cycle amplification that B2.5-EIRENE achieves through
   plasma-side iteration. With kinetic D+ via `fix chem/adas mode
   kinetic`, D+ particles would transport via the Boris pusher, hit
   walls, and recycle through `surf_react wall_pwi` — giving OE the
   same amplification EIRENE gets from B2 coupling.
2. **Standalone-EIRENE-on-same-plasma comparison.** Current reference
   is the SOLPS-EIRENE COUPLED converged state. A cleaner apples-to-
   apples target is standalone EIRENE run on the same frozen plasma
   (the binary and runner are ready in `eirene/`).
3. **b2-only wall path.** `--wall-source b2` not yet implemented;
   needed for non-SOLPS codes (OEDGE, SOLEDGE3X).
4. **Proper Bohm flux near strike points.** Some SOLPS cells at the
   sheath edge appear to hold upstream Te values (~2 keV) rather than
   target (~10 eV); investigate how `convert_solps_plasma.py` picks
   up b2fstate temperatures.
