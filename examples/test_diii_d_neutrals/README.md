# test_diii_d_neutrals

OpenEdge **plasma–neutral interaction** demonstration on a DIII-D
axisymmetric geometry with a fixed SOLPS plasma background. Drives
wall recycling + volumetric chemistry (ionisation, recombination, CX,
dissociation) and writes per-cell source dumps so the user can see
where each reaction channel fires across the poloidal plane.

Short 10,000-step run. Extend `run` in the deck for longer statistics.

## Layout

```
test_diii_d_neutrals/
  input/
    plasma.h5         SOLPS plasma (ne, Te, Ti, upar) on the EIRENE mesh
                      + per-cell wall_face_area + wall_surf_cell map
                      + embedded /equilibrium/* (psi, r, z, btf, rtf)
    wall.surf         SPARTA wall segments (axi: x = Z, y = R)
    wall.recycle      TRIM + thermal absorb-and-reemit spec
    dg.equ            DIII-D equilibrium (psi) for fix reflect/psi
    neutral.species   Species: D2, D, D+
    neutral.reactions ADAS / Janev reaction set (iz, recomb, CX, dissoc)
  openedge/
    in.diii_d_neutrals  the deck
    run_openedge.sh     launcher (source oneapi + mpirun)
  scripts/
    NOTES_fnum.md       fnum sizing recipe
  output/               run outputs (.grid dumps)
```

## Running

```bash
cd openedge
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 16 ~/buildOpenEdge/src/spa_mpi -in in.diii_d_neutrals \
    > ../output/run.log 2>&1
```

`NP=64 ./run_openedge.sh` for 64 ranks on one node.

Wall-clock on 64 ranks (mora) for the 10,000-step default: ~2–3 s.
Linearly scale to see 100k / 500k / 1M by editing the final `run N`.

## Coordinate convention

True SPARTA axisymmetric (`boundary o ao p`, `x = Z` axial,
`y = R` radial). All per-volume / per-area diagnostics are full-3D
quantities — no `2π · R̄` post-multiply. See `CLAUDE.md` §
"Coordinate convention" for the slot-mapping details.

## Physics pieces in the deck

- **`fix plasma/data`** — loads `plasma.h5` (mesh + plasma + equilibrium).
  Provides ne, Te, Ti, B-field, psi to every consumer.
- **`surf_react wall_pwi`** — incoming D+/D/D2 recycle via TRIM fast
  reflection (hot) + thermal absorb-and-reemit at 2 eV.
- **`fix emit/surf/recycle`** — primary neutral source. Reads ne/Te/Ti at
  each wall segment's adjacent B2 cell (via `mesh/wall_surf_cell`),
  computes Bohm flux `Γ = ne · cs · sin(α) · face_area`, emits
  `0.5 · R · Γ` of the recycling mixture per step (`R = 0.99`).
- **`fix chem/adas`** — volumetric D ionisation, D+ recombination, CX,
  and D2 dissociation rates from ADAS/Janev tables. 20-column per-cell
  source tally exposed as `f_fchem[*]`.
- **`fix reflect/psi ... action absorb`** — inner core boundary at
  `psi_norm = 0.95`. Neutrals that wander core-side are removed.

## What gets dumped

`output/diii_d_neutrals.grid.XXXXXX` (every 1000 steps) — per-cell:

| col | quantity                   | units        |
|-----|----------------------------|--------------|
|  1  | cell id                    |              |
|  2  | xc (= Z, axial)            | m            |
|  3  | yc (= R, radial)           | m            |
|  4  | D2 density                 | m⁻³          |
|  5  | D density                  | m⁻³          |
|  6  | ionisation rate            | 1 / m³ s     |
|  7  | recombination rate         | 1 / m³ s     |
|  8  | charge-exchange rate       | 1 / m³ s     |
|  9  | dissociation rate          | 1 / m³ s     |
| 10  | ionisation energy source   | W / m³       |
| 11  | recombination energy src   | W / m³       |
| 12  | charge-exchange energy src | W / m³       |
| 13  | dissociation energy src    | W / m³       |

`f_fchem[5..16]` (momentum source per reaction) is still computed by
`fix chem/adas` — add the columns to the `dump d1` line if you need
them for Gkeyll coupling.

## Regenerating input files (one-time)

```bash
python3 ../../tools/converters/convert_solps_plasma.py \
    <SOLPS case dir> \
    --b2fgmtry <SOLPS baserun b2fgmtry> \
    --equ-file <SOLPS baserun dg.equ> \
    --mesh-extra <SOLPS baserun mesh.extra> \
    --plasma-out input/plasma.h5 \
    --wall-out input/wall.surf \
    --wall-source mesh-extra
```

`plasma.h5` is mesh-only (~1 MB for DIII-D). All the plasma, the
equilibrium, and the wall-to-B2-cell map live in it; no separate
`bfield.h5` or `.equ` at run time.
