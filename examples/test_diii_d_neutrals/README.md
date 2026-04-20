# test_diii_d_neutrals

One-to-one EIRENE vs. OpenEdge benchmark for neutral transport on the
DIII-D geometry (SOLPS case `run_lore2023_reference`, converged D-only).

Both codes drive the **same** single D₂ puff on the **same** frozen
plasma through the **same** geometry. Everything else (walls outside
the puff region, CX, recombination, species set) is identical.

## Layout

```
test_diii_d_neutrals/
  input/                  # shared plasma + geometry + species/reaction data
    plasma.h5             # (te, ne, ti) on SOLPS (R,Z) grid
    bfield.h5             # B-field (not strictly needed for neutrals)
    wall.surf             # OpenEdge surface file (68 wall segments)
    wall.recycle          # OpenEdge surf_react recycle spec
    neutral.species       # OpenEdge species file (D2, D, D+)
    neutral.reactions     # OpenEdge Mode-A reactions (diss + iz + CX)
    eirene_truth.h5       # SOLPS-EIRENE balance.nc extract — reference truth
  eirene/                 # standalone EIRENE driver
    fort.1.solps_ref      # baseline SOLPS-ITER-generated input deck (625 lines)
    Database -> ...       # symlink to EIRENE databases (AMJUEL/HYDHEL/TRIM)
    run_eirene.sh
  openedge/               # OpenEdge driver
    in.diii_d_neutrals         # single-puff case (stop_at_np, exhausts)
    in.diii_d_neutrals_recycle # distributed-divertor puff, steady-state
    run_openedge.sh
  scripts/                # helpers
    extract_eirene_sources.py  # rebuild eirene_truth.h5 from balance.nc
    NOTES_fnum.md              # fnum sizing recipe
  output/                 # run outputs (logs, .grid dumps, plots)
  compare.py              # per-cell S_iz, S_diss comparison + plot
```

## How to run

**OpenEdge side:**

```bash
cd openedge
./run_openedge.sh in.diii_d_neutrals_recycle     # ~3.5 min wall on 16 ranks
```

Stats every 200 steps, dumps every 2000 steps to `../output/diii_d_recycle.grid`.

**EIRENE side** (once input deck is stripped down — see below):

```bash
cd eirene
./run_eirene.sh fort.1         # or fort.1.solps_ref to reproduce SOLPS truth
```

## EIRENE standalone

Binary at `/home/cloud/eirene_standalone/EIRENE/binRelease/eirene` —
ITER Org repo, commit `f8f63fa0`, Intel ifx + MPI. See
`/home/cloud/eirene_standalone/EIRENE/README.md` for build details.

The `fort.1.solps_ref` is the SOLPS-ITER-generated input deck from
`run_lore2023_reference` and drives the full 7-stratum recycling case.
To get a 1-to-1 comparison we need to derive a stripped-down deck
(`fort.1.onepuff`) that has:

  - only stratum 1 active (one divertor surface, D₂ at fixed rate)
  - absorbing walls everywhere else
  - plasma read from our `input/plasma.h5` (or an EIRENE-format copy)

That stripped deck is the current work item. See `NOTES_design.md`.

## Comparison metric

`compare.py` reads the OpenEdge dump and `eirene_truth.h5`, interpolates
to a common (R,Z) grid, and produces:

  - per-cell log-residual maps: `log10(S_OE / S_EIRENE)`
  - volume-integrated totals: `∫ S_iz dV`, `∫ S_diss dV`, `∫ S_cx dV`
  - 2×3 panel of OE vs. EIRENE S_iz, S_diss, and OE densities

Target agreement for the 1-to-1 case: < 20% on integrals, within a
factor ≈2 per cell away from the puff point.

## Status (2026-04-20)

- OpenEdge side: works. See `openedge/in.diii_d_neutrals_recycle`.
- EIRENE side: standalone binary built and ready; input-deck stripping
  in progress (`NOTES_design.md`).
- Reference truth: `input/eirene_truth.h5` extracted from `balance.nc`
  of the original SOLPS run — represents the full 7-stratum coupled
  state, so it is the *end* goal, not the 1-to-1 benchmark target.
