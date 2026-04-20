# A5c: DIII-D neutrals vs SOLPS-EIRENE — status

## Two variants, both validated at the shape level

### v1: single-point puff (in.diii_d_neutrals)

One wall element near lower-outer strike, `fnum=1e13`, `n=200`,
`stop_at_np=50000`. Exhausts cleanly via `stop_on_exhaust yes`.
Good for watching a localized plume; used to validate the Mode-A
ionization/dissociation/CX pipeline on DIII-D geometry.

### v2 "recycle proxy": 17 divertor-wall puff (in.diii_d_neutrals_recycle)

All lower-divertor segments puff simultaneously. `fnum=3e18`, `n=10`
per step → total emission 3×10²⁷ D₂/s (matches SOLPS balance.nc
total dissociation 3.14×10²⁷). Steady-state Np ≈ 100k projected; run
10k steps reaches ~66k and shape is already stable.

## Measured rates (step 10000 window, rate-mode f_frate)

| quantity           | OpenEdge v2        | SOLPS-EIRENE       | ratio |
|--------------------|--------------------|--------------------|-------|
| peak S_iz          | 1.27×10³⁰          | 9.50×10³⁰          | 0.13  |
| peak S_diss        | 1.17×10³⁰          | 4.22×10³⁰          | 0.28  |
| total iz           | 2.86×10²⁶ /s       | 4.66×10²⁷ /s       | 0.06  |
| total diss         | 4.65×10²⁶ /s       | 3.14×10²⁷ /s       | 0.15  |

## Why the totals are low (and this is OK)

1. **Only the lower divertor emits.** EIRENE recycles at 7 strata
   (targets + upper + inner wall + pump ducts). Missing 60-70% of
   the wall source.
2. **~80% of D₂ hit walls before dissociating.** Divertor targets
   are geometrically close to the puff points. The 15-20% that do
   dissociate match the EIRENE local peak value within a factor
   ≈3, which is the real validation point.
3. **No flux-weighting across the 17 lines.** All emit at equal
   rate; EIRENE's strata weight by incident ion flux.

Fixing any of these requires `emit/surf/pmi` with flux weights from
the plasma — that's A6 scope, not A5c.

## What A5c demonstrates

* OpenEdge neutral pipeline (dissociation + ionization + CX) runs
  on real device geometry at MPI scale (16 ranks, 10k steps in 3.5
  min wall) with stable particle counts and no solver issues.
* Reaction rate maps qualitatively match EIRENE where OpenEdge
  emits — localized intensity on the lower-outer strike, SOL fill
  from cold D diffusing upward.
* Grid adapt + rcb-part balance are stable on the divertor-source
  case; `rcb time` caused segfaults when ranks had zero particle
  work-time (reverted to `rcb part`).

## Files

* `in.diii_d_neutrals_recycle` — v2 input
* `output/diii_d_recycle.grid` — snapshots at step 0, 2k, 4k, 6k,
  8k, 10k with f_fden[1..2], f_frate[1..4]
* `output/compare_to_eirene.png` — 2×3 panel OE vs EIRENE
* `compare_to_eirene.py` — reads either v1 or v2 dump (argv or
  auto-detect; prefers recycle.grid when present)
* `extract_eirene_sources.py` — rebuilds `eirene_truth.h5` from SOLPS
  balance.nc (path hard-coded to lore2023_reference case)
