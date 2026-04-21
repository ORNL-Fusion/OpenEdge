# `fix emit/surf/recycle` — wall-recycling neutral source

Launches neutrals from SPARTA wall surfaces at a rate equal to the local
plasma Bohm wall flux × a total recycling coefficient. Mirrors the physics
of EIRENE strata 1–5 (SOLPS recycling strata) but driven purely from the B2
plasma state (ne, Te, Ti) and geometry — no EIRENE output consumed.

## Syntax

```
fix <ID> emit/surf/recycle <mixture> <group> <plasma_fix_ID> \
    [mass <amu>] [R <0..1>] [twall <K>]
```

## Bohm flux formula

Stangeby (2000) ch. 2:
```
Γ = n_i · c_s · sin(α_B),   c_s = sqrt((Te+Ti)/m_ion)
```
with `sin(α_B)` the geometric projection of B onto the wall inward normal.

## Emission rate per SPARTA task

A *task* is the unit of work: one wall surface in one cell after
`adapt_grid` refinement.

```
dot{N}_task = 0.5 · R · Γ_dom · A_seg[isurf] · area_share[itask]
```

- **`A_seg[isurf]`** is `mesh/wall_surf_area[isurf]` from plasma.h5 — the
  converter aggregates the B2 face area of every B2 boundary cell that
  chose this segment as its nearest, so it is the full SOLPS flux budget
  claimed by segment `isurf` regardless of whether one or many B2 cells
  map to it.
- **`area_share[itask] = task.area / Σ_{tasks with same isurf} task.area`**,
  computed at init time with an `MPI_Allreduce` so every rank sees the
  global denominator. Σ over all tasks of an isurf equals 1, so the
  per-isurf total is exactly `0.5 · R · Γ · A_seg`, independent of how
  `adapt_grid` splits the segment.
- **`Γ_dom`** uses the dominant B2 cell's `(ne, Te, Ti)` for the segment
  (the cell that contributed the largest face area). Sub-1.5× error when
  neighboring cells differ in plasma; refinable later by writing a
  per-segment plasma-weighted average.
- **1/2 prefactor**: balances D⁺ → D₂ recombination at the wall; the
  mixture fractions control the atom/molecule split.

## Re-emission velocity

Half-Maxwellian flux at `twall` along the inward normal. TRIM
fast-reflection channel not yet implemented.

## Init diagnostic (printed once per init at rank 0)

```
[emit/surf/recycle] tasks=N, mapped=M (P%)
[emit/surf/recycle] Bohm-flux rate (raw SPARTA segment area, sin_alpha=1) = X /s
[emit/surf/recycle] Bohm-flux rate (B2-aggregated surf_area, sin_alpha=1) = Y /s [USING THIS]
[emit/surf/recycle] wall mapping (global): K unique B2 cells, A m^2 of T total face area (P%)
```

- Both rates are MPI-global. The "raw SPARTA segment area" line uses
  `tasks[i].area` (the cone-frustum `2π·R·L` in SPARTA axi mode, the
  per-radian poloidal length in 2D Cart). The "B2-aggregated surf_area"
  is the actual emission formula above and what the runtime uses. They
  should agree within the geometric mismatch between the SPARTA wall
  mesh and the B2 boundary cells.
- "wall mapping" sums `mesh_wall_face_area[c]` over **dominant** cells
  only — under-reports when many B2 cells per segment, since
  `mesh_wall_surf_area[isurf]` (the actually-used quantity) carries the
  full aggregation.

## Wall → B2-cell mapping — three fallback paths

The fix needs to know, for each wall surface, which SOLPS B2 boundary cell
owns it. Three paths, tried in order:

1. **Topological (preferred)** — `fix_plasma_data` reads
   `mesh/wall_surf_cell[iseg]` from plasma.h5, written by the converter
   (see `docs/converters/wall_geometry.md`). Direct index lookup. Falls
   through if the chosen cell has `wall_face_area == 0`.
2. **Geographic fallback** — nearest B2 boundary cell by centroid
   distance, restricted to cells with `wall_face_area > 0`. Triggered
   when (1) is unavailable.
3. **Raw SPARTA area** — last-resort emission using `tasks[i].area` (with
   no `area_share`) when neither (1) nor (2) is available. Only correct
   when SPARTA wall and B2 boundary coincide exactly.

## Related

- Use `rcb part` (not `rcb time`) for `fix balance` in wall-sourced cases —
  see `memory/feedback_rcb_time_segfault.md`.
