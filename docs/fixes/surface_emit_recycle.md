# `fix surface/emit/recycle` — wall-recycling neutral source

Launches neutrals from SPARTA wall surfaces at a rate equal to the local
plasma Bohm wall flux × a total recycling coefficient. Mirrors the
physics of EIRENE strata 1–5 (SOLPS recycling strata) but driven purely
from the B2 plasma state (ne, Te, Ti) and geometry — no EIRENE output
consumed.

> **Renamed 2026-04-22.** Formerly `fix emit/surf/recycle`. Keyword
> grammar below is unchanged.

## Syntax

```
fix <ID> surface/emit/recycle <mixture> <group> <plasma_fix_ID> \
    [mass <amu>] [R <0..1>] [twall <K>] \
    [twall_species <sp1> <T1> [<sp2> <T2> ...]]
```

- **`twall`** — default emission temperature [K] used by the Maxwellian
  flux sampler and internal (rot/vib) energy sampler for every species
  in the mixture.
- **`twall_species`** — optional per-species overrides. Consumes
  `<sp> <T>` pairs until a recognised next keyword (or end of args).
  A named species must exist both in SPARTA's species table and in the
  mixture; otherwise init errors out. Species in the mixture not named
  here fall back to the scalar `twall`.

Typical recycled-neutral decomposition (DIII-D-style graphite wall):

```
twall 500.0 \
twall_species D 23210.0 \     # Franck-Condon ~2 eV from D2 dissociation
              D2 500.0         # wall-thermalised molecular channel
```

## Bohm flux formula

Stangeby (2000) ch. 2:

```
Γ = n_i · c_s · sin(α_B),   c_s = sqrt((Te+Ti)/m_ion)
```

with `sin(α_B)` the geometric projection of B onto the wall inward
normal.

## Emission rate per SPARTA task

A *task* is the unit of work: one wall surface in one cell after
`adapt_grid` refinement.

```
dot{N}_task = 0.5 · R · Γ_dom · A_seg[isurf] · area_share[itask]
```

- **`A_seg[isurf]`** — `mesh/wall_surf_area[isurf]` from plasma.h5.
  The converter aggregates the B2 face area of every B2 boundary cell
  that chose this segment as its nearest, so it is the full SOLPS flux
  budget claimed by segment `isurf` regardless of how many B2 cells
  map to it.
- **`area_share[itask] = task.area / Σ_{tasks with same isurf} task.area`**,
  computed at init time with an `MPI_Allreduce` so every rank sees the
  global denominator. Σ over all tasks of an isurf equals 1, so the
  per-isurf total is exactly `0.5 · R · Γ · A_seg`, independent of how
  `adapt_grid` splits the segment.
- **`Γ_dom`** — uses the dominant B2 cell's `(ne, Te, Ti)` for the
  segment (the cell that contributed the largest face area). Sub-1.5×
  error when neighbouring cells differ in plasma; refinable later by
  writing a per-segment plasma-weighted average.
- **1/2 prefactor** — balances D⁺ → D₂ recombination at the wall;
  mixture fractions control the atom/molecule split.

## Re-emission velocity

Half-Maxwellian flux at `twall` (or per-species `twall_species`) along
the inward normal. Rotational and vibrational energies are sampled at
the same temperature via `Particle::erot` / `Particle::evib`. TRIM
fast-reflection channel is handled separately by
`surf_react surface/pwi` reflection rules — the recycle fix provides
the thermal-return channel.

## Init diagnostic (printed once per init at rank 0)

```
[surface/emit/recycle] tasks=N, mapped=M (P%)
[surface/emit/recycle] Bohm-flux rate (raw SPARTA segment area, sin_alpha=1) = X /s
[surface/emit/recycle] Bohm-flux rate (B2-aggregated surf_area, sin_alpha=1) = Y /s [USING THIS]
[surface/emit/recycle] wall mapping (global): K unique B2 cells, A m^2 of T total face area (P%)
```

- Both rates are MPI-global. "raw SPARTA segment area" uses
  `tasks[i].area` (the cone-frustum `2π·R·L` in SPARTA axi mode, the
  per-radian poloidal length in 2D Cart). "B2-aggregated surf_area" is
  the formula above and what the runtime uses. They should agree within
  the geometric mismatch between the SPARTA wall mesh and the B2
  boundary cells.
- "wall mapping" sums `mesh_wall_face_area[c]` over **dominant** cells
  only — under-reports when many B2 cells per segment, since
  `mesh_wall_surf_area[isurf]` (the actually-used quantity) carries the
  full aggregation.

## Wall → B2-cell mapping — three fallback paths

The fix needs to know, for each wall surface, which SOLPS B2 boundary
cell owns it. Three paths, tried in order:

1. **Topological (preferred)** — `fix background` reads
   `mesh/wall_surf_cell[iseg]` from plasma.h5, written by the converter
   (see `docs/converters/wall_geometry.md`). Direct index lookup. Falls
   through if the chosen cell has `wall_face_area == 0`.
2. **Geographic fallback** — nearest B2 boundary cell by centroid
   distance, restricted to cells with `wall_face_area > 0`. Triggered
   when (1) is unavailable.
3. **Raw SPARTA area** — last-resort emission using `tasks[i].area`
   (with no `area_share`) when neither (1) nor (2) is available. Only
   correct when SPARTA wall and B2 boundary coincide exactly.

## Wall-normal convention

**Unified 2026-04-21.** Every OpenEdge surface fix —
`fix emit/surf`, `fix surface/emit/puff`, `fix surface/emit/recycle`,
`fix surface/emit/source`, `surf_collide diffuse`,
`surf_react surface/pwi` — now uses the **SPARTA canonical** convention:
wall normals point INTO the fluid (plasma), and emission /
outgoing-reflection velocity is along `+normal`.

Practical consequences for decks:

| wall.surf producer | normals point | use `invert`? |
|---|---|---|
| `convert_solps_plasma.py` | INTO plasma | **no** |
| `convert_s3x_plasma.py` | INTO plasma | **no** |
| `convert_oedge_plasma.py` | INTO plasma | **no** |

The historical `emit/surf/recycle` vs `emit/surf` sign mismatch (fixed
in `fix_surface_emit_recycle.cpp`) is gone. Old decks that carried
`read_surf ... invert` as a workaround should drop `invert` when
upgrading — otherwise emission will face outward through the wall.

Decks that use `read_surf core.surf invert ...` for the separate
**core** boundary (psi_norm = const contour written by
`tools/extract_psi_contour.py` or the converter `--core-out` flag)
remain correct — core.surf genuinely needs the opposite orientation
from wall.surf.

## Related

- Use `rcb part` (not `rcb time`) for `fix balance` in wall-sourced
  cases — see the `feedback_rcb_time_segfault` memory entry.
- `surf_react surface/pwi` — TRIM / absorb-re-emit reflections that
  coexist with the recycle fix on the same wall group. The recycle
  fix provides the thermal-return channel; surface/pwi provides the
  fast reflection probability from the TRIM tables.
- `fix volume/chem/adas mode neutral` — consumes the recycled neutrals
  upstream via ionization / CX / dissociation, closing the wall-plasma
  loop.
