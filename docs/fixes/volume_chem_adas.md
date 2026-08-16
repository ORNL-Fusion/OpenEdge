# `fix volume/chem/adas`

## Description

This fix applies ADAS-based volumetric chemistry with competing Poisson
channel selection. It supports ionization, recombination, charge
exchange (CX), and dissociation, and it plays the role of EIRENE's
collision operator in OpenEdge standalone runs.

> **Renamed 2026-04-22.** Formerly `fix chem/adas`. Old decks should
> sweep that token; the keyword grammar below is unchanged apart from the
> new `output` keyword and the removal of `adas_dir`.
>
> **2026-04-24:** the `plasma TeSRC NeSRC` keyword (cell-centered Te/ne
> override) was removed. Te/ne now come exclusively from the per-particle
> plasma cache so the sheath Boltzmann correction is applied
> consistently. Migration: feed the deck's plasma through
> `fix background` (use `constant ...` for uniform test cases) and
> activate the cache with `global bfield_compute <fix-or-compute-ID>`,
> sheath, or GCA.

## Syntax

```
fix ID volume/chem/adas Nevery <species|Z> <reactions_file|auto|element> \
    [mode kinetic|neutral] \
    [source_species <sp1> [sp2] ...] \
    [stop_on_exhaust yes|no] [exhaust_threshold <N>] \
    [units counts|rate|batch <N> <R_puff>|batch_fix <emit_id> <R_puff>] \
    [output summary|detailed] \
    [ionization yes|no] [recombination yes|no] \
    [cx yes|no] [dissociation yes|no]
```

- **`<species|Z>`** — projectile element symbol (`D`, `C`, `W`) or
  numeric atomic number (`1`, `6`, `74`). Isotope labels collapse to the
  same `Z` — ADAS rates are charge-dependent only.
- **`<reactions_file|auto|element>`** — source of the reaction list,
  tried in this order:
  1. Literal path (`../input/neutral.reactions`) — used as-is.
  2. Element symbol (`D`, `W`) — pulls the opaque string from
     `processes.h5:/volume/reactions/<elem>/catalog` via
     `ProcessLibrary::load_reactions_catalog`, writes to a
     per-process tempfile, parses with the existing text parser.
  3. `auto` — same as passing the arg[3] element symbol.
  4. If (2) misses (element absent from processes.h5), falls back to
     the text file at
     `${OPENEDGE_ROOT}/database/ingest/reactions/<elem>.reactions`.

  Typical: `fix fchem volume/chem/adas 1 D auto ...`.
- **`output summary|detailed`** — tally layout (see below).
- **Channel toggles** (`ionization`, `recombination`, `cx`,
  `dissociation`) default to `yes`. Setting one to `no` disables every
  reaction of that type at init without editing the file — useful for
  ablation runs.

Rate data lives in the consolidated `database/processes.h5` under
`/volume/rates/{scd,acd,ccd}/<elem>` (ionization / recombination / CX)
plus `/volume/thresholds/<elem>` and `/volume/pec/<elem>` if present.
The old per-element `database/adas/ADAS_Rates_<Z>.h5` fallback was
dropped in 2026-04-22; so was the `adas_dir` keyword.

## Plasma source

Te/ne (and Ti, vpar, B for CX channels) are read from the per-particle
plasma cache populated by `Update::cache_plasma_particles()`. The cache
needs an upstream provider AND an activator:

| Provider | Activator |
|---|---|
| `fix background file plasma.h5 ...` | `global bfield_compute <fix-ID>`, sheath, GCA, or Boris with bfield source |
| `fix background constant temp_e ... dens_e ...` | same — pcache works in test cases too |
| `compute plasma/fields all file plasma.h5 ...` | `global bfield_compute <compute-ID>`, sheath, or GCA |

When sheath is active, the Boltzmann correction
`ne_local = ne_upstream · exp(-φ/Te)` is applied inside the sheath
engagement distance and feeds through to chem/adas without any extra
plumbing — see [`sheath.md`](sheath.md) for details.

A pure-neutral deck (no charged particles, no pusher) still needs an
activator. The simplest is `global bfield_compute pd` against a
`fix pd background constant ...` provider — Boris won't engage on
neutrals, but the activator populates the cache.

Init diagnostic (rank 0) prints which reactions fired and why any were
skipped:

```
[volume/chem/adas] ionization    : 20 active, 0 skipped (of 20)
[volume/chem/adas] recombination : 20 active, 0 skipped (of 20)
[volume/chem/adas] cx            : disabled by toggle (2 in file)
[volume/chem/adas] dissociation  : 1 active, 0 skipped (of 1)
[volume/chem/adas] output summary (6 cols)  ADAS tables: SCD ACD CCD PLT PRB -
```

## Restrictions and behavior

- This fix reads Te/ne from the per-particle plasma cache, not directly
  from a cell-centered override.
- A pure-neutral deck still needs a cache activator such as
  `global pusher plasma <ID>`.
- Reactions whose required species are missing from the deck's species
  table are skipped at init.

## Truncated charge-state ladders

If a reactions file lists ionization/recombination for charge states
beyond the species defined in the deck (e.g. deck has `W0..W10+` but the
file continues `W10+ -> W11+`), the runtime silently skips those
reactions — `find_species` returns −1, the reaction is marked inactive,
particles can't climb past the highest defined charge state. The init
"missing species" line surfaces the drop explicitly.

Physical consequence: the top defined charge state acts as a terminal
sink — ions accumulate there. Fine when the top state is beyond the
SOL/edge-relevant range; check before trusting densities at the top.

## Rate styles (reactions file)

- **`A` (ADAS)** — bilinear interpolation on HDF5 rate tables
  `⟨σv⟩(Te, ne)` from ADF11 data (SCD/ACD/CCD) in processes.h5.
- **`J` (Janev)** — 9-term polynomial `ln⟨σv⟩ = Σ bₙ (ln Te)ⁿ` from
  HYDHEL/Janev 1987. Used for molecular dissociation.

## Reaction types (in reactions file)

| type | meaning | charge change |
|---|---|---|
| `I` | ionization | +1 |
| `R` | recombination | −1 |
| `E` | charge exchange (with background H) | −1 |
| `D` | dissociation (1→2 products, creates new particle) | — |

- CX rate data from ADAS CCD files (`ccd89_*.dat`), stored as
  `ChargeExchangeRateCoeff` in processes.h5.
- Dissociation uses deferred particle creation to avoid array
  invalidation during iteration.
- After CX or dissociation, product velocity is re-sampled from a
  shifted Maxwellian at local Ti and bulk flow (EIRENE-like), when the
  per-particle plasma cache provides Ti, vpar, and B-field.
- Per-type reaction tally printed every 10 000 steps.

## Mode A (EIRENE semantics)

`mode neutral` deletes the neutral on ionization instead of relabeling
it as an ion, and compresses the particle array via `dellist` at
`end_of_step`. Combined with
`source_species D D2 stop_on_exhaust yes [exhaust_threshold N]`, the
run halts cleanly when the alive source population drops to `N`
(default 0). `exhaust_threshold` skips the slow-converging fat tail;
`exhaust_armed` internally guards against spurious exit during ramp-up.

## Per-cell tally: `summary` vs `detailed`

Exposed as `f_ID[*]` for `dump grid`, `fix ave/grid`, or the library
API. Two layouts selectable with `output`:

### `output summary` (default) — 6 columns

Plasma-frame source moments (see `docs/neutral_plasma_coupling/main.tex`
§4). Direct consumer of this layout is the Gkeyll outer loop.

| col | quantity | units |
|---|---|---|
| 1 | Sp — particle source (signed, ion − neutral when mode=neutral) | m⁻³ s⁻¹ |
| 2–4 | Sm_{x,y,z} — momentum source vector | kg m⁻² s⁻² |
| 5 | Qe — electron power source (PLT ionization + PRB recombination) | W m⁻³ |
| 6 | Qi — ion power source (sheath + thermal + CX) | W m⁻³ |

### `output detailed` — 20 columns

Per-reaction-type breakdown, quantity-major across 4 types (I, R, CX, D):

| cols | quantity | units |
|---|---|---|
| 1–4 | count per cell per reaction type | — |
| 5–8 | Σ m·vx at reaction events | kg m/s |
| 9–12 | Σ m·vy | kg m/s |
| 13–16 | Σ m·vz | kg m/s |
| 17–20 | Σ ½ m \|v\|² | J |

## `units` keyword

- **`counts`** (default) — raw cumulative totals since fix start.
- **`rate`** — window-averaged SI rate (m⁻³ s⁻¹, N/m³, W/m³); zeroed
  each Nevery window. Use with `fix ave/grid … ave running` for a
  smooth SOLPS/Gkeyll input.
- **`batch <N> <R_puff>`** — EIRENE-style MC. Each of the N
  trajectories carries weight `w = R_puff / N` [events/s] (divided by
  cell volume for source density). Cumulative across the run — final
  value is the steady-state source for one puff.
- **`batch_fix <emit_id> <R_puff>`** — same as `batch`, but `N` is
  pulled from the paired emit fix's cumulative emit count each step.
  Tracks ramp-up automatically, no hand-matching with `stop_at_np`.

## Examples

Typical deuterium chemistry:

```text
fix pd    background file plasma.h5 static yes
global pusher plasma pd
fix fchem volume/chem/adas 1 D auto mode neutral units rate
```

Disable charge exchange:

```text
fix fchem volume/chem/adas 1 D auto cx no
```

## Data pipeline

`database/ingest/build_processes_h5.py` converts ADF11 ASCII files
directly to the consolidated `processes.h5`:

- `acd` (recombination), `scd` (ionization), `ccd` (charge exchange)
- `plt` (line radiation), `prb` (recombination + bremsstrahlung)
- ionization potentials from ADF11 headers

Set `ADAS_ADF11_DIR` or symlink into `database/adas/adf11/` before
re-running.

## Shipped reactions files

`database/adas/reactions/<elem>.reactions` is the canonical location
picked up by `auto` / element-symbol arg[4].

### `D.reactions`

| type | reaction | rate | notes |
|---|---|---|---|
| `I` | `D → D+` | ADAS SCD | ionization |
| `R` | `D+ → D` | ADAS ACD | recombination |
| `E` | `D+ → D` | ADAS CCD | ion-side CX; inactive in Mode A (no kinetic D+) |
| `E` | `D → D`  | ADAS CCD | **neutral-side CX** (EIRENE-dominant channel for puff neutrals) |
| `D` | `D2 → D + D` | Janev HYDHEL H.2 2.2.5 | molecular dissociation |

### Channel toggle examples

```
fix fchem volume/chem/adas 1 D auto cx no                            # disable both CX
fix fchem volume/chem/adas 1 D auto recombination no dissociation no # ion + CX only
fix fchem volume/chem/adas 1 D auto ionization no recombination no   # CX + dissoc only
```

Each toggle disables every reaction of that type at init (same code
path as missing-species filtering), so there's no runtime overhead.

## Reactions file example

```
D --> D+
I A 1.0 0.0 0.0 0.0 0.0

D+ --> D
E A 1.0 0.0 0.0 0.0 0.0

D2 --> D + D
D J -2.787e+01 1.052e+01 -4.973e+00 1.451e+00 -3.063e-01 4.433e-02 -4.096e-03 2.160e-04 -4.929e-06
```

## Related

- `fix surface/emit/puff` — hard-capped surface emission for Mode A
  puffs. Pair with `units batch_fix ID R_puff` so the tally scales
  from the emit fix's cumulative count.
- `surf_react surface/pwi` — surface recycling (1→1 TRIM reflect, 1→1
  absorb-and-re-emit, 1→2 molecular return) sourced from processes.h5.
- `compute volume/emissivity/grid` — consumes `Qe` from `output summary`
  together with ADAS PEC coefficients in processes.h5.
