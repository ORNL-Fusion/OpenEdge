# `fix chem/adas` — volumetric neutral reactions

ADAS-based volumetric chemistry with competing Poisson channel selection.
Supports ionization, recombination, charge exchange (CX), and dissociation.
Plays the role of EIRENE's collision operator in OpenEdge standalone runs.

## Syntax

```
fix ID chem/adas Nevery Z reactions_file \
    adas_dir PATH plasma TeSRC NeSRC \
    [mode kinetic|neutral] \
    [source_species <sp1> [sp2] ...] \
    [stop_on_exhaust yes|no] [exhaust_threshold <N>] \
    [units counts|rate|batch <N> <R_puff>|batch_fix <emit_id> <R_puff>]
```

## Rate styles

- **`A` (ADAS)**: bilinear interpolation on HDF5 rate tables `⟨σv⟩(Te, ne)`
  from ADF11 data (SCD/ACD/CCD).
- **`J` (Janev)**: 9-term polynomial `ln⟨σv⟩ = Σ bₙ (ln Te)ⁿ` from
  HYDHEL/Janev 1987. Used for molecular dissociation.

## Reaction types (in reactions file)

| type | meaning | charge change |
|---|---|---|
| `I` | ionization | +1 |
| `R` | recombination | −1 |
| `E` | charge exchange (with background H) | −1 |
| `D` | dissociation (1→2 products, creates new particle) | — |

- CX rate data from ADAS CCD files (`ccd89_*.dat`); same format as ACD/SCD,
  stored as `ChargeExchangeRateCoeff` in HDF5.
- Dissociation uses deferred particle creation to avoid array invalidation
  during iteration.
- After CX or dissociation, product velocity is re-sampled from a shifted
  Maxwellian at local Ti and bulk flow (EIRENE-like), when the per-particle
  plasma cache provides Ti, vpar, and B-field.
- Per-type reaction tally printed every 10,000 steps.

## Mode A (EIRENE semantics)

`mode neutral` deletes the neutral on ionization instead of relabeling it as
an ion, and compresses the particle array via `dellist` at end_of_step.
Combined with `source_species D D2 stop_on_exhaust yes [exhaust_threshold N]`,
the run halts cleanly when the alive source population drops to `N`
(default 0). `exhaust_threshold` skips the slow-converging fat tail.
`exhaust_armed` guards against spurious exit during batch ramp-up.

## 20-column per-cell source tally (`array_grid`)

Exposed as `f_ID[*][col]` for `dump grid`, `fix ave/grid`, or the library
API. Layout is quantity-major across 4 reaction types (ion, rec, CX, dissoc):

| cols | quantity | units |
|---|---|---|
| 1–4   | count per cell per reaction type | — |
| 5–8   | sum of m·vx at reaction events | kg·m/s |
| 9–12  | sum of m·vy | kg·m/s |
| 13–16 | sum of m·vz | kg·m/s |
| 17–20 | sum of ½·m·\|v\|² | J |

## Tally units (`units` keyword)

- **`counts`** (default): raw cumulative totals since fix start.
- **`rate`**: window-averaged SI rate (m⁻³s⁻¹, N/m³, W/m³); zeroed each
  Nevery window. Use with `fix ave/grid … ave running` for smooth
  SOLPS/Gkeyll input.
- **`batch N R_puff`**: EIRENE-style MC. Each of the `N` trajectories
  carries weight `w = R_puff / N` [events/s] (divided by cell volume for
  source density). Cumulative across the run — final value is the
  steady-state source for one puff.
- **`batch_fix <emit_id> R_puff`**: same as `batch`, but `N` is pulled from
  the paired emit fix's cumulative emit count each step. Tracks ramp-up
  automatically so you don't hand-match `N` with the emit fix's
  `stop_at_np`.

## Data pipeline

`database/adas/adas.py` converts ADF11 ASCII files to HDF5. Supports `acd`
(recombination), `scd` (ionization), `ccd` (charge exchange). Set
`ADAS_ADF11_DIR` or symlink into `database/adas/adf11/`.

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

- `fix emit/surf/puff` — hard-capped surface emission for Mode A puffs.
  Pair with `units batch_fix ID R_puff` so the tally scales from the
  emit fix's cumulative count.
- `surf_react recycle` — surface recycling (1→1 exchange, 1→2 dissociation,
  absorption) with cosine re-emission at specified return energy.
