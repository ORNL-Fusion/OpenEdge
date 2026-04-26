# `surf_react surface/pwi` — wall plasma–wall interactions

TRIM-backed reflection, absorb-and-re-emit, and dissociation reactions at
SPARTA wall surfaces. Reads reflection data from the consolidated
`database/processes.h5` (`/surface/reflection/<proj>_on_<target>/`) so no
per-pair files need to live beside the deck.

> **Renamed / consolidated 2026-04-22.** This style replaces two older
> ones:
>
> - `surf_react recycle` (legacy: cosine-reemission exchange only)
> - `surf_react wall_pwi` (intermediate rename)
>
> The reactions-file grammar has been extended to add the `T` (TRIM
> reflect) channel; the `A` / `E` / `D` / `R` channels are carried over
> with compatible semantics.

## Syntax

```
surf_react ID surface/pwi <reactions_file> \
    [twall <K>] [twall_surf <per-surf-attr>] \
    [trim_dir <path>] \
    [R_surf <per-surf-attr>]
```

- **`<reactions_file>`** — ASCII file with recycling / reflection rules
  (see below). Typical deck has it in `input/wall.recycle`.
- **`twall <K>`** — uniform wall temperature (K) for re-emission
  samplers (half-Maxwellian at twall along inward normal).
- **`twall_surf <attr>`** — per-surf wall temperature attribute, read
  from `fix surf/custom` or a precomputed attribute on the surf group.
  Mutually exclusive with `twall`.
- **`trim_dir <path>`** — only needed when falling back to legacy
  per-pair TRIM files instead of processes.h5. In the standard OpenEdge
  workflow this is left unset; reflection tables come from processes.h5.
- **`R_surf <attr>`** — per-surf recycling coefficient attribute. When
  set, the `R` value of any `A`-type reaction is overridden per surface
  segment.

## Reaction types

| type | meaning | reactant count | product count |
|---|---|---|---|
| `T` | TRIM reflect (E, θ -> reflection probability + outgoing energy + direction from processes.h5) | 1 | 1 |
| `A` | absorb + re-emit (probability `R`; optional f_mol for atom → molecule partition) | 1 | 1 |
| `E` | exchange (1 → 1, deterministic, legacy) | 1 | 1 |
| `D` | dissociation (1 → 2, shared KE absorbs bond energy) | 1 | 2 |
| `R` | recombination / pure absorb (no re-emit) | 1 | 0 |

Channels are evaluated **in order listed in the file, per reactant
species**; the first to cross a uniform random draw fires. So the
typical pattern is:

1. `T <pair>` — TRIM reflects with probability R_N(E, θ) from the
   table. Random draw against R_N; if under, reflect; otherwise fall
   through.
2. `A <R> [<f_mol>]` — catch-all for non-reflected particles: absorb,
   then re-emit as atom with prob `R (1 − f_mol)`, as molecule with
   prob `R · f_mol / 2`, or pump with prob `1 − R + R f_mol / 2`.

## TRIM reflection (`T` channel)

Looks up `database/processes.h5:/surface/reflection/<pair>/` via the
`ProcessLibrary` loader. Pair names are lowercase: `d_on_w`, `o_on_w`,
`w_on_w`, `d_on_c`, etc.

The table carries energy / angle grids plus reflection probability
`R_N(E_in, θ)`, outgoing energy bounds `[E_out_min, E_out_max]`, and
angular quantiles for the outgoing direction (polar + azimuth). The
runtime samples a reflected velocity from those quantiles — preserving
the shape of the TRIM distribution rather than imposing a cosine.

Missing pair → fatal error at init listing the missing key; drop the
reaction or add the table.

## Absorb + re-emit (`A` channel)

Syntax (per reactant):

```
<reactant> --> <product>
A <R> [<f_mol>]
```

- `R` ∈ [0, 1] — probability of re-emission (else pumped = removed).
- `f_mol` ∈ [0, 1] — optional, fraction of recycled atoms that come
  back as the molecular product species. Default 0 (purely atomic
  re-emission). For D, `f_mol = 0.95` matches EIRENE / WTD standard
  (Franck-Condon partition).

Re-emission velocity: half-Maxwellian at `twall` (or `twall_surf`) along
the inward surface normal.

`R = 0` collapses to pure absorb (used for W on W: any W that doesn't
TRIM-reflect just vanishes into the wall).

## Reactions file example — D on C (graphite)

```
# D+ recycling: neutralise + TRIM reflect as D, else absorb-and-re-emit
# as 95% D2 / 5% D.
D+ --> D
T d_on_c
D+ --> D2
A 0.99 0.95

D --> D
T d_on_c
D --> D2
A 0.99 0.95

D2 --> D2
A 0.99
```

## Reactions file example — W on W (impurity wall)

```
# W reflection: TRIM reflect neutral W, else absorbed (no re-emit).
W --> W
T w_on_w
W --> W
A 0.0

W+ --> W
T w_on_w
W+ --> W
A 0.0

# (repeat for W2+ … W20+ as needed)
```

## Wall-normal convention

Matches the unified 2026-04-21 convention: inward normals, outgoing
velocity along `+normal`. Converters (`convert_solps_plasma.py`,
`convert_s3x_plasma.py`) emit wall.surf with that orientation; no
`invert` on the corresponding `read_surf`.

## Related

- `database/processes.h5` — source of TRIM reflection tables. Rebuild
  with `database/ingest/build_processes_h5.py` after ingesting new
  TRIM output.
- `fix surface/emit/recycle` — thermal-return channel that complements
  the TRIM reflect path on the same wall group. `surface/pwi` handles
  the fast-reflection probability; `emit/recycle` handles the
  Bohm-flux-driven thermal return.
- `compute surface/physical/sputter` — sputter yield diagnostic on the same
  surfaces (analytic Eckstein for now; TRIM sputter migration
  pending).
