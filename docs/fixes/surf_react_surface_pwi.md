# `surf_react surface/pwi` — wall plasma–wall interactions

TRIM-backed reflection, absorb-and-re-emit, and dissociation reactions at
SPARTA wall surfaces. Reads reflection data from the consolidated
`database/processes.h5` (`/surface/reflection/<proj>_on_<target>/`) so no
per-pair files need to live beside the deck.

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
- **`trim_dir <path>`** — optional override pointing at a directory of
  per-pair TRIM files (one `<proj>_on_<target>.h5` each). In the
  standard workflow this is left unset; reflection tables come from
  `database/processes.h5`.
- **`R_surf <attr>`** — per-surf recycling coefficient attribute. When
  set, the `R` value of any `A`-type reaction is overridden per surface
  segment.

## Reaction types

| type | meaning | reactant count | product count |
|---|---|---|---|
| `T` | TRIM reflect (E, θ -> reflection probability + outgoing energy + direction from processes.h5) | 1 | 1 |
| `A` | absorb + re-emit (probability `R`; optional f_mol for atom → molecule partition) | 1 | 1 |
| `S` | additive (self-)sputter: yield Y(E, θ) from the named table; products launched with a Thompson spectrum (below) | 1 | 0–n |
| `E` | exchange (1 → 1, deterministic species swap) | 1 | 1 |
| `D` | dissociation (1 → 2, shared KE absorbs bond energy) | 1 | 2 |
| `R` | recombination / pure absorb (no re-emit) | 1 | 0 |

### Sputter-product energies (`S` channel)

Sputtered atoms are launched with the truncated Thompson spectrum
`f(E) ∝ E/(E+Es)³ · [1 − √((E+Es)/(Emax+Es))]` where `Es` is the
**full surface binding energy** read from the table's `Es_eV` attribute
(W: 8.68 eV) and `Emax = γ·E_in − Es` is the per-impact kinematic
cutoff (γ = 4·M₁M₂/(M₁+M₂)², so fast impacts eject a harder tail than
slow ones — no user knob needed). The spectrum peaks at `Es/2`
automatically; see the "Thompson: `Ub` is the FULL surface binding
energy" section of `surface_emit_source.md` for why the inserted value
must never be pre-halved.

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

## Areal-density surface state

The mixed-material surface model tracks per-surf, per-species **areal
densities** ("adens") in a homogeneous **reaction zone**, from which
**reaction-zone concentrations** c_i = δ_i/δ_TOT are derived each
sync; the remainder of the zone is **bulk** (substrate) material.

```
surf_react ID surface/pwi <reactions_file> ... \
    adens_surf <attr> \
    [adens_init <species> <atoms/m^2>] ... \
    [rzone <atoms/m^2>] \
    [adens_erosion <computeID> <col> <species> [noconc]] ... \
    [conc_feedback yes|no] [adens_nevery <N>]
```

- **`adens_surf <attr>`** — create/reuse the per-surf DOUBLE array
  (one column per species, atoms/m^2). Retained particles credit their
  species column (**influx**); sputtered atoms debit the sputtered
  species column (**erosion flux**). Derived customs: `<attr>_net`,
  `<attr>_dep`, `<attr>_ero`, `<attr>_conc`.
- **`adens_init <species> <val>`** — initial areal density (e.g. a
  boronization layer), applied once at attribute creation.
- **`rzone <val>`** — reaction-zone total areal density used to derive
  concentrations. WallDYN specifies the same quantity as a thickness
  (`RZoneWidth`, Å) converted through the mixture density
  1/n_tot = Σ c_i/n_i; here it is the areal density directly.
- **`adens_erosion <computeID> <col> <species> [noconc]`** — debit the
  named species by a background-driven gross-erosion flux from a
  per-surf compute (e.g. `surface/physical/sputter ... erosion_flux`).
  Debits are concentration-limited (scaled by the eroded material's
  reaction-zone concentration) unless `noconc` is given — use `noconc`
  when the compute already evaluates composition-resolved (compound)
  yield tables, otherwise the concentration is double-counted.
- **`conc_feedback yes|no`** — `no` freezes composition feedback:
  `mat` weights become 1 and compound tables evaluate at their c=1
  endpoint (fresh lead-element surface). For A/B comparisons.
- **`adens_nevery <N>`** — sync interval for folding per-rank deltas
  into the owned array (default 1).

Charge states pool into one **material** per element (W, W+, W2+ …
share one inventory; concentrations are pooled *before* clipping at
zero, so redeposited-ion credits cancel neutral-column debits — see
2026-08-03 fix). Sputter (`S`) reaction channels accept
`mat <species>` (linear concentration weighting of a pure-pair yield)
or `conc <species>` (composition coordinate of a 3D compound table —
no linear rescaling; the table carries the composition dependence).

## Related

- `database/processes.h5` — source of TRIM reflection tables. Rebuild
  with `database/ingest/build_processes_h5.py` after ingesting new
  TRIM output. Sputter tables (2D pure-pair and 3D compound
  `<proj>_on_<mix>_<elem>` with a `C` composition axis) live under
  `/surface/sputter/`; ingest with
  `tools/converters/add_sputter_tables.py`.
- `fix surface/emit/recycle` — thermal-return channel that complements
  the TRIM reflect path on the same wall group. `surface/pwi` handles
  the fast-reflection probability; `emit/recycle` handles the
  Bohm-flux-driven thermal return.
- `fix surface/emit/source ... conc_scale <attr>_conc <col>` — emission
  scaled by a reaction-zone concentration column. Drop `conc_scale`
  when the driving compute is compound-mode (already c-weighted).
- `compute surface/physical/sputter` — background-driven gross-erosion
  flux on the same surfaces; supports 2D tables, IEAD convolution, and
  compound mode (`compound <mix> conc <attr> <species>`).
