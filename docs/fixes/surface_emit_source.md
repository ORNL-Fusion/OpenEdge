# `fix surface/emit/source` — sputtered-impurity neutral source

Launches sputtered impurity neutrals (W, C, B, …) from wall surfaces at
a rate dictated by a `compute surface/physical/sputter` erosion-flux output. Pairs
with `compute surface/physical/sputter ... erosion_flux` to produce the W / C / B
source term for kinetic-impurity transport.

> **Renamed 2026-04-22.** Formerly `fix emit/surf/pmi`. Keyword grammar
> below is current.

## Syntax

```
fix ID surface/emit/source <mixture> <group> <compute_id> \
    [flux_index <col>] \
    [perspecies yes|no] \
    [normal yes|no] \
    [n <N_per_step>] \
    [nlaunch <N_per_surf>] \
    [nlaunch_total <N_per_step_global>] \
    [flux_thresh <f>] \
    [source_thresh <f>] \
    [model thermal | thermal_tsurf <name> | thompson <Ub> [cos_n <n>] [emax <E>] | fixed_energy <E>] \
    [region <rID>]
```

- **`<mixture>`** — emitted impurity mixture (e.g. `tungstenSource`,
  typically with mass and temperature set to reflect the sputter
  energy spectrum; keep W at frac 1 so only neutral W is emitted, the
  ionization chain builds up the charge states via
  `fix volume/chem/adas`).
- **`<group>`** — surface group ID to emit from (the same `wall` group
  as the `compute surface/physical/sputter`).
- **`<compute_id>`** — reference to a `compute surface/physical/sputter` whose
  output column `flux_index` is the erosion flux
  [particles · m⁻² · s⁻¹]. Typical: `cpmiO` or `cpmiW`.
- **`flux_index <col>`** — 1-based column of the compute's per-surf
  array to use as the emission flux. Default 1 (the first column),
  which matches `compute … erosion_flux` being the only output.
- **`perspecies yes`** — emit one particle per species in the mixture
  per step (incompatible with `n > 0` and `nlaunch_total`). Useful for
  debugging species-level emission.
- **`normal yes`** — inject along the surface inward normal (overrides
  the mixture `vstream`).
- **`n <N>`** — emit exactly N per step per task (CONSTANT mode).
- **`nlaunch <N>`** — emit approximately N per active surf per step
  (weighted launch; redistributes if a surf's flux is low).
- **`nlaunch_total <N>`** — emit approximately N per step globally,
  weighted across surfs by the local erosion flux. Preferred for
  production impurity runs — controls total MC noise.
- **`flux_thresh <f>`** — skip surfs with erosion flux below `f`
  [m⁻² s⁻¹]. Prevents tiny-noise contributions from sub-threshold
  segments flooding MC.
- **`source_thresh <f>`** — skip emission entirely when the sum of
  `flux_index` across all surfs in the group drops below `f` (useful
  for running until the erosion source has transient-decayed, or for
  gated one-shot puffs alongside `volume/chem/adas stop_on_exhaust`).
- **`region <rID>`** — restrict emission to a region.

## Emission energy and angle (`model`)

| model | energy distribution |
|---|---|
| `thermal` (default) | half-Maxwellian flux at the task temperature |
| `thermal_tsurf <name>` | half-Maxwellian at the per-element surface temperature in custom attribute `<name>` |
| `thompson <Ub> [cos_n <n>] [emax <E>]` | Thompson sputter spectrum (below) |
| `fixed_energy <E>` | monoenergetic at E (eV) |

All models use a cos^n angular distribution about the surface normal
(n = 1, the Knudsen cosine, unless `cos_n` is given).

### Thompson: `Ub` is the FULL surface binding energy

The sampled spectrum is the truncated Thompson distribution of
Mellet PPCF **59** (2017) eq. (10) / Guterl NME **27** (2021) eq. (2.2):

```
f(E) ∝ E / (E + Ub)³ · [1 − √((E + Ub)/(emax + Ub))],   0 ≤ E ≤ emax
```

**`Ub` is the surface binding energy itself — for tungsten 8.68 eV,
not 4.34.** The ½ is built into the formula: f(E) peaks at `Ub/2`
automatically, which is exactly the familiar "sputtered atoms come off
near half the binding energy" rule. That rule describes the *output*
spectrum produced by inserting the full `Ub`; inserting a pre-halved
value shifts the whole spectrum a factor of two too soft (peak at
`Ub/4`) and inflates prompt redeposition. Both reference papers state
the inserted value explicitly: E_b = U_S = 8.68 eV for W.

With `thompson 8.68 emax 80`: peak 4.34 eV, median ≈ 10 eV,
mean ≈ 14.5 eV.

### `emax`: the high-energy cutoff (set it in production)

Pure Thompson has a ~1/E² tail whose mean diverges: without `emax`,
~1e-4 of emitted atoms sample above 100·Ub — unphysical keV-scale
neutrals that fly ballistically across the machine and contaminate
far/shadow regions. The physical cutoff is the kinematic limit of the
collision cascade:

- Guterl (NME 2021): E_c ≈ 2·T_e for low-Z projectiles in attached
  conditions (~30–40 eV at WEST divertor targets);
- Mellet (PPCF 2017)/Mousel: E_max = γ⟨E_imp⟩·((m₁+2m₂)/(2m₁+m₂))⁶;
- single-binary-collision estimate γ·E_inc − Ub ≈ 80 eV for O→W at
  ~300 eV impact.

`emax` applies the smooth Mellet roll-off (the bracketed factor above),
not a hard clip, so the tail dies gradually toward `emax`.

## Pairing with `compute surface/physical/sputter`

```
compute cpmiW surface/physical/sputter wall background pd \
    target W projectiles all \
    mass_amu 2.01410177811 static yes \
    erosion_flux

fix fpw  particle/weight
fix femit surface/emit/source tungstenSource wall cpmiW \
    perspecies no normal yes \
    nlaunch_total 100 source_thresh 0.0
```

- `cpmiW` emits a single column of erosion flux
  [particles · m⁻² · s⁻¹]; `femit` multiplies it by the axi ring area
  of each surface element to get the per-segment source rate, then
  draws the configured macroparticle launch count per step.
- `fix particle/weight` is *required* when per-surface emission rates
  vary (which is always, since the flux has spatial structure): each
  macro gets a `pweight` proportional to its segment's flux, so the
  spatial density is recovered correctly with
  `compute grid/weighted … nrho_w`. Without `particle/weight`, peak
  density is over-estimated near flat regions and under-estimated at
  sputter-heavy segments.
- `nlaunch_total` sets the per-step macro count; pick based on target
  MC noise:
  - `nlaunch_total=10` — good for quick peak-density check (~1 min
    runs).
  - `nlaunch_total=100` — converged SOL-wide mean (minutes to hours
    depending on domain).
  - `> 100` — no extra physics value on current test cases; just more
    CPU.

## Wall-normal convention

`normal yes` emits along the inward surface normal. Pairs with the
unified 2026-04-21 convention (converters write wall.surf with inward
normals; no `invert` on `read_surf`).

## Related

- `compute surface/physical/sputter` — erosion-flux calculator that drives the
  source rate.
- `fix particle/weight` — per-macro weight infrastructure. Strictly
  required for spatially-variable emission rates.
- `fix volume/chem/adas` — ionization / recombination chain for the
  sputtered impurity neutrals once they are in the volume.
- `fix force/thermal`, `fix cross_field_diffusion`, `fix coulomb/binary` —
  transport forces on the ionized impurity species.
