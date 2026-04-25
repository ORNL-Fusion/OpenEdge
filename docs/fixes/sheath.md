# Sheath physics — nested under `global pusher ... sheath`

Per-particle sheath physics applied at and near walls. Two execution
modes (kick vs. spatial), one combined model (Coulette-Manfredi close
to wall + Borodkina tail beyond `s ≈ 60 λ_D`). Driven by a
`compute nearest_surf/grid` providing cell-level wall geometry and the
pusher's `plasma` provider for upstream Te/ne/Ti.

Implementation: `src/OPENEDGE/sheath_models.{h,cpp}` for the physics,
`update.cpp` for the per-particle and per-subcycle integration.

## Syntax

Sheath is a sub-keyword of `global pusher`:

```
global pusher mode boris|hybrid plasma <ID> \
              [subcycles N] \
              [other pusher options...] \
              sheath off|kick|spatial \
                     [geom <nearest_surf/grid-ID>] \
                     [mD_amu <amu>]
```

- **`sheath off`** (default) — no sheath overlay.
- **`sheath kick`** — velocity boost at wall collision. Recommended for
  IEAD / impact-energy diagnostics. No per-subcycle E-field, no gyro-
  orbit blowups at grazing angles.
- **`sheath spatial`** — sheath E-field integrated per Boris subcycle
  as particles approach the wall. Use when you need the spatial profile
  inside the magnetic presheath (transport across `λ_D`–`L_MPS`).
- **`geom <ID>`** — `compute nearest_surf/grid` providing per-cell
  distance to the nearest wall surface, surface index, and outward
  normal. Required when sheath is on.
- **`mD_amu <amu>`** — background ion mass for sheath scales (default
  D = 2.014).

The plasma source comes from the parent `global pusher plasma <ID>` —
sheath shares it, no separate provider.

## Internal auto-defaults

The user API intentionally omits `dmax`, `pot_mult`, and `model` — they
are computed internally from local plasma + geometry:

- **`dmax`** (engagement cutoff) — `max(5·L_MPS, 10·λ_D)` per particle,
  evaluated at the particle's local `Te, Ti, ne, |B|, α`. The factor of
  5 captures the slowly-decaying CM Chodura tail; the floor at `10·λ_D`
  keeps the cutoff sensible when α is near 90° and `L_MPS → 0`. See
  `sheath_auto_dmax()` in `update.cpp`.
- **`pot_mult`** (wall drop in units of Te) — auto Bohm-Stangeby
  floating wall: `0.5·ln[(mD/2π·me)/(1+Ti/Te)]`.
- **Model** — combined Coulette-Manfredi (kinetic PIC fit, close to
  wall) + Borodkina magnetic-presheath tail (`s ≥ 60 λ_D`), with linear
  blend over `60 ≤ s ≤ 120`. Single physically-motivated choice; the
  pure-Borodkina path remains in `sheath_models.cpp` for unit tests but
  is not exposed via the user API.

## α-angle (Chodura metrics)

The angle between **B** and the outward wall normal sets the magnetic-
presheath thickness `L_MPS = ρ_i · tan(α_n)`. Computed at every sheath
evaluation by `chodura_metrics()` from `(B, n)` at the particle
position, not assumed perpendicular. The kick and spatial modes both
work at grazing angles — wall drop and MPS scale degrade gracefully as
α → 0 (clamping `tan(α_n)` to ≤ 30 keeps `L_MPS` finite at exactly
grazing incidence).

## prepare / evaluate split

The Boris subcycle loop hits the sheath E-field every step, but
upstream `Te, ne, |B|, α` and the derived scales (`λ_D, ρ_i, L_MPS,
fd, φ_0`, the model fit coefficients) are constant over a single Boris
call:

- **`sheath_prepare_coulette_manfredi`** — runs the expensive
  transcendentals once per Boris call, returns a `SheathEmagCoeffs`
  struct.
- **`sheath_emag_at_distance(coeffs, d)` / `sheath_phi_at_distance(...)`**
  — per-subcycle: 2-4 `exp()` and a handful of multiplies.

Dominant per-step cost saving for spatial sheath runs at moderate-to-
high subcycle counts.

## Boltzmann electron-density correction

When sheath is active, `Update::cache_plasma_particles()` overwrites
the per-particle `ne_vec` with

```
ne_local = ne_upstream · exp(-φ(d_particle) / Te)
```

inside `d_max`, where `φ(d_particle)` is evaluated by the same
prepared-sheath coefficients used by the pusher. Outside `d_max` the
upstream value passes through unchanged.

**Consequence for `fix volume/chem/adas`:** the fast-path consumer
reads `ne_vec[ip]` directly and sees the depleted electron density
inside the sheath, *without* any plumbing in the chem fix itself.
Ionization sources fall off correctly into the magnetic presheath as
the bulk density depletes — this is what makes near-wall neutral
transport quantitative without a separate sheath model in the chem
path.

## Required setup — kick mode (IEADs)

```
fix           pd       background file plasma.h5 ...
compute       cgeom    nearest_surf/grid all <wall-group> dist nx ny nz surfid
global        pusher mode boris plasma pd subcycles 5 \
                     sheath kick geom cgeom
```

## Required setup — spatial mode (sheath E-field profile)

```
compute       cplasma  plasma/fields all file plasma.h5 ...
compute       cgeom    nearest_surf/grid all <wall-group> dist nx ny nz surfid
global        pusher mode boris plasma cplasma subcycles 50 \
                     sheath spatial geom cgeom
```

Higher subcycle counts are typical for spatial mode (50–500) so the
sheath E-field is sampled along the gyro-orbit at sub-`λ_D`
resolution.

## Related

- [`pusher.md`](pusher.md) — the parent `global pusher` API.
- `compute nearest_surf/grid` — per-cell wall geometry (distance,
  surface index, outward normal). Required input.
- [`volume_chem_adas.md`](volume_chem_adas.md) — consumer of the
  Boltzmann-corrected `ne` from the per-particle plasma cache.
