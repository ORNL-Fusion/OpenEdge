# Charged-particle pusher — `global pusher ...`

Single hierarchical keyword for the charged-particle pusher
(Boris full-orbit or Boris/GCA hybrid) plus an optional sheath
overlay. Replaces the older `global boris_*`, `global gca`,
`global bfield_compute`, and `global sheath` keywords.

## Syntax

```
global pusher mode boris|hybrid \
              [subcycles N] \
              [plasma <compute-or-fix-ID>] \
              [gca_switch <factor>] \
              [dump yes|no] [dump_every N] \
              [bad_dt_check yes|no] [bad_dt_limit <max>] \
              [sheath off|kick|spatial \
                      [geom <nearest_surf/grid-ID>] \
                      [mD_amu <amu>]]
```

Multiple `global pusher ...` lines accumulate; each call processes its
own keywords and leaves the others at their current value.

## Top-level keywords

- **`mode boris|hybrid`** (default `boris`) —
  - `boris`: full-orbit Boris pusher.
  - `hybrid`: Boris/GCA hybrid. Particles run as full Boris orbits
    when the local field is well-resolved; switch to a GCA push when
    `ρ_L > L_B / gca_switch` (i.e. when the orbit can't be resolved
    on the timestep). GCA uses RK4 with Littlejohn corrections.
- **`subcycles N`** (default 1) — velocity/position substeps per move.
  Typical values: 1 for ballistic neutrals (no charged push), 5 for
  kick-only sheath, 50–500 for spatial sheath profiles.
- **`plasma <ID>`** — upstream provider for B (always) plus Te/ne for
  sheath/diagnostics. Either a `compute plasma/fields` or a
  `fix plasma/data`. Activates the per-particle plasma cache
  (`pcache`) — required for any deck where downstream fixes (e.g.
  `fix volume/chem/adas`) need per-particle Te/ne/Ti.
- **`gca_switch <factor>`** (default 2.5) — GCA-vs-Boris switching
  threshold for hybrid mode. Larger factor → GCA fires earlier (more
  particles use GCA). Ignored when `mode boris`.
- **`dump yes|no`**, **`dump_every N`** — debug trace of E/B at each
  pusher call. Off by default. Useful for unit tests / single-particle
  validation.
- **`bad_dt_check yes|no`** (default yes), **`bad_dt_limit <max>`**
  (default 0.5) — warn once when `|q/m|·|B|·dt_sub` exceeds the limit
  (i.e. when the Larmor period is poorly resolved on the subcycle).

## Sheath sub-tree

```
sheath off|kick|spatial \
       [geom <nearest_surf/grid-ID>] \
       [mD_amu <amu>]
```

- **`sheath off`** (default) — no sheath overlay.
- **`sheath kick`** — velocity boost at wall collision (recommended
  for IEAD / impact-energy diagnostics).
- **`sheath spatial`** — sheath E-field integrated per Boris subcycle
  along the approach to the wall.
- **`geom <ID>`** — `compute nearest_surf/grid` providing per-cell
  distance to nearest wall, outward normal, surface index. Required
  when sheath is on.
- **`mD_amu <amu>`** (default D = 2.014) — background ion mass for
  Bohm sound speed and Debye length.

The plasma source for the sheath comes from the pusher's `plasma`
keyword — there is no separate sheath plasma. Internal scales
(`dmax`, `pot_mult`, model blend) are computed automatically; see
[`sheath.md`](sheath.md) for the physics details.

## Defaults

```
mode boris
subcycles 1
plasma   <none — pcache disabled>
gca_switch 2.5
dump no
dump_every 1
bad_dt_check yes
bad_dt_limit 0.1
sheath off
mD_amu 2.014
```

## Examples

**Pure neutral run, no charged push** — pcache only, for test
decks that need `fix volume/chem/adas`:

```
fix    pd plasma/data constant temp_e 20 dens_e 1e19
global pusher plasma pd
```

(No `mode` needed — defaults to `boris`, but with no charged particles
the pusher is a no-op. The `plasma pd` line is what activates pcache.)

**IEAD run with sheath kick:**

```
fix     pd plasma/data file plasma.h5 static yes
compute cgeom nearest_surf/grid all wall dist nx ny nz surfid
global  pusher mode boris plasma pd subcycles 5 \
               sheath kick geom cgeom
```

**Spatial sheath profile:**

```
compute cplasma plasma/fields all file plasma.h5 ...
compute cgeom   nearest_surf/grid all wall dist nx ny nz surfid
global  pusher mode boris plasma cplasma subcycles 100 \
               sheath spatial geom cgeom
```

**Boris/GCA hybrid for impurity transport:**

```
compute cplasma plasma/fields all file plasma.h5 ...
global  pusher mode hybrid plasma cplasma subcycles 50 \
               gca_switch 2.5
```

## Related

- [`sheath.md`](sheath.md) — sheath physics details (kick vs. spatial,
  model blend, Boltzmann ne correction, prepare/evaluate split).
- `compute nearest_surf/grid` — per-cell wall geometry (required when
  sheath is on).
- `compute plasma/fields` / `fix plasma/data` — plasma providers.
- [`volume_chem_adas.md`](volume_chem_adas.md) — ionization/recombination
  consumer of the per-particle pcache filled by the pusher.
