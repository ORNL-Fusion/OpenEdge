# `fix droplet/emit` — droplet injection from a region

Inject droplets at a chosen rate into a region, with control over size,
mass, temperature, and emission angle / speed. Used to seed the droplet
pipeline (drag, charge, evaporation) for transient melt-injection
benchmarks and PFC-loss studies.

## Syntax

```
fix ID droplet/emit <mix-ID> <region-ID> [keyword value]*
```

| keyword | meaning |
|---|---|
| `n <Ndrop>` | total droplet count to emit (single-shot mode) |
| `nevery <N>` | injection cadence (continuous mode) |
| `density <rho>` | droplet material density [kg/m³] |
| `temperature <T>` | initial droplet temperature [K] |
| `vstream <vx vy vz>` | bulk velocity (physical R, Z, φ) [m/s] |
| `speed <V>` | scalar emission speed (sampled isotropically) |
| `magVelocity <V>` | magnitude with `incidentAngle` direction |
| `incidentAngle <deg>` | launch angle from surface normal |
| `fractions <f1 f2 ...>` | radius distribution (per radius bin) |
| `region <region-ID>` | spatial region for injection |
| `normal yes\|no` | launch along surface inward normal (yes) vs. isotropic (no) |
| `subsonic <P> <T>` | subsonic boundary-condition emission |
| `perspecies yes\|no` | distribute the count per mixture group |
| `twopass` | run two-pass insertion to balance per-cell counts |

## Example

Continuous injection of 2.5 mm Li droplets at the upper-baffle inlet:

```
region inlet block 0.0 0.1 1.0 1.05 -0.05 0.05
fix femit droplet/emit dropmix inlet \
    nevery 100 density 534 temperature 773 \
    speed 1.0 normal yes
```

## Files

- `src/OPENEDGE/fix_droplet_emit.{h,cpp}`
- See `examples/test_droplet` for end-to-end emit + drag + charge.
