# `surf_collide diffuse` — diffuse wall scatter (with per-species T_wall)

OpenEdge's `surf_collide diffuse` extends the base diffuse
model with a `twall_species` keyword: each species can carry its own
re-emission temperature. This matters for hydrogenic recycling, where
atomic D should desorb at the Franck–Condon temperature
(~$23\,210\,\mathrm{K}$ ≈ 2 eV) while molecular D₂ desorbs
wall-thermalised (~500 K).

## Syntax

```
surf_collide ID diffuse <Twall> <accommodation> \
    [translate <Vx> <Vy> <Vz>] [rotate <Wx> <Wy> <Wz>] \
    [twall_species <sp1> <T1> <sp2> <T2> ...]
```

| arg | meaning |
|---|---|
| `Twall` | default wall temperature (K) used for any species not in `twall_species` |
| `accommodation` | thermal-accommodation coefficient $\alpha \in [0, 1]$. `1.0` = full diffuse re-emission |
| `translate <Vx Vy Vz>` | superimposed wall translation (rare) |
| `rotate <Wx Wy Wz>` | rotation rate (idem) |
| `twall_species <sp> <T> ...` | per-species re-emission temperature in K. Repeat for each species. |

## Example

Recycling-driven gas puff with hot D atoms and cold D₂ molecules:

```
surf_collide wall_diffuse diffuse 500.0 1.0 \
    twall_species D 23210.0 D2 500.0
surf_modify wall collide wall_diffuse
```

This gives Franck–Condon-temperature atomic D back into the plasma
(captures the ~2 eV birth energy) while molecules thermalise to the
wall, matching the standard EIRENE recycling convention.

## Files

- `src/OPENEDGE/surf_collide_diffuse.{h,cpp}`
- See [`fix surface/emit/recycle`](surface_emit_recycle.md) which
  uses this surf_collide as the standard wall-recycling channel.
