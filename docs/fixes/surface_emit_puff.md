# `fix surface/emit/puff` — hard-capped surface emission

Like `fix emit/surf` but with a hard cap on total emitted particles so a
one-shot EIRENE-style batch can terminate cleanly. Designed to pair
with `fix volume/chem/adas … mode neutral` (Mode A).

> **Renamed 2026-04-22.** Formerly `fix emit/surf/puff`. Keyword set
> below is unchanged from the old name.

## Syntax

```
fix ID surface/emit/puff mixture group \
    [n <N_per_step>] [normal yes|no] [stop_at_np <N_total>] \
    [perspecies yes|no] [region <rID>]
```

- **`mixture`** — SPARTA mixture ID for the emitted species.
- **`group`** — surface group ID to emit from (use `group surf ...` to
  pre-select the divertor or puff location).
- **`n <N_per_step>`** — emit exactly `N` particles per step (CONSTANT
  mode). `n 0` reverts to flow-based emission using the mixture's
  `vstream` velocity.
- **`normal yes`** — inject along the surface inward normal (into the
  fluid). Overrides the mixture `vstream` direction.
- **`stop_at_np <N>`** — latch emission off once `N` total particles
  have been emitted globally. No further injection after that; combined
  with `fix volume/chem/adas … stop_on_exhaust yes` this gives a clean
  "puff once, track to completion, exit" cycle.
- **`perspecies yes`** — emit one particle of each species in the
  mixture per step (incompatible with `n > 0`).
- **`region <rID>`** — additionally restrict emission to surface
  segments inside the named region.

## Pairing with `fix volume/chem/adas` tally

Use `units batch_fix <emit_ID> R_puff` on the chemistry fix so the
source tally scales from the emit fix's actual cumulative emitted
count. Lets the tally ramp up correctly during the hard-cap ramp.

## Wall normal convention

Emission along `+normal` = into the fluid. Pair with wall.surf written
by the OpenEdge converters so normals already point inward (see
CLAUDE.md "Unified wall-normal convention"). If you hand-built the
surf file, verify the walk order; otherwise use `normal no` with an
explicit `mixture vstream` vector.
