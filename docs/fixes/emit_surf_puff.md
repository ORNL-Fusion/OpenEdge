# `fix emit/surf/puff` — hard-capped surface emission

Like `fix emit/surf` but with a hard cap on total emitted particles, so a
one-shot EIRENE-style batch can terminate cleanly. Designed to pair with
`fix chem/adas … mode neutral` (Mode A).

## Syntax

```
fix ID emit/surf/puff mixture group \
    [n <N_per_step>] [normal yes|no] [stop_at_np <N_total>] \
    [perspecies yes|no] [region <rID>]
```

- `n 200` — emit 200 particles per step (CONSTANT mode). `n 0` reverts to
  flow-based emission using the mixture vstream.
- `normal yes` — inject along the surface normal (vs. mixture vstream).
- `stop_at_np N` — latch emission off once `N` total particles have been
  emitted. Once latched, no further injection; combined with
  `fix chem/adas … stop_on_exhaust yes` this gives a clean
  "puff once, track to completion, exit" cycle.

## Pairing with `fix chem/adas` tally

Use `units batch_fix ID R_puff` on `fix chem/adas` so the tally scales from
the paired emit fix's actual cumulative count.
