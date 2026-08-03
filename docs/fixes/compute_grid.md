# `compute grid` — per-cell DSMC moments

OpenEdge's `compute grid` matches SPARTA upstream's
[`compute grid` API](https://sparta.github.io/doc/compute_grid.html)
exactly — same 22 value keywords (`n`, `nrho`, `nfrac`, `mass`,
`massrho`, `massfrac`, `u`, `v`, `w`, `usq`, `vsq`, `wsq`, `ke`,
`temp`, `erot`, `trot`, `evib`, `tvib`, `pxrho`, `pyrho`, `pzrho`,
`kerho`), same per-group output, same `post_process_grid` machinery.

The OpenEdge override carries one **upstream bug-fix patch** — a
bounds check on `particles[i].icell` against `nglocal` before the
tally write. Without it, after `fix balance` migrates cells (or
`fix adapt` resizes the grid), particles can briefly hold an `icell`
from the prior layout, causing a heap-use-after-free that
nondeterministically segfaults large multi-rank runs (caught at scale
on a 64-rank `test_diii_d_neutrals` case). The patch skips any
particle with `icell ∉ [0, nglocal)`; it is resorted into a valid
cell on the next sort/move.

## Syntax

See the [SPARTA upstream
documentation](https://sparta.github.io/doc/compute_grid.html) for
the full API and value-keyword reference. There are no OpenEdge-
specific keywords or behavioural differences beyond the patch above.

## When to use `compute grid/weighted` instead

If your run uses `fix particle/weight` (i.e. weighted-launch emit
fixes are active), use [`compute grid/weighted`](compute_grid_weighted.md)
— that variant accumulates `pweight` per particle so densities and
moments are correctly normalised across spatially-varying emission
rates. With `compute grid`, all particles count equally regardless
of their weight, which is wrong for source-driven cases.

## Files

- `src/OPENEDGE/compute_grid.{h,cpp}` — 99% upstream + the icell
  bounds check
- Mirrored to `src/compute_grid.{h,cpp}` per the three-copy rule
