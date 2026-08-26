# Performance: grid refinement near surface sources

> **2026-04-24 caveat:** the `fix balance ... rcb part`/`rcb time` recipes
> below crash on np>1 — the mere presence of `fix balance` in the fix
> list induces a rank segfault in source-driven decks (bisected against
> `test_diii_d_neutrals` np=4). Both weighting modes are affected. Until
> the upstream `fix_balance` / RCB issue is patched, **omit `fix
> balance` from np>1 decks** and rely on the initial
> `balance_grid rcb cell` + adapt-driven rebalance only. You will run
> imbalanced (5–20× max/min) but still get correct results and most of
> the multi-rank speedup. The `fix adapt` recipes below remain valid.

For wall-source cases (PMI sputtering, divertor emission, evaporation),
particles spawn at the wall and cluster in a small set of cells near the
strike point. Two performance bottlenecks this creates:

1. **SurfColl checks dominate Move time.** Each particle move tests against
   every surface element in its current cell. Coarse cells contain dozens
   of surface elements → dozens of intersection tests per particle per
   step.
2. **MPI load imbalance.** RCB partitions cells (atoms), so all particles
   in one cell go to one rank. With particles concentrated in a few cells,
   one rank does all the work and the rest sit in MPI barriers (`Other`
   bucket becomes 80–90% of loop time).

Both are solved by `adapt_grid` near the wall, which (a) reduces the number
of surface elements per cell and (b) gives RCB more granularity to
subdivide the dense region.

## Standard recipe

Place after `read_surf` and `surf_modify`, before any `fix` commands:

```
# define refinement region(s) covering the wall area where particles cluster
region   rdiv_lo block 1.82 3.2 -0.94 -0.40 -INF INF   # lower divertor
region   rdiv_up block 1.82 3.2  0.40  0.80 -INF INF   # limiter / upper

# refine cells that overlap surfaces inside the region
# - thresh 0.001: refine if longest surf segment in cell > 1mm
# - cells 2 2 1: each parent splits into 2x2x1 children (2D)
# - maxlevel 5 + iterate 5: actually drives 5 levels of refinement
#   (iterate is the cap, NOT maxlevel — they must match or iterate must
#    be >= maxlevel)
adapt_grid all refine surf all 0.001 maxlevel 5 cells 2 2 1 region rdiv_lo all iterate 5
adapt_grid all refine surf all 0.001 maxlevel 5 cells 2 2 1 region rdiv_up all iterate 5
balance_grid rcb cell
```

## Critical ordering for MPI runs with `gridcut 0.0`

```
create_grid 100 100 1
balance_grid rcb cell        # FIRST balance — needed for adapt_grid in MPI
read_surf ...
surf_collide / surf_react / surf_modify ...
region ... ; adapt_grid ...  # refinement after surfaces have collision models
balance_grid rcb cell        # SECOND balance — distribute refined cells
```

- Skip the first `balance_grid` and MPI runs error out with *"Cannot mark
  grid cells as inside/outside surfs because ghost cells do not exist"*.
- Run `adapt_grid` before `surf_modify` and you get *"surface elements
  not assigned to a collision model"*.

## Measured impact

### `test_west_axi`, 100×100 base grid, ~13K particles

| config | wall (1 rank) | wall (32 ranks) | SurfColl checks |
|---|---|---|---|
| no refinement | 8.72 s | n/a (load imbalance) | 83.8 M |
| ml=5 it=5 refine | 5.03 s | **0.95 s** | 8.5 M (10× fewer) |

Total speedup: ~9× on a single rank (Move halved by fewer surface checks),
~13× on 32 ranks (refinement also makes RCB partitioning effective).

### `test_diii_d_neutrals` (wall-recycling)

With `fix surface/emit/recycle` driving wall emission:

| config (np=16) | ms/step @ ~50k particles |
|---|---|
| static `adapt_grid` `maxlevel 3`, no fbal/fadapt | ~110 ms |
| `maxlevel 5` static + `fadapt 200` + `fbal rcb part` | **~15 ms** |

**~7–8× speedup**, moves bottleneck off MPI load imbalance onto actual
particle work. Per-rank particle counts become balanced (min/max ratio
~1.5 instead of ~10).

## Tuning notes

- `thresh` (third arg to `refine surf`) is the surf-element length below
  which refinement stops. Default `0.001` (1 mm) works for typical fusion
  meshes; lower only if your wall mesh has sub-mm features.
- `iterate N` is the cap on refinement depth, **not** `maxlevel`. Set
  `iterate >= maxlevel` or refinement stops early. Common mistake.
- **Wider regions outperform tight regions** even though they use more
  cells: the surface-check reduction across the *whole wall* is a bigger
  win than concentrated refinement at the strike. Prefer broad geometric
  bands over narrowly-tuned ones.
- `fix balance ... rcb time` (every 500 steps) outperforms `rcb part` for
  impurity transport — `time` weights cells by measured CPU cost,
  capturing per-particle work intensity (Boris subcycles, surf checks).
- **Exception for wall-sourced cases (`emit/surf/pmi`, `emit/surf/recycle`):**
  use `rcb part`, not `rcb time`. When the first balance fires several
  ranks have zero particle work-time (no tasks emitted yet) and the RCB
  recursion in `rcb time` mode segfaults on NaN weights. `rcb part`
  partitions by particle count instead — stable with idle ranks.
  See `memory/feedback_rcb_time_segfault.md`.

## Runtime adaptive refinement: `fix adapt`

Static `adapt_grid` only sets up the grid once. For source-driven cases
where the cluster grows over time (more particles emitted than removed),
add `fix adapt` to refine cells whose particle count exceeds a threshold
*during* the run. Combined with `fix balance ... rcb time`, this gives an
extra **2–4× speedup** on top of the static refinement at moderate rank
counts (np=4–16).

### Recipe

```
# Runtime adaptive refinement: split any cell holding > 500 particles.
# Refines monotonically (no coarsen) so the grid never loses granularity.
# Capped by maxlevel + setup adapt_grid.
fix fadapt adapt 200 all refine particle 500 0 maxlevel 8 cells 2 2 1
fix fbal   balance 200 1.1 rcb time
```

- The `0` is the coarsen threshold (always required by the parser, even
  with `refine` only — leave at 0 for monotonic refinement).
- `200` (Nevery for fadapt and fbal) is the sweet spot. More aggressive
  settings (every 100, threshold 200) actively hurt — refine + balance
  overhead overwhelms the gain.
- `fix balance` alone every 200 steps does **not** help; the win requires
  fadapt + balance working together. Don't drop balance frequency without
  also enabling fadapt.

### Measured impact (`test_west_axi`, ~130K particles, nlaunch=100)

| ranks | static refine only | + fix adapt | speedup from fadapt |
|-------|--------------------|-------------|---------------------|
| 4     | 36.8 s              | 19.6 s       | 1.88× |
| 8     | 30.0 s              | **9.7 s**    | **3.09×** |
| 16    | 21.2 s              | 5.6 s        | 3.78× |
| 32    | 14.3 s              | ~7.0 s       | 2.0× (variance) |

Total speedup vs single rank with no refinement: **~20× at np=16** for
this case. np=8 is the practical sweet spot for divertor-source cases at
this resolution — beyond that, refinement granularity becomes the new
bottleneck.

### Critical ordering for `fix adapt`

```
# ---- Diagnostics FIRST ----
compute cden grid all species nrho
fix    fden ave/grid all 1 1000 1000 c_cden[*] ave one
fix    frate ave/grid all 1 1000 1000 f_fchem[*] ave one

# ---- Runtime refinement + balance AFTER all fix ave/grid ----
fix fadapt adapt 200 all refine particle 500 0 maxlevel 5 cells 2 2 1
fix fbal   balance 500 1.1 rcb part
```

The code errors out with *"Fix adapt must come after fix ave/grid"* if the
ordering is wrong. `fix adapt` invalidates any cell-to-value mapping
every time it refines, so the `ave/grid` fixes must already be set up to
hear about it.
