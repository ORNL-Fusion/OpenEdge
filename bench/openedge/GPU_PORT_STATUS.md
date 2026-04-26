# OpenEdge Kokkos coverage

Status of GPU/Kokkos backends for OpenEdge fixes and computes, ordered
by impact on the test_west_axi hot path. ✓ = working; ⚠ = stale or
broken; — = no Kokkos version yet.

## Working

| Style | Source | Notes |
|---|---|---|
| `compute nearest_surf/grid/kk` | `OPENEDGE/KOKKOS/compute_nearest_surf_grid_kokkos.{cpp,h}` | sheath geometry lookups |
| `compute plasma/fields/kk` | `OPENEDGE/KOKKOS/compute_plasma_fields_kokkos.{cpp,h}` | E, B, gradients |
| `compute surface/physical/sputter/kk` | `OPENEDGE/KOKKOS/compute_surface_physical_sputter_kokkos.{cpp,h}` | Eckstein yields |
| `fix volume/chem/adas/kk` | `OPENEDGE/KOKKOS/fix_volume_chem_adas_kokkos.{cpp,h}` | needs verification after style-typo fix |

## Missing — high priority for test_west_axi

These run per-particle every step on the impurity hot path. Without
Kokkos versions, GPU runs shuttle particles host↔device on each step.

| Style | Effort | Notes |
|---|---|---|
| `fix coulomb/binary/kk` | ~1 week | replaces deleted `coll/nanbu/kk`; partial reuse possible |
| `fix coulomb/background/kk` | ~1 week | as above |
| `fix force/thermal/kk` | 2–3 days | per-particle, point-query plasma |
| `fix cross_field_diffusion/kk` | 2–3 days | per-particle random walk + pinch |
| `surf_react surface/pwi/kk` | ~1 week | TRIM table interpolation, branch-heavy |
| `fix surface/emit/source/kk` | 2–3 days | per-segment Bohm-flux source |
| `fix background/kk` | ~3 days | mesh interpolation kernels (already mostly device-side via plasma/fields) |

## Missing — lower priority

Diagnostics and surface state — invoked less frequently or used only
in specific decks.

| Style | Effort | Notes |
|---|---|---|
| `compute grid/weighted/kk` | 1 day | small variant of stock `compute grid/kk` |
| `compute volume/emissivity/grid/kk` | 2–3 days | PEC table interpolation |
| `compute surface/chemical/{adatom,evaporation}/kk` | 2–3 days each | LM-deck only |
| `fix bfield/{grid,particle}/kk` | 1 day | replaced in most decks by `compute plasma/fields/kk` |
| `fix efield/{grid,particle}/kk` | 1 day | same |
| `fix particle/weight/kk` | 1 day | trivial — pweight holder, no per-step loops |
| `fix surface/emit/{puff,recycle}/kk` | 2–3 days each | similar to `surface/emit/source` |
| `fix surface/state/lm/kk` | low | LM physics, called rarely |

## Skipped (CPU-only by design)

The droplet/dust kinetics and inner-boundary reflection do not need
GPU acceleration --- droplets are relatively few (<10⁴) and reflect/psi
runs at boundary crossings only.

| Style | Reason |
|---|---|
| `fix droplet/{charge,drag,emit,evaporate,viscous}` | small particle counts |
| `fix reflect/psi` | per-cross only, not per-step |
| `fix force/gravity` | trivial cost |

## How to verify Kokkos coverage matches an active deck

Run the deck under Kokkos:

```bash
mpirun -np 4 ~/buildOpenEdge/src/spa_kokkos_cuda \
       -k on g 4 -sf kk -in in.west
```

The `-sf kk` switch makes SPARTA prefer the `/kk` style for any
fix/compute that has one. Watch the log for any warning about a fix
falling back to the CPU path --- that's where the host↔device shuffle
happens.
