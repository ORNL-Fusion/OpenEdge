# `fix coulomb/binary` — binary Coulomb scatter between kinetic species

Pairwise Nanbu Coulomb operator between the kinetic test particles
themselves. Use when there are enough test particles per cell that
binary scatter dominates over scatter against the background plasma
(rare in dilute impurity-transport runs; more common in single-species
benchmarks or dense divertor source-driven cases).

## Syntax

```
fix ID coulomb/binary <Nevery> background <plasma_fix>
```

| arg | meaning |
|---|---|
| `<Nevery>` | apply every $N$ steps |
| `background <plasma_fix>` | the plasma fix providing $\ln\Lambda$ scaling |

The background reference is required because the Coulomb logarithm
$\ln\Lambda$ depends on local $T_e, n_e$, but the actual scatter
events happen between pairs of kinetic particles in the same cell.

## Physics

Pair-up in each cell, then apply the same Nanbu rotation as
[`coulomb/background`](coulomb_background.md) but with the partner
particle's velocity replacing the background drift. Energy and
momentum are conserved per pair; `Nevery` divides the per-step
collision frequency by `Nevery` so total scattering rate is preserved
under sub-sampling.

## Example

```
fix pd     background file plasma.h5 static yes
fix fcoulp coulomb/binary 1 background pd
```

## Files

- `src/OPENEDGE/fix_coulomb_binary.{h,cpp}`
- Common parsing: `src/OPENEDGE/fix_coulomb_base.{h,cpp}`
