# `surf_react recycle` — surface recycling

Incoming ions are neutralized and re-emitted with cosine angular
distribution at a specified energy.

## Syntax

```
surf_react ID recycle reactions_file
```

## Reaction types

| type | meaning |
|---|---|
| `E` | exchange (1 → 1) |
| `D` | dissociation (1 → 2) |
| `R` | recombination / absorption |

Each product has a specified return energy [eV]. Velocity is sampled from
a cosine distribution relative to the surface normal.

## Reactions file example

```
D+ --> D
E 1.0 3.0

D --> D
E 1.0 0.025
```

Format: `type probability energy1 [energy2]`

- `1.0` — recycling probability (100%)
- `3.0` — return energy [eV] (Franck-Condon for D atoms)
- `0.025` — thermal energy (~300 K wall temperature)
