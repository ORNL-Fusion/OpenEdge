# `fix droplet/viscous` — viscous force on droplets

Viscous drag in the magnetised limit, including ion-collisional and
Coulomb-pickup contributions. Complements
[`fix droplet/drag`](droplet_drag.md) (Epstein / collisionless drag)
when the droplet is moving slowly enough that thermal collisions
dominate momentum transfer.

## Syntax

```
fix ID droplet/viscous <Nevery> <A_bg> <Z_bg> \
    plasma <Te-src> <Ti-src> <Ni-src> <Vpar-src> \
    bfield <Br-src> <Bt-src> <Bz-src> \
    [model epstein|coulomb] \
    [coulomb/chi <chi>] [coulomb/delta <delta>] [coulomb/lnlambda <ll>] \
    [gravity <gx> <gy> <gz>] \
    [mass <m>] [radius <r>] [temp <T>] \
    [diag yes|no] [diag/every <N>]
```

| arg | meaning |
|---|---|
| `A_bg, Z_bg` | background ion mass (amu) + charge |
| `plasma <Te> <Ti> <Ni> <Vpar>` | per-cell sources, typically `c_<plasma_fields>[col]` |
| `bfield <Br> <Bt> <Bz>` | per-cell B sources |
| `model epstein\|coulomb` | drag-physics selector |
| `gravity <gx> <gy> <gz>` | optional body-force in physical (R, Z, φ) |
| `diag yes` + `diag/every N` | enable per-droplet diagnostic dump every N steps |

The four plasma sources and three B sources match
`compute plasma/fields` columns; you can chain them through that
compute or read directly from `fix background`.

## Example

```
fix pd  background file plasma.h5 static yes
compute cpf plasma/fields all background pd te ti ni upar br bt bz
fix fvis droplet/viscous 1 2.0 1 \
    plasma c_cpf[1] c_cpf[2] c_cpf[3] c_cpf[4] \
    bfield c_cpf[5] c_cpf[6] c_cpf[7] \
    model coulomb gravity 0 -9.81 0
```

## Files

- `src/OPENEDGE/fix_droplet_viscous.{h,cpp}`
- See `examples/test_droplet_viscous_axi` and `examples/test_droplet_viscous_3d`.
