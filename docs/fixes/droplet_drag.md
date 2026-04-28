# `fix droplet/drag` — drag force on droplets

Adds Epstein (free-molecular) or Coulomb-collisional drag to each
droplet, plus optional gravity. The drag force is applied as a
velocity update per timestep using the local plasma flow and
temperature from `fix background`.

## Syntax

```
fix ID droplet/drag <Nevery> <Z_bg> <A_bg> background <plasma_fix> \
    [model epstein|coulomb] \
    [coulomb/chi <chi>] [coulomb/delta <delta>] [coulomb/lnlambda <ll>] \
    [gravity <gx> <gy> <gz>] \
    [radius <r>] [mass <m>] [temp <T>]
```

| arg | meaning | default |
|---|---|---|
| `Z_bg, A_bg` | background ion charge + mass (amu) | required positional |
| `background <fix_id>` | plasma source for $\rho, T, V_\|$ | required |
| `model epstein` | free-molecular drag (mean free path ≫ droplet) | default |
| `model coulomb` | charged-particle pickup with three tuning constants | optional |
| `gravity <gx> <gy> <gz>` | extra body-force in physical (R, Z, φ) coords | `0 0 0` |
| `coulomb/chi`, `coulomb/delta`, `coulomb/lnlambda` | Coulomb-mode parameters | model-specific |

## Physics

**Epstein drag** — Northrup–Stannard form for a sphere in a flowing
collisionless plasma:

$$
\mathbf{F}_d = -\frac{8}{3}\sqrt{\tfrac{2\pi m_i T_i}{\pi}}
              \, n_i\, r^2\, (\mathbf{v}_d - \mathbf{V}_\|).
$$

**Coulomb drag** — used when the droplet is strongly charged so the
Yukawa-screened cross-section dominates. The three constants
$(\chi, \delta, \ln\Lambda)$ pick between standard parametrisations
(Khrapak vs. Khrapak-Morfill, screening mode).

## Example

```
fix pd    background file plasma.h5 static yes
fix fdrag droplet/drag 1 1 2.0 background pd model epstein gravity 0 -9.81 0
```

## Files

- `src/OPENEDGE/fix_droplet_drag.{h,cpp}`
- See `examples/test_droplet_drag` for the validation case.
