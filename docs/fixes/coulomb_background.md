# `fix coulomb/background` — Coulomb scatter against background plasma

Pitch-angle scattering of kinetic test particles off the background
plasma using the Nanbu test-particle Coulomb operator. Background `Te,
ne` (and optionally `Ti, ni, V_∥, B`) are read from `fix background`
or per-cell computes; the operator applies an isotropised energy /
angle update consistent with the local Spitzer collision frequency.

## Syntax

```
fix ID coulomb/background <Nevery> background <plasma_fix> \
    [<extra-keywords>]
```

| arg | meaning |
|---|---|
| `<Nevery>` | apply the operator every $N$ steps |
| `background <plasma_fix>` | the `fix background` ID providing $T_e$, $n_e$, $T_i$, $n_i$, $V_\|$, $B$ at the particle position |

Alternative form pulling the background from explicit per-cell
computes:

```
fix ID coulomb/background <Nevery> plasma <Te-src> <Ne-src>
```

with `<Te-src>`, `<Ne-src>` of the form `c_<compute>[col]`.

## Physics

Spitzer collision frequency

$$
\nu_s = \frac{n_e\, e^4\, \ln\Lambda}
            {4\pi\, \varepsilon_0^2\, m_p\, v^3} ,
$$

is sampled per timestep; the operator rotates each test-particle
velocity by a Gaussian-distributed angle of width
$\sqrt{2 \nu_s \Delta t}$ and applies an energy update toward the
background drift / temperature. See Nanbu (PRE 1997) for the
single-particle scheme. Coefficients are taken from `fix background`
when available.

## Example

```
fix pd     background file plasma.h5 static yes
fix fcoulb coulomb/background 5 background pd
```

`Nevery = 5` is a typical compromise between fidelity and cost; for
strongly-collisional regions drop to 1.

## Files

- `src/OPENEDGE/fix_coulomb_base.{h,cpp}` — common parsing + RNG setup
- `src/OPENEDGE/fix_coulomb_background.{h,cpp}` — test-particle operator
- Companion: [`fix coulomb/binary`](coulomb_binary.md) for binary
  (test-test) Coulomb scatter.
