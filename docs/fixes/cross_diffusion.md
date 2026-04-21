# `fix cross_diffusion` — anomalous perpendicular diffusion + pinch

Applied as position displacements at `END_OF_STEP`.

## Syntax

```
fix ID cross_diffusion Nevery \
    bfield BxSRC BySRC BzSRC \
    [D_perp VAL | bohm TeSRC [scale VAL]] \
    [pinch Vr Vz] \
    [gradient_pinch Cp neSRC gradNeR gradNeZ]
```

## Diffusion modes

- **Constant** — `D_perp 1.0` gives D⊥ = 1.0 m²/s.
- **Bohm** — `bohm c_cplasma[Te_col] scale 0.1` gives `D = scale · Te / (16eB)`.
  Default `scale = 1.0`.

## Pinch modes

- **Constant** — `pinch -50.0 0.0` adds a constant velocity in (R, Z).
- **Gradient-driven** — `gradient_pinch Cp neSRC gradNeR gradNeZ` gives
  `V = Cp · D⊥ · ∇⊥(ne)/ne`. Typical `Cp = 1–3` (ITG turbulence).

## Dimensionality

- **2D**: displacement in poloidal perpendicular direction only.
- **3D**: two perpendicular directions via Gram-Schmidt.

Particles that diffuse outside the domain are reverted (no loss).

## Gradient source

`compute plasma/fields` output columns `grad_ne_r`, `grad_ne_z` provide the
electron density gradient (finite differences).

**Mesh-only plasma.h5 caveat (2026-04-20+)**: `grad_ne_{r,z}` return zero
on mesh-only files. Migrate to per-particle FD queries against
`fix plasma/data` until gradient support returns.
