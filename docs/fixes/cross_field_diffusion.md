# `fix cross_field_diffusion` — anomalous perpendicular diffusion + pinch

Applied as position displacements at `END_OF_STEP`:

```
dx_perp (2D) = sqrt(2 · D_perp · dt) · ξ · ê_perp
dx_perp (3D) = sqrt(2 · D_perp · dt) · (ξ₁ · ê₁ + ξ₂ · ê₂)
```

plus an optional convective pinch for impurity ions.

## Syntax

Two modes depending on how B-field and gradient sources are plumbed.

### Mode A — `background` (recommended, mesh-native)

```
fix ID cross_diffusion Nevery background <plasma_fix_ID> \
    [D_perp VAL | bohm [scale VAL]] \
    [pinch Vr Vz] \
    [gradient_pinch Cp]
```

Reads B and (if needed) Te / ne from a `fix background` instance.
Gradient_pinch in this mode uses per-particle finite differences on
`pd` instead of explicit variable inputs.

### Mode B — legacy explicit sources

```
fix ID cross_diffusion Nevery \
    bfield BxSRC BySRC BzSRC \
    [D_perp VAL | bohm TeSRC [scale VAL]] \
    [pinch Vr Vz] \
    [gradient_pinch Cp neSRC gradNeR gradNeZ]
```

## Diffusion models

- **Constant** — `D_perp 1.0` gives D⊥ = 1.0 m²/s.
- **Bohm** — `bohm [scale 0.1]` gives `D = scale · Te / (16 e B)`.
  Default `scale = 1.0`. In `background` mode, Te is read from `pd`
  directly; in Mode B, pass `bohm c_cplasma[Te_col]`.

## Pinch modes

- **Constant** — `pinch -50.0 0.0` adds a constant velocity in
  `(R, Z)` [m/s].
- **Gradient-driven** — `gradient_pinch Cp` gives
  `V = Cp · D⊥ · ∇⊥(ne)/ne`. Typical `Cp = 1–3` (ITG turbulence).
  In `background` mode, ∇ne is computed by FD on `pd`; in Mode B
  you pass explicit `neSRC gradNeR gradNeZ`.

## Dimensionality

- **2D**: displacement in the poloidal perpendicular direction only.
- **3D**: two perpendicular directions via Gram-Schmidt orthogonalisation
  of `b̂`.

Particles that diffuse outside the SPARTA domain have their step
reverted on the next check (no loss from diffusion alone; wall
collisions still apply normally).

## Gradient source compatibility

| plasma.h5 layout | Mode A (`background`) | Mode B (explicit srcs) |
|---|---|---|
| mesh-only (new converters, 2026-04-22+) | ✅ FD via `pd->interp2D` on mesh | ❌ `compute plasma/fields` returns 0 for grad_ne on mesh-only |
| legacy regular-grid | ✅ | ✅ |

**Prefer Mode A.** Mode B is kept only for legacy decks.

Note: the D_perp + constant-`pinch` path needs only B, so it works on
mesh-only plasma.h5 with either mode. Only `gradient_pinch` and `bohm`
pull Te / ne / grad_ne and therefore care about the layout.

## Related

- `fix force/thermal` — Braginskii parallel forces on impurities
  (complementary to this perpendicular channel).
- `fix coulomb/binary` — Coulomb scatter; adds isotropic velocity
  scatter, while this fix adds perpendicular position diffusion.
- The convergence scan in
  `examples/test_west_axi/analysis/run_convergence_scan.sh` (pre-
  cleanup) A/B-compared decks with and without this fix on the WEST
  density map.
