# Plan: WarpX-generated IEAD lookup (future)

The current `test_iead/` validates the OpenEdge sheath model against a
custom Fortran Boris tracker. That's fine as a unit test but it doesn't
give us a drop-in `f(E, θ, φ | τ, ζ, α_B)` lookup table for production
runs. This note captures the plan to generate our own — the same role
STYX's `Sheathdata/sheath1D_database_{means,distributions}` plays in
SOLEDGE3X-EIRENE, except STYX ships an empty reader on this cluster and
the IRFM-generated Mellet 2017 tables aren't available here.

## Why our own

- Mellet 2017 (Plasma Phys. Control. Fusion 59 035006) provides **no fit
  for the joint f(E, θ)**. Only averages `<E_imp>` (Eq. 8 + Table 1) and
  qualitative scalings for `<α>`. The paper itself states that the full
  distribution is tabulated from PIC and used as a lookup.
- STYX has a working 5D sampler (`styx/src/styx_sample_sheath1D.f90`,
  case 4) but the `Sheathdata/` tables are not in the S3X checkout.
  All local `eirene_coupling.txt` files run `sheath_model = 0` — the
  classic shifted-Maxwellian + `2 Ti + 3 Te` path — so the PIC lookup
  is dormant in practice.
- Generating our own means:
  (a) we match our exact species set (D+, Ne0..Ne10+, Li0..Li3+, W0..W20+, …),
  (b) we own the data and can regenerate as physics choices evolve,
  (c) the output format can be native HDF5 in the plasma.h5 convention.

## Code choice: WarpX

- 1D Cartesian electrostatic with a static tilted magnetic field is a
  native mode in WarpX (`warp.picmi`).
- Absorbing wall BC + per-particle impact diagnostics give direct
  (E, v, x) at impact. Standard in WarpX examples.
- Actively maintained at LBNL, GPU-capable, Python PICMI input decks
  — easy to script parameter sweeps.
- FBPIC was considered and rejected: it's Fourier-Bessel cylindrical,
  designed for laser-plasma, a bad fit for a slab sheath with tilted B.

Alternative for the first pass: a ~300-line custom NumPy Boris + Poisson
1D-ES code. Faster to iterate than wiring up WarpX, and at this
geometric simplicity the output is identical. WarpX only wins once we
want self-consistent impurity back-reaction on the potential (later).

## Parameter grid (following Mellet 2017)

| Axis   | Range                         | Count |
|--------|-------------------------------|-------|
| τ      | {0.5, 1, 2, 5}                | 4     |
| α_B    | {1°, 2°, 3°, 4°}              | 4     |
| ζ      | log-spaced in [10⁻¹, 10²]     | ~15   |

~240 runs per species. Each 1D-ES sheath is seconds on CPU. Most
computationally significant is the species count: per Mellet, the
sheath-entrance parallel-velocity distribution `f_||,se` has to be
solved once per (A, Z) via the Chung-Hutchinson multispecies Vlasov —
the table itself is then indexed only by (τ, ζ, α_B) because the
species bit enters at sampling time.

Start narrow: **D+ only**, then expand to the species we actually need
(W charge states for erosion, Ne/Li for impurity transport).

## Output format (HDF5)

```
/sheath_iead/
    tau          (N_tau,)           [dimensionless]
    ksi          (N_ksi,)           [dimensionless, = ζ]
    alphaB       (N_alphaB,)        [degrees]
    E_axis       (N_E,)             [normalised by Te]
    alpha_axis   (N_alpha,)         [degrees]
    beta_axis    (N_beta,)          [degrees]
    E_mean       (N_tau, N_ksi, N_alphaB)                     [Te units]
    alpha_mean   (N_tau, N_ksi, N_alphaB)                     [rad]
    beta_mean    (N_tau, N_ksi, N_alphaB)                     [rad]
    Edistr       (N_E, N_tau, N_ksi, N_alphaB)
    adistr       (N_alpha, N_E, N_tau, N_ksi, N_alphaB)
    bdistr       (N_beta, N_alpha, N_E, N_tau, N_ksi, N_alphaB)
```

This mirrors the layout read by `styx_sample_sheath1D.f90` case 4
(`sheath1D(isurf)%{E, alpha, beta, Edistr, adistr, bdistr}`), so an
equivalent C++ sampler in OpenEdge can be a line-by-line port.

## Consumer in OpenEdge

New fix `fix emit/surf/sheath_iead` (or extension to `fix emit/surf/pmi`):
given per-surface (τ, ζ, α_B) computed from the local plasma (via
`fix background`) and B-field projection, sample (E, α, β) from the
HDF5 table and emit the particle along the wall normal with the
appropriate velocity rotation.

## Validation before mass production

Must reproduce:

- Mellet Fig 3: `f_||,se` for D+ and Ne10+ at τ=1 (sheath-entrance
  parallel velocity distributions)
- Mellet Fig 4: `<E_||,se>/k_B T_i` vs (√A, Z) with the 2D quadratic
  fit from their Table 1
- Mellet Fig 7: `<α_sh>` low-density (ζ=0.1) and high-density (ζ=150)
  asymptotes vs 1/Z and √(A/Z)

Once those match to ~10%, roll to the full parameter grid.

## Status

**Not started.** File-holder. To pick up: start with a 1D NumPy Boris +
Poisson to reproduce Fig 3 first (fast iteration), then port to WarpX
only when we commit to production-quality data generation.
