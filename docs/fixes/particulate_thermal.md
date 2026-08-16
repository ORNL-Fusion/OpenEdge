# `fix particulate/thermal`

Evolve particulate temperature, phase change, evaporation or sublimation,
mass, and radius. The model can optionally launch evaporated atoms and apply
an asymmetric evaporation-recoil force.

## Syntax

```text
fix ID particulate/thermal Nevery MIXTURE_ID background BACKGROUND_ID \
    [material NAME] [heating flux|oml|auto] \
    [heatflux/scale S] [ion_mass_amu M] [twall_K T] \
    [alpha_e A] [rocket_eta ETA] [emit_into MIXTURE_ID]
```

## Heating closures

- `heating flux` uses
  `Q = 0.25 sqrt(q_par^2 + q_perp^2)` from the plasma background. This is
  the default and represents an imposed fluid heat-flux closure.
- `heating oml` uses local electron and ion collection. It consumes the
  charge from `fix particulate/charge` when available and otherwise uses a
  hydrogenic floating-potential estimate.
- `heating auto` selects OML collection when the Debye length is defined and
  terminates if `R_d/lambda_D > 1`, because a finite-size collection closure
  is not yet implemented.

`heatflux/scale` multiplies the selected heat input and defaults to `1`.
The two closures represent different physical assumptions; `auto` is a
validity guard, not an experimental calibration.

## Thermal and mass model

The surface energy balance contains plasma heating, radiative exchange with
`twall_K`, and latent heat carried by the evaporated flux. Material density,
heat capacity, emissivity, melting point, latent heats, atomic mass, and
Antoine coefficients come from `material`.

Evaporation uses an Antoine vapor pressure and Hertz-Knudsen flux with
accommodation coefficient `alpha_e`. An apparent-heat-capacity band handles
melting. Adaptive internal substeps limit each update to 25 K and 2% radius
change.

- `material` defaults to `Li`.
- `twall_K` defaults to `300` K.
- `ion_mass_amu` defaults to `2.0`.
- `alpha_e` defaults to `1.0` and must lie in `(0, 1]`.
- `rocket_eta` defaults to zero and must lie in `[0, 1]`.
- `emit_into` creates kinetic vapor particles in the selected mixture.

The global scalar reported by the fix is the cumulative number of physical
atoms evaporated across all ranks.

## Restrictions

The current implementation supports 2D Cartesian and 2D axisymmetric
geometry. Quantitative OML heating requires the particulate to remain inside
the OML validity range.

## Example

```text
fix heat particulate/thermal 1 particulates background pd \
         material Li heating auto alpha_e 1.0 \
         twall_K 300 emit_into lithium_vapor
```
