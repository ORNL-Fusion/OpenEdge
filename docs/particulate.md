# Particulate models

OpenEdge represents a finite-size condensed object with the particle's
`radius`, `mass`, and `temp` state. The same representation covers solid
powder, molten droplets, and objects that change phase during a run.

## Commands

| Command | Purpose |
|---|---|
| [`material`](fixes/material.md) | Define density, heat capacity, phase-change, vapor-pressure, emission, and strength properties. |
| [`fix particulate/emit`](fixes/particulate_emit.md) | Inject particulates from selected surfaces. |
| [`fix particulate/charge`](fixes/particulate_charge.md) | Calculate a local charge state. |
| [`fix particulate/drag`](fixes/particulate_drag.md) | Apply ion, neutral, electric, and gravitational forces. |
| [`fix particulate/thermal`](fixes/particulate_thermal.md) | Evolve temperature, phase change, evaporation, mass, and radius. |
| [`surf_collide particulate/bounce`](fixes/surf_collide_particulate_bounce.md) | Apply sticking or restitution at a wall. |

## Recommended setup

Define particulate species with positive radius, mass, and temperature,
place those species in a dedicated mixture, and use that mixture to scope the
particulate fixes. Charged atomic species can use the Boris/GCA pusher while
finite-size particulates bypass it:

```text
global pusher mode hybrid plasma pd skip particulates

fix emit particulate/emit particulates injector ...
fix charge particulate/charge 1 background pd mixture particulates material Li
fix drag particulate/drag 1 2.0 1 background pd \
         mixture particulates material Li
fix heat particulate/thermal 1 particulates background pd \
         material Li heating auto
```

The particulate fixes use a symmetric start/end-of-step update. Keep their
`Nevery` values consistent unless a convergence study demonstrates that a
different cadence is acceptable.

## Coordinate convention

User-supplied directions use physical `(R, Z, phi)` components. OpenEdge maps
them to the active Cartesian or axisymmetric storage layout. This applies to
emission `dir` and drag `gravity` vectors.

## Validity

The charging, drag, and heating closures have different validity ranges. A
successful run does not establish that a closure is valid for a particular
particulate size. In particular, OML charging and heating require a
particulate small compared with the plasma screening scale. Use
`fix particulate/charge ... validity error` for runs that must remain inside
that approximation.
