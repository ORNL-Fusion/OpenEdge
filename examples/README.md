# OpenEdge examples

Each leaf directory is runnable on its own: an `in.*` deck, its `input/`
data, and a `scripts/` folder with checks or plots. Outputs land in
`output/` and are ignored by git.

```bash
cd examples/verification/pushers/orbit
./run.sh /path/to/spa_mpi
```

Smoke-test the whole suite from the repository root:

```bash
./regression/run_regression.sh --exe /path/to/spa_mpi
```

## Verification

Focused checks that ask whether OpenEdge solves the implemented model
correctly, against analytical solutions, numerical references, or other
codes. Every case has a deterministic PASS/FAIL gate.

| Directory | Purpose |
|---|---|
| `verification/collisions/coulomb/` | Coulomb slowing-down and binary thermalization |
| `verification/efield_polarization/` | Polarization drift in a time-dependent E field |
| `verification/ionization_recombination/` | ADAS ionization and recombination balance |
| `verification/particulates/dustt2005/uniform_plasma_benchmark/` | DUSTT-2005 grain model against analytic reference results |
| `verification/particulates/dustt2005/cat_solps_droplet_transport/` | DUSTT-2005 droplet transport in CAT geometry |
| `verification/particulates/dis2021/uniform_plasma_comparison/` | DIS-2021 verification and controlled comparison with DUSTT-2005 |
| `verification/pushers/orbit/` | Boris and GCA orbits |
| `verification/pushers/hybrid/` | Boris/GCA near-wall handoff |
| `verification/surface_emission/constant_flux/` | Constant-flux emission and cadence scaling |
| `verification/surface_pwi/deposit_tagging/` | Deposit tagging and layer seeding |

## Workflows

Device-scale cases that demonstrate a complete modeling chain.

| Directory | Purpose |
|---|---|
| `workflows/particulates/cat_liquid_metal_divertor/` | CAT liquid-metal surface sources and droplets |
| `workflows/particulates/st40_lithium_powder_dropper/` | ST40 lithium powder injection |
| `workflows/particulates/west_boron_powder_dropper/` | WEST boron powder injection |
| `workflows/impurity_transport/rfpie_tungsten_transport/` | RFPIE tungsten sputtering and transport |
| `workflows/impurity_transport/west_tungsten_transport/` | Axisymmetric WEST tungsten transport |

## Visualization

| Directory | Purpose |
|---|---|
| `visualization/paraview/` | Native grid, surface, and particle VTK XML output; no Python conversion required |

## Local work

Unpublished cases go under `wip/`, which git ignores entirely. Keep a
second copy of anything important there.

## Naming rules

- lowercase `snake_case` directory names, named for the physics, not the author
- `in.openedge` for a workflow's canonical deck
- `input/` for simulation dependencies, `scripts/` for checks, plots and notebooks, `output/` for generated products
- species files named by role: `atoms.species`, `droplets.species`, `grains.species`
