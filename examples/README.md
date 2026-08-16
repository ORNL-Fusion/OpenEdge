# OpenEdge examples

The public examples are organized by purpose. Each leaf directory is intended
to be runnable on its own and normally contains an `in.*` deck plus its input
data, checks, or plotting scripts.

## Verification

`verification/` contains focused checks that ask whether OpenEdge solves the
implemented model correctly. They use analytical solutions, numerical
references, or code-to-code comparisons with deterministic PASS/FAIL criteria.
These are the best cases for regression testing. Comparisons against physical
experiments are validation and should be labeled separately.

| Directory | Purpose |
|---|---|
| `verification/collisions/coulomb/` | Coulomb slowing-down and binary thermalization |
| `verification/efield_polarization/` | Polarization-drift verification |
| `verification/ionization_recombination/` | ADAS ionization, recombination, and charge exchange |
| `verification/particulates/dustt/` | Grain dynamics comparison with DUSTT |
| `verification/particulates/droplet_transport/` | Droplet-mover integration gate in CAT geometry |
| `verification/pushers/orbit/` | Boris and GCA orbit verification |
| `verification/pushers/hybrid/` | Hybrid near-wall pusher verification |
| `verification/surface_emission/constant_flux/` | Constant-flux emission and cadence scaling |

## Workflows

`workflows/` contains larger scientific cases, device applications, and
notebook-driven demonstrations. They can participate in smoke tests, but are
primarily intended to demonstrate complete modeling workflows.

| Directory | Purpose |
|---|---|
| `workflows/particulates/cat_liquid_metal_divertor/` | CAT liquid-metal surface sources and droplet response |
| `workflows/particulates/st40_lithium_powder_dropper/` | ST40 lithium-powder dropper |
| `workflows/particulates/west_boron_powder_dropper/` | WEST boron-powder injection |
| `workflows/impurity_transport/rfpie_tungsten_transport/` | RFPIE tungsten sputtering and transport |
| `workflows/impurity_transport/west_tungsten_transport/` | Axisymmetric WEST tungsten transport |

ParaView integration examples live under `visualization/paraview/`.

## Local WIP

Unpublished or incomplete research cases belong under `wip/`. The complete
directory is ignored by Git and is not part of the public repository. Do not
rely on that directory as the only copy of important work: ignored files can
be removed by `git clean -fdX`.

Use `NOTES.local.md` or a `*.local.pdf` suffix for private notes stored inside
an otherwise public case. Public case documentation should use `README.md`.

## Running a case

From the selected leaf directory:

```bash
mpirun -np 4 /path/to/spa_mpi -in in.case_name
```

Run the registered smoke-test suite from the repository root:

```bash
./regression/run_regression.sh --exe /path/to/spa_mpi
```

Generated logs, dumps, figures, and `output/` directories are ignored. Large
background files may need to be generated locally; consult the leaf
`README.md` for case-specific instructions.

## Naming rules

- Use lowercase `snake_case` directory names.
- Name a leaf for the physics or workflow, not the person developing it.
- Put focused PASS/FAIL checks in `verification/` and complete applications in
  `workflows/`.
- Use `in.openedge` for a workflow's single canonical deck.
- Keep simulation dependencies in `input/`, analysis notebooks and helpers in
  `scripts/`, generated products in `output/`, and local-only cases in `wip/`.
- Name species files by role: `atoms.species`, `droplets.species`, or
  `grains.species`.
