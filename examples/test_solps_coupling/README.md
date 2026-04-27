# test_solps_coupling — OpenEdge ↔ SOLPS-ITER coupling for Li sources

OpenEdge runs Li droplet transport (Antoine evaporation + Epstein drag +
OML charging + Boris/sheath pusher) against a SOLPS-ITER plasma background.
Each coupling chunk:

1. **OpenEdge** advances droplets by `n_oe_steps` against the current
   `plasma.h5`. Per-step particle dump → `coupled_test/openedge/output/particles.txt`
   carries the per-cell Li source rate (mass loss × atomic mass) for the chunk.
2. **Driver** (`tools/coupling/openedge_solps_driver.py`) reads the OpenEdge
   particle dump, integrates the Li source onto the SOLPS B2 grid, and
   relaxes it into SOLPS as a volumetric source term.
3. **SOLPS** runs `n_solps_steps` to update the plasma. Driver calls
   `solps.write_plasma_h5(plasma_h5, nr, nz)` to regenerate the regular-grid
   `plasma.h5` for the next OpenEdge chunk.

Repeats `n_coupling_steps` times.

## Layout

```
coupled_test/                 active coupling workflow
  config.json                 driver knobs (chunk lengths, MPI ranks, paths)
  run_coupling.sh             entry point: ./run_coupling.sh
  run_solps.csh               wrapper invoked by the driver
  openedge/
    in.coupled                template deck (driver edits Ndump per chunk)
    plasma.h5                 regenerated each chunk by solps.write_plasma_h5
    output/particles.txt      OpenEdge → SOLPS source data
input/                        shared by all decks
  wall.surf, core.surf        SPARTA surface geometry (2D Cart, x=R, y=Z)
  particle.source             50-droplet launch state (2.5 mm Li)
  droplet.species             Li droplet species table
analysis/                     trajectory / mass-loss / convergence plotters
matlab/, *.m                  Matlab post-processors
source.py                     particle.source generator (regenerate if you
                              need a different launch distribution)
test_coupling_api.cpp         C-API smoke test for openedge_extract_fix /
                              get_ngrid / reload_plasma (build standalone)
```

## Run

Requires SOLPS-ITER tree at `/home/cloud/local/solps/solps-iter-3.0.8-devel`
and a SOLPS case at `coupled_test/solps/v0_1/` (gitignored — bring your own).

```bash
cd coupled_test
./run_coupling.sh
```

## Physics knobs in `in.coupled`

- `heatflux_scale` — multiplier on the heat-flux fed into Antoine evaporation.
  2.0 matches the reference run; lower values lengthen droplet lifetime.
- `rocket_eta 0.5` — recoil efficiency from asymmetric evaporation along
  −∇Te. Set to 0 to disable.
- `model epstein` on `fix droplet/drag` — Epstein subsonic drag. `coulomb`
  also available.

## Coordinate layout

SPARTA-native axisymmetric (`boundary o ao p`, `x = Z`, `y = R`,
`z = phi`). Wall and core surfaces have line endpoints in `(Z, R)`
(see CLAUDE.md axi conventions). `wall.surf` is the same SOLPS-Li
geometry used by `examples/test_droplet/`; `core.surf` is the
psi_norm = 0.90 contour around the magnetic axis.

## Adding more Li source channels

The current deck tracks **droplet evaporation** only. To add other Li source
terms (e.g. physical sputtering of solid Li, ad-atom desorption from the
liquid film, plasma-induced surface ionization), add the matching fix:

| Channel | Fix | Notes |
|---|---|---|
| droplet evap (here) | `fix droplet/evaporate` | already wired |
| physical sputter | `compute surface/physical/sputter` + `fix surface/emit/source` | needs `database/processes.h5` Li sputter tables |
| LM Antoine + ad-atom | `fix liquid_metal` | needs Tsurf field |
| recycling neutrals | `fix surface/emit/recycle` | tracked separately, not a Li channel |

Each new source dumps into the same `output/particles.txt`; the driver's
source tally already iterates over particle species so multi-channel works
without driver changes.
