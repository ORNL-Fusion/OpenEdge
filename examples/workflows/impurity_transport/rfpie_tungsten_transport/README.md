# RFPIE tungsten sputtering and transport

He plasma on a biased W target in the RFPIE geometry, built from the RFPIE
CAD and Langmuir-probe profiles.

- Background: measured He radial profiles (density shape-preserving, Te an
  even-polynomial fit), quasineutral He+, uniform in z.
- Source: He-on-W RustBCA yield from `database/processes.h5`, Thompson W
  emission (Us = 8.68 eV, Emax = 80 eV).
- Target: one watertight surface; the plasma-facing top is retiled into
  radial rings so wall-flux sampling resolves the peaked profile. Each tile
  carries `sheath = [Vdc, Vrf_peak, phase]`. The sheath is a thin potential
  sheet, not a resolved Debye sheath.
- Transport: W to W4+ with ADAS charge-state evolution; pweight-aware
  interval-averaged density per charge state in `output/rfpie_w_density.*.dump`.
- Sampling: 64 x 64 x 96 cells, 100 markers per step, roulette at 200
  markers per cell and species, two-stage RCB balancing.

The chamber STL is kept for plotting only; the deck uses the open Cartesian
box as its outer boundary.

## Files

| File | Purpose |
|---|---|
| `in.openedge` | Deck; defaults Vdc = -500 V, Vrf = 0, 64 phase samples |
| `input/config.json` | Geometry and voltage parameters |
| `input/target_face.surf`, `input/target.surf` | Retiled target surfaces |
| `input/plasma_he.h5` | Reconstructed He background |
| `scripts/build_geometry.py`, `build_plasma.py` | Rebuild the inputs from CAD and probe data |
| `scripts/check_case.py` | Case consistency check |
| `scripts/analysis.ipynb` | Plasma, geometry, yield and W density diagnostics |
| `scripts/build_he_on_w.py`, `install_he_on_w.py` | Regenerate the He-on-W yield table in `processes.h5` |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
python scripts/check_case.py
jupyter nbconvert --to notebook --execute --inplace scripts/analysis.ipynb
```

Defaults: dt = 20 ns, 10 000 steps (200 us), about 16 plume transit
times. Changing frequency: edit `input/config.json`, rebuild
`target_face.surf`, and keep `rffreq` consistent in both input files. For
time-resolved RF at 13.56 MHz use one phase sample, disable the PMI cache,
and set dt to about 2 ns. The DC case is the cleaner starting point.
