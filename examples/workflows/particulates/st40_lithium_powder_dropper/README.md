# ST40 lithium powder dropper

This axisymmetric workflow injects the measured shot-14449 lithium-powder
source into a static SOLPS-13589 background. Grains charge, experience plasma
drag and gravity, ablate into Li, and feed Li charge-state transport.

## Layout

| Path | Purpose |
|---|---|
| `in.openedge` | Canonical OpenEdge deck |
| `input/` | Plasma, wall, species, chemistry, and measured source history |
| `input/plasma_st40_solps.h5` | SOLPS-13589 background — not in git (`examples/**/input/*.h5` is repo-ignored); regenerate with `tools/converters/convert_solps_plasma.py` from the SOLPS-13589 run |
| `scripts/analysis.ipynb` | Case inspection and result analysis |
| `scripts/build_geometry.py` | Rebuild the axisymmetric wall and dropper segment |
| `scripts/make_srate.py` | Rebuild the time-dependent powder source deck |
| `scripts/plot_lpd.py` | Plot Li particles and density |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
```

The deck runs a 0.5 s demo slice of the shot (`nsteps 250000`); for the
full 5 s source history run with `-var nsteps 2500000`.

The source timing follows shot 14449, while transport uses a fixed background
from shot 13589. The case is therefore suitable for qualitative plume and
trajectory comparisons, not pixel-by-pixel comparison with the Photron video.
