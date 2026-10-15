# ST40 lithium powder dropper

Li powder injected into the SOLPS-13589_2 background. Grains ablate into Li,
Li is removed at ionization, and the per-cell ionization and electron-energy
sources are handed to SOLPS.

```
solps/        SOLPS run (baserun/, run_dir/), not tracked
openedge/     in.openedge, input/, scripts/, notebooks/
coupling/     SOLPS <-> OpenEdge loop (config.json, run_coupling.sh)
```

## Run

From `openedge/`:

```bash
scripts/build_background_13589_2.sh                 # ../solps -> input/plasma_13589.h5
mpirun -np 4 ~/build_oe/src/spa_mac_mpi -in in.openedge
python scripts/build_13589_2_li_notebook.py         # notebooks/st40_13589_2_li_analysis.ipynb
```

For a run kept elsewhere: the notebook script takes `--outdir <dir> --log <log>`.

Defaults: 200k steps (0.4 s), constant feed 3.0e21 atoms/s, `chem_mode neutral`.
Sources are the 20 ms window averages `output/source_SpSe.<step>`.

After a SOLPS regrid, rebuild the core sink:
`python ../../../../../tools/extract_psi_contour.py --plasma-h5 input/plasma_13589.h5 --out input/st40_core.surf --psi-norm 0.615`

## Coupling

See `coupling/README.md`. The SOLPS case must contain Li species.
