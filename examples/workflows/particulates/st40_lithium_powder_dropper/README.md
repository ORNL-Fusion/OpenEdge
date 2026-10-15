# ST40 lithium powder dropper

Axisymmetric injection of Li powder (shot 14449 feed rate) into the static
SOLPS-13589_2 background. Grains charge, feel plasma drag and gravity, ablate
into Li atoms; the atoms are removed at ionization and their per-cell
ionization and electron-energy sources are the hand-over to SOLPS.

## Layout

| Path | Purpose |
|---|---|
| `solps/` | Erin's SOLPS-13589_2 dump, not tracked: `baserun/` (the regridded b2fgmtry sent Oct 14) and `run_dir/` (b2fstate, fort.31/33/34/35/46, gfile) |
| `openedge/in.openedge` | Self-contained deck: domain, absorbing walls and core contour, background, grain physics, Li ionization, source diagnostics |
| `openedge/input/` | Wall and core surfaces, species, `li.reactions` (Li -> Li+ only), `plasma_13589.h5` (built, not tracked), `srate.deck` (measured S(t), optional) |
| `openedge/scripts/build_background_13589_2.sh` | Build `input/plasma_13589.h5` from `../solps` (total heat flux as in SOLPS-routines `calc_fht`, exponential wall-gap extension, core hole left as vacuum) |
| `openedge/scripts/build_13589_2_li_notebook.py` | Generate and execute `notebooks/st40_13589_2_li_analysis.ipynb` from a run: snapshot, grain and Li density, inventories, Sp/Qe sources, B2-mapped sources |
| `openedge/scripts/launch_eos.slurm`, `sync_eos.sh` | EOS launcher (nsteps from the deck, 16 ranks) and case mirror (`push`, `pull <run>`) |
| `openedge/scripts/check_equilibrium_orientation.py`, `plot_lpd.py`, `build_geometry.py`, `make_srate.py` | Equilibrium orientation gate; plume plots; wall surface from the EIRENE input; measured S(t) deck |
| `coupling/` | SOLPS coupling: `config.json`, `run_coupling.sh`, `run_solps.csh`, `SOLPS_ADD_LI.md` (what the SOLPS case needs) |

## Run

All commands run from `openedge/`.

```bash
mpirun -np 4 ~/build_oe/src/spa_mac_mpi -in in.openedge                      # 0.4 s -> output/, log.openedge
/Users/42d/venvs/openedge-env/bin/python scripts/build_13589_2_li_notebook.py    # notebook from output/
```

On EOS (`/scratch/home_eos/42d/OpenEdge/examples/workflows/particulates/st40_lithium_powder_dropper/openedge`):

```bash
scripts/sync_eos.sh push
ssh eos 'cd .../openedge && sbatch scripts/launch_eos.slurm'      # -> runs/s<steps>_n<ranks>_<job>/
scripts/sync_eos.sh pull s200000_n16_<job>
/Users/42d/venvs/openedge-env/bin/python scripts/build_13589_2_li_notebook.py --outdir runs/<run> --log runs/<run>.log
```

Defaults: 200k steps of 2 us (200 ms powder flight, ionization steady from
step 120k, nine 20 ms source windows), constant feed of 3.0e21 atoms/s
(`ngrain 50`, `nweight 1` every 200 steps), `fnum 2e11`, `chem_mode neutral`.
`-var chem_mode kinetic` keeps Li+ as tracked particles with its transport
fixes (standalone checks only). Do not add `global weight cell radius`: the
axisymmetric cell weighting clones and deletes grains crossing cells.

The 20 ms window averages `output/source_SpSe.<step>` (Sp = Li+ birth rate,
Qe = electron energy sink including PLT radiation) are what
`tools/coupling/source_spse_to_solps.py` maps onto the B2 grid.

## Regenerating the background

```bash
scripts/build_background_13589_2.sh                    # from ../solps -> input/plasma_13589.h5
python ../../../../../tools/extract_psi_contour.py --plasma-h5 input/plasma_13589.h5 \
    --out input/st40_core.surf --psi-norm 0.615        # core sink at the inner B2 row (after a regrid)
```
