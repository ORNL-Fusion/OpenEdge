# ST40 lithium powder dropper

Li powder injected into the SOLPS-13589_2 background. Grains ablate into Li,
Li is removed at ionization, and the per-cell ionization and electron-energy
sources are handed to SOLPS.

```
solps/        SOLPS run (baserun/, run_dir/), not tracked
openedge/     in.openedge, input/, scripts/, notebooks/
openedge/3d/  full-torus 3-D deck: where the Li atoms land (no coupling)
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

## 3-D

From `openedge/3d/`, with a VTK-enabled build (`dump surf/vtk`; `-var vtk 0` for text dumps). Keep the grid coarse (3 cm default), the chem fix does per-cell work every step:

```bash
python scripts/make_geometry.py                     # input/st40_*_3d.surf + .stl, st40_box.inc, st40_groups.inc
mpirun -np 4 ~/build_oe_vtk97/src/spa_mac_mpi -in in.openedge
```

Same deck knobs as 2-D. Outputs: `landing.<step>.vtu` (per wall triangle and 20 ms window:
`v_area` [m^2], `v_grain_hits`, `v_grain_rate` [grains/s], `v_Li_hits`, `v_Li_flux`
[atoms/m^2/s], `v_Li_rate` [atoms/s]; absorbing walls, so flux = deposition),
`li_dens.*.vtu`, `state.*.vtu`.
Analysis: `python scripts/build_3d_landing_notebook.py --outdir <run> --log <log>` -> `notebooks/st40_3d_landing.ipynb`.
Camera comparison: `python scripts/make_camera_video.py output --video <ST40 mp4>` -> `sim_vs_camera.mp4`
(synthetic pinhole view next to the fast-camera frame; camera position and time offset are flags).
Grain radius 25 um (the ST40 "50 um" powder read as a diameter), `nweight 8`; `-var nsrc v_srate -var nweight 80 -var nsteps 250000` for the measured 14449 pulse.
Background: `python ../scripts/far_sol_decay.py ../input/plasma_13589.h5 input/plasma_farsol.h5` replaces the SOLPS far-SOL plateau above
psi_N 1.1 by an exponential decay (lambda_n 2 cm, lambda_t 4 cm); the raw SOLPS file is `-var plasma ../input/plasma_13589.h5`.
Deck defaults: kinetic (Li+..Li3+ transported), fnum 2e13, `balwt time`, dumps every 5000 steps;
`-var chem_mode neutral -var fnum 2e11` removes Li at ionization (the run behind `output/` and the landing notebook).
Without VTK (`-var vtk 0`) the video script reads the text `state` dump; the notebook builder needs the `.vtu` dumps.

## Coupling

See `coupling/README.md`. The SOLPS case must contain Li species.
