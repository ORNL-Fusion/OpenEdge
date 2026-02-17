# test_oml_charging

Standalone droplet charging test for `fix droplet/charge` using full OML current balance.

## Run

From this directory:

```bash
mpirun -np 4 /path/to/spa_mpi -in in.oml_charging
```

## Notes

- Plasma inputs (`Te`, `Ti`, `ne`, `ni`) come from `compute plasma/fields`.
- Droplet radius comes from each particle's `radius` field.
- This test intentionally does not include drag or evaporation.
- Charging diagnostics are printed by `fix droplet/charge` every `diag/every` steps.

## Plot space + time charging

After the run creates `case.oml`, generate a 2x2 subplot figure:

```bash
python3 plot_oml_charging.py --dump case.oml
```

Output:

- `Figs/oml_charging_subplots.png`
- Panels: early/mid/late 2D charge maps (`R-Z`, color by `Zd=q/e`) + time traces (`Zd min/avg/max`)

## Plot R-Z trajectories

Use the dedicated trajectory plotter:

```bash
python3 plot_traj_rz.py --dump case.out
```

or from the richer dump:

```bash
python3 plot_traj_rz.py --dump case.oml
```

Output:

- `Figs/traj_rz.png`
