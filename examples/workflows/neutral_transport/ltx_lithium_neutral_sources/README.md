# LTX-β neutral sources on the SOLEDGE3X plasma

OpenEdge computes, for one SOLEDGE3X plasma solution of LTX-β (shot 103955,
Li wall), the volume sources a fluid or kinetic plasma code needs from the
neutrals: particle, momentum and energy source moments of Li⁺ and H⁺ and the
neutral densities, on a uniform R–Z grid. It does what the coupled neutral code
does in one plasma iteration: one batch of every source is emitted at step 1
and followed until it is gone.

Sources
- Li: H⁺ wall flux of the plasma solution × Bohdansky yield (`compute
  surface/physical/sputter`), Thompson energies; Li ionization by ADAS rates.
- H: the same H⁺ wall flux recycled through the Li wall (`surf_react
  surface/pwi`: TRIM reflection, thermal re-emission as H₂, R = 0.5), an H₂ puff
  at the inboard midplane, then ionization, charge exchange and the molecular
  chain H₂ → H₂⁺ → H + H⁺ (`fix volume/chem/adas`, HYDHEL rates).

Run
```
./scripts/build_background.sh                       # input/plasma_ltx.h5, input/ltx_wall_axi.surf from the S3X run
mpirun -np 4 ~/build_oe/src/spa_mac_mpi -in in.openedge     # ~2 s
python scripts/export_sources.py                    # output/ltx_li_sources.h5 and .csv
python scripts/build_notebook.py --execute          # notebooks/ltx_li_sources.ipynb
```
Python: `/Users/42d/venvs/openedge-env/bin/python` (h5py, scipy, matplotlib, nbformat).

How it works
- `units flight` on the two chem fixes: every event adds its particle weight
  over cell volume and time step and nothing is reset, so at the end of the
  flight the fix array is the steady-state source in SI; the trailing columns
  are the densities of the transported species. One dump, no averaging.
- `stop_on_exhaust` on the hydrogen fix ends the run at the first output once
  fewer than `nexhaust` particles remain (a few 0.01 eV H₂ in the far SOL).
- The export deposits the adaptive cells conservatively onto a 5 mm grid and
  writes `S_p, S_m_{r,z,phi,par}, Q_e, Q_i, n_Li0, psi_n`, the same under `H/`
  with `n_H0, n_H2, n_H2plus`, the wall profiles and the totals.

Results against the SOLEDGE3X-EIRENE tallies of the same plasma

| quantity | OpenEdge | reference |
|---|---|---|
| H⁺ to the wall | 6.83e20 /s | 6.83e20 /s |
| H⁺ source | 7.60e20 /s | 7.42e20 /s |
| H⁺ born via H₂⁺ | 2.65e20 /s | 2.64e20 /s |
| Q_e(H) | −3.49 kW | −3.28 kW |
| Q_i(H) | −0.86 kW (± 10 W over seeds) | −0.87 kW |
| Li eroded by H⁺ | 1.73e19 /s | 1.88e19 /s |
| Li eroded by H atoms | 0.60e19 /s | 0.57e19 /s |
| Li ionized | 2.24e19 /s | 2.38e19 /s |
