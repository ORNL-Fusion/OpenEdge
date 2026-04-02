# DIII-D carbon test

Small 2D carbon case for DIII-D.

**Data download required:** `./download_data.sh test_d3d_mateja` (from repo root)

Files:
- `input/plasma.h5`
- `input/204953_3000.x16.equ`
- `input/plasma.species`
- `input/surfaces/wall.txt`

Notes:
- Inner core uses `fix reflect/psi`.
- B comes from `global bfield_compute cwest`.
- `input/plot_oedge_native.py` is a local check script.

Ignored local output:
- `output/`
- `log.openedge`
- `input/Figs/`
