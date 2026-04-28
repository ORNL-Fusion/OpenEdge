# DIII-D carbon test

> **Note (2026-04-20):** Currently runs in legacy 2D Cartesian layout
> (per-radian wedge convention). Migration to SPARTA-native axisymmetric
> is queued — see `CLAUDE.md` § "Migration cookbook".

Small 2D carbon case for DIII-D.

**Data download required:** `./download_data.sh test_d3d_mateja` (from repo root)

Files:
- `input/plasma.h5`
- `input/204953_3000.x16.equ`
- `input/plasma.species`
- `input/surfaces/wall.txt`

Notes:
- Inner core uses `fix reflect/psi`.
- B comes from `global pusher plasma cwest`.
- `input/plot_oedge_native.py` is a local check script.

Ignored local output:
- `output/`
- `log.openedge`
- `input/Figs/`
