# Rebuild the IEAD database from scratch

Step-by-step recipe to regenerate `iead_database.h5` on a fresh
machine. Total wall time ≈ 20 minutes on an 8-core x86 box.

## 0. What you need

- `gfortran` (any recent version — the code uses Fortran 2003 + OpenMP)
- Python ≥ 3.8 with `numpy`, `h5py`, `matplotlib`
- ~50 MB scratch disk for the per-cell CSVs
- 1 CPU node (the sweep is embarrassingly parallel within OpenMP per
  call; no MPI / no GPU required)

## 1. Vendor or clone sheath_tracker_v2

`sheath_tracker_v2.f90` is the forward solver. It pushes 50 k test
ions per (charge state, projectile mass) through a magnetised
Chodura-Bohm sheath using a Boris pusher and writes the impact
`(Z, E_eV, theta_from_normal_deg)` to CSV.

Vendored copy lives at `OpenEdge/tools/iead/sheath_tracker_v2.f90`.
If you are working from a developer checkout where `tools/iead/` is
not yet populated, copy it from `/home/cloud/SHEATH/sheath_tracker_v2.f90`.

```bash
cd OpenEdge/tools/iead
gfortran -O3 -fopenmp -o sheath_tracker_v2 sheath_tracker_v2.f90
```

The build script `sheath_tracker_v2.build.sh` wraps the same one-liner.

## 2. Run the sweep

```bash
cd OpenEdge/tools/iead
mkdir -p iead_database/{csv,nml,log}
OMP_NUM_THREADS=8 python3 sweep_driver.py
```

The driver loops over `(tau_i, psi_j, Z_k)` and writes
`iead_database/csv/t<i>_p<j>_z<Z>.csv`. Re-running picks up where it
left off (each cell skips if its CSV already exists and is non-empty).
Add `--reset` to overwrite, or `--limit N` for a smoke test.

Expected runtime: ~16 min on 8 OpenMP threads.

## 3. Pack the CSVs into a single HDF5 table

```bash
python3 pack_database.py
```

This reads `iead_database/index.h5` for the sweep axes and
`iead_database/csv/*.csv` for the impact records, builds a 2-D
histogram in `(Etilde = E/(Z*Te), theta_from_normal)` per cell with
`(120, 30)` uniform bins on `[0, 60] x [0, 90deg]`, and writes
`iead_database/iead_database.h5` (~9 MB, gzip-compressed).

Output schema is documented in `README.md`.

## 4. Sanity check (optional)

```bash
python3 -c "
import h5py, numpy as np
with h5py.File('iead_database/iead_database.h5') as g:
    f = g['f'][...]
    Et = 0.5 * (g['Etilde_bin_edges'][1:] + g['Etilde_bin_edges'][:-1])
    th = 0.5 * (g['theta_bin_edges_deg'][1:] + g['theta_bin_edges_deg'][:-1])
    Z  = g['Z_grid'][...]
    # Pick an interior cell and report <Etilde> and <theta>
    i, j, k = 4, 4, 0    # mid-grid, Z=1
    p = f[i, j, k]
    pE  = p.sum(axis=1); pT = p.sum(axis=0)
    print(f'<Etilde> = {(Et * pE).sum():.2f}  (= <E>/(Z*Te) at Z={int(Z[k])})')
    print(f'<theta>  = {(th * pT).sum():.1f} deg')
"
```

For an interior `(tau, psi)` cell at Z=1 expect `<Etilde> ≈ 5` (so
`<E> ≈ 5*Te`) and `<theta>` in the 50°–70° range (set by `psi`).

## 5. Install into the OpenEdge database tree

```bash
mkdir -p OpenEdge/database/iead
cp iead_database/iead_database.h5  OpenEdge/database/iead/
cp iead_database/README.md         OpenEdge/database/iead/
cp iead_database/IEAD_REGENERATE.md OpenEdge/database/iead/
# place the Mellet PDF here once you have a copy locally
# cp .../Mellet_2017_PPCF_59_035006.pdf OpenEdge/database/iead/
```

`compute surface/physical/sputter ... iead auto` will then pick it up
via `OPENEDGE_ROOT` / the compile-time database dir / cwd-relative
`database/`, mirroring `processes.h5` resolution.

## 6. Update / extend

To extend the parameter coverage (e.g. add Z=11..20 for higher-Z
impurities, or refine the `psi` grid near grazing) edit the `*_GRID`
arrays at the top of `sweep_driver.py` and rerun. The packer reads the
axes from `index.h5` so it picks up the new shape automatically.

To regenerate at a different reference (Te, ne, B), edit the
`TE_FIXED`, `NE_FIXED`, `B_FIXED` constants in `sweep_driver.py`. The
shape of the IEAD in `(Ehat, theta)` is approximately invariant in
typical SOL/divertor conditions, so this is rarely needed.

## Versioning note

The HDF5 file carries the sweep conditions and `m_proj` scheme as
attributes. If you change them, please bump the `scheme` string in
`pack_database.py` and document the change in `README.md`.
