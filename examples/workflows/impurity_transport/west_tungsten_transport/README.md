# WEST tungsten transport

Axisymmetric (x = Z, y = R) W transport in WEST geometry: O-on-W sputtering
from RustBCA tables, Boris mover with spatial sheath, Coulomb, thermal-gradient
and cross-field forces (D_perp 0.3, pinch -0.6 from SOLEDGE3X), ADAS
ionization to W20+, and a PWI wall with TRIM reflection and Thompson
re-emission. W markers come from a four-band wall source (lower and upper
divertor, LFS, HFS).

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
```

Defaults: 8000 warmup + 2000 diagnostic steps at dt = 2e-8 s, a few
minutes on a laptop. Production: `-var nwarm 100000 -var ndiag 10000`.
Other variables: `nLo/nUp/nLfs/nHfs` (markers per step per band), `Dperp`.

## Outputs

`output/` holds the grid dump (total and neutral W density), `wall_ehist.dat`
impact energy and angle histograms, the warmup restart, and a particle dump.
`scripts/analysis.ipynb` produces density maps, radial profiles, the
ionization-length check and the convergence trace for every `D<val>` run
it finds.

## Rebuilding inputs from SOLEDGE3X

```bash
python3 ../../../../tools/converters/convert_s3x_plasma.py <run_dir> \
    --plasma-snapshot plasmaFinal.h5 \
    --plasma-out input/plasma.h5 --wall-out input/wall.surf --geometry axi
python3 scripts/subdivide_surf.py input/wall.surf input/wall_fine.surf \
    --maxlen 0.02 --region 0.30 1.0 0 99 0.01
python3 scripts/make_core_surf.py input/plasma.h5 input/core.surf --level 0.1
```

The wall is wound normals-in, so read it without `invert`; the flow volume
should be about 26 m^3. `input/plasma.h5` is git-ignored.
