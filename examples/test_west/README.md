# WEST tungsten transport — axisymmetric test (test_west)

W-impurity transport in real WEST geometry, 2D axisymmetric (x = Z,
y = R, axis at y = 0). Chain: O-on-W physical sputtering (RustBCA
tables) -> Boris mover with spatial sheath -> Coulomb, thermal-gradient
and cross-field forces (S3X-matched D_perp 0.3, pinch -0.6) -> ADAS
ionization W..W20+ -> PWI wall (TRIM reflection, Thompson re-emission).
W markers come from a four-band stratified wall source (lower/upper
divertor, LFS, HFS) so every wall region gets statistics.

## Run

    mpirun -np 4 spa_mac_mpi -in in.axi_west_emission

Defaults run 10k steps (8k warmup + 2k diagnostics, dt 2e-8) — a
few minutes on a laptop. Production: `-var nwarm 100000 -var ndiag
10000`. Other knobs: `-var nLo/nUp/nLfs/nHfs <N>` (markers/step per
band), `-var Dperp <val>`, `-var sheathmode boundary` (pre-Aug-2026
sheath model for A/B).

## Outputs and analysis

Everything lands in `state/` (git-ignored): consolidated grid dump
(total + neutral W density), `wall_ehist.dat` impact energy/angle
histograms, warmup restart, OVITO particle dump. `analysis.ipynb`
makes the density maps, radial profiles, ionization-length check,
convergence trace, and the 2D total-W map; it auto-discovers however
many `D<val>` runs exist.

## Rebuilding inputs from a SOLEDGE3X run

    cd tools/converters
    python3 convert_s3x_plasma.py <run_dir> --plasma-snapshot plasmaFinal.h5 \
        --plasma-out input/plasma.h5 --wall-out input/wall.surf --geometry axi
    cd input
    python3 subdivide_surf.py wall.surf wall_fine.surf --maxlen 0.02 \
        --region 0.30 1.0 0 99 0.01
    python3 make_core_surf.py plasma.h5 core.surf --level 0.1

The wall file is wound normals-in: read WITHOUT `invert`; interior
flow volume must come out ~26 m^3. `input/plasma.h5` (~12 MB) is
git-ignored; regenerate with the converter above.
