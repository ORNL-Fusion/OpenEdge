# WEST tungsten transport — axisymmetric

W-impurity transport in real WEST geometry, 2D axisymmetric (x = Z, y = R,
axis at y = 0), kick-drift Boris mover. Chain: O-on-W physical sputtering
(Eckstein/PWI) → transport with sheath (boundary mode), Coulomb collisions
with the background, Braginskii thermal-gradient forces, cross-field
diffusion → ADAS ionization W..W20+ → wall deposition + self-sputtering.

Single case (flat layout): 1.5 MW SOLEDGE3X background, Te median
17 eV, max ~145 eV. NOTE: the wall file is wound normals-in already,
so the deck reads it WITHOUT `invert` (flow volume ~26 m^3, not ~36).
(The diGenova semi-detached variant was removed Aug 2026; recover from
git history if needed.)

## Run

    mpirun -np 4 spa_mac_mpi -in in.axi_west_emission -var Dperp 0.5 -log log.d05

Short test: append `-var nwarm 20000 -var ndiag 5000`. Statistics knobs:
`-var nLo/nUp/nMain <N>` (markers per step per wall band, stratified
source, flux-conserving; emit runs `nevery 1`).
Full convergence needs ~500k warmup steps (self-sputtering slow tail).

## Analysis

- `analysis.ipynb` — wall geometry, W density maps and radial profiles,
  ionization length, erosion/deposition along the wall, convergence,
  impact-energy spectrum + self-sputter yield.

## Rebuilding inputs from a SOLEDGE3X run

    cd tools/converters
    python3 convert_s3x_plasma.py <run_dir> --plasma-snapshot plasmaFinal.h5 \
        --plasma-out input/plasma.h5 --wall-out input/wall.surf \
        --geometry axi
    cd input
    python3 subdivide_surf.py wall.surf wall_fine.surf --maxlen 0.02 \
        --region 0.30 1.0 0 99 0.01
    python3 make_core_surf.py plasma.h5 core.surf --level 0.1

Check the wall winding after conversion (read_surf with/without `invert`;
the interior flow volume is ~26 m^3).

## Data

`input/` — WEST wall/core surfaces, species, PWI recycle table, and
the surface-generation scripts. `input/plasma.h5` (~12 MB) is git-ignored;
it regenerates from /Users/42d/soledge/1p5MW/run_dir with the converter
above. Run artifacts land in `state/` (git-ignored).
