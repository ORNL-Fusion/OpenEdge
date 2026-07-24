# WEST tungsten transport — axisymmetric

W-impurity transport in real WEST geometry, 2D axisymmetric (x = Z, y = R,
axis at y = 0), kick-drift Boris mover. Chain: O-on-W physical sputtering
(Eckstein/PWI) → transport with sheath (boundary mode), Coulomb collisions
with the background, Braginskii thermal-gradient forces, cross-field
diffusion → ADAS ionization W..W20+ → wall deposition + self-sputtering.

Two cases, same deck, different SOLEDGE3X backgrounds/geometry:

- `axi_diGenova_plasma_bkg/` — Di Genova setup; cold (semi-detached)
  strike points (Te ~ 2-3 eV), so W sources are weak (~1e15 atoms/s).
- `axi_1p5MW_plasma_bkg/` — 1.5 MW case (antenna closer to the core),
  hotter background (Te median 17 eV, max ~145 eV). NOTE: its wall file
  is wound normals-in already, so the deck reads it WITHOUT `invert`
  (flow volume must come out ~26 m^3, not ~36).

## Run

    cd axi_1p5MW_plasma_bkg && mpirun -np 4 spa_mac_mpi -in in.axi_west_emission -var Dperp 0.5 -log log.d05

Short test: append `-var nwarm 20000 -var ndiag 5000`. Statistics knob:
`-var n <N>` (launches per emission event, flux-conserving).
Full convergence needs ~500k warmup steps (self-sputtering slow tail).

## Analysis

- `<case>/analysis.ipynb` — wall geometry, W density maps and radial profiles,
  ionization length, erosion/deposition along the wall, convergence,
  impact-energy spectrum + self-sputter yield.
- `axi_diGenova_plasma_bkg/analysis/*.py` — standalone plotting tools inherited from the old
  `test_west` example (2D density maps, charge states, wall flux). They take
  `--dump` arguments; note they predate the axi column convention (file
  x = Z, y = R) and may need that swap.

## Rebuilding inputs from a SOLEDGE3X run

    cd tools/converters
    python3 convert_s3x_plasma.py <run_dir> --plasma-snapshot plasmaFinal.h5 \
        --plasma-out <case>/input/plasma.h5 --wall-out <case>/input/wall.surf \
        --geometry axi
    cd <case>/input
    python3 subdivide_surf.py wall.surf wall_fine.surf --maxlen 0.02 \
        --region 0.30 1.0 0 99 0.01
    python3 make_core_surf.py plasma.h5 core.surf --level 0.1

Check the wall winding after conversion (read_surf with/without `invert`;
the interior flow volume is ~26 m^3).

## Data

`<case>/input/` — WEST wall/core surfaces, species, PWI recycle table, and
the surface-generation scripts. `input/plasma.h5` (~12 MB) is git-ignored;
the diGenova one recovers with:

    git show db597e4:examples/test_west_axi/input/plasma.h5 > axi_diGenova_plasma_bkg/input/plasma.h5

the 1p5MW one regenerates from /Users/42d/soledge/1p5MW/run_dir with the
converter above. Run artifacts land in `<case>/state/` (git-ignored).
