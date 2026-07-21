# WEST tungsten transport: axisymmetric vs Cartesian

Two matched W-impurity transport cases in real WEST geometry that differ
**only in geometry mode**, to isolate the effect of the axisymmetric
(2πR ring) vs planar Cartesian (flat slab) treatment.

- `axi/`  — 2D axisymmetric (x = Z, y = R). The physically correct tokamak
  reduced model (matches SOLEDGE/ERO); kick-drift Boris mover.
- `cart/` — 2D planar, **identical** physics and inputs (same
  `wall_fine.surf`, `plasma.h5`, sheath, PWI + self-sputtering, friction,
  cross-field diffusion, ADAS). The only difference is flat-slab geometry.
  Uses `fix background ... rz_axes axi` so the planar run keeps the axi
  x=Z, y=R convention (otherwise planar assumes R=x, Z=y and the wall
  samples outside the plasma mesh → no emission).

## Run

    cd axi  && mpirun -np 4 spa_mac_mpi -in in.axi_west_emission  -var Dperp 0.5
    cd cart && mpirun -np 4 spa_mac_mpi -in in.cart_west_emission -var Dperp 0.5

Short runs: append `-var nwarm 20000 -var ndiag 5000 -var sourceThreshW 1e5`.
Full convergence needs ~500k warmup steps (self-sputtering slow tail).

## Analysis

`axi/input/analysis.ipynb` — per-case diagnostics plus axi-vs-cart comparison
cells: deposition/erosion vs wall coordinate, summary table, self-normalized
density ratio, and convergence + impact-energy overlays.

## Data

`input/plasma.h5` (12 MB WEST SOLEDGE3X background) is git-ignored. Recover:

    git show db597e4:examples/test_west_axi/input/plasma.h5 > axi/input/plasma.h5
    cp axi/input/plasma.h5 cart/input/plasma.h5
