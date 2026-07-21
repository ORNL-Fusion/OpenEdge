# Axisymmetric Boris mover acceptance tests

Charged-particle transport in 2D axisymmetric domains (x = Z axial,
y = R radial, axis at ylo). No collisions, no chemistry, no sinks: the
only physics is the Boris push and the axisymmetric move/remap
machinery in `Update::move()`.

The axi mover traces every move as a straight 3D chord
(`xnew = x + dt*v`, constant `v`) and handles gyration around the axis
geometrically (`axi_remap`). The Boris pusher therefore runs in
kick-drift form in axi mode: velocity kicks at fixed position, then one
linear drift per step (see `Pusher::push_boris_2d`). These inputs are
the regression tests for that contract.

## Cases

Uniform-B box (this directory):

- `in.axi_bz` — uniform axial B. Ions gyrate in the (R, phi) plane, the
  motion handled entirely by `axi_remap` and the curved-cell-face
  crossing tests. This is the case that loses particles if the mover
  and pusher disagree.
- `in.axi_bt` — uniform toroidal B. Ions gyrate in the poloidal (Z, R)
  plane; checks in-plane orbits with the remap nearly passive.

Real WEST geometry (`west/`):

- `west/in.axi_west` — WEST wall (SOLEDGE3X 3MW case) with the matching
  plasma.h5 B-field, specular walls, 100 eV W+, divertor grid
  refinement (split cells). Closed vessel: np must stay constant.
  Requires `west/input/plasma.h5` (12 MB, not in git — see below).
- `west/in.axi_west_emission` — work-in-progress full-physics demo
  (sputter-driven W emission); not a pass/fail test.
- `west/subdivide_surf.py` — splits wall segments to a max length
  (finer emission/erosion resolution); `wall_fine.surf` was generated
  with `--maxlen 0.02 --region 0.30 0.85 1.8 3.25 0.01`.

## Run

    mpirun -np 4 spa_mac_mpi -in in.axi_bz
    mpirun -np 4 spa_mac_mpi -in in.axi_bt
    cd west && mpirun -np 4 spa_mac_mpi -in in.axi_west

## Pass criteria

1. `Axisymm bad moves = 0` in the run summary.
2. `np` in the stats output stays at its initial value: the domains are
   closed, so any drop is particles lost by the mover. In the WEST case
   `nexit` must also stay 0 (a nonzero value means a particle tunneled
   through the wall and left through the outer box boundary).
3. `c_T` (temperature) stays flat: a static B field does no work, so a
   drift means the mover is corrupting velocities.

These criteria fail on builds before the kick-drift fix: the subcycled
Boris endpoint was inconsistent with the chord the axi tracer assumes,
so particles ended up outside their cell after remap and were discarded
(`naxibad`). With subcycles 10, the uniform-B case lost 19659/20000
particles in 20000 steps and the WEST case 9528/10000 in 10000 steps.

## Data

`west/input/plasma.h5` is too large for git (it was recovered from
commit db597e4, the old `examples/test_west_axi` case). Regenerate it
from that commit with

    git show db597e4:examples/test_west_axi/input/plasma.h5 > west/input/plasma.h5

or add it to the `download_data.sh` release tarballs.
