# Axi mover fix — status and plan

Working branch: `feature/axi-kick-drift` (local). Build: `/Users/42d/build_oe`
(`cmake -C cmake/presets/mac_mpi.cmake -DPKG_OPENEDGE=ON`, binary
`src/spa_mac_mpi`). Old symptom-patch branch `feature/axi-mover-fixes` is
kept for reference until this merges, then delete it (optionally tag
`archive/axi-mover-fixes` first).

## Done (Phase 1)

Root cause: `Update::move()`'s axi machinery (`axi_horizontal_line`,
`axi_line_intersect`, `axi_remap`) traces every move as a straight chord
`xnew = x + dt*v` with constant v, but the subcycled Boris pusher returned
a curved endpoint with a rotated final velocity. Crossing tests disagreed
with the endpoint; particles landed outside their cell after remap and
were discarded (`naxibad` = "Axisymm bad moves" in the run summary).

- `src/pusher.cpp` (+ `src/OPENEDGE/` mirror): kick-drift Boris in axi —
  velocity kicks at fixed position, single linear drift per step; the
  per-subcycle position advance and wall-clip/cell-exit early returns are
  planar-2D only now. `Pusher::init()` refuses hybrid/gca in axi domains.
- `src/update.cpp` (+ mirror): cross-diffusion kick applied to PKEEP only
  (PINSERT could read `dx_cd` out of bounds).
- Tests here: uniform-B (`in.axi_bz`, `in.axi_bt`) and real WEST geometry
  (`west/in.axi_west`, specular walls, real plasma.h5 B-field).

Evidence (subcycles 10): uniform axial B lost 19659/20000 in 20k steps
before the fix, 0 after; WEST geometry lost 9528/10000 in 10k steps
before, 0 after, temperature flat to 7 digits.

`west/subdivide_surf.py` splits wall tiles; `wall_fine.surf` = <=2 cm
everywhere, <=1 cm in the upper divertor/baffle (Z > 0.30), for finer
emission/erosion resolution. Flow volume matches wall.surf to 12 digits.

## Phase 2 — cross-field diffusion chord consistency

`Update::move()` still adds `dx_cd` to `xnew` without touching `v`
(update.cpp, "Apply cross-field diffusion displacement"). In axi the
R-face and surface crossing tests use v, so the kick reintroduces exactly
the chord mismatch the pusher fix removed (INTERIOR verdict -> remap
outside cell -> naxibad), and walls never see the diffusive displacement.

Fix: fold the kick into the step's velocity before xnew is computed
(v += dx_cd/dtremain for PKEEP), so the traced chord includes diffusion
and surf collisions see it. Decide whether to strip the kick from v after
the move or accept the one-step velocity perturbation (GITR-style codes
accept it). Test: WEST case + `fix fcd cross_field_diffusion` with a big
D_perp; criteria: naxibad = 0, W density near walls smooth, wall flux
consistent as D_perp -> 0.

## Phase 3 — full-physics WEST axi validation

Build up from `west/in.axi_west_emission` (currently: sputter-driven W
emission, specular wall, no chem):

1. Sheath on (`sheath kick` or `sheath spatial` + `nearest_surf/grid`):
   in axi the sheath E is now evaluated at the fixed pre-step position —
   verify near-wall orbits still deposit (naxibad ~ 0, no pile-up).
2. Diffuse wall + `surf_react surface/pwi` (needs wall.recycle +
   database/processes.h5).
3. `volume/chem/adas` W ionization chain — W+..W10+ all Boris-pushed.
4. Acceptance: naxibad = 0, W mass balance closes
   (emitted = deposited + inventory), deposition profile sane vs the
   Cartesian `test_west` run.
5. Optional gold standard: single W+ in the static field, check canonical
   angular momentum p_phi = m*R*v_phi + q*psi is conserved.

## Phase 4 — platform and merge

- KOKKOS: `src/KOKKOS/update_kokkos.cpp` (+ OPENEDGE KOKKOS) still has the
  old behavior if it implements its own axi/Boris path — audit and mirror.
- hybrid/GCA in axi: either implement kick-drift for the Boris branch of
  hybrid + GCA-consistent linear segments, or leave the init() error.
- `west/input/plasma.h5` (12 MB) into the `download_data.sh` tarball flow
  (recover with: `git show db597e4:examples/test_west_axi/input/plasma.h5`).
- Merge to main; delete `feature/axi-mover-fixes`.

## Gotchas / notes

- SPARTA forbids external y-force in axi (update.cpp:431) — same contract
  the pusher fix restores; any new axi force must go through the chord.
- `create_particles n N` can create N-1 (volume rounding, warns) — the
  conservation criterion is "constant from initial", not "== N".
- dt must resolve the gyroperiod in axi (kick-drift): bad_dt_check warns
  at |q/m||B|dt_sub > 0.5. For W+ at 5 T use dt <= ~2e-7 with 10 subcycles.
- read_surf uses `invert` (converter walks CW in (Z,R)).
