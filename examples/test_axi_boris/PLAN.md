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

## Phase 2 — cross-field diffusion chord consistency (DONE, pending review)

Implemented in update.cpp (+ OPENEDGE mirror), "Apply cross-field
diffusion displacement": the kick is folded into the step velocity
(v += dx_cd/dtremain, PKEEP only) alongside xnew, so the traced chord
xnew = x + dtremain*v stays exact for the axi crossing/remap tests and
walls see the diffusive flux. The kick is then STRIPPED from v at the
first velocity-transforming event — surf collision (before sheath kick /
collide, so PWI never sees the phantom kick energy ~ dx/dt, which can
dwarf thermal), boundary collide, psi reflect, mid-move migration (on the
sender; the receiver rebuilds the chord from v and could never strip) —
or at post_move_bookkeeping when the step completes cleanly.

Two failure modes found by the D_perp sweep and fixed:
- Strip-after-move only: wall reflections transform v with the kick
  inside, the post-hoc subtraction leaves O(kick) residuals -> runaway
  heating (T -> 3e9 K at D=10) + naxibad. Hence strip-at-event.
- Leak through walls: the chord reaches a surf only because of the kick,
  so the stripped v may be non-incident (v.n >= 0); specular reflection
  of that v aims it THROUGH the wall and `exclude = minsurf` lets it out
  (1292 escapes at D=1). Fixed with a graze path: non-incident after
  strip -> no collision physics/tallies, particle continues inward from
  the wall point.

Verified (in.axi_west_emission, D_perp = 0/0.1/1/10, chem on): naxibad 0,
boundary exits 0, np constant 6740, no particles outside the closed wall
(residual "outside" flags are on-wall points at dump precision ~1e-6 m),
T flat to 1e-3 over 20k steps (exact at D=0), W inventory closes across
the sweep. Maps + radial profiles in west/input/analysis.ipynb.

Quantitative acceptance test: `in.axi_diffusion` + `check_diffusion.py` —
uniform axial B (via new `bfield_const` option in fix
cross_field_diffusion), H+ ring, no E; the poloidal kick is then exactly
radial so <(R-R0)^2> = 2*D*t. Measured D = 0.5064 for input 0.5 (+1.3%),
0.0975 for 0.1 (-2.5%), ~0 at D=0 (spread = gyroradius). Rerun after any
mover/diffusion change. Still worth doing: dt-halving and Nevery
(1 vs 10) invariance of the WEST profiles.

Open item: reflective box boundaries ('r') with diffusion would need the
same non-incident guard at domain->collide (this case uses outflow).

## Phase 3 — full-physics WEST axi validation

`west/in.axi_west_emission` now runs the full chain: sputter-driven W
emission + depositing PWI wall + ADAS chain + sheath boundary + diffusion
+ friction. Status of the original steps:

1. Sheath on: DONE — `sheath boundary` mode (sub-grid barrier, see the
   sheath section below). naxibad = 0, near-wall orbits deposit.
2. Depositing wall: DONE — `surf_react surface/pwi input/wall.recycle`
   (TRIM w_on_w reflect + absorb), diffuse `surf_collide` catch-all.
3. ADAS ionization: DONE — `volume/chem/adas 1 W auto`, W+..W10+ all
   Boris-pushed. Neutral ionization length verified lambda_ion ~ 1.1 cm
   (neutral-W density decay vs distance-to-wall, analysis.ipynb).
4. Acceptance NOT DONE: W mass balance closure (emitted = deposited +
   in-flight + core-lost) still to check; deposition profile vs Cartesian
   `test_west` still to compare.
5. Optional gold standard NOT DONE: single W+ canonical angular momentum
   p_phi = m*R*v_phi + q*psi conservation.

Also added this session (not originally in the plan):
- Core-boundary mesh exclusion: `west/make_core_surf.py` extracts the
  psi_norm=0.1 flux surface from plasma.h5 -> `input/core.surf` (normals
  into the SOL). Read as a specular surf so the deep core is marked
  inside/solid (flow volume 27.3 -> 17.4 m^3, annulus only). REPLACES the
  old `fix reflect/psi` core boundary.
- Particle-based mesh adaptation: `fix adapt ... refine particle` in the
  WARMUP phase (unfix + reset_timestep before diagnostics, so the frozen
  grid keeps the ave/grid tally valid).
- Diagnostic-tally fixes: reset_timestep 0 before the diagnostic run so the
  ave/grid|ave/surf window completes (else it silently outputs zeros);
  erosion_rate is a STATIC per-surf value dumped directly (ave/surf zeroes
  it because it accumulates per-collision tallies, not static values).

## Di Genova 2021 (NF 61 106019) parity — gap analysis

Reference: WEST #55797 ERO2.0 run. Key finding of the audit — OpenEdge
already has EVERY physics ingredient he used, and west/input/plasma.h5
carries all the background data (ne, Te, Ti, per-species parr_flow, E-field
e_r/e_t/e_z, grad_te/grad_ti, full B). The gap is turning modules on and
matching his config, not writing new physics. One structural change (the
depositing wall) is the gatekeeper for everything downstream.

Di Genova ingredient -> OpenEdge status (in in.axi_west_emission):
- Background from SolEdge-EIRENE ............ HAVE, ON (fix background pd).
- Boris in E/B .............................. HAVE, ON (E currently NOT read
  from pd->e_* outside sheath; his run also sets E=0 outside sheath, so OK,
  but confirm we're not silently dropping his e_r/e_z if we later want it).
- Cross-field D_perp anomalous diffusion .... HAVE, ON (Phase 2, verified).
- W source = O-on-W physical sputtering ..... HAVE, ON (compute
  surface/physical/sputter, Thompson 4.4). VERIFY incident energy uses the
  sheath form E_i = 2*Ti + 3*Z*Te and Eckstein/SDTrimSP yield, per paper.
- W ionization W+..W10+ (ADAS) .............. HAVE, ON.
- Sheath E-field (accelerates ions, sets E_i and prompt redep): DONE —
  `sheath boundary` mode (sub-grid potential barrier). ON.
- Depositing/absorbing wall (net erosion = gross - redeposition;
  prompt/local/far): DONE — `surf_react surface/pwi` (TRIM w_on_w reflect
  + absorb). ON. (Was the #1 gap; now closed.)
- Collisional friction/slowing on Maxwellian background (his Fokker-Planck
  operator; the DOMINANT redeposition + screening mechanism): DONE +
  VERIFIED (fix coulomb/background 1 background pd 2.0 1.0). D+ parallel-
  flow drag flushes W to the divertor. parr_flow SIGN verified two ways:
  (1) parr_flow*bhat poloidal projection converges on the targets (down in
  lower half, up in upper, stagnation ~midplane); (2) empirically ~98% of
  the W density ends up in the lower divertor. Uses SCALAR parr_flow*bhat
  (plasma.h5 has no vector flow components); OpenEdge B direction matches
  SOLEDGE parr_flow sign convention. Also verified correct in axi (velocity
  kick at END_OF_STEP, respects kick-drift chord).
- W self-sputtering (W-on-W; changes n_W^bound by ~4-5x low density,
  fig 6c/d): code SUPPORTS it (compute surface/physical/sputter takes a
  projectiles list) — currently O only. Add W to the projectiles + iterate
  the deposited-W flux to convergence. Config change, not new code.
- Braginskii thermal force ................. HAVE (fix thermal_force), OFF.
  Di Genova did NOT include it (lists as future work) — leave off for
  parity, available if wanted.
- Drifts .................................... he switched OFF; we have none.

Priority order — items 1-3 DONE (depositing wall, sheath, friction all ON
and verified). Remaining:
4. Self-sputtering: add W to sputter projectiles, iterate to convergence
   (his loop). Config change (code supports it). Expect n_W^bound up
   ~4-5x at low density. Do AFTER the baseline comparison (B) so its
   effect can be measured against a reference.
5. VERIFY sputter incident-energy model and yield fit vs his
   E_i = 2Ti+3ZTe + SDTrimSP.

**NEXT (task B): comparison diagnostics.** All inputs now exist (depositing
wall + friction + core surface), so these are the highest-value work — they
are what let us actually compare to Di Genova. s-axis machinery from
analysis.ipynb's wall-vs-s cell is reusable.
- Wall EROSION + DEPOSITION flux vs wall coordinate s (his fig 6a): DONE
  (analysis.ipynb fig-6a cell). Erosion compute outputs erosion_flux
  (c_cero[1]) + erosion_rate (c_cero[2]).
  DEPOSITION-TALLY BUG FOUND + FIXED: stock `compute surf nflux_incident`
  is pweight-UNAWARE (uses the struct .weight, not the OpenEdge pweight
  custom that carries the real macroparticle count ~1e5). It undercounted
  deposition by the pweight/fnum ratio -> erosion appeared ~4e4x above
  deposition. Fix: new `compute surf/weighted` (src/OPENEDGE/, mirrors
  compute grid/weighted; subclasses ComputeSurf, overrides surf_tally to
  weight by pweight, divides normflux by fnum since pweight is already the
  real count). Deposition now uses it -> erosion/deposition peak ratio
  0.05 (was ~4e4), i.e. comparable as physics requires. Re-run the full
  case for the quantitative fig-6a.
- W penetration factor tau_W = N_in / phi_W (his eq 1, tables 3-5): DONE
  (analysis.ipynb penetration cell). N_in = confined W (psi_norm<1),
  phi_W = sum erosion_rate. Result tau_W = 8e-6 s (log10 -5.1) — IN his
  range (-3..-8). ~20% of W penetrates inside the separatrix.
- Core-boundary <n_W^bound> (his figs 6c/d, 7a/b): DONE (same cell) but
  LOW (~2e5 vs his ~1e14). CAVEAT: our core.surf is SPECULAR (reflects),
  his inner boundary ABSORBS -> W accumulates as core content there. For a
  comparable n_W^bound, switch core.surf to absorbing (surf_collide vanish
  or a counting react). Penetration profile n_W(psi_norm) also plotted:
  peaks at separatrix, falls ~5 decades inward (W is edge-localized).
- Impact-energy PDF at the wall (his fig 7c, <E_in>_LDP = 112 eV). NOT
  DONE — needs a per-collision incident-KE tally at the wall.
- Sputtered-energy PDF (his fig 6b) — Thompson tail, self-sput cutoff.
  NOT DONE — needs an emitted-particle KE tally.

## Sheath resolution + boundary mode (done; effect masked by missing friction)

Problem found: `sheath spatial` under-resolves the Debye sheath. In axi
kick-drift, position is FROZEN during subcycles (only v is kicked), so the
sheath is sampled once per step at the start position — more subcycles do
NOT help. Grid (~300 um finest) and drift/step (~100 um) are both ~10x the
divertor lambda_D (~32 um), so ions are never sampled in the strong-field
layer. Field diagnostic confirmed: model gives correct fields (1e5 V/m at
hot divertor, 50 V/m at cold far-SOL walls), but the aggregate seen is weak
because cold walls dominate by count.

Implemented `sheath boundary` mode (pusher.cpp + update.cpp): sub-grid
thin-sheet potential barrier. Inbound ion -> impact-energy kick (as `kick`);
outbound near-wall ion -> energy check vs Z*e*phi_total, reflect if it
can't escape (prompt redep), else decelerate. Per-particle "sheath_paid"
int custom fires the escape charge once per transit. Resolution-independent
(uses sheath_phi_at_distance, exact energy conservation at any dt). Modes
now: off | kick | spatial | boundary. Cache (per-wall-element, static
plasma) also added — ~33% faster AND fixes a per-particle spatial-sheath
temperature runaway (re-derived sheath-edge coeffs from the ion's
penetrating position). Diagnostics behind `dump yes`: per-step "sheath
step" (near-wall/engaged/reflect/escape counts, |E|), per-element "sheath
elem" (field profile). OE_NO_SHEATH_CACHE env toggle for A/B.

Verified: boundary fires (~20 reflect/step at D=0.1), naxibad=0, clean.
Originally np(boundary)==np(spatial) grew unbounded because W barely
reached the walls WITHOUT friction — that was the diagnosis that motivated
turning friction on. RESOLVED: with fix coulomb/background ON (now done +
verified), W is flushed to the divertor (~98% in the lower divertor) and
deposits. Re-evaluate the sheath's effect via the deposition PROFILE (task
B fig-6a), not total np. dt is now 5e-9 (was 5e-8); dt is a -var for
convergence checks.

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
