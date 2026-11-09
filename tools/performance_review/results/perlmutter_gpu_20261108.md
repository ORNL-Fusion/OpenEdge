# OpenEdge on Perlmutter A100: where the GPU time goes (2026-11-08)

Source: `origin/development` a508aff6 (= perf_handoff/BASE_COMMIT.txt, latest on GitHub) +
`uncommitted_build_fixes.patch` + the per-grid sync fix described in section 5
(`experiments/kokkos_pergrid_sync.patch`, uncommitted on branch `perf-review`).
Binaries built today: `$PSCRATCH/openedge-build-a508aff-gpu/src/spa_kokkos_cuda_perlmutter`
(nvcc_wrapper, sm_80, Kokkos CUDA+OpenMP, GTL-linked, `-lineinfo`) and
`$PSCRATCH/openedge-build-a508aff-cpu/src/spa_mpi` (Cray MPICH, `-O3 -g`). The pre-existing
`openedge-build-kkhost4` binary is an OpenMP host build from 2026-09-21 (85 source files behind)
and was not used.

Case: `west_mb8_continuous_ops_20260922`, from the bundle under
`$PSCRATCH/slag_mb_pilot_source_on_matched/` (the checkout's `examples/wip` has no `production/`
directory). The brief's `teprobe_fixed_off` case does not exist in this bundle's `case_matrix.csv`;
`prevention_fixed_on` (fresh start, fixed strike point, external W injection on) was used
instead, through the unmodified `run_case.sh` with `RUN_STEPS_OVERRIDE`, output cadence = run
length, `DENSITY_REPEAT = steps/250`, `CHECKPOINT_STEPS_OVERRIDE=0`, `STATS_EVERY=1000`.
A wrapper (`results/perlmutter_gpu_20261108/oe_wrap.py`) only changes `post no` to `post yes` in
the generated schedule copy and appends the Kokkos arguments
`-k on g 4 -sf kk -pk kokkos react/retry yes gpu/aware yes comm threaded` (the production IPC line
from `scripts/perlmutter_gpu_env.sh`). Production inputs were not modified.

All runs on one GPU node (nid200341, 4x A100 40 GB, 64 Milan cores), interactive allocation
61003229. CPU runs use the same node (plain `spa_mpi`, 4/8/48 ranks, 1 thread each).
Raw tables, profiles, counter output and scripts: `results/perlmutter_gpu_20261108/`.
Run directories: `$PSCRATCH/openedge-runs/perf_review_20261108/<tag>/`.

## 1. Which per-step kernels run on the device (hypothesis 1 of the brief)

Hypothesis 1 ("no Kokkos versions of fix_background, pusher, update, surf_react_surface_pwi") is
out of date. The source has `src/KOKKOS/update_kokkos.cpp` (5381 lines: OpenEdge move loop, Boris
pusher `oe_boris3d`, spatial sheath, device pcache fill), `pusher_kokkos.h` (device mesh queries
`MeshKokkos::locate_tri_at_point`, `query_bfield_at_point`, `query_scalars_at_point`),
`surf_react_surface_pwi_kokkos.cpp`, and Kokkos versions of every fix in the deck except
`fix background` and `fix particle/weight`, which have no per-step hooks. The runtime confirms
it: the run prints `pcache: DEVICE fill active (mask 0x37, sheath ne correction)`,
`mesh B/E bound from fix background: 59936 tris`, `chem device path active`,
`cross_field_diffusion device fill`, `device emission ACTIVE from step 1`, and the end-of-run
ledger `Kokkos host fallbacks this run` lists only `finject` (surface/emit/source, thermal model,
nevery 1000: 20 host calls per rank in 20000 steps).

| Operation (deck) | Where | Host<->device traffic per step |
|---|---|---|
| background lookup for the pusher (pcache fill, `TagUpdatePcacheFill`) | device, every step (sheath forces nevery 1); one CSR hash-bucket walk per particle, no hint | none |
| move loop + Boris + sheath + surf collide/react (`TagUpdateMove`, 2.7 passes/step) | device | 60 B scalars D2H+H2D per pass; react/retry backups D2D: particles 2.1 MB + PWI sigma 2.2 MB + 4 small per pass |
| fix volume/chem/adas | device | event counters D2H (4 B x 2) + events |
| fix coulomb/background, force/thermal (2 half kicks), cross_field_diffusion | device | dx_cd 97 kB H2D |
| fix surface/emit/source femit | device | 2 counters D2H |
| fix surface/emit/source finject (thermal model) | host, every 1000 steps | full particle round trip |
| PWI ledger (sigma_nevery 100) | device deltas, host fold + Allreduce | 2.2 MB D2H + conc 2.2 MB H2D per pass on the following step |
| stats (compute reduce is a host compute) | host, every 1000 steps | full particle D2H |
| fix balance rcb part 1000 | host RCB, device particle migration | see section 5 |
| compute surface/physical/sputter (static yes) | host, cached once | none |
| dumps, restart | host, output steps only | particle D2H |

So particles do not bounce between host and device every step. What dominates is (a) a load-balance
pathology in the multi-rank Kokkos path and (b) a per-step fixed cost of synchronisations.

## 2. Wall time, 20000 steps from a fresh start

| run | ranks | loop s | ms/step | final markers | Mp-step/s | note |
|---|---|---|---|---|---|---|
| cpu4 | 4 CPU | 439.2 | 22.0 | 83186 | 3.8 | |
| cpu8 | 8 CPU | 244.5 | 12.2 | 83084 | 6.8 | |
| cpu48 | 48 CPU | 38.4 | 1.9 | 24454 | 12.7 | 64598 markers lost at the periodic caps |
| gpu1 | 1 A100 | 33.4 | 1.7 | 83842 | 50.2 | |
| gpu4 (unpatched) | 4 A100 | 179.3 | 9.0 | 83767 | 9.3 | Modify 79 % = fix balance |
| gpu4 no-rebalance control | 4 A100 | 44.0 | 2.2 | 83848 | 38.1 | balance 1000 -> 1000000 in a generated copy |
| gpu4 patched (section 5) | 4 A100 | 43.2 | 2.2 | 83840 | 38.8 | fix balance active, 20 rebalances |

The 48-rank CPU run is not the same physics: the production deck ships `global gridcut 0.0` with
`surf_collide PERIODIC toroidal`, and at 48 ranks 64598 markers (317k in 100k steps) are dropped
with "Particles lost at periodic caps (teleport target not local)". The 4- and 8-rank CPU runs and
all GPU runs lose none, so their populations are 3.4x larger. The cap loss is the decomposition
dependence already documented for the monoblock case; the deck needs `gridcut 0.035`.

Steady-state production regime, 100000 steps (fix balance active, marker count saturating):

| run | loop s | ms/step late | markers at 100k | lost at caps |
|---|---|---|---|---|
| cpu48_100k | 288.4 | 3.3 | 46715 | 316847 |
| gpu4_100k (unpatched) | 800.6 | 8.5 (Modify 590 s) | 179467 | 0 |
| gpu4p_100k (patched) | 236.2 | 2.5 | 180135 | 0 |

## 3. Per-step cost on the device (nsys + Kokkos Tools, 4 ranks, rank 0, 5000 steps)

Kokkos-label NVTX connector (`kp_nvtx.cpp`) + `nsys -t cuda,nvtx`; steady-state window of 400 steps
away from rebalances:

| quantity | per step (rank 0 of 4, ~20k markers) |
|---|---|
| GPU kernel time | 0.56 ms (18 launches) |
| Kokkos events | 232 (35 deep_copy fence pairs, 16 kernels, 2.7 move passes) |
| cudaDeviceSynchronize (host) | 140 calls, 1.1 ms |
| cudaMemcpy (host, blocking) | 63 calls, 1.1 ms |
| H2D | 21 copies, 0.48 MB (dx_cd 97 kB x1, 81 kB x3.6, rest <4 kB) |
| D2H | 23 copies, 25 kB (scalars, counters) |
| D2D | 16 copies, 12.8 MB (react/retry backups: particles 2.1 MB + sigma 2.2 MB + dep/sheath per pass) |
| wall per step | 2.5 ms (median) |

The step is latency bound: kernels occupy the GPU 0.56 ms of 2.5 ms; the rest is ~140 host-side
synchronisations (every `deep_copy` of a scalar or small view fences twice) and three move passes
per step each with backup, launch, scalar readback and migration plan. The move kernel itself runs
at 78 us per pass for 20k particles. One rank alone (gpu1) is faster than four at this size
(1.7 vs 2.2 ms/step) because the migration (2.3 passes/step, Comm 33 % at 4 ranks) and the
imbalance (Move min/max 11.5/17.3 s) cost more than the 4x smaller per-rank kernels gain. Kernel
time grows slowly with N (patched gpu4_100k: 2.5 ms/step at 180k markers vs 2.2 at 80k), so the
node pays off only once there are well over 100k markers per rank.

Top kernels by GPU time (rank 0, 5000 steps): `TagUpdateMove` 0.83 s, `TagUpdatePcacheFill`
0.63 s, `TagFixForceThermal` 0.34 s, `TagFixCoulombBg` 0.30 s, `TagFixCrossFieldDiffusion` 0.18 s,
`TagFixChemAdas` 0.10 s, migration kernels 0.25 s. Pcache fill + the three force fixes each walk the
mesh hash bucket again per particle (5 to 6 locations per particle per step); the subagent
inventory and `pusher_kokkos.h:557-600` confirm there is no per-particle hint on the device.

## 4. The rebalance pathology (why 4 GPUs were 5x slower than 1)

`OE_FIX_TIMING=1000` on the unpatched 4-rank run: `fbal balance` 41.76 s of the 42.99 s Modify
bucket in 5000 steps (5 calls); `OE_CELLMIG_DIAG=1`: `migrate_cells` 15.3 / 6.4 / 6.8 / 6.8 / 5.9 s
per call on rank 0 with `nmigrate=0` on that rank; inside `migrate_cells_only` the time is in
`pack` 4-10 s, `compress` 0.5-5.6 s, `unpack` 0.1-4.7 s on the ranks that send or receive cells and
in `plan` (waiting in `Irregular::create_data_variable`) on the others. In the 100k-step run the
per-1000-step time alternates 4.1 s / 12 s with the rebalance volume; Modify is 590 s of 800 s.

gdb stack sampling of rank 0 during rebalances (`gdb_stack_samples_gpu4_rebalance.txt`, 46
samples): `ModifyKokkos::unpack_grid_one` -> `ParticleKokkos::sync` -> `DualView::sync_device`
-> `cudaMemcpy` / `cudaDeviceSynchronize`, plus `Irregular::create_data_variable` ->
`MPI_Reduce_scatter` waits. The NVTX trace of the rebalance step on rank 0 counts 297 full particle
array D2H copies (2.1 MB each), 297 H2D, 5940 custom-vector H2D copies (131 kB each) and 891
ivector copies: one round trip per migrating cell.

Mechanism (`src/KOKKOS/modify_kokkos.cpp:191-310`): every per-grid hook (`pack_grid_one`,
`unpack_grid_one`, `copy_grid_one`, called once per migrating cell) brackets each gridmigrate fix
with `particle_kk->sync(fix->execution_space, fix->datamask_read)` and `modify(..., datamask_modify)`.
`fix background` sets `gridmigrate = 1` only to receive `grid_changed()` and has no Kokkos class,
so it carries the base-class `ALL_MASK` masks and Host space; `fix volume/chem/adas/kk` is also
gridmigrate (rate-cache rows follow cells) with Device space and `PARTICLE|SPECIES|CUSTOM`. Per cell
the pair therefore does sync(Host, ALL) D2H, modify(Host, ALL), sync(Device, P|S|C) H2D. At ~6000
cells per rank (4 ranks on the 88x7x42 grid) that is 6-15 s per rebalance; at 48 ranks (580 cells)
it is 12x smaller, which is why the CPU-node comparisons never showed it. This is the same trap
recorded for `surface/emit/source/kk` on 2026-09-08, through a different pair of fixes.

## 5. Fix applied (uncommitted, branch `perf-review`), validation

1. `src/fix_background.cpp` and `src/OPENEDGE/fix_background.cpp`: `datamask_read = datamask_modify
   = EMPTY_MASK` (the fix has no per-step particle hooks and its per-grid hooks are the base no-ops).
   Alone this ABORTS at the first rebalance ("particle cell index out of range after rebalance"):
   the chem fix's `sync(Device)` then runs under `auto_sync = 1`, whose blanket `modify(Host)` pushes
   the stale host mirror over the particles that `CommKokkos::migrate_cells` had just migrated on
   the device. The old ping-pong had been masking this.
2. `src/KOKKOS/modify_kokkos.cpp`: the six per-grid wrappers (`pack/unpack/copy/add_grid_one`,
   `reset_grid_count`, `grid_changed`) skip the particle `sync`/`modify` for fixes with
   `kokkos_flag` (they sync their own per-grid views, as `FixAveGridKokkos::pergrid_sync` does);
   host fixes keep the upstream behaviour.

Result: fix balance 0.142 s per 5 calls (28 ms per rebalance, `migrate_cells` 8 ms), 4-rank loop
179.3 -> 43.2 s for 20000 steps, identical to the no-rebalance control; tallies unchanged within
noise (reactions 1068891 vs 1069166, surface reactions 379848 vs 380064, markers 83767 vs 83840).
`tools/check_package_parity.sh`: ok. Regression with the patched binaries (`regression/run_regression.sh`, 4 ranks, srun in the
allocation): GPU `--kk` 21 passed / 0 failed / 1 skipped (st40 input absent) including kokkos_parity
(CPU==GPU) and pwi_deposit_tagging; CPU 20 passed / 0 failed / 2 skipped (st40, parity needs --kk).
Logs: `results/perlmutter_gpu_20261108/regression_{gpu,cpu}_patched.txt`.

## 6. What is left, ranked by measured share

| # | item | evidence | expected effect | risk / effort |
|---|---|---|---|---|
| 1 | Per-grid sync fix above | 75 % of 4-rank wall, 100 % of the 4-vs-1 GPU inversion | done: 4.1x on this deck; production 48-rank CPU-node equivalent now 1 node = 4 A100 at 2.2 ms/step vs 48 cores at 1.9-3.3 ms/step with 3.4x more markers retained | none to physics; regression 21/21 GPU, 20/20 CPU |
| 2 | Fence/launch floor: 35 deep_copy fence pairs + 140 cudaDeviceSynchronize per step | 1.1 ms of the 2.5 ms step is host-side sync at 20k markers; kernels 0.56 ms | scalar readbacks through pinned buffers and async copies on the move stream; merge the three collider `nsingle` readbacks and the PWI scalars; one combined event readback for chem | medium: ordering of retry/rollback; 3-5 days |
| 3 | react/retry backups each pass: sigma 2.2 MB + particles 2.1 MB D2D, 2.7 passes/step | 12.8 MB/step D2D, 16 copies and fences; sigma backup is the full global surf array per pass | snapshot only the active-particle range and a sparse sigma journal | medium: rollback exactness; 3-5 days |
| 4 | Move passes and migration: 2.3-2.7 passes/step, each with plan Alltoall and scalar round trips (Comm 33 % at 4 ranks) | gpu4 patched: Comm 14.1 s of 43 s | fold entry/exit scalars into one pinned readback per pass; skip the migration plan when no rank has leavers (one Allreduce is already paid) | low; 2 days |
| 5 | Repeated hash-bucket walks per particle (pcache, Boris B, E per subcycle, coulomb, thermal, cd) | pcache fill kernel 0.63 s vs move 0.83 s (rank 0, 5000 steps); no device hint | device per-particle hint (last triangle) as on the CPU, or one shared location per step | low with exact fallback; 3-5 days |
| 6 | `finject` host emission every 1000 steps (thermal model) and the stats `compute reduce` host sync | 2 full particle round trips per 1000 steps | port the thermal emission model; a reduce/kk compute | low; 1-2 days |
| 7 | Deck: `gridcut 0.0` with toroidal caps | 65k-317k markers lost at 48 ranks, none at 4 | fixed in the NERSC bundle deck (section 8); EOS decks still at 0.0 | physics of the campaign |

## 8. Deck fix: gridcut 0.035 (applied 2026-11-08 on the NERSC bundle copy)

`in.fresh_history` shipped `global gridcut 0.0` with `surf_collide PERIODIC toroidal` caps; the cap
teleport needs the far cap's cell to be stored locally, so at 48 ranks 64598 markers (20000 steps)
and 316847 (100000 steps) were dropped silently as "lost at periodic caps". Set to `gridcut 0.035`
(the cap distance, box y extent; backup `in.fresh_history.pre_gridcut_20261108`, bundle manifest
refreshed, `verify_bundle.sh` OK). `in.restart_history` inherits gridcut from the restart file.
Re-measured with the patched binaries:

| run (gridcut 0.035) | loop s | ms/step late | markers | lost at caps | boundary exits | c_cWtotal at end |
|---|---|---|---|---|---|---|
| cpu48_gc, 20k steps | 66.8 | 3.3 | 83871 | 0 | 15 | |
| gpu4p_gc, 20k steps | 36.3 | 1.8 | 83707 | 0 | 13 | |
| gpu1_gc, 20k steps | 33.7 | 1.7 | 83502 | 0 | 15 | |
| cpu48_gc_100k | 636.1 | 7.5 | 144174 | 0 | 48771 | 2.509e13 |
| gpu4p_gc_100k | 199.2 | 2.05 | 179946 | 0 | 40037 | 2.514e13 |

With the losses removed the 48-rank CPU run carries the same population as the 4-rank runs and takes
66.8 s instead of 38.4 s for 20000 steps; one GPU node is now 1.8x (20k steps) to 3.2x (100k steps)
faster than 48 CPU cores on the same deck, and 4 A100 are only 7 % faster than 1 A100 at 84k
markers. The GPU run also lost the 1.8 comm passes per step it had been spending on cap teleports
(2.3 before). Open check for the physics owner: at 100k steps the CPU and GPU runs differ in marker
count (144k vs 180k) while the weighted W inventory `c_cWtotal` agrees to 0.2 %, and both show
40-49k open-boundary exits over 100k steps (13-15 in the first 20k), i.e. markers leave through the
open box faces once the population has filled the wedge; that is the leak the EOS side is probing.
Tables: `results/perlmutter_gpu_20261108/finish_*_gc*.txt`.

## 9. Concurrency sweep: cases per node-hour versus single-case latency

Production is a campaign of independent cases, so throughput per node matters as much as the
latency of one case. Single-rank runs on one GPU each, gridcut 0.035, patched binary, allocation
61006018 (nid200292). "wall" is from `run_case.sh` start to the end of the log (setup about 15 s
for 20k steps, 60 s for 100k steps with dumps); loop is SPARTA's loop time.

| configuration | cases | loop per case | wall per case | batch wall | cases per node-hour |
|---|---|---|---|---|---|
| 1 case alone, 1 A100, 20k steps | 1 | 33.7 s | 50 s | 50 s | 72 |
| 4 cases, 1 per GPU, 20k steps | 4 | 32.5-32.7 s | 45-46 s | 54 s | 267 |
| 8 cases, 2 per GPU (no MPS), 20k steps | 8 | 95.6 s mean, 633 s max | 24-659 s | 659 s | 44 |
| 4 cases, 1 per GPU, 100k steps, dumps every 20k | 4 | 248-250 s | 311-312 s | 312 s | 46 |
| 48 CPU ranks, 20k steps (production line, 1 case per node) | 1 | 66.8 s | 72 s | 72 s | 50 |
| 48 CPU ranks, 100k steps | 1 | 636 s | about 680 s | 680 s | 5.3 |

Four cases per node run at full single-GPU speed (the node has 64 cores; each case used 8). Two
processes per GPU without MPS time-slice the device and one of each pair stalls (633 s loop), so
oversubscription needs MPS and was not pursued. The 16-run batch in the raw data (`conc4_*`)
serialised on the GPU gres and only shows that 16 serial runs all take 33.3 s.

In the realistic 100k-step interval (population grows to 180k markers, dumps and restart every
20k steps, output 0.8 % of the loop), one GPU node completes 4 cases in 312 s against one case in
about 680 s on 48 CPU cores: 8.7x the cases per node-hour including setup, 10.2x on loop time.
The single-case latency gain is 2.5x (249 vs 636 s loop) with one GPU per case, or 3.2x with four
cooperating GPUs (199 s). The campaign-level 10x is therefore available today by packing four
cases per node; the per-case 10x needs the per-step work below.

## 10. Per-step critical path (what a single-case 10x needs)

Non-overlapping budget of one step at 4 cooperating ranks, rank 0, 20k markers per rank, from the
NVTX trace (steady window) and nsys API sums; the step takes 2.2 ms wall:

| component | per step | evidence |
|---|---|---|
| GPU kernel execution | 0.56 ms | cuda_gpu_kern_sum; move 2x78 us, pcache 125 us, thermal 2x42 us, coulomb 76 us, cd 46 us, chem 25 us, migration ~50 us |
| host blocking in fences | ~0.7 ms | 140 cudaDeviceSynchronize/step (5.4 s / 5000 steps); 35 deep_copy fence pairs + 14 scalar-copy pairs + 16 scalar-to-scalar pairs |
| small blocking copies (control scalars, counters) | ~0.3 ms | 63 cudaMemcpy/step: update scalars H2D+D2H per pass, 4 collide pre-zero + 4 readbacks per pass, PWI scalars, chem counters, emit counters, comm pmeta |
| migration plan and imbalance wait | ~0.3-0.5 ms | Comm 26-33 % of the loop at 4 ranks (7-14 s / 20k steps), 1.8-2.3 passes per step, Alltoall per pass |
| launch and host bookkeeping | ~0.3 ms | 16 launches, view rebinding, functor copies per pass |

Order of work agreed on 2026-11-08, each item measured on 1 GPU and on 4 cooperating GPUs before
the next, with the regression suite and the gpu_parity stage after each:
1. compact control readbacks: one device control block zeroed with one async fill per pass and
   read back with one copy; collide counters read once per step instead of per pass;
2. asynchronous copies on the same execution-space instance for every fill and backup that is
   not consumed by the host, one fence before each host consumer;
3. persistent buffers (no per-pass view allocation in migration/backups; already partly done);
4. backups restricted to the state that rollback needs (active particle range, sparse sigma journal);
then continuation queues (one pass per step for reaction products) and a shared exact gather
(one triangle location per particle per phase shared by pcache, Boris B/E, coulomb, thermal, cd).
0.5 ms/step is the experimental target: 0.3 ms of kernels after fusion plus 0.2 ms of host work.

## 11. Item 1 done: compact control readbacks (uncommitted, measured)

Change (files `src/KOKKOS/surf_collide_{diffuse,vanish,transparent,toroidal,specular,piston}_kokkos.{h,cpp}`,
`src/KOKKOS/update_kokkos.cpp`): each surface-collide style's `post_collide()` now only issues its
counter readback as an execution-space copy and a new `post_collide_apply()` folds it into the host
tallies; `UpdateKokkos::move` fences once per pass for all styles instead of two global fences per
style. The move loop's per-pass scalar upload and the react/retry sheath and GC custom backups are
execution-space copies too (ordered on the same instance ahead of the move kernel). Semantics are
unchanged: every host consumer still sits behind a fence.

| run, 20000 steps, gridcut 0.035 | before item 1 | after item 1 |
|---|---|---|
| 1 A100 loop | 33.7 s | 31.4 s (-7 %) |
| 4 A100 loop | 36.3 s | 33.6 s (-7 %) |
| 4 A100 Move bucket (avg) | 12.9 s | 10.7 s |
| Kokkos events per step, steady state, rank 0 | 232 | 131 |
| fence events per step | 105 | 61 |
| cudaDeviceSynchronize per step (whole 5000-step run) | 141 | 25 |
| median step wall, rank 0 of 4 | 2.54 ms | 1.74 ms |
| reactions / surface reactions (4 ranks) | 1067565 / 379708 | 1064744 / 378582 |

The per-step wall at 4 ranks is now 1.68 ms (loop/steps) against 0.54 ms of kernels. What remains
in the steady-state trace, per step: 12 value fills (`deep_copy(view, 0)` in the fixes, chem, emit,
PWI: 0.27 ms of fences), 10 small view copies (comm pmeta and index_send, PWI scalars, chem event
readbacks, dx_cd upload: 0.24 ms), the two blocking scalar readbacks after the move kernel (0.25 ms,
mostly waiting for the kernel itself), and the fix kernels. Next edits in the agreed order: value
fills through an execution-space `deep_copy(exec, view, 0)` (item 2, mechanical), a pinned control
block shared by update/collide/PWI/comm read once per pass (rest of item 1), then the migration
plan skip and the kernel fusion. Regression on this binary (GPU `--kk`, incl. kokkos_parity): 21 passed, 0 failed, 1 skipped (`results/perlmutter_gpu_20261108/regression_gpu_item1.txt`); the CPU binary is untouched by item 1.

## 12. Reconciled timing table and checkpoint

Every wall-time figure in this report, keyed by what changed between them. All runs:
`prevention_fixed_on`, 20000 steps from a fresh start unless marked 100k, same node class
(A100 node, 64 Milan cores), binaries from this tree. "deck" is the gridcut of
`in.fresh_history`; "code" is the commit whose source the binary was built from.

| code | deck | 48 CPU | 4 CPU | 1 A100 | 4 A100 | 4 A100 100k | 48 CPU 100k |
|---|---|---|---|---|---|---|---|
| 540f0719 (base + build fixes) | gridcut 0.0 | 38.4 s (24k markers, 64598 lost) | 439.2 s | 33.4 s | 179.3 s | 800.6 s | 288.4 s (47k, 316847 lost) |
| 6e663da3 (+ per-grid sync fix) | gridcut 0.0 | | | | 43.2 s | 236.2 s | |
| 6e663da3 | gridcut 0.035 | 66.8 s (84k markers) | | 33.7 s | 36.3 s | 199.2 s (180k) | 636.1 s (144k) |
| 32b3f05b (+ item 1) | gridcut 0.035 | | | 31.4 s | 33.6 s | | |

The 48-rank CPU figures at gridcut 0.0 are not comparable with anything else (lost population);
the first two GPU rows at gridcut 0.0 are comparable with each other only. Speed-ups quoted in the
text: 179.3/43.2 = 4.15x and 800.6/236.2 = 3.39x (sync fix, gridcut 0.0); 66.8/33.6 = 2.0x
(1 node vs 48 cores, 20k steps) and 636/199 = 3.2x (100k steps) on the corrected deck.

Checkpoint (branch `perf-review`, tracking `origin/development`, nothing pushed):

| commit | content | validation |
|---|---|---|
| 540f0719 | build fixes from the EOS review (cmake compile options, ADAS Kokkos eirene_mode) | builds CUDA+OpenMP and CPU |
| 6e663da3 | correctness/performance: no particle sync around per-grid hooks of Kokkos fixes; fix background EMPTY masks | GPU regression 21/0/1 incl. kokkos_parity; CPU 20/0/2 |
| 32b3f05b | performance item 1: async collide counter readbacks, one fence per pass | GPU regression 21/0/1 incl. kokkos_parity; CPU suite not rerun (no CPU-path file touched: the change is confined to `src/KOKKOS/`) |
| 7204f483 | tools/performance_review (this report, raw data, scripts) | n/a |

Binaries of 32b3f05b: `spa_kokkos_cuda_perlmutter` sha256 7d5927baab7f2020..., `spa_mpi` 97d66cd0edadd93b...
(copies kept as `$PSCRATCH/oe-perf-tools/spa_kokkos_cuda_item1_32b3f05b`). Skipped in every
regression pass: `st40_core_species_boundary` (its input deck is not in this checkout) and, in CPU
mode, `kokkos_parity` (needs the --kk binary; it ran and passed in the GPU passes).

## 13. Item 2 done: asynchronous fills and small copies (commit eac15682)

Three unprofiled repeats per configuration, 20000 steps, gridcut 0.035, same node:

| binary | 1 A100 loop s (3 repeats) | 4 A100 loop s (3 repeats) |
|---|---|---|
| item 1 (32b3f05b) | 31.54 31.37 31.41 | 33.29 33.18 33.51 |
| item 2 (eac15682) | 29.27 29.17 29.11 | 31.64 31.13 31.16 |

Repeat spread is below 0.3 s, the change is 2.2 s (7 %) on one GPU and 2.0 s (6 %) on four; reactions
1067089-1068734 and surface reactions 379027-380153 across all runs, i.e. within the run-to-run noise.
Re-profile (rank 0 of 4, steady window): 88 Kokkos events per step (232 at the start, 131 after
item 1), 16 fence events per step (105, 61), median step 1.68 ms (2.54, 1.74); cudaDeviceSynchronize
has left the API top list, cudaMemcpyAsync is now the largest host item (36 per step, 0.6 ms): these
are the control copies that still target pageable host views, so the driver stages them synchronously.
GPU regression 21/0/1 incl. kokkos_parity. Retained.

## 14. Measured production latency, campaign throughput, projected targets

Measured (code eac15682 unless stated, deck gridcut 0.035, prevention_fixed_on):

| metric | value | reference |
|---|---|---|
| single case, 4 A100, 20k steps | 31.3 s (1.57 ms/step) | 48 CPU ranks 66.8 s: 2.1x |
| single case, 1 A100, 20k steps | 29.2 s (1.46 ms/step) | 2.3x |
| single case, 4 A100, 100k steps (code 6e663da3) | 199 s, 2.05 ms/step at 180k markers | 48 CPU ranks 636 s: 3.2x |
| single case, 1 A100, 100k steps with dumps (code 6e663da3) | 249 s | 2.6x |
| campaign, 4 cases per node, 100k steps (code 6e663da3) | 46 cases per node-hour | 48 CPU ranks 5.3: 8.7x (10.2x on loop time) |

The 100k-step figures predate items 1 and 2 (those were measured at 20k steps); scaling their 13 %
onto the 100k runs is a projection, not a measurement.

Projected single-case targets, 4 cooperating A100, per step (kernels today 0.54 ms of 1.57 ms):

| stage | expected step | basis |
|---|---|---|
| pinned control block: all control scalars/counters in CudaHostPinnedSpace views, one async readback per pass | 1.2-1.3 ms | 36 pageable cudaMemcpyAsync per step at ~15 us each |
| migration plan skip when the global leaver count is zero (all ranks call the Allreduce; a rank with no leavers may still receive) | 1.1-1.2 ms | 1.8-2.3 passes per step; second and third passes often move nothing |
| shared exact gather: per-particle triangle hint written by the pcache fill, tested first with the same containment predicate in the 16 device query sites, bucket walk as fallback | 0.9-1.0 ms | pcache + three force kernels re-walk the bucket; CPU hint hit rate 99.55 % |
| selective fusion: start-of-step (emit, cd fill, thermal half kick) and end-of-step (chem, coulomb, thermal half kick) kernels sharing one location | 0.6-0.7 ms | 6 launches + 6 syncs per step; operator order within each phase preserved as a sequence inside the kernel |
| continuation queue: reaction products pushed in the same pass | 0.5-0.6 ms | 2.1 move launches per step today |

0.5 ms/step stays an experimental target: it assumes every stage lands at the optimistic end and
no new critical path appears. Each stage follows the item-1/2 protocol (three repeats on one and
four GPUs, re-profile, regression + parity) before it is retained.

## 15. Item 3: pinned control block (measured, two variants)

Persistent pinned host staging (`OEPinnedSpace` = CudaHostPinnedSpace under CUDA, HostSpace otherwise,
`kokkos_type.h`) for every small control transfer: the move loop's 15-int control block (upload and
per-attempt readback), the six surface-collide counter readbacks (now issued inside the attempt
loop so the one per-attempt fence covers them; the style hooks are split into `post_collide_issue`,
`post_collide`, `post_collide_apply`), the PWI tally readback, the emission counters, and the
migration `pmeta` readback. Host views keep their types; the pinned view is a staging copy next to
them, and the host<->pinned transfer is a `memcpy`. Device-side ordering is unchanged (one instance);
diagnostic values (pcache diag, sheath diag) stay on the device as before.

Three unprofiled repeats each, 20000 steps, same node and deck, same session:

| binary | 1 A100 loop s | 4 A100 loop s |
|---|---|---|
| item 2 (eac15682) | 29.35 29.27 29.39 | 31.13 31.21 30.89 |
| pinned block, host<->pinned via Kokkos deep_copy (rejected) | 30.88 30.84 31.12 | 33.69 33.56 33.58 |
| pinned block, host<->pinned via memcpy (retained candidate) | 27.99 27.91 27.97 | 29.80 29.65 29.44 |

The first variant is slower by 5-8 % in every repeat: a Kokkos `deep_copy` between pageable and
pinned host views is a cudaMemcpy with fences, which re-introduced the synchronisation it was meant
to remove. The memcpy variant gains 4.7 % on both one and four GPUs; reactions 1064907-1069380 and
surface reactions 378833-379621, within noise.

Profile, rank 0 of 4, 5000 steps, item 2 vs retained candidate. Counting scopes: the host API
issues 36.0 cudaMemcpyAsync + 1.2 cudaMemcpy + 8.1 cudaMemcpyToSymbolAsync = 45.3 copy calls per
step; the GPU-side memory operations are 15.5 H2D + 17.7 D2H + 12.0 D2D = 45.2 per step, the same
set seen from the device. The 8.1 ToSymbol copies are Kokkos functor-argument uploads to constant
memory (one per `cuda_parallel_launch_constant_memory` launch), so only 7.4 of the H2D operations
carry data. Copy calls and bytes are unchanged by the pinned block (H2D 434 kB, D2H 40 kB, D2D
9.3 MB per step), as intended: it changes how the host waits, not what moves. Kernels 0.54 ms per
step in both. Bytes unchanged (H2D 434 kB, D2H 40 kB, D2D
9.3 MB per step), kernels 0.54 ms per step both. Host blocking time per step moved and shrank: item
2 cudaMemcpyAsync 0.59 ms + stream/event/device syncs 0.21 ms = 0.80 ms; candidate cudaMemcpyAsync
0.24 ms + cudaStreamSynchronize 0.30 ms + event/device 0.17 ms = 0.71 ms. The copies no longer
stall individually; the explicit fences now carry the wait for the GPU. The projected 0.6 ms saving
was not realised: the measured saving is about 0.09 ms per step (1.4 s per 20000 steps), because most
of what the staged copies had been waiting for was the kernels themselves. Regression on the retained candidate (GPU `--kk` incl. kokkos_parity): 21 passed, 0 failed, 1 skipped
(`results/perlmutter_gpu_20261108/regression_gpu_item3.txt`). Committed as f806e9ac; binary kept as
`$PSCRATCH/oe-perf-tools/spa_kokkos_cuda_item3_f806e9ac`.

## 16. Production rerun on the retained binary (f806e9ac = items 1-3)

Same deck (gridcut 0.035), same node class, 100000 steps from a fresh start:

| metric | before items 1-3 (6e663da3) | retained (f806e9ac) | 48 CPU ranks | gain over 48 cores |
|---|---|---|---|---|
| single case, 4 cooperating A100, loop | 199.2 s | 165.0 s (1.75 ms/step late, 179k markers) | 636.1 s | 3.9x |
| single case, 1 A100 with dumps every 20k, loop | 249 s | 225 s | 636.1 s | 2.8x |
| 4 cases per node (1 GPU each), wall incl. setup | 312 s | 241 s | about 680 s per case | 59.8 vs 5.3 cases per node-hour: 11.3x |

Items 1-3 together took 17 % off the single-case 4-GPU production run and 10 % off the single-GPU
run. The campaign figure now exceeds 10x per node-hour on this deck; the single-case figure stands at
3.9x, with 0.54 ms of kernels inside a 1.65 ms step at 4 ranks (loop/steps). The remaining host
share is the per-pass structure itself (2.1 move launches, migration plan and exchange, 10 fences)
and the fix kernels; those are the shared-gather and fusion items, not yet started.

## 17. Migration-plan skipping: measured, not implemented

`OE_COMM_TIMING` with two new counters (commit pending with the gather work), 20000 steps, gridcut 0.035:

| | 4 A100 | 1 A100 |
|---|---|---|
| migrate passes per rank | 56034 (2.8 per step) | 39999 (2.0 per step) |
| passes with nothing to exchange on any rank | 36714 (65 %) | 39999 (100 %) |
| passes with nothing leaving a given rank (summed over ranks) | 145645 of 224136 | 20021 |
| plan (Alltoall) seconds, rank 0 / slowest rank | 1.84 / 6.67 | 0.01 |
| exchange seconds | 2.37 | 0 |
| migrate total seconds, rank 0 / slowest | 6.32 / 11.27 | 1.28 |

Two thirds of the passes move nothing anywhere (reaction-product continuation passes and settled
entry/exit passes), and each still pays the counts Alltoall. That Alltoall is also the only thing
that establishes global emptiness: a rank with no leavers may still receive, so participation
cannot be dropped, and any cheaper test (an Allreduce of the leaver count) is itself a collective
of the same order at 4 ranks. The avoidable part is therefore the Alltoall call overhead alone,
about 33 us per pass on rank 0 (1.84 s / 56034), i.e. under 0.06 ms per step, while the 4.8 s
difference between rank 0 and the slowest rank in `plan` is imbalance wait that any collective
would absorb. Not worth a code path. The lever that this measurement does expose is the pass count:
2.8 migrate passes per step at 4 ranks (2.0 at 1 rank, all empty) is the continuation-queue item.

## 18. Item 4: shared exact gather, stage 1 (triangle hint in the pcache fill and the mover)

Design. The CPU keeps a validated per-particle "last triangle" (`fix background` custom
`oe_mesh_tri_<id>`, int, tri+1, 0 = none) that follows the particle through sorting, cloning,
migration and rebalance because it is a SPARTA custom. The device now reads the same custom:
`MeshKokkos::locate_tri_hinted` tests the hinted triangle first and accepts it only when the point
is strictly inside (every barycentric coordinate beyond the 1e-10 band). A strictly interior point
lies in exactly one triangle of a proper mesh, so the answer equals the bucket walk's; points in the
band (shared edges, vertices) and misses fall through to the unchanged CSR bucket walk, which keeps
the original first-hit order. This is a location hint only, not a cache of interpolated fields:
every query still reads the fields of the triangle found at the queried position, including the
per-subcycle E query at intermediate pusher positions (a local copy of the hint follows them). The
pcache fill writes the hint back; the mover reads it. Stage 1 covers the pcache fill (plasma scalars
+ B), the mover's start-position B, per-subcycle mesh E and the sheath scalar fallback.

Three interleaved unprofiled repeats vs f806e9ac, 20000 steps, gridcut 0.035 (first two repeats
2026-11-08 on nid003348, third 2026-11-09 on nid001044):

| binary | 1 A100 loop s | 4 A100 loop s | 4 A100 Move bucket avg |
|---|---|---|---|
| item 3 (f806e9ac) | 28.34 28.02 28.19 | 29.41 29.17 29.27 | 8.59 8.59 8.57 |
| item 4 stage 1 | 27.05 26.91 26.91 | 29.07 29.07 29.02 | 7.99 8.07 7.98 |

Gain 4.3 % on one GPU, 0.8 % on four (the four-rank step is latency bound; the kernels it shortens
are a smaller share there). Kernel time per step at 4 ranks 0.542 -> 0.511 ms. Reactions and
surface reactions within the run-to-run band. GPU regression 21/0/1 incl. kokkos_parity. Retained.
Stage 2 (hint read in the coulomb, thermal and cross-field kernels) is measured next.

## 19. Item 4 stage 2: hint read in the coulomb, thermal and cross-field kernels

The three force fixes bind the same custom (read-only; the pcache fill owns the write) and use the
hinted locator and B query. Three interleaved repeats vs 78335f65 (stage 1), 2026-11-09, nid001044:

| binary | 1 A100 loop s | 4 A100 loop s | 4 A100 Modify bucket r1 |
|---|---|---|---|
| stage 1 (78335f65) | 27.04 26.90 27.16 | 28.73 28.59 28.54 | 5.05 s |
| stage 2 | 26.16 26.22 26.12 | 28.54 28.77 28.77 | 4.78 s |

Gain 3.2 % on one GPU, flat on four (the fix kernels are 0.1 ms of a 1.43 ms step there and the
step is latency bound). Retained for the one-GPU-per-case campaign configuration; it does not
regress the four-rank case. Note the node-to-node spread: today's four-rank stage-1 baseline (28.6 s)
is below yesterday's stage-1 measurement on another node (29.05 s), which is why every comparison
is made from interleaved repeats on the same node. GPU regression 21/0/1 incl. kokkos_parity. Retained (commit below).

## 20. Item 5: react/retry backups restricted to the moved range

`UpdateKokkos::backup/restore` now snapshot and roll back only `[pstart,pstop)`, the particles the
pass moves (plus the same range of the sheath and GC customs); the kernel never writes other
particles, products appended past `nlocal` are discarded by the retry anyway, and an empty pass
(entry/exit passes on ranks that received nothing) copies nothing. The PWI sigma/dep and collider
backups are unchanged. Three interleaved repeats vs 5efd24d8, 2026-11-09, nid001044:

| binary | 1 A100 loop s | 4 A100 loop s |
|---|---|---|
| stage 2 (5efd24d8) | 26.30 26.21 26.05 | 28.40 28.37 28.57 |
| item 5 | 25.76 25.70 25.52 | 28.47 28.45 28.37 |

Gain 2.0 % on one GPU, flat on four; tallies within noise. Regression: REGRESSION_I6.

## 21. Where the four-rank step stands and what is left

At 4 cooperating A100 the step is 1.42 ms (28.4 s / 20000) with 0.50 ms of kernels; at 1 A100 the
step is 1.28 ms. The host-side remainder is now dominated by SPARTA's own pass structure under
`gridcut >= 0`: about 2.1 move launches, 2.8 migrate calls (one plan Alltoall each, 65 % of them
exchanging nothing) and the per-pass control fence. Reaction-product continuation passes are rare
(0.08 per step), so a continuation queue would not pay here; the second pass per step is the
entry/exit pass that any rank's particle crossing out of the ghost layer forces on all ranks.

Kernel side (rank 0, per launch): pcache fill 146 us is now the largest kernel, ahead of the move
(74-78 us). Its cost is the per-particle sheath section: nearest-plane refinement over the cell's
`csurfs` and `prepare_borodkina` for every particle, although the potential is only applied when
the wall distance is below the window `mps_end_m` (about 10 rho_i sin alpha plus a few Debye
lengths). An exact early-out needs a run-wide upper bound of that window (max over the mesh of
rho_i and lambda_D) so that particles farther from the cell's nearest element than the bound skip
the refinement and the coefficient prep; the result is unchanged because phi is not applied there.
That and the start/end-of-step fusion (6 launches and their syncs into 2) are the next items; both
are measured with the same protocol before being retained.

## 22. Production rerun on 5efd24d8 (items 1-4, hint stages 1-2)

Same deck (gridcut 0.035), 100000 steps from a fresh start, nid001044, 2026-11-09:

| metric | 6e663da3 (before items) | f806e9ac (items 1-3) | 5efd24d8 (+ hint 1-2) | 48 CPU ranks | gain |
|---|---|---|---|---|---|
| single case, 4 A100, loop | 199.2 s | 165.0 s | 156.3 s (1.65 ms/step late, 180k markers) | 636.1 s | 4.1x |
| single case, 1 A100 with dumps every 20k, loop | 249 s | 225 s | 186 s | 636.1 s | 3.4x |
| 4 cases per node (1 GPU each), batch wall incl. setup | 312 s | 241 s | 202 s | about 680 s per case | 71.3 vs 5.3 cases per node-hour: 13.4x |

The triangle hint pays most where the kernels matter: the one-GPU 100k case gained 17 % (225 to
186 s) against 4 % at 20000 steps, because at 180k markers per GPU the hinted lookups are a larger
share of the step than at 84k. Campaign throughput per node is now 13.4x the 48-core line and the
single-case latency 4.1x; 10x single-case needs 63.6 s, i.e. a further 2.5x on 156 s.

## 7. Reproduction

```bash
# allocation: salloc --no-shell -q interactive -C gpu -N 1 -t 4:00:00 -A m5186_g --gpus-per-node=4
export OE_JOBID=<jobid>
results/perlmutter_gpu_20261108/run_one.sh gpu  gpu4   4 20000     # CUDA GTL binary, IPC launch line
results/perlmutter_gpu_20261108/run_one.sh cpu  cpu48 48 20000     # spa_mpi
OE_FIX_TIMING=1000 OE_CELLMIG_DIAG=1 results/perlmutter_gpu_20261108/run_one.sh gpu gpu4_ft 4 5000
OE_NSYS_RANKS=0 results/perlmutter_gpu_20261108/run_one.sh gpu-nsys gpu4_nsys 4 5000   # kp_nvtx.so + nsys
results/perlmutter_gpu_20261108/nsys_summary.sh <run>/profile_gen/nsys_r0.nsys-rep 5000
```
The shell that launches these may itself live inside a Slurm job (jupyter node): `run_one.sh`
unsets every `SLURM_*` variable before setting the allocation id, and everything `srun` execs
must be on `$PSCRATCH`, not in a node-local `/tmp`.
