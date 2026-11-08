# OpenEdge WEST performance review

Repeated mesh queries are expensive, but the proposed explanation that OpenEdge lacks a device pusher and PWI implementation does not match the current source. The WEST Kokkos-host run selects device background sampling, chemistry, and movement, with no reported host fallbacks. Its operation trace instead exposes many dispatches, fences, and large rollback snapshots. Unconditional Cartesian background tabulation also fails the initial accuracy test, especially at plasma and wall boundaries.

The retained code changes repair two Kokkos build blockers. The small CPU lookup shortcut is preserved as an experiment because release timings do not establish a useful speedup. No approximate background, physics change, database edit, or production input edit is included.

## Benchmark and provenance

Measurements were taken on exclusive EOS compute nodes, excluding `eos01`, where the user's three existing WEST jobs continued. No local benchmark or build was left running on the laptop. Mora was reachable and idle but supplied no CUDA device; the `nersc` SSH alias did not resolve.

The source snapshot came from local revision `a508aff6cc14b19c34024d6f39fcaa839c49ebd1`, including the existing untracked diagnostic sources. EOS uses AMD EPYC 9654 nodes with 192 physical cores, GCC 15.2, OpenMPI 4.1.7, and HDF5 1.14.5. The CPU profile build uses `-O3 -g -pg -DNDEBUG`; release timing builds use `-O3 -DNDEBUG`. MPI binding remains the benchmark script's `--bind-to none`.

The case is `teprobe_fixed_off`, 20,000 steps, `dt=1e-8`, grid `88x7x42`, 59,936 background triangles, and 1,231 wall-flux samples. Output and restart cadence is 20,000 steps. `DENSITY_REPEAT=80` is required: the brief's command otherwise combines a 25,000-step averaging window with a 20,000-step output interval and fails during setup. Generated private decks enable post-run timing and mesh diagnostics. Production decks remain unchanged.

The plasma and `processes.h5` SHA256 hashes match the laptop copies; see [data hashes](results/data.sha256) and [host details](results/host.txt). The remote database is read through a symlink and was not modified. Files and full profiles remain under:

```text
eos:/scratch/home_eos/42d/oe_perf_review_20261107/
  source/                 baseline source plus retained build fixes
  source_patched/         experimental scalar memo candidate
  evidence/               original CPU profiles, Kokkos counts and audit
  patched_evidence/       candidate profiles and wall comparisons
  timing_evidence/        paired release-build measurements
  baseline_np8/           production-format outputs and restart
  baseline_np48/
```

## CPU profiles

| Measurement | 8 ranks | 48 ranks |
| --- | ---: | ---: |
| Loop time, 20,000 steps | 363.478 s | 78.7187 s |
| Mean time per step | 18.174 ms | 3.936 ms |
| Final markers | 85,515 | 69,217 |
| Particle moves | 997,105,660 | 813,497,700 |
| Move | 22.02% | 13.91% |
| Coll | 18.92% | 11.69% |
| Modify | 21.07% | 13.11% |
| Pcache | 9.51% | 5.82% |
| Chem | 9.79% | 5.93% |
| Adens | 1.54% | 13.97% |
| Comm | 0.17% | 1.28% |
| Output | 0.03% | 0.14% |
| Other | 16.95% | 34.13% |

These are instrumented loop timings, not release performance predictions. The 4.62x elapsed-time ratio is not a fixed-work strong-scaling result: rank-local random streams and migration produce different marker histories. Timer buckets use OpenEdge's internal attribution; `Other` must not be equated with measured MPI time. Startup and the final standalone `write_restart` are outside the loop timing.

The 8-rank flat profile attributes 12.96% of sampled application self CPU time to triangle walking, 9.31% to Nanbu background scattering, 8.27% to the Boris implementation, 7.20% to B queries, and 11.90% to point-line plus point-triangle distance functions. There are 10.9997 billion triangle-walk calls, 4.6462 billion B queries, and 2.8116 billion ADAS table interpolations over the complete application execution. Approximately 99.55% of performed mesh lookups use hints; the cost is repeated validated queries, not a full 60,000-triangle scan for every marker. These gprof figures include setup and exclude most uninstrumented shared-library work. [8-rank flat profile](results/cpu_np8_gprof_flat.txt), [48-rank flat profile](results/cpu_np48_gprof_flat.txt).

At 48 ranks, the final population is 1,442 markers/rank on average, minimum zero, maximum 1,795. Move time ranges from 0.207 to 21.003 seconds across ranks. Seven toroidal cells therefore do not impose a seven-rank limit, but particle-count balance leaves different workloads, empty ranks, and a much larger surface-ledger/coordination share. A single final imbalance number does not describe the entire run.

## Kokkos path and operation counts

`src/KOKKOS/pusher_kokkos.h`, `update_kokkos.cpp`, and `surf_react_surface_pwi_kokkos.cpp` contain the previously suspected missing functionality. WEST prints `pcache: DEVICE fill active`, binds mesh B/E, enables the ADAS device path, and reports no host fallbacks. The sputter compute's Kokkos style delegates to its CPU implementation, but its `static yes` cache avoids repeating wall-flux searches every step.

Runtime activation needs checking independently of the executable name. The production `run_case.sh` does not add `-k on` or `-sf kk`; `sparta.cpp` defaults `kokkosflag` to zero and constructs the CPU mover unless activation is requested. Merely substituting a CUDA binary in that script therefore does not enable Kokkos. The review wrapper adds the flags explicitly. Whether historical Perlmutter runs did so is not established here. `OE_PROFILE_KOKKOS_ARGS` lets the wrapper accept an explicit CUDA configuration without editing a production deck.

The 8-rank, one-thread-per-rank OpenMP run took 349.221 seconds for 20,000 steps and ended with 85,369 markers. This is a different backend/build from the CPU gprof run and is not a GPU speedup measurement.

The counter plugin requests no additional global fencing. It records the complete application, including initialization:

| Callback measurement | Per rank per step |
| --- | ---: |
| `parallel_for` dispatches | 26.58 |
| Fence callbacks | 67.25 |
| Requested deep-copy payload | 12.97 MB |
| Surface sigma rollback payload | 6.13 MB |
| Particle rollback payload | 5.45 MB |
| Sheath bank and previous-potential rollback payload | 0.73 MB |

All observed copy spaces are Host on this backend. Requested payload is not measured memory-bus traffic, and OpenMP fence callbacks are not evidence of CUDA synchronization duration. CUDA can also implement fills differently. GPU kernel durations, H2D/D2H bytes, and PCIe bottlenecks remain unmeasured. The collector is ready for a CUDA run. [Complete counts](results/kokkos_counts.json).

## Ranked improvements

Effort ranges below are engineering estimates. Numerical ceilings describe the measured profile slice that could be removed; they are not promised whole-application gains.

| Rank | Change and evidence | Gain or measurable target | Physics risk | Effort |
| --- | --- | --- | --- | --- |
| 0 | Verify runtime `-k on` and `-sf kk`, build flags, and selected styles in historical GPU jobs. The production launcher supplies neither activation flag. | Recover the exact command first; no speedup estimate is justified without it. | None for inspection. | Minutes. |
| 1 | Share exact mesh location within a query stage; add device particle hints or Cartesian candidate lists. Billions of repeated walks/B/scalar queries dominate sampled CPU work. | Eliminating the 12.96% walk self-time slice alone gives a 1.15x sample-space ceiling. Repeated field queries offer additional scope; elapsed gain needs measurement. | Low if candidate order, intermediate positions, and fallback semantics remain exact. | 3–7 days for common gather; longer for a grid accelerator. |
| 2 | Reduce retry snapshots and unnecessary synchronization. Requested snapshot payload dominates the 12.97 MB/rank-step copy total; 67 fence callbacks/rank-step. | Measure active-particle subviews and sparse surface journals against the present 11.58 MB/rank-step sigma-plus-particle snapshot payload. CUDA gain unknown. | Medium: preserve rollback ledgers, customs, and RNG state. | 3–10 days plus CUDA validation. |
| 3 | Reuse sheath geometry at identical positions and avoid redundant distance/field preparation. Distance functions consume 11.90% of CPU self samples. | Removing that entire slice gives a 1.14x sample-space ceiling; practical gain is lower. | Medium near edges and trial/replay positions. | 3–7 days. |
| 4 | Balance by observed particle/operator cost and investigate surface reductions. At 48 ranks Adens is 13.97%, with empty ranks and broad move-time variation. | Removing all Adens time would cap that isolated benefit at 1.16x loop speed. Instrument MPI waits before predicting a gain or changing cadence. | Low for decomposition; higher for changing synchronization cadence. | 3–7 days. |
| 5 | Share ADAS interpolation brackets or use exact rate reuse. 2.812 billion interpolations, about 2.82 per particle move. | Interpolation plus bracket setup account for 4.64% of CPU self samples, a 1.05x isolated ceiling. Precomputed approximate cell rates need accuracy gates. | Low for identical-input reuse; high for sheath or energy-dependent approximations. | 3–10 days. |
| 6 | Measure cell sorting and hot-field SoA/AoSoA before replacing `OnePart`. CPU record size is 120 bytes; custom vectors are already SoA. | No cache-counter, bandwidth, or sorting A/B result is available. No numeric speedup is assigned. | Medium: migration/custom alignment and RNG association. | Several days for sorting experiments; weeks for container changes. |
| 7 | Use existing distributed output modes before designing collective HDF5. Current filenames serialize dump/restart data through one writer. | Loop output is only 0.03–0.14% for this cadence; final standalone restart cost is not included. | Low for changing file organization, with readback tests. | 1–3 days. |

Static-background field tabulation is a separate structural project. The audit finds electron-density NRMSE of 10.19% with centers and 4.71% with linear fits across cell boxes; near-wall errors and negative fits preclude a blanket switch. Prefer exact candidate lists and conservative validity masks first. [Design and accuracy gates](DESIGN.md), [full accuracy CSV](results/background_accuracy.csv).

The O(N) wall-flux searches are real, but the static sputter/source setup prevents them from dominating the timed WEST loop. Replacing their search structure is lower priority for this case.

## Changes and validation

Two build fixes remain active:

1. Remove ADAS Kokkos references to the deleted `eirene_mode` member and update the associated comments.
2. Keep SPARTA compiler options on the CMake project directory instead of overwriting `MPI::MPI_CXX` interface options. Feature definitions stored only on that imported interface were absent from the combined Kokkos build's compilation flags, producing dummy/real class redefinitions. This also makes project options available with MPI stubs.

The experimental [scalar memo patch](experiments/exact_scalar_memo.patch) is not applied to the production source. It passed the exact query digest on 419,552 vertices, edge midpoints, and centroids, including intermediate-position query sequences. Both digests are `5c750beea1f4faa8`. Candidate CPU profiles and wall comparisons are retained for review, but paired release timings show no useful gain at either rank count.

Each release run advances the same rank-specific baseline restart from step 20,000 through 25,000, with three alternating-order pairs on the same exclusive node:

| Ranks | Baseline median ms/step | Candidate median ms/step | Baseline range | Candidate range | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 24.0092 | 24.0412 | 23.9258–24.1672 | 23.9962–24.2662 | 0.9987x |
| 48 | 4.3461 | 4.3495 | 4.3229–4.4153 | 4.3344–4.3619 | 0.9992x |

After normalizing aggregate time by particle moves, the candidate is 0.36% slower at 8 ranks and 0.47% faster at 48 ranks. These small mixed results do not support retaining the extra branch. [Full measurements](results/measurements.json).

The candidate's 20,000-step wall-ledger changes are small relative to the observed unchanged-baseline variation, but this is not an ensemble equivalence proof:

| Comparison | Retained W change | Deposition normalized L1 | Erosion normalized L1 | Net normalized L1 |
| --- | ---: | ---: | ---: | ---: |
| Candidate vs baseline, 8 ranks | 0.667% | 0.391% | 0.137% | 1.500% |
| Candidate vs baseline, 48 ranks | 2.897% | 2.476% | 0.492% | 5.883% |
| Unchanged baseline repeat, 48 ranks | 11.983% | 2.254% | 0.451% | 5.203% |

These are area-integrated quantities for all 35 monoblocks. The JSON reports include every block's net/deposition/erosion atoms and the wall identity `net = dep - ero`. Global escaped weighted W is inferred from the other inventories, not independently measured; complete global W-budget closure is still outstanding. No approximate physics change has been accepted on that basis.

Baseline and candidate CPU suites each passed 20 tests; the final retained builds also passed [20 CPU tests](results/regression_final_cpu.txt) and [20 Kokkos-host tests](results/regression_final_kkhost.txt). ST40 is absent from this checkout. The default GPU-only parity runner is skipped on the host backend; a private copy with OpenMP flags passes its thermal, weighted grid/surface, population, restart, and E+B comparisons. The optional B-on-W smoke input is absent. The working-tree package checker reports the same two pre-existing unlisted diagnostic files; excluding only those untracked files in a scratch snapshot, [package parity and make yes-openedge pass](results/parity_final_tracked_install.txt). See the reports in [results](results/).

## Reproduction and remaining work

The Slurm scripts assume isolated `source` and `source_patched` snapshots under `OE_REVIEW_ROOT`, with representative and regression data already present. Do not point them at a live production checkout. Output directories refuse overwrites. The installed EOS VTK prefix and module names are site-specific.

```bash
sbatch tools/performance_review/eos_baseline.slurm
# Once the CPU binary and generated setup exist:
sbatch tools/performance_review/eos_audit.slurm
# Optional experimental worktree, paired timings, and repeat control:
sbatch tools/performance_review/eos_patch.slurm
sbatch tools/performance_review/eos_timing.slurm
sbatch tools/performance_review/eos_repeat.slurm
python3 tools/performance_review/summarize.py /path/to/review_root
```

The remaining measurements are a CUDA/Nsight trace, cache/bandwidth counters and layout/sorting A/B tests, explicit MPI wait attribution, isolated final-restart timing, and an independent weighted escape tally. The current evidence establishes where to investigate without establishing the cause of a particular historical Perlmutter result.

The transferable lessons are configurable sorting and measured cost balance from [WarpX](https://warpx.readthedocs.io/en/latest/usage/parameters.html), SoA/tiled particle access from [AMReX](https://amrex-codes.github.io/amrex/docs_html/Particle.html), occupancy-dependent vectorization from [Smilei](https://smileipic.github.io/Smilei/Understand/vectorization.html), and modular particle kernels from [PICSAR](https://github.com/ECP-WarpX/picsar). SPARTA's own [acceleration guidance](https://sparta.github.io/doc/Section_accelerate.html) explains the penalty of host-only styles, but the current WEST path must be assessed from its trace rather than file-name absence.
