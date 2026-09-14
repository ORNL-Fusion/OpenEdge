#!/bin/bash
# -----------------------------------------------------------------------
#  OpenEdge Regression Test Runner
#
#  Runs all registered regression tests and reports pass/fail.
#  A test passes if the run exits 0, its log contains no ERROR lines, and
#  (when <case>/regression_reference.json exists) its end-of-run metrics
#  are within the stored tolerances (regression/metrics.py).
#  Record: name|dir|deck|required-data-file|flags|args
#  (flags: nokk = skip under --kk; args: extra command-line args, e.g. -var gridcut 0.02)
#
#  Usage:
#    ./regression/run_regression.sh [--np N] [--exe PATH] [--filter PATTERN]
#
#  Options:
#    --np N          Number of MPI ranks (default: 4)
#    --exe PATH      Path to the sparta binary (default: auto-detect)
#    --filter PAT    Only run tests matching glob pattern
#    --nsteps N      Requested smoke-test steps (default: 1000). Workflow
#                    cases are always capped at 1000 steps.
#    --verbose       Show full output on failure
#    --update-ref    Write <case>/regression_reference.json from this run
#                    (intended for the CPU path; keeps existing tolerances)
#    --parity-cpu-exe PATH  With --kk: also run examples/verification/gpu_parity
#                    (CPU binary PATH vs the --kk binary, deterministic compare)
# -----------------------------------------------------------------------

set +u  # Intel setvars uses unset variables

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/examples"

# ADAS / database lookups inside the code resolve through OPENEDGE_ROOT
export OPENEDGE_ROOT="$ROOT_DIR"

# Defaults
NP=4
EXE=""
FILTER="*"
NSTEPS=1000
WORKFLOW_MAX_STEPS=1000
VERBOSE=0
KKMODE=0
LAUNCHER=""    # override 'mpirun -np N' (e.g. --launcher "srun -n 4")
UPDATE_REF=0
PARITY_CPU_EXE=""
METRICS="$SCRIPT_DIR/metrics.py"
PY="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np)      NP="$2"; shift 2 ;;
    --exe)     EXE="$2"; shift 2 ;;
    --filter)  FILTER="$2"; shift 2 ;;
    --nsteps)  NSTEPS="$2"; shift 2 ;;
    --kk)      KKMODE=1; shift ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --update-ref) UPDATE_REF=1; shift ;;
    --parity-cpu-exe) PARITY_CPU_EXE="$2"; shift 2 ;;
    *)         echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Find executable
if [[ -z "$EXE" ]]; then
  for cand in "$HOME/build_oe/src/spa_mac_mpi" \
              "$HOME/buildOpenEdge/src/spa_mpi" \
              "$ROOT_DIR/build/src/spa_mac_mpi"; do
    if [[ -x "$cand" ]]; then EXE="$cand"; break; fi
  done
  if [[ -z "$EXE" ]]; then
    echo "ERROR: sparta binary not found. Use --exe PATH" >&2
    exit 1
  fi
fi

# Kokkos mode: run every case through the -sf kk path. GPU backend when
# the binary is a CUDA build, host OpenMP backend otherwise.
KKARGS=()
MODE=cpu
if [[ $KKMODE -eq 1 ]]; then
  # detect the backend from the binary itself (linked CUDA runtime), not its name
  # OE_KK_NGPU=4 with an srun line that leaves all GPUs visible (--gpu-bind=none): Kokkos picks by rank
  # gpu/aware no: the MPI on this machine is not GPU-aware unless
  # MPICH_GPU_SUPPORT_ENABLED=1 (multi-rank runs segfault otherwise)
  GPUAWARE=no; [[ "${MPICH_GPU_SUPPORT_ENABLED:-0}" == "1" ]] && GPUAWARE=yes
  if ldd "$EXE" 2>/dev/null | grep -qi 'libcudart'; then   # libcuda.so.1 alone is also linked by the OpenMP build on Perlmutter
    KKARGS=(-k on g ${OE_KK_NGPU:-1} -sf kk -pk kokkos react/retry yes gpu/aware $GPUAWARE comm threaded); MODE=gpu
  elif ldd "$EXE" >/dev/null 2>&1; then
    KKARGS=(-k on t 1 -sf kk -pk kokkos react/retry yes); MODE=kkhost
  else
    case "$EXE" in
      *cuda*) KKARGS=(-k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware $GPUAWARE comm threaded); MODE=gpu ;;
      *)      KKARGS=(-k on t 1 -sf kk -pk kokkos react/retry yes); MODE=kkhost ;;
    esac
  fi
fi

# Source Intel MPI if available
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------
#  Test registry: "name|directory|input_file|requires"
#  directory is relative to examples/. requires (optional) is a file,
#  relative to the test directory, that must exist or the test is
#  skipped (e.g. git-ignored plasma files that need regeneration).
# -----------------------------------------------------------------------
declare -a TESTS=(
  "ionization_recombination|verification/ionization_recombination|in.ionization_recombination"
  "efield_polarization|verification/efield_polarization|in.input|"
  "coulomb_background|verification/collisions/coulomb|in.background|"
  "coulomb_binary|verification/collisions/coulomb|in.binary|"
  "dustt2005_uniform_plasma|verification/particulates/dustt2005/uniform_plasma_benchmark|in.grain|input/grain.species"
  "dis2021_uniform_comparison|verification/particulates/dis2021/uniform_plasma_comparison|in.compare|"
  "pusher_gca|verification/pushers/orbit|in.gca|"
  "pusher_boris|verification/pushers/orbit|in.boris|"
  "pusher_hybrid_axiring|verification/pushers/hybrid|in.axiring|input/source.axi256"
  "st40_core_species_boundary|workflows/particulates/st40_lithium_powder_dropper/verification/core_species_boundary|in.openedge|../../input/plasma_st40_solps_corefill.h5"
  "constant_flux|verification/surface_emission/constant_flux|in.constant_flux|"
  "d2_chemistry|verification/d2_chemistry|in.d2_chem|"
  "dustt2005_cat_transport|verification/particulates/dustt2005/cat_solps_droplet_transport|in.openedge|input/plasma.h5"
  "west_boron_powder_dropper|workflows/particulates/west_boron_powder_dropper|in.openedge|../../impurity_transport/west_tungsten_transport/input/plasma.h5"
  "cat_liquid_metal_divertor|workflows/particulates/cat_liquid_metal_divertor|in.openedge|input/plasma_attached.h5"
  "west_tungsten_transport|workflows/impurity_transport/west_tungsten_transport|in.openedge|input/plasma.h5"
  "rfpie_tungsten_transport|workflows/impurity_transport/rfpie_tungsten_transport|in.openedge|input/plasma_he.h5"
  # coverage variants (audit 2026-09-12 item 16): cut-cell ghost decomposition and chunked balance/restart memory
  "west_tungsten_gridcut|workflows/impurity_transport/west_tungsten_transport|in.openedge|input/plasma.h5||-var gridcut 0.02"
  "west_tungsten_memlimit|workflows/impurity_transport/west_tungsten_transport|in.openedge|input/plasma.h5||-var memlimit 1"
)

# -----------------------------------------------------------------------
#  Temporary input with a bounded total step count. Initialization `run 0`
#  commands are preserved; multiple advancing runs share the requested budget.
# -----------------------------------------------------------------------
make_regression_input() {
  local infile="$1"
  local nsteps="$2"
  local tmpfile="${infile}.regression"
  local nruns
  nruns=$(awk '$1 == "run" && $2 != "0" { n++ } END { print n + 0 }' "$infile")
  awk -v total="$nsteps" -v nruns="$nruns" '
    BEGIN { remaining = total; seen = 0 }
    $1 == "run" && $2 != "0" {
      seen++
      if (seen < nruns) {
        later = nruns - seen
        steps = (remaining > later) ? 1 : 0
      } else {
        steps = remaining
      }
      remaining -= steps
      printf "run                 %d\n", steps
      next
    }
    { print }
  ' "$infile" > "$tmpfile"
  echo "$tmpfile"
}

# -----------------------------------------------------------------------
#  Run tests
# -----------------------------------------------------------------------
PASS=0
FAIL=0
SKIP=0
declare -a RESULTS=()

echo "========================================================================"
echo "  OpenEdge Regression Tests"
echo "  Executable: $EXE"
echo "  MPI ranks:  $NP"
if [[ $KKMODE -eq 1 ]]; then echo "  Kokkos:     ${KKARGS[*]}  (backend: $MODE)"; fi
if [[ $UPDATE_REF -eq 1 ]]; then echo "  Reference:  writing regression_reference.json from this ($MODE) run"; fi
echo "  Steps:      $NSTEPS"
echo "========================================================================"
echo ""

for entry in "${TESTS[@]}"; do
  IFS='|' read -r name testdir infile requires flags xargs <<< "$entry"
  # legacy 4-field form: "nokk" in the requires slot means no data file
  if [[ "$requires" == "nokk" ]]; then flags="nokk"; requires=""; fi

  if [[ "$name" != $FILTER ]]; then
    continue
  fi

  dir="$EXAMPLES_DIR/$testdir"
  printf "%-40s " "$name"

  if [[ ! -d "$dir" ]]; then
    RESULTS+=("SKIP  $name  (directory not found)")
    echo "SKIP (no dir)"
    ((SKIP++))
    continue
  fi
  if [[ ! -f "$dir/$infile" ]]; then
    RESULTS+=("SKIP  $name  (input file $infile not found)")
    echo "SKIP (no deck)"
    ((SKIP++))
    continue
  fi

  # Some lightweight verification fixtures are generated rather than
  # versioned.  Let a fresh checkout prepare them before applying the
  # required-data check below (large external data still remain skips).
  if [[ -n "$requires" && ! -e "$dir/$requires" &&
        -f "$dir/scripts/make_input.py" ]]; then
    generator_log="$dir/make_input.log"
    if ! (cd "$dir" && "${PYTHON:-python3}" scripts/make_input.py \
          > "$generator_log" 2>&1); then
      RESULTS+=("FAIL  $name  (fixture generator failed; see make_input.log)")
      echo "FAIL (fixture generation)"
      ((FAIL++))
      if [[ "$VERBOSE" -eq 1 ]]; then
        echo "--- Last 20 lines of $generator_log ---"
        tail -20 "$generator_log" 2>/dev/null || true
        echo "---"
      fi
      continue
    fi
  fi
  if [[ -n "$requires" && ! -e "$dir/$requires" ]]; then
    RESULTS+=("SKIP  $name  (missing $requires - regenerate it first)")
    echo "SKIP (missing data)"
    ((SKIP++))
    continue
  fi

  if [[ $KKMODE -eq 1 && "$flags" == *nokk* ]]; then
    RESULTS+=("SKIP  $name  (not supported under -sf kk by design: 2D/axi pusher, GCA or sheath kick)")
    printf "%-40s SKIP (nokk)\n" "$name"
    ((SKIP++))
    continue
  fi

  mkdir -p "$dir/output" 2>/dev/null || true
  case_nsteps=$NSTEPS
  if [[ "$testdir" == workflows/* && $case_nsteps -gt $WORKFLOW_MAX_STEPS ]]; then
    case_nsteps=$WORKFLOW_MAX_STEPS
  fi
  tmpinput=$(make_regression_input "$dir/$infile" "$case_nsteps")

  logfile="$dir/regression.log"
  ok=1
  if [[ -n "$LAUNCHER" ]]; then
    (cd "$dir" && $LAUNCHER "$EXE" "${KKARGS[@]}" -in "$(basename "$tmpinput")" \
        $xargs -log none > "$logfile" 2>&1) || ok=0
  else
    (cd "$dir" && mpirun -np "$NP" "$EXE" "${KKARGS[@]}" -in "$(basename "$tmpinput")" \
        $xargs -log none > "$logfile" 2>&1) || ok=0
  fi
  if [[ $ok -eq 1 ]] && grep -q "^ERROR" "$logfile"; then ok=0; fi
  # a rank dying under srun/mpirun can still return 0 through the launcher;
  # treat launcher-reported task failures as FAIL too
  if [[ $ok -eq 1 ]] && grep -qE "Segmentation fault|srun: error|DUE TO TASK FAILURE|Kokkos::abort|cudaError" "$logfile"; then ok=0; fi

  # end-of-run metrics: extract, then compare against the stored reference
  # (or write the reference with --update-ref)
  metrics_note=""
  if [[ $ok -eq 1 ]]; then
    # keyed by case name: several cases can share one directory
    mfile="$dir/regression_metrics_${name}_$MODE.json"
    reffile="$dir/regression_reference_${name}.json"
    if "$PY" "$METRICS" extract "$logfile" "$mfile" 2>/dev/null; then
      if [[ $UPDATE_REF -eq 1 ]]; then
        "$PY" "$METRICS" update "$mfile" "$reffile" > /dev/null && metrics_note="(reference updated)"
      elif [[ -f "$reffile" ]]; then
        cmp_out=$("$PY" "$METRICS" compare "$mfile" "$reffile")
        if [[ $? -ne 0 ]]; then
          ok=0; metrics_note="(metrics outside reference bands: $(echo "$cmp_out" | tail -1 | cut -d' ' -f3-))"
          echo "$cmp_out" >> "$logfile"
          if [[ "$VERBOSE" -eq 1 ]]; then echo "$cmp_out"; fi
        else
          metrics_note="(metrics within reference)"
        fi
      else
        metrics_note="(no reference)"
      fi
    fi
  fi

  if [[ $ok -eq 1 ]]; then
    RESULTS+=("PASS  $name  $metrics_note")
    echo "PASS $metrics_note"
    ((PASS++))
  else
    RESULTS+=("FAIL  $name  $metrics_note")
    echo "FAIL"
    ((FAIL++))
    if [[ "$VERBOSE" -eq 1 ]]; then
      echo "--- Last 20 lines of $logfile ---"
      tail -20 "$logfile" 2>/dev/null || true
      echo "---"
    fi
  fi

  rm -f "$tmpinput"
done

# -----------------------------------------------------------------------
#  gpu_parity: deterministic CPU-vs-GPU comparison (needs both binaries)
# -----------------------------------------------------------------------
if [[ "gpu_parity" == $FILTER || "*" == "$FILTER" ]]; then
  printf "%-40s " "gpu_parity"
  pdir="$EXAMPLES_DIR/verification/gpu_parity"
  if [[ $KKMODE -ne 1 || -z "$PARITY_CPU_EXE" ]]; then
    RESULTS+=("SKIP  gpu_parity  (needs --kk and --parity-cpu-exe <cpu binary>)")
    echo "SKIP (needs --kk and --parity-cpu-exe)"
    ((SKIP++))
  else
    if [[ -n "$LAUNCHER" ]]; then lc="$LAUNCHER"; lg="$LAUNCHER"; else lc="mpirun -np $NP"; lg="mpirun -np $NP"; fi
    if (cd "$pdir" && EXE_CPU="$PARITY_CPU_EXE" EXE_GPU="$EXE" LAUNCH_CPU="$lc" LAUNCH_GPU="$lg" \
        bash run.sh > "$pdir/regression.log" 2>&1); then
      RESULTS+=("PASS  gpu_parity  (grid/weighted + thermal kick CPU==GPU)"); echo "PASS"; ((PASS++))
    else
      RESULTS+=("FAIL  gpu_parity  (see examples/verification/gpu_parity/regression.log)"); echo "FAIL"; ((FAIL++))
      if [[ "$VERBOSE" -eq 1 ]]; then tail -20 "$pdir/regression.log"; fi
    fi
  fi
fi

# -----------------------------------------------------------------------
#  pwi_deposit_tagging: surf_react surface/pwi ledger invariants (deposit_as,
#  seeding, yscale) checked between runs of the same binary; any backend
# -----------------------------------------------------------------------
if [[ "pwi_deposit_tagging" == $FILTER || "*" == "$FILTER" ]]; then
  printf "%-40s " "pwi_deposit_tagging"
  ddir="$EXAMPLES_DIR/verification/surface_pwi/deposit_tagging"
  if [[ -n "$LAUNCHER" ]]; then ld="$LAUNCHER"; else ld="mpirun -np $NP"; fi
  bwinp=$(awk '$1=="variable" && $2=="inp" {print $4}' "$ddir/in.bw_smoke")
  skipbw=0; [[ -e "$ddir/$bwinp/plasma_58245.h5" ]] || skipbw=1   # the 70 MB B-on-W plasma smoke needs its file
  lrtol=1e-9; [[ "$MODE" == gpu ]] && lrtol=0.05   # GPU tallies are atomic: two runs are different realizations
  if (cd "$ddir" && SPA="$EXE" SPA_ARGS="${KKARGS[*]}" LAUNCH="$ld" SKIP_BW=$skipbw LEDGER_RTOL=$lrtol \
      bash run.sh > "$ddir/regression.log" 2>&1); then
    RESULTS+=("PASS  pwi_deposit_tagging  (ledger invariants$( [[ $skipbw -eq 1 ]] && echo ', bw smoke skipped'))"); echo "PASS"; ((PASS++))
  else
    RESULTS+=("FAIL  pwi_deposit_tagging  (see examples/verification/surface_pwi/deposit_tagging/regression.log)"); echo "FAIL"; ((FAIL++))
    if [[ "$VERBOSE" -eq 1 ]]; then tail -20 "$ddir/regression.log"; fi
  fi
fi

# -----------------------------------------------------------------------
#  Summary
# -----------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================================================"
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
echo ""

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
