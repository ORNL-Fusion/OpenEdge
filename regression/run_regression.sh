#!/bin/bash
# -----------------------------------------------------------------------
#  OpenEdge Regression Test Runner
#
#  Runs all registered regression tests and reports pass/fail.
#  A test passes if the run exits 0 and its log contains no ERROR lines.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np)      NP="$2"; shift 2 ;;
    --exe)     EXE="$2"; shift 2 ;;
    --filter)  FILTER="$2"; shift 2 ;;
    --nsteps)  NSTEPS="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
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
  "ionization_recombination|verification/ionization_recombination|in.ionization_recombination|"
  "efield_polarization|verification/efield_polarization|in.input|"
  "coulomb_background|verification/collisions/coulomb|in.background|"
  "coulomb_binary|verification/collisions/coulomb|in.binary|"
  "particulate_dustt|verification/particulates/dustt|in.grain|"
  "pusher_gca|verification/pushers/orbit|in.gca|"
  "pusher_boris|verification/pushers/orbit|in.boris|"
  "constant_flux|verification/surface_emission/constant_flux|in.constant_flux|"
  "lithium_droplet_transport|workflows/particulates/lithium_droplet_transport|in.openedge|"
  "west_boron_powder_dropper|workflows/particulates/west_boron_powder_dropper|in.openedge|../../impurity_transport/west_tungsten_transport/input/plasma.h5"
  "cat_liquid_metal_divertor|workflows/particulates/cat_liquid_metal_divertor|in.openedge|input/plasma_attached.h5"
  "west_tungsten_transport|workflows/impurity_transport/west_tungsten_transport|in.openedge|input/plasma.h5"
  "rfpie_tungsten_transport|workflows/impurity_transport/rfpie_tungsten_transport|in.openedge|input/plasma_he.h5"
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
echo "  Steps:      $NSTEPS"
echo "========================================================================"
echo ""

for entry in "${TESTS[@]}"; do
  IFS='|' read -r name testdir infile requires <<< "$entry"

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
  if [[ -n "$requires" && ! -e "$dir/$requires" ]]; then
    RESULTS+=("SKIP  $name  (missing $requires - regenerate it first)")
    echo "SKIP (missing data)"
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
  (cd "$dir" && mpirun -np "$NP" "$EXE" -in "$(basename "$tmpinput")" \
      -log none > "$logfile" 2>&1) || ok=0
  if [[ $ok -eq 1 ]] && grep -q "^ERROR" "$logfile"; then ok=0; fi

  if [[ $ok -eq 1 ]]; then
    RESULTS+=("PASS  $name")
    echo "PASS"
    ((PASS++))
  else
    RESULTS+=("FAIL  $name")
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
