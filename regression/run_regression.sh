#!/bin/bash
# -----------------------------------------------------------------------
#  OpenEdge Regression Test Runner
#
#  Runs all registered regression tests and reports pass/fail.
#  A test passes if spa_mpi exits with code 0 (ran to completion).
#
#  Usage:
#    ./Regression/run_regression.sh [--np N] [--exe PATH] [--filter PATTERN]
#
#  Options:
#    --np N          Number of MPI ranks (default: 4)
#    --exe PATH      Path to spa_mpi binary (default: auto-detect)
#    --filter PAT    Only run tests matching glob pattern
#    --nsteps N      Override run steps for all tests (default: 1000)
#    --verbose       Show full output on failure
# -----------------------------------------------------------------------

set +u  # Intel setvars uses unset variables

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/examples"

# Defaults
NP=4
EXE=""
FILTER="*"
NSTEPS=1000
VERBOSE=0

# Parse arguments
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
  if [[ -x "$HOME/buildOpenEdge/src/spa_mpi" ]]; then
    EXE="$HOME/buildOpenEdge/src/spa_mpi"
  else
    echo "ERROR: spa_mpi not found. Use --exe PATH" >&2
    exit 1
  fi
fi

# Source Intel MPI if available
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------
#  Test registry: each entry is "name|directory|input_file|setup_cmd"
#  setup_cmd is optional (use "" for none)
# -----------------------------------------------------------------------
declare -a TESTS=(
  "collide_slowdown|test_collide|in.nanbu_slowdown|"
  "collide_thermalize|test_collide|in.nanbu_thermalize|"
  "gravity_3d|test_gravity_3d|in.gravity3d|"
  "gca_boris|test_gca|in.boris|"
  "gca_gca|test_gca|in.gca|"
  "ionization_recombination|test_ionization_recombination|in.ionizationRecombination_oxygen|"
  "particle_weight_standard|test_particle_weight|in.standard|"
  "particle_weight_weighted|test_particle_weight|in.weighted|"
  "west_axi|test_west_axi|in.west_regression|"
  "iead|test_iead|in.iead_alpha0|python3 create_case.py"
  "polarization|test_polarization|in.input|"
)

# -----------------------------------------------------------------------
#  Temporary input file with overridden step count
# -----------------------------------------------------------------------
make_regression_input() {
  local infile="$1"
  local nsteps="$2"
  local tmpfile="${infile}.regression"

  # Copy original and override the run command (handles both numeric and variable forms)
  sed -E "s/^run[[:space:]]+.+/run                 ${nsteps}/" "$infile" > "$tmpfile"
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
  IFS='|' read -r name testdir infile setup <<< "$entry"

  # Apply filter
  if [[ "$name" != $FILTER ]]; then
    continue
  fi

  dir="$EXAMPLES_DIR/$testdir"
  if [[ ! -d "$dir" ]]; then
    RESULTS+=("SKIP  $name  (directory not found)")
    ((SKIP++))
    continue
  fi

  printf "%-40s " "$name"

  # Run setup if needed
  if [[ -n "$setup" ]]; then
    if ! (cd "$dir" && eval "$setup" > /dev/null 2>&1); then
      RESULTS+=("FAIL  $name  (setup failed: $setup)")
      echo "FAIL (setup)"
      ((FAIL++))
      continue
    fi
  fi

  # Check input file exists
  if [[ ! -f "$dir/$infile" ]]; then
    RESULTS+=("SKIP  $name  (input file $infile not found)")
    echo "SKIP"
    ((SKIP++))
    continue
  fi

  # Create regression input with overridden step count
  mkdir -p "$dir/output" 2>/dev/null || true
  tmpinput=$(make_regression_input "$dir/$infile" "$NSTEPS")

  # Run test
  logfile="$dir/regression.log"
  if (cd "$dir" && mpirun -np "$NP" "$EXE" < "$(basename "$tmpinput")" > "$logfile" 2>&1); then
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

  # Clean up temporary input
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
