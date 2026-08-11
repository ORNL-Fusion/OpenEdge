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
#    --nsteps N      Override run steps for all tests (default: 1000)
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
  "gravity|test_gravity|in.gravity3d|"
  "ionization_recombination|test_ionization_recombination|in.ionization_recombination|"
  "efield_polarization|test_efield_polarization|in.input|"
  "coulomb_background|test_coulomb|in.background|"
  "coulomb_binary|test_coulomb|in.binary|"
  "droplet_emission|test_droplet|in.droplet_emission|"
  "tsurf_emission|test_tsurf_emission|in.tsurf_emission|input/plasma.h5"
  "st40_li_dropper|test_st40_li_dropper|in.st40|input/plasma_st40_solps.h5"
  "grain_boron_west|test_grain_boron|in.b_west_drop|../test_west/input/plasma.h5"
  "west_axi_emission|test_west|in.axi_west_emission|input/plasma.h5"
  "constant_flux|test_constant_flux|in.constant_flux|"
  "pusher_gca|test_gca_vs_boris|in.gca|"
  "pusher_boris|test_gca_vs_boris|in.boris|"
)

# -----------------------------------------------------------------------
#  Temporary input with overridden step count (replaces every `run` line)
# -----------------------------------------------------------------------
make_regression_input() {
  local infile="$1"
  local nsteps="$2"
  local tmpfile="${infile}.regression"
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
  tmpinput=$(make_regression_input "$dir/$infile" "$NSTEPS")

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
