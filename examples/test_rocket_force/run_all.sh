#!/bin/bash
# Run all rocket force test cases
# Usage: ./run_all.sh [path_to_spa_mpi]

set -u

resolve_spa() {
    if [[ $# -ge 1 && -n "$1" ]]; then
        printf '%s\n' "$1"
        return 0
    fi

    if [[ -n "${SPA_MPI:-}" ]]; then
        printf '%s\n' "$SPA_MPI"
        return 0
    fi

    local candidate
    for candidate in \
        "$HOME/buildOpenEdge/src/spa_mpi" \
        "$HOME/build_oe/src/spa_mac_mpi" \
        "$HOME/build_oe/src/spa_mpi" \
        "$HOME/buildOpenEdge/src/spa_mac_mpi" \
        "$HOME/OpenEdge/src/spa_mpi" \
        "$HOME/OpenEdge/src/spa_mac_mpi" \
        "$HOME/OpenedgeGPU/src/spa_mpi"
    do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v spa_mpi >/dev/null 2>&1; then
        command -v spa_mpi
        return 0
    fi

    return 1
}

SPA=$(resolve_spa "${1:-}") || {
    echo "Could not find an executable spa_mpi."
    echo "Pass it explicitly: ./run_all.sh /path/to/spa_mpi"
    echo "Or set SPA_MPI=/path/to/spa_mpi"
    exit 1
}

echo "=== Rocket Force Test ==="
echo "Using: $SPA"

# Generate input files and HDF5 data
python3 create_case.py || exit 1

# Run three eta cases
failures=0
for eta in 0p0 0p5 1p0; do
    echo ""
    echo "--- Running eta=$eta ---"
    logfile="run_eta_${eta}.log"
    if mpirun -np 1 "$SPA" -in "in.eta_${eta}" >"$logfile" 2>&1; then
        tail -n 12 "$logfile"
    else
        echo "Run failed for eta=$eta. Full log: $logfile"
        tail -n 40 "$logfile"
        failures=1
    fi
done

if [[ $failures -ne 0 ]]; then
    echo ""
    echo "One or more runs failed. Comparison skipped."
    exit 1
fi

echo ""
echo "--- Comparing results ---"
python3 compare_rocket.py
