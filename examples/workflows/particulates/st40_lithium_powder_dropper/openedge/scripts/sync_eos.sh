#!/usr/bin/env bash
# Mirror this case to EOS (push) or fetch its runs/ back (pull).
set -euo pipefail
mode="${1:-dry-run}"
case_dir="$(cd "$(dirname "$0")/.." && pwd)"
remote_case=/scratch/home_eos/42d/OpenEdge/examples/workflows/particulates/st40_lithium_powder_dropper/openedge
ex=(--exclude output/ --exclude runs/ --exclude notebooks/
    --exclude .ipynb_checkpoints/ --exclude __pycache__/ --exclude .DS_Store --exclude 'log.*' --exclude 'slurm-*.out')
case "$mode" in
  dry-run) rsync -avn "${ex[@]}" "$case_dir/" "eos:$remote_case/" ;;
  push)    ssh eos "mkdir -p '$remote_case/runs'"; rsync -av --delete "${ex[@]}" "$case_dir/" "eos:$remote_case/" ;;
  pull)    mkdir -p "$case_dir/runs"      # all runs, or one (dir + .log): scripts/sync_eos.sh pull s400000_n64_3894
           rsync -av "eos:$remote_case/runs/${2:-}*" "$case_dir/runs/" ;;
  *) echo "usage: scripts/sync_eos.sh dry-run|push|pull [run_name]" >&2; exit 2 ;;
esac
