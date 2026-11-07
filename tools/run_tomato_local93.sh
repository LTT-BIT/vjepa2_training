#!/usr/bin/env bash
# Local raw inputs and derived outputs; no SSHFS or remote reads.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mode="${1:-all}"
stage="${2:-run}"
case "$mode" in all) modes=(current_only future_only joint);; current_only|future_only|joint) modes=("$mode");; *) echo 'Usage: bash tools/run_tomato_local93.sh [all|current_only|future_only|joint] [run|prepare|extract|train]' >&2; exit 2;; esac
case "$stage" in run|prepare|extract|train) ;; *) echo 'Invalid stage; choose run, prepare, extract, or train.' >&2; exit 2;; esac
for selected_mode in "${modes[@]}"; do
    output_dir="$repo_dir/runs/tomato/local93_20261107/$selected_mode"
    mkdir -p "$output_dir"
    log_file="$output_dir/launcher_${stage}_$(date +%Y%m%d_%H%M%S).log"
    printf 'Mode=%s Stage=%s Output=%s\n' "$selected_mode" "$stage" "$output_dir"
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "$repo_dir/.venv-tomato/bin/python" -u -m tools.train_tomato_vjepa_phase "$stage" \
        --config "$repo_dir/configs/tomato_four_stage_local93.yaml" \
        --prediction-mode "$selected_mode" --output-dir "$output_dir" \
        2>&1 | tee "$log_file"
done
