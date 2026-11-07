#!/usr/bin/env bash
# Default is CPU data validation only. Training/extraction requires an explicit stage.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-joint}"
stage="${2:-prepare}"
case "$mode" in current_only|future_only|joint) ;; *) echo 'Invalid prediction mode' >&2; exit 2;; esac
case "$stage" in prepare|extract|train|run) ;; *) echo 'Invalid stage' >&2; exit 2;; esac
cd "$repo_dir"
exec "$repo_dir/.venv-tomato/bin/python" -u -m tools.train_tomato_vjepa_phase "$stage" \
  --config "$repo_dir/configs/tomato_four_stage_local94.yaml" \
  --prediction-mode "$mode" --output-dir "$repo_dir/runs/tomato/local94_20261107/$mode"
