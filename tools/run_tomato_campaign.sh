#!/usr/bin/env bash
# Explicit full head-training campaign. Frozen feature extraction must finish first.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -f runs/tomato/local94_20261107/joint/feature_index.json ]]; then
  echo 'Run full94 frozen feature extraction first; no concurrent cache writers.' >&2
  exit 2
fi
exec .venv-tomato/bin/python -u -m tools.train_tomato_campaign \
  --config configs/tomato_campaign94.yaml \
  --campaign-dir runs/tomato/campaign94_20261107/experiments
