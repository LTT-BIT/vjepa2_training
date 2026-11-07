#!/usr/bin/env bash
# Remote raw inputs, local derived outputs. No source writes, no training changes.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mode="${1:-all}"
stage="${2:-run}"
case "$mode" in all) modes=(current_only future_only joint);; current_only|future_only|joint) modes=("$mode");; *) echo 'Usage: bash tools/run_tomato_remote93.sh [all|current_only|future_only|joint] [run|prepare|extract|train]' >&2; exit 2;; esac
case "$stage" in run|prepare|extract|train) ;; *) echo 'Invalid stage; choose run, prepare, extract, or train.' >&2; exit 2;; esac
python_bin="$repo_dir/.venv-tomato/bin/python"
config_file="$repo_dir/configs/tomato_four_stage_remote93.yaml"
mount_dir="$repo_dir/runs/tomato/remote93_mount"
remote_source='tairan28:/mnt/workspace/train_platform/datasets/tomato_sharpa/normal/no-block-more'
result_root="$repo_dir/runs/tomato/remote93_20261107"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
created_mount=0
cleanup() {
    result_code=$?
    trap - EXIT
    if ((created_mount)); then
        fusermount3 -u "$mount_dir" || echo "Could not unmount $mount_dir; inspect open readers." >&2
    fi
    exit "$result_code"
}
trap cleanup EXIT
mkdir -p "$mount_dir"
if mountpoint -q "$mount_dir"; then
    [[ "$(findmnt -rn -M "$mount_dir" -o SOURCE)" == "$remote_source" ]] || { echo 'Existing mount has unexpected remote source.' >&2; exit 1; }
    findmnt -rn -M "$mount_dir" -o OPTIONS | tr ',' '\n' | rg -qx ro || { echo 'Existing mount must be read-only.' >&2; exit 1; }
else
    sshfs_bin="$(command -v sshfs || true)"
    if [[ -z "$sshfs_bin" ]]; then
        sshfs_bin="$repo_dir/runs/tomato/remote_io_probe_20261107/client/unpacked/usr/bin/sshfs"
    fi
    [[ -x "$sshfs_bin" ]] || { echo 'SSHFS missing: install sshfs, or restore the locally unpacked probe client.' >&2; exit 1; }
    "$sshfs_bin" "$remote_source" "$mount_dir" -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,BatchMode=yes
    created_mount=1
fi
for selected_mode in "${modes[@]}"; do
    case "$selected_mode" in
        current_only) run_name=current_only_seed42;;
        future_only) run_name=future_only_h1s_seed42;;
        joint) run_name=joint_h1s_lc1_lf0p5_seed42;;
    esac
    output_dir="$result_root/$run_name"
    mkdir -p "$output_dir"
    log_file="$output_dir/launcher_${stage}_$(date +%Y%m%d_%H%M%S).log"
    printf 'Mode=%s Stage=%s Output=%s\n' "$selected_mode" "$stage" "$output_dir"
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "$python_bin" -u -m tools.train_tomato_vjepa_phase "$stage" \
        --config "$config_file" --prediction-mode "$selected_mode" --output-dir "$output_dir" \
        2>&1 | tee "$log_file"
done
