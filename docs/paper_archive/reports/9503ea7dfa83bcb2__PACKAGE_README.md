# LOCAL93 Joint Memory research

Independent research package. The production trainer, Current-only/Future-only loaders,
GUI and handshake implementation are imported read-only. No Agent or robot interface.

Use the repository `.venv-tomato/bin/python` and choose a **new** experiment directory:

```bash
EXPERIMENT_DIR=runs/tomato/joint_memory_ablation_MY_UNIQUE_RUN
.venv-tomato/bin/python -m tools.tomato_joint_memory.run --output-dir "$EXPERIMENT_DIR" --check-only
.venv-tomato/bin/python -m tools.tomato_joint_memory.run --output-dir "$EXPERIMENT_DIR" --smoke
.venv-tomato/bin/python -m tools.tomato_joint_memory.run --output-dir "$EXPERIMENT_DIR" --run
.venv-tomato/bin/python -m tools.tomato_joint_memory.run --output-dir "$EXPERIMENT_DIR" --verify
.venv-tomato/bin/python -m tools.tomato_joint_memory.run --output-dir "$EXPERIMENT_DIR" --report-only
```

`--check-only` rehashes source assets and caches, validates the existing locked manifest,
58 measured positions and the original 75/9/9 source split. Missing/changed artifacts
fail; no extraction, missing-value filling or new split fallback. All modes use the
same 4000D frozen features. The baseline configuration's existing metadata mismatch
policy is preserved. `--verify` rehashes protected baseline inputs and outputs.

`--smoke` runs two epochs at seed42 for execution correctness, then `--run` trains
B0R/T0/T1/T2/K1/P1 for seeds42/43/44. P1 uses five source-isolated OOF teachers;
val/test beliefs use the fixed full-train baseline. A single common receipt locks
all checkpoints and validation results before historical test evaluation. Existing
training/evaluation folders are never overwritten. Interrupted runs retain failure
logs; partial test recovery requires auditing the locked receipt and completed
artifacts (the actual 2026-11-12 recovery script is in that run's supplemental/).

Prediction rows save paired probabilities in the fixed class order declared by the
protocol. `future_valid` only controls supervision/scoring. The actual production
handshake receives a strict whitelist of timestamps and probabilities, never labels.
A separate C1 replay disables ACK for diagnosis only.

`protocol.lock.json` records code, architecture, hyperparameters, selection,
OOF membership, fixed C0/C1 clocks and promotion thresholds. New checkpoint format
`tomato-joint-memory-v1` is loaded explicitly, with feature/protocol/statistics and
checkpoint SHA checks. It is not presented as an original production checkpoint.

Read the dated report in docs/ for real results, failed attempts and amendments.
Synthetic tests establish correctness only. All classification results use the
actual LOCAL93 frozen features; actual Encoder timing is a separate supplement
and must not be confused with the cached training/evaluation path.
