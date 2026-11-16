# Candidate-only handshake ablation (2026-11-13)

This offline LOCAL93 experiment compares G0 logistic, G1 MLP, G2 causal GRU and G3 MLP with censored time-bin auxiliary supervision. Each receives the identical four-anchor probability/own-state context and may only HOLD/REQUEST the single causal Future candidate. Original H0 and frozen R1 are controls. No physical robot control or Encoder execution.

The only production refactor extracts ACK-only `Handshake.receive_evidence` and the original candidate path. The default `prediction` calls both; full event-level parity is required. `CandidateRequestPort` overrides the old adapter's ingest and **never** neutralizes Future probabilities.

```bash
PY=.venv-tomato/bin/python
EXP=runs/tomato/handshake_candidate_gate_v3_NEW_UNIQUE
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --check-only
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --smoke
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --run
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --verify
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --report-only
$PY -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
```

P0 checks immutable raw inputs and both prior experiments, OOF membership/normalization, original core identity and H0/R1 event parity. `protocol.lock.json` seals executable experiment sources before smoke/main fitting. All train outputs refuse overwrite. `--run` first trains all12 v0 policies, performs train-only own-state diagnostics, then globally applies at most one aggregation/refit to all12. v1 is mandatory if triggered. Early stopping uses fixed H0-state validation gate BCE, not Active performance. Three-seed risk and mean-benefit rules are fixed before evaluation; seed42 represents a qualifying group. Only H0 and the one locked winner enter historically exposed test.

`--verify` checks protected input hashes, protocol source hashes, reloads every policy, reproduces all validation events/metrics, checks GT-stripped inference and batch/stream numerical agreement, and audits every saved train/val/test/stress actual commit for clock/TTL/session/legal-edge/version invariants. `logs/tests.json` is the captured pytest result recorded by the operator before report generation. Decisions are gzip JSONL to retain exact causal features without large text duplication. Handoff excludes videos, features, Encoder and full model weights.

The score is weighted supervised research confidence, not a calibrated physical completion probability. Candidate coverage uses GT only in the scorer. Auxiliary time bins never choose actions. Bin boundaries and right-censoring are tested, including last recording tails; no fabricated FINISH. Historical val/test have been observed repeatedly; bootstrap is exploratory and source-paired, not independent-seed significance.
