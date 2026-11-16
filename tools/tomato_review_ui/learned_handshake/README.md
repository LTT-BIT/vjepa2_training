# Learned handshake V2 (offline research)

Frozen B0 Joint probabilities, unchanged original ACK/timer/planner. No V-JEPA,
Memory, physical outcome model, GUI default or robot integration is trained here.

Use `.venv-tomato/bin/python`, and a **new unique** output directory:

```bash
PY=.venv-tomato/bin/python
EXP=runs/tomato/learned_handshake_v2_MY_NEW_RUN
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --check-only
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --phase-a
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --smoke
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --run
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --verify
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --report-only
```

The config explicitly names the original LOCAL93 and previous compatible OOF
experiment. Identity mismatch blocks execution; no alternate dataset or split is
selected. This campaign reused all five teachers after parameter/normalization/
source/cache SHA checks; no new teacher or Encoder calls were needed. It does not
implement an automatic teacher-rebuild fallback for future incompatible datasets.

`phase-a` verifies full historical H0 parity for 84 train/val sources and exactly
23/54 narrow hits with 98.233s Wrong Active on val. R1/R2 are fixed rules. R3 is
not run because the referenced V1 accumulator threshold was unavailable.

Train policies L0/L1a/L1b/L2/L3a at seeds42/43/44. L0 uses explicit
0.5*0.01*sum(nonbias W squared), optimizer decay0; others use AdamW decay.01. The complete current/past input
schema is 4x11 probability/time/mask context plus48 own-state/candidate features,
92 flattened values. L0 uses current probabilities+48 state/difference fields.
No truth, annotation, source ID, split, task-stage index or recording length enters
the network. Only the replay driver/scorers/target generator know recording bounds.

Labels use the actual next planner tick. Right-censored next boundaries have
masked next/time labels, not artificial hazard negatives; no executable tick means
no supervision. Late catchup retains positive action supervision. Event/source/
time-stratum weights and train-only statistics avoid stable-negative domination.
An optional single train-only aggregation is triggered by >10% Active time shift;
v1 is then used without retrospectively picking the better val-Active version.

The learner replaces candidate/REQUEST logic, not just an H0 veto. All requests
still traverse source/session/policy identity, freshness, external task mask,
original planning commits and original Current ACK/late ACK/hard recovery.
Future is neutralized only during the core ACK call to suppress its old candidate
branch, then real evidence is restored immediately. H0 itself calls the original
replay untouched. Malformed policy scores/times cause serialized HOLD events.

Policy checkpoints use `tomato-learned-handshake-v2`, SHA sidecars, feature and
normalization bindings. The GUI does not load them automatically. Completed output
folders are never overwritten. Historical test is evaluated only for H0 and the
unique validation-selected policy, once; all other main comparisons are val-only.

The actual 2026-11-12 run also contains supplemental scripts for K1 Future swap,
complete CPU/GPU latency, independent saved-event recomputation and full checkpoint
reload parity. These are post-selection diagnostics, never threshold searches.
The detailed report and lightweight handoff retain their commands and evidence;
full train rollout JSONL stays local. No claim of closed-loop physical safety,
completion detection, live watchdog behavior or Encoder+Pi05 concurrency is made.
