"""Shared annotation, training-text and deployment phase contract."""

import math

from openpi.policies.fr3_sharpa_protocol import ORDER

# Converter blocks are left arm, right arm, left hand, right hand. The policy
# order is left arm/hand then right arm/hand; do not infer order from dimensions.
_SOURCE_BLOCKS = (ORDER[:7], ORDER[29:36], ORDER[7:29], ORDER[36:])
SOURCE_STATE_NAMES = tuple(
    f"{joint}.{kind}" for block in _SOURCE_BLOCKS for kind in ("position", "velocity") for joint in block
)
SOURCE_ACTION_NAMES = tuple(
    f"{joint}.{kind}"
    for block, kind in zip(_SOURCE_BLOCKS, ("position", "position", "command", "command"), strict=True)
    for joint in block
)

PHASES = ("pickup_1", "harvest_1", "return_truss", "adjust", "pickup_2", "harvest_2", "return_stem")
COARSE_SEQUENCE = ("pickup", "harvest", "return", "adjust", "pickup", "harvest", "return")
ACTION_PROMPTS = {
    "pickup": "Right hand grasps and lifts the truss to the center",
    "harvest": "Right hand holds truss while left hand picks tomatoes into left basket",
    "return": "Right hand returns the held object to the right plate",
    "adjust": "Right fingers reposition the truss on the right plate",
}
# Seven progress states select four action instructions. Both returns share one
# instruction; their different successors are determined by the progress state.
PROMPTS = tuple(ACTION_PROMPTS[action] for action in COARSE_SEQUENCE)
# A is a concise overall goal, not an enumeration of every motion. The complete
# procedure remains in the annotation and this metadata; B uses the phase texts.
GOAL = "Harvest all tomatoes into the left basket and return stem to right plate."
PROCEDURE = (
    "Pick up the truss with the right hand. Hold it steady while the left hand picks tomatoes into the left basket. "
    "Put the truss on the right plate, adjust it with the right fingers, and pick it up again. "
    "Pick all remaining tomatoes and return the bare stem to the right plate."
)
MAX_TOKENS = 256
STATE_DIM = 58


def reviewed_segments(document, length):
    """Require the reviewed seven-occurrence coarse template; never invent gaps."""
    rows = document.get("segments", [])
    if document.get("stage") != "coarse" or tuple(r.get("label") for r in rows) != COARSE_SEQUENCE:
        raise ValueError("Expected reviewed seven-segment coarse tomato template; inspect exceptions manually")
    end = 0
    for row in rows:
        a, b = row.get("start_frame"), row.get("end_frame")
        if type(a) is not int or type(b) is not int or a != end or b <= a or row.get("reviewed") is not True:
            raise ValueError("All coarse segments must be reviewed, contiguous, nonempty frame intervals")
        end = b
    if end != length:
        raise ValueError("Coarse segments must cover the complete episode")
    return rows


def phase_at(rows, frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("Invalid frame")
    for i, row in enumerate(rows):
        if row["start_frame"] <= frame < row["end_frame"]:
            return i
    raise ValueError("Frame outside annotated episode")


def input_text(prompt, bins):
    if len(bins) != STATE_DIM or any(type(v) is not int or not -1 <= v <= 255 for v in bins):
        raise ValueError("Expected 58 discretized state values in [-1, 255]")
    return f"Task: {prompt.strip()}, State: {' '.join(map(str, bins))};\nAction: "


def audit_token_budget(tokenizer):
    """Bound all possible state bins, not just the English text or a zero state.

    SentencePiece's metaspace separates numeric state values. Verify each bin
    tokenizes independently in this template before using the additive bound.
    Refuse an unfamiliar tokenizer instead of silently relying on this property.
    """
    reports = []
    for prompt in dict.fromkeys((GOAL, *PROMPTS)):
        base = tokenizer.encode(input_text(prompt, [0] * STATE_DIM), add_bos=True)
        increases = []
        for value in range(-1, 256):
            n = len(tokenizer.encode(input_text(prompt, [value] + [0] * (STATE_DIM - 1)), add_bos=True))
            increase = n - len(base)
            uniform = len(tokenizer.encode(input_text(prompt, [value] * STATE_DIM), add_bos=True))
            if uniform != len(base) + STATE_DIM * increase:
                raise ValueError("Tokenizer state-boundary behavior changed; perform a new exhaustive budget audit")
            increases.append(increase)
        maximum = len(base) + STATE_DIM * max(increases)
        if maximum > MAX_TOKENS:
            raise ValueError(f"Prompt can exceed {MAX_TOKENS} tokens: {maximum}: {prompt}")
        reports.append({"prompt": prompt, "max_input_tokens": maximum})
    return reports


class PhaseDecision:
    """Debounced phase proposals; consumes model predictions, never robot commands.

    Seven occurrences distinguish the repeated coarse classes. No automatic
    completion or elapsed-time transitions; the final phase means return-stem,
    not proof that the task has completed. Reset between recordings/robot runs.
    """

    def __init__(self, threshold=0.8, confirmations=3):
        if not 0 < threshold <= 1 or type(confirmations) is not int or confirmations < 1:
            raise ValueError("Invalid decision thresholds")
        self.threshold, self.confirmations = threshold, confirmations
        self.reset()

    def reset(self):
        self.phase, self.count, self.last_time = 0, 0, None

    def update(self, probabilities, timestamp):
        if (
            len(probabilities) != len(PHASES)
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)
            or not math.isfinite(timestamp)
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("Expected seven probabilities and a strictly increasing observation timestamp")
        self.last_time = timestamp
        proposed = max(range(len(PHASES)), key=lambda i: probabilities[i])
        accepted = proposed == self.phase + 1 and probabilities[proposed] >= self.threshold
        self.count = self.count + 1 if accepted else 0
        changed = self.count >= self.confirmations
        if changed:
            self.phase, self.count = proposed, 0
        uncertain = probabilities[self.phase] < self.threshold
        return {
            "phase": PHASES[self.phase],
            "phase_index": self.phase,
            "action_class": COARSE_SEQUENCE[self.phase],
            "prompt": PROMPTS[self.phase],
            "changed": changed,
            "needs_review": uncertain or proposed not in (self.phase, self.phase + 1),
            "task_complete": False,
        }
