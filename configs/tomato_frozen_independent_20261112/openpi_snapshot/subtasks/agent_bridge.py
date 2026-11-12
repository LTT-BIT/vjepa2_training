"""Four-class prediction -> ordered prompt interface, without robot/RTC side effects.

The caller supplies transport, explicit session initialization and a safe replan
boundary. This module never flushes queues or interprets final return as success.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.subtasks.tomato import ACTION_PROMPTS, COARSE_SEQUENCE, PHASES

PROTOCOL = 'tomato-stage-prediction-v1'
LABELS = ('pickup', 'harvest', 'return', 'adjust')


def named_agent_history(history, policy_joint_names, agent_joint_names):
    """Reorder measured [16,58] positions by joint names; never by dimension alone."""
    if (len(history) != 16 or tuple(policy_joint_names) != tuple(ORDER)
            or len(agent_joint_names) != 58 or len(set(agent_joint_names)) != 58):
        raise ValueError('Expected canonical OpenPI 16x58 measured position history')
    names = [n + '.position' for n in policy_joint_names]
    if set(names) != set(agent_joint_names):
        raise ValueError('Agent/policy joint name mismatch')
    indices = [names.index(n) for n in agent_joint_names]
    if any(len(row) != 58 or any(not math.isfinite(x) for x in row) for row in history):
        raise ValueError('Nonfinite or malformed history')
    return [[row[i] for i in indices] for row in history]


def validate_prediction(value, *, observation_id, session_id, now, max_age_s=2.):
    if (value.get('protocol') != PROTOCOL or value.get('labels') != list(LABELS)
            or value.get('valid') is not True or value.get('error') is not None
            or value.get('observation_id') != observation_id or value.get('session_id') != session_id
            or not isinstance(value.get('model_version'), str) or not value['model_version']):
        raise ValueError('Invalid prediction identity/schema')
    t, end = value.get('input_timestamp'), value.get('result_timestamp')
    if (type(t) not in (int, float) or type(end) not in (int, float)
            or not all(math.isfinite(x) for x in (t, end, now, max_age_s))
            or max_age_s <= 0 or not 0 <= now - t <= max_age_s or not t <= end <= now):
        raise ValueError('Stale/future/invalid prediction timestamps')
    horizon = value.get('future_horizon_s')
    if type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Invalid future horizon')
    mode = value.get('prediction_mode')
    active = {'current_only': ('current',), 'future_only': ('future',), 'joint': ('current', 'future')}.get(mode)
    if active is None:
        raise ValueError('Invalid prediction mode')
    for name in ('current', 'future'):
        p, z = value.get(name + '_probabilities'), value.get(name + '_logits')
        if name not in active:
            if p is not None or z is not None:
                raise ValueError('Inactive head must be null')
            continue
        if (not isinstance(p, list) or len(p) != 4 or not isinstance(z, list) or len(z) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in p + z)
                or any(x < 0 or x > 1 for x in p) or not math.isclose(sum(p), 1., abs_tol=1e-5)):
            raise ValueError('Expected four finite normalized probabilities and logits')
        exp = [math.exp(x - max(z)) for x in z]; total = sum(exp)
        if any(not math.isclose(a, b / total, abs_tol=1e-5) for a, b in zip(p, exp, strict=True)):
            raise ValueError('Probabilities do not match logits')
    return value


@dataclass(frozen=True)
class PromptContext:
    phase: str
    phase_index: int
    action_class: str
    prompt: str
    version: int
    input_timestamp: float
    changed: bool
    reason: str
    source: str
    future_warning: str | None
    session_id: str
    model_version: str
    observation_id: str
    confidence: float
    needs_review: bool
    task_complete: bool = False


class PredictionPromptProvider:
    """Conservative ordered adapter. Future predicts warnings, never early commands.

    Assumes a human explicitly starts at pickup_1. Four-class probabilities alone
    cannot recover lost progress after an arbitrary mid-task restart.
    """
    def __init__(self, *, threshold=.8, confirmations=2, max_age_s=2.):
        if not 0 < threshold <= 1 or type(confirmations) is not int or confirmations < 1 or max_age_s <= 0:
            raise ValueError('Invalid decision configuration')
        self.threshold, self.confirmations, self.max_age_s = threshold, confirmations, max_age_s
        self.session_id = None
        self._sessions = set()

    def reset(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('Explicit nonempty session ID required')
        if session_id in self._sessions:
            raise ValueError('Use a new session ID when resetting task progress')
        self._sessions.add(session_id)
        self.session_id = session_id
        self.phase = self.count = self.version = 0
        self.last_timestamp = None
        self.model_version = None

    def accept_prediction(self, value, *, observation_id, now):
        if self.session_id is None:
            raise ValueError('Initialize a session before predictions')
        validate_prediction(value, observation_id=observation_id, session_id=self.session_id,
                            now=now, max_age_s=self.max_age_s)
        t = value['input_timestamp']
        if self.last_timestamp is not None and t <= self.last_timestamp:
            raise ValueError('Repeated or out-of-order observation')
        if self.model_version is not None and value['model_version'] != self.model_version:
            raise ValueError('Model changed during session; explicitly reset')
        p = value['current_probabilities']
        if p is None:
            raise ValueError('Future-only model cannot drive current-stage prompts')
        if self.last_timestamp is not None and t - self.last_timestamp > self.max_age_s:
            self.count = 0
        self.last_timestamp, self.model_version = t, value['model_version']
        current = LABELS[max(range(4), key=lambda i: p[i])]
        next_class = COARSE_SEQUENCE[self.phase + 1] if self.phase + 1 < len(PHASES) else None
        confirmed = current == next_class and max(p) >= self.threshold
        self.count = self.count + 1 if confirmed else 0
        changed = self.count >= self.confirmations
        if changed:
            self.phase += 1; self.version += 1; self.count = 0
        future = value['future_probabilities']
        warning = None
        if future is not None and max(future) >= self.threshold:
            candidate = LABELS[max(range(4), key=lambda i: future[i])]
            if self.phase + 1 < len(PHASES) and candidate == COARSE_SEQUENCE[self.phase + 1]:
                warning = candidate
        action = COARSE_SEQUENCE[self.phase]
        reason = 'confirmed_next_phase' if changed else ('awaiting_confirmation' if confirmed else 'hold_current_phase')
        return PromptContext(PHASES[self.phase], self.phase, action, ACTION_PROMPTS[action], self.version,
                             t, changed, reason, 'vjepa2', warning, self.session_id, self.model_version, observation_id,
                             float(p[LABELS.index(action)]), p[LABELS.index(action)] < self.threshold)


def bind_chunk(context, chunk_id):
    """Metadata alongside an inference request/result; never changes action values."""
    return dict(chunk_id=chunk_id, prompt_version=context.version, phase=context.phase,
                prompt=context.prompt, input_timestamp=context.input_timestamp,
                session_id=context.session_id, model_version=context.model_version,
                observation_id=context.observation_id)


def chunk_matches_prompt(binding, context):
    """Caller checks at its existing safe replan boundary; no queue/RTC operations."""
    return (binding.get('prompt_version') == context.version and binding.get('phase') == context.phase
            and binding.get('prompt') == context.prompt and binding.get('session_id') == context.session_id
            and binding.get('model_version') == context.model_version)
