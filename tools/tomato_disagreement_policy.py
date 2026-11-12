"""Offline research policies; no annotations, robotics, model fitting or wall-clock priors.

Future can prepare, cancel or briefly defer; never authorizes a Prompt transition.
The bounded arm adds <=1 fresh Current observation per uninterrupted candidate.
"""
import numpy as np
from .tomato_state_recovery import CLASSES, StableLabel

ARMS = ('B0', 'advisory', 'bounded_conflict', 'current_k3')


class DisagreementPolicy:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Unknown fixed research arm')
        self.arm = arm
        self.phase = self.version = 0
        self.last = None
        self.warning = StableLabel()
        self.events = []
        self._clear()

    def _clear(self):
        self.current_count = self.future_count = 0
        self.future_label = None
        self.prepared = None
        self.prepare_time = None
        self.hold_used = False
        self.warning.break_streak()

    def update(self, current, future, timestamp, *, now=None):
        before = self.phase
        try:
            c, f = np.asarray(current, float), np.asarray(future, float)
            t = float(timestamp)
            arrival = t if now is None else float(now)
            valid = (all(p.shape == (4,) and np.isfinite(p).all() and (p >= 0).all()
                         and abs(p.sum()-1) < 1e-4 for p in (c, f))
                     and np.isfinite([t, arrival]).all() and 0 <= arrival-t <= 1
                     and (self.last is None or t > self.last))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            self._clear()
            return dict(valid=False, time=None, phase=self.phase, previous_phase=before,
                        changed=False, prompt_version=self.version, warning=None,
                        resynchronized=False, reason='invalid_stale_or_reordered')
        if self.last is not None and t-self.last > .75:
            self._clear()
        self.last = t
        warning = self.warning.update(f, t)
        ci, fi = int(c.argmax()), int(f.argmax())
        self.future_count = self.future_count+1 if fi == self.future_label and f[fi] >= .8 else 1 if f[fi] >= .8 else 0
        self.future_label = fi
        reason = 'terminal' if self.phase == 6 else 'current_evidence_incomplete'
        if self.phase < 6:
            target = CLASSES[self.phase+1]
            if self.arm != 'B0':
                preparation_valid = fi == target and f[fi] >= .5
                if self.prepared is not None and (not preparation_valid or t-self.prepare_time >= 1.5):
                    self.events.append(dict(time=t, kind='cancel_preparation', phase=self.phase+1))
                    self.prepared = self.prepare_time = None
                if preparation_valid and self.prepared is None:
                    self.prepared, self.prepare_time = self.phase+1, t
                    self.events.append(dict(time=t, kind='prepare', phase=self.prepared))
            qualifies = ci == target and c[target] >= .8
            self.current_count = self.current_count+1 if qualifies else 0
            if not qualifies:
                self.hold_used = False
            required = 3 if self.arm == 'current_k3' else 2
            if self.current_count >= required:
                # A short upcoming phase may already be absent at t+1.5.
                # The next-after-target class is legitimate lookahead, not conflict.
                support = {target}
                if self.phase+2 < len(CLASSES):
                    support.add(CLASSES[self.phase+2])
                conflict = self.future_count >= 2 and fi not in support
                if self.arm == 'bounded_conflict' and conflict and not self.hold_used:
                    self.hold_used = True
                    reason = 'bounded_future_conflict_wait_one_fresh_current'
                    self.events.append(dict(time=t, kind='defer_once', phase=self.phase+1))
                else:
                    self.phase += 1
                    self.version += 1
                    reason = 'current_confirmed_after_bounded_wait' if self.hold_used else 'current_confirmed'
                    self.current_count = 0
                    self.hold_used = False
                    self.prepared = self.prepare_time = None
                    self.events.append(dict(time=t, kind='commit', phase=self.phase, version=self.version))
        return dict(valid=True, time=t, phase=self.phase, previous_phase=before,
                    changed=self.phase != before, prompt_version=self.version,
                    warning=warning, resynchronized=False, reason=reason,
                    prepared_phase=self.prepared, future_argmax=fi, current_argmax=ci,
                    current_confirmations=self.current_count, future_confirmations=self.future_count,
                    estimated_transition_time=None, real_latency_measured=False)
