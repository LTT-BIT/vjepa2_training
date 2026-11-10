"""Causal, opt-in phase estimation. No labels, episode clocks, models or robot I/O.

Warnings reproduce B0 verbatim. Estimated seven-position posterior is a sequence
filter over four-class emissions, NOT seven-way classifier probabilities.
"""
from dataclasses import dataclass
import math
import numpy as np

PHASES = ('pickup_1', 'harvest_1', 'return_truss', 'adjust',
          'pickup_2', 'harvest_2', 'return_stem')
CLASSES = (0, 1, 2, 3, 0, 1, 2)
MODES = ('baseline', 'detect', 'hypotheses', 'resync')


class StableLabel:
    """Streaming equivalent of tomato_boundary_metrics.change_events."""
    def __init__(self, threshold=.5, confirmations=2):
        self.threshold = threshold
        self.confirmations = confirmations
        self.accepted = self.pending = None
        self.count = 0

    def break_streak(self):
        self.pending = None
        self.count = 0

    def update(self, probabilities, timestamp):
        label = int(np.argmax(probabilities))
        if probabilities[label] < self.threshold:
            self.break_streak()
            return None
        self.count = self.count + 1 if self.pending == label else 1
        self.pending = label
        if self.count < self.confirmations or label == self.accepted:
            return None
        previous, self.accepted = self.accepted, label
        if previous is not None:
            return dict(time=float(timestamp), previous=previous, label=label)
        return None


class B0State:
    """Unchanged B0 warnings + historical ordered Current .8/K2 Prompt rule."""
    def __init__(self):
        self.warnings = StableLabel()
        self.phase = self.count = 0

    def update(self, current, future, timestamp):
        warning = self.warnings.update(future, timestamp)
        before = self.phase
        if self.phase < 6:
            target = CLASSES[self.phase + 1]
            qualifies = int(np.argmax(current)) == target and current[target] >= .8
            self.count = self.count + 1 if qualifies else 0
            if self.count >= 2:
                self.phase += 1
                self.count = 0
        return dict(warning=warning, phase=self.phase, changed=self.phase != before)


@dataclass(frozen=True)
class Hypothesis:
    phase: int
    log_mass: float
    best_log_mass: float
    # At most six transitions; each carries actual causal Current evidence.
    path: tuple = ()


class OrderedBeam:
    def __init__(self, width=3):
        if width != 3:
            raise ValueError('v1 fixes beam width before evaluation')
        self.width = width
        self.states = [Hypothesis(0, 0., 0.)]
        self.runs = np.zeros(4, dtype=int)
        self.starts = np.zeros(4)
        self.minima = np.ones(4)

    def break_streak(self):
        self.runs[:] = 0

    def update(self, current, timestamp):
        label = int(np.argmax(current))
        for k in range(4):
            if k == label and current[k] >= .5:
                if self.runs[k] == 0:
                    self.starts[k], self.minima[k] = timestamp, current[k]
                else:
                    self.minima[k] = min(self.minima[k], current[k])
                self.runs[k] += 1
            else:
                self.runs[k] = 0
        children = {}
        for h in self.states:
            advance = h.phase < 6 and self.runs[CLASSES[h.phase + 1]] >= 2
            for dest in (h.phase, h.phase + 1) if advance else (h.phase,):
                prior = math.log(.5) if advance else 0.
                emission = math.log(max(float(current[CLASSES[dest]]), 1e-12))
                log_mass = h.log_mass + prior + emission
                best = h.best_log_mass + prior + emission
                path = h.path
                if dest != h.phase:
                    k = CLASSES[dest]
                    path += (dict(phase=dest, start=float(self.starts[k]),
                                  end=float(timestamp), count=int(self.runs[k]),
                                  min_probability=float(self.minima[k])),)
                if dest in children:
                    old = children[dest]
                    children[dest] = Hypothesis(dest, float(np.logaddexp(old.log_mass, log_mass)),
                                               max(old.best_log_mass, best),
                                               path if best > old.best_log_mass else old.path)
                else:
                    children[dest] = Hypothesis(dest, log_mass, best, path)
        ranked = sorted(children.values(), key=lambda h: (-h.log_mass, h.phase))[:self.width]
        z = float(np.logaddexp.reduce([h.log_mass for h in ranked]))
        self.states = [Hypothesis(h.phase, h.log_mass-z, h.best_log_mass-z, h.path) for h in ranked]
        return [dict(phase=h.phase, probability=math.exp(h.log_mass), path=list(h.path)) for h in self.states]


class StateRecovery:
    """One instance per session. Explicit reset; invalid/gapped observations cannot
    complete evidence streaks. A/B use exactly original B0 behavior on valid data.
    Time-head probabilities may be attached by callers but never veto a decision.
    """
    def __init__(self, mode='baseline', *, session_id='replay', model_id='frozen-b0'):
        if mode not in MODES or not session_id or not model_id:
            raise ValueError('Explicit valid strategy/session/model identity required')
        self.mode, self.session_id, self.model_id = mode, session_id, model_id
        self.base = B0State()
        self.beam = None if mode == 'baseline' else OrderedBeam()
        self.phase = self.version = 0
        self.last = None
        self.pending = None
        self.count = self.inconsistent_count = 0
        self.inconsistent_label = None

    def reset(self, session_id, model_id=None):
        self.__init__(self.mode, session_id=session_id, model_id=model_id or self.model_id)

    def _break_streak(self):
        self.base.count = 0
        self.base.warnings.break_streak()
        self.count = self.inconsistent_count = 0
        self.pending = self.inconsistent_label = None
        if self.beam is not None:
            self.beam.break_streak()

    def update(self, current, future, timestamp, *, now=None, session_id=None, model_id=None):
        try:
            c, f = np.asarray(current, dtype=float), np.asarray(future, dtype=float)
            t = float(timestamp)
            arrival = t if now is None else float(now)
            valid = all(p.shape == (4,) and np.isfinite(p).all() and (p >= 0).all()
                        and abs(p.sum()-1) < 1e-4 for p in (c, f))
            valid = valid and np.isfinite([t, arrival]).all() and 0 <= arrival-t <= 1.
            valid = valid and (self.last is None or t > self.last)
            valid = valid and session_id in (None, self.session_id) and model_id in (None, self.model_id)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self._break_streak()
            return dict(valid=False, phase=self.phase, changed=False, warning=None,
                        reason='invalid_stale_or_identity_requires_reset')
        # Preserve B0 gap semantics; conservative recovery streaks require fresh 2Hz history.
        if self.last is not None and t-self.last > .75 and self.beam is not None:
            self.beam.break_streak()
            self.pending, self.count, self.inconsistent_count = None, 0, 0
        self.last = t
        base = self.base.update(c, f, t)
        before = self.phase
        hypotheses = self.beam.update(c, t) if self.beam is not None else []
        resync, proof = False, []
        if self.mode in ('baseline', 'detect'):
            self.phase = base['phase']
        else:
            best = hypotheses[0]
            candidate = best['phase'] if best['probability'] >= .8 else None
            self.count = self.count+1 if candidate is not None and self.pending == candidate else int(candidate is not None)
            self.pending = candidate
            if candidate is not None and self.count >= 2 and candidate > self.phase:
                distance = candidate-self.phase
                proof = [v for v in best['path'] if self.phase < v['phase'] <= candidate]
                complete = [v['phase'] for v in proof] == list(range(self.phase+1, candidate+1))
                witnessed = complete and all(v['count'] >= 2 and v['min_probability'] >= .5 and v['end'] <= t for v in proof)
                arrival_confirmed = self.beam.runs[CLASSES[candidate]] >= 2 and c[CLASSES[candidate]] >= .8
                # Require .8 for BOTH arrival observations when resynchronizing.
                arrival_confirmed = arrival_confirmed and self.beam.minima[CLASSES[candidate]] >= .8
                if witnessed and (distance == 1 or (self.mode == 'resync' and distance <= 2 and arrival_confirmed)):
                    self.phase = candidate
                    resync = distance > 1
        changed = self.phase != before
        if changed:
            self.version += 1
        label = int(c.argmax())
        incompatible = label != CLASSES[self.phase] and (self.phase == 6 or label != CLASSES[self.phase+1]) and c[label] >= .8
        self.inconsistent_count = self.inconsistent_count+1 if incompatible and self.inconsistent_label == label else int(incompatible)
        self.inconsistent_label = label if incompatible else None
        alert = self.mode != 'baseline' and self.inconsistent_count >= 2
        return dict(valid=True, time=t, phase=self.phase, phase_name=PHASES[self.phase],
                    changed=changed, previous_phase=before, prompt_version=self.version,
                    warning=base['warning'], baseline_phase=base['phase'],
                    inconsistent=bool(alert), hypotheses=hypotheses,
                    estimated_phase=hypotheses[0]['phase'] if hypotheses else None,
                    resynchronized=resync, witness=proof if changed else [],
                    reason='witnessed_resync' if resync else 'ordered_evidence' if changed else 'hold')


class StateInconsistencyDetector:
    """Attachable report-only observer of an existing maintained phase.

    Four-class emissions can contradict an ordered state, but cannot identify
    which repeated position is correct. This class never proposes a reset.
    One-step lag is deliberately not a contradiction (it is the allowed next
    class); therefore a negative report does NOT establish correct state.
    """
    def __init__(self):
        self.last = None
        self.key = None
        self.count = 0

    def update(self, current, timestamp, maintained_phase):
        try:
            c = np.asarray(current, dtype=float)
            t = float(timestamp)
        except (TypeError, ValueError):
            self.key, self.count = None, 0
            return dict(valid=False, inconsistent=False, recommends_reset=False)
        valid = (isinstance(maintained_phase, (int, np.integer)) and 0 <= maintained_phase < 7
                 and c.shape == (4,) and np.isfinite(c).all() and (c >= 0).all()
                 and abs(c.sum()-1) < 1e-4 and np.isfinite(t)
                 and (self.last is None or t > self.last))
        if not valid:
            self.key, self.count = None, 0
            return dict(valid=False, inconsistent=False, recommends_reset=False)
        if self.last is not None and t-self.last > .75:
            self.key, self.count = None, 0
        self.last=t
        k=int(c.argmax());p=int(maintained_phase)
        incompatible = c[k] >= .8 and k != CLASSES[p] and (p == 6 or k != CLASSES[p+1])
        key=(p,k) if incompatible else None
        self.count=self.count+1 if key is not None and key==self.key else int(key is not None)
        self.key=key
        return dict(valid=True,inconsistent=self.count>=2,count=self.count,
                    phase=p,current_class=k,confidence=float(c[k]),recommends_reset=False)
