"""Opt-in research filter. Does not receive truth, predict timestamps or Prompt state."""
import math
import numpy as np


class CausalProbabilityFilter:
    """Timestamp-aware EMA; one fresh observation per update, no interpolation.

    Half-life .5s is fixed to one B0 observation period before evaluation.
    A gap >.75s resets history; callers must also break event confirmation streaks.
    """
    def __init__(self, half_life_s=.5, max_gap_s=.75):
        if not np.isfinite([half_life_s, max_gap_s]).all() or min(half_life_s, max_gap_s) <= 0:
            raise ValueError('Positive finite time constants required')
        self.half_life_s, self.max_gap_s = half_life_s, max_gap_s
        self.last = self.value = None

    def update(self, probabilities, timestamp):
        p = np.asarray(probabilities, dtype=float)
        t = float(timestamp)
        if (p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any()
                or abs(p.sum()-1) > 1e-4 or not np.isfinite(t)
                or (self.last is not None and t <= self.last)):
            raise ValueError('Fresh increasing observation and four-class probability required')
        reset = self.last is None or t-self.last > self.max_gap_s
        if reset:
            self.value = p.copy()
        else:
            decay = math.exp2(-(t-self.last)/self.half_life_s)
            self.value = decay*self.value+(1-decay)*p
        self.last = t
        return self.value.copy(), reset
