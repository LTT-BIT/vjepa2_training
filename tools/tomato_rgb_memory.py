"""Experimental measured-state + RGB memory, independent of stage/result labels.

Features must be extracted from real synchronous video windows. This module
neither detects ROIs nor derives physical success from motion or predictions.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from tools.tomato_dataset import fingerprint


@dataclass(frozen=True)
class RGBStateWindow:
    session_id: str
    episode_id: str
    observation_id: str
    encoder_signature: str
    roi_signature: str
    arrival_timestamp: float
    times: object                 # [16], actual selected video timestamps
    state_times: object           # [16], aligned measured-state timestamps
    global_visual: object         # [3,1024], pooled video features
    local_visual: object          # [1024], pooled crop video feature
    positions: object             # [16,58], measured positions, never actions
    boxes: object                 # [16,4], source-image crop xyxy
    roi_valid: object             # [16], false means fresh full-view fallback
    roi_available_at: object      # [16], when each box became available
    source_camera: str
    source_size: tuple            # (width,height)
    roi_mode: str                 # causal_prediction or manual_offline


class RGBStateMemory:
    protocol = 'tomato-rgb-state-memory-v2'

    def __init__(self, session_id, episode_id, joint_names, encoder_signature,
                 roi_signature, *, capacity=16, mode='online', max_gap=.75):
        names = tuple(joint_names)
        if (len(names) != 58 or len(set(names)) != 58
                or not all(isinstance(n, str) and n for n in names)
                or sum('fr3' in n for n in names) != 14):
            raise ValueError('Named 14-arm +44-hand measured-state schema required')
        if not all(isinstance(s, str) and s for s in
                   (session_id, episode_id, encoder_signature, roi_signature)):
            raise ValueError('Explicit session, episode, encoder and ROI identities required')
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('Bounded capacity must be 1..64')
        if mode not in ('online', 'offline_diagnostic') or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('Explicit mode and positive gap required')
        self.session_id, self.episode_id = session_id, episode_id
        self.joint_names = names
        self.encoder_signature, self.roi_signature = encoder_signature, roi_signature
        self.capacity, self.mode, self.max_gap = capacity, mode, max_gap
        self.rows = deque(maxlen=capacity)
        self.last = self.last_arrival = None

    def observe(self, window):
        if ((window.session_id, window.episode_id, window.encoder_signature, window.roi_signature)
                != (self.session_id, self.episode_id, self.encoder_signature, self.roi_signature)):
            raise ValueError('Cross-session, episode, encoder or ROI input rejected')
        arrays = {}
        for name, shape in dict(times=(16,), state_times=(16,), global_visual=(3, 1024),
                                local_visual=(1024,), positions=(16, 58), boxes=(16, 4),
                                roi_available_at=(16,)).items():
            a = np.asarray(getattr(window, name), dtype=np.float64)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Expected finite {name} {shape}')
            arrays[name] = a
        valid = np.asarray(window.roi_valid)
        if valid.shape != (16,) or valid.dtype != np.bool_:
            raise ValueError('Explicit boolean per-frame ROI validity required')
        t, st = arrays['times'], arrays['state_times']
        end = float(t[-1])
        if (t[0] < 0 or np.any(np.diff(t) < 0) or np.any(np.diff(st) < 0)
                or not np.allclose(np.diff(t), .1, atol=1e-6, rtol=0)
                or np.any(t-st < -1e-8) or np.any(t-st > .05+1e-8)
                or not math.isfinite(window.arrival_timestamp)
                or not 0 <= window.arrival_timestamp-end <= 1
                or self.last is not None and end <= self.last
                or self.last_arrival is not None and window.arrival_timestamp < self.last_arrival):
            raise ValueError('Fresh causal 16-frame 10Hz aligned windows required')
        if (not isinstance(window.observation_id, str) or not window.observation_id
                or any(r['observation_id'] == window.observation_id for r in self.rows)):
            raise ValueError('Unique observation identity required')
        if window.source_camera not in ('cam0', 'cam1', 'cam2'):
            raise ValueError('Known source camera required')
        size = np.asarray(window.source_size)
        if size.shape != (2,) or not np.issubdtype(size.dtype, np.integer) or np.any(size <= 0):
            raise ValueError('Positive integer source dimensions required')
        b = arrays['boxes']
        if np.any(b[:, 2:] <= b[:, :2]):
            raise ValueError('Ordered source crop boxes required')
        if not np.allclose(b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]):
            raise ValueError('Square source crops required')
        if np.any((b[:, 2] <= 0) | (b[:, 3] <= 0) | (b[:, 0] >= size[0]) | (b[:, 1] >= size[1])):
            raise ValueError('Crop must intersect source image')
        if window.roi_mode not in ('causal_prediction', 'manual_offline'):
            raise ValueError('Explicit ROI provenance required')
        if self.mode == 'online' and (window.roi_mode != 'causal_prediction'
                                     or np.any(arrays['roi_available_at'] > t+1e-8)):
            raise ValueError('Offline/manual or future-available ROI cannot enter online memory')
        if np.any(arrays['roi_available_at'] < 0):
            raise ValueError('ROI availability must be nonnegative')
        # Validate before mutating: rejected windows never reset history.
        gap = self.last is not None and end-self.last > self.max_gap
        if gap:
            self.rows.clear()
        row = dict(observation_id=window.observation_id, timestamp=end,
                   source_camera=window.source_camera, source_size=tuple(map(int, size)),
                   roi_mode=window.roi_mode, roi_valid=valid.copy(), **arrays)
        self.rows.append(deepcopy(row))
        self.last, self.last_arrival = end, window.arrival_timestamp
        return dict(gap_reset=gap, stored=len(self.rows), physical_success_inferred=False,
                    suggests_prompt_update=False, robot_action=None)

    def read(self, *, now, limit=None):
        if self.last is None:
            return dict(valid=False, reason='no_observation')
        if not math.isfinite(now) or not 0 <= now-self.last <= 1:
            return dict(valid=False, reason='stale_or_future_query')
        if limit is None:
            limit = self.capacity
        if type(limit) is not int or not 1 <= limit <= self.capacity:
            raise ValueError('Read limit must be within configured capacity')
        rows = list(self.rows)[-limit:]
        return deepcopy(dict(valid=True, protocol=self.protocol, rows=rows,
            schema_signature=fingerprint(dict(protocol=self.protocol, joint_names=self.joint_names,
                encoder=self.encoder_signature, roi=self.roi_signature, mode=self.mode)),
            endpoint_span_s=rows[-1]['timestamp']-rows[0]['timestamp'],
            raw_coverage_s=rows[-1]['timestamp']-rows[0]['times'][0],
            mode=self.mode, joint_names=self.joint_names,
            stage_inputs=False, outcome_inputs=False, learned_reader_trained=False,
            suggests_prompt_update=False, robot_action=None))

    def reset(self, session_id, episode_id):
        if session_id == self.session_id:
            raise ValueError('New session ID required')
        self.__init__(session_id, episode_id, self.joint_names, self.encoder_signature,
                      self.roi_signature, capacity=self.capacity, mode=self.mode, max_gap=self.max_gap)
