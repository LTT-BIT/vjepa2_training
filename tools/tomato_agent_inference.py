"""Checkpoint-backed four-class inference; no robot or seven-class checkpoint imports."""
from __future__ import annotations

import time
import numpy as np

from . import tomato_dataset as ds
from . import tomato_phase_model as model

PROTOCOL = 'tomato-stage-prediction-v1'


class TomatoAgent:
    def __init__(self, checkpoint, pretrained_dir, device='cuda'):
        self.saved, self.head = model.load_checkpoint(checkpoint, device=device)
        self.encoder, provenance = model.load_encoder(pretrained_dir, device, self.saved['contract']['seed'])
        if provenance != self.saved['pretrained']:
            raise ValueError('Checkpoint/encoder provenance mismatch')
        self.model_version = ds.digest(checkpoint)
        self.joint_names = self.saved['joint_names']
        self.device = device

    def predict(self, rgb, joints, joint_names, *, observation_id, session_id, timestamp, max_age_s=2.0):
        started = time.time()
        if (not isinstance(observation_id, str) or not observation_id or not isinstance(session_id, str)
                or not session_id or not np.isfinite(timestamp) or max_age_s <= 0
                or not 0 <= started - timestamp <= max_age_s):
            raise ValueError('Invalid identity or stale/future observation timestamp')
        if (len(joint_names) != 58 or len(set(joint_names)) != 58
                or set(joint_names) != set(self.joint_names)):
            raise ValueError('Expected all 58 named measured position joints')
        q = np.asarray(joints, dtype=np.float32)
        if q.shape != (16, 58):
            raise ValueError('Expected [16,58] joint history')
        # OpenPI order differs; align explicitly by names before feature construction.
        q = q[:, [joint_names.index(n) for n in self.joint_names]]
        features = model.fuse_features(self.encoder, np.asarray(rgb)[None], q[None])
        logits = model.predict_heads(self.head, features, self.saved['mean'], self.saved['std'])
        result = dict(protocol=PROTOCOL, model_version=self.model_version, observation_id=observation_id,
                      session_id=session_id, input_timestamp=timestamp, result_timestamp=time.time(),
                      labels=list(ds.LABELS), future_horizon_s=self.saved['future_horizon_s'],
                      prediction_mode=self.saved['prediction_mode'], valid=True, error=None)
        for name in ('current', 'future'):
            value = logits[name + '_logits']
            if value is None:
                result[name + '_logits'] = result[name + '_probabilities'] = None
            else:
                z = value[0].astype(np.float64); p = np.exp(z - z.max()); p /= p.sum()
                result[name + '_logits'] = z.tolist(); result[name + '_probabilities'] = p.tolist()
        result['latency_ms'] = (result['result_timestamp'] - started) * 1000
        return result
