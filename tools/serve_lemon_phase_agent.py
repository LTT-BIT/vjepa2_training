#!/usr/bin/env python3
"""Private JSON-lines worker, started by PhaseAgentClient; no robot commands."""
import argparse
import base64
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tools import lemon_phase_model as model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = sys.stdout

    def send(message):
        output.write(json.dumps(message, allow_nan=False) + '\n')
        output.flush()

    try:
        # Reserve stdout for protocol messages; native/model diagnostics belong on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            saved, encoder, head = model.load_model(args.model, args.device)
            if (saved.get('labels') != list(model.V3_LABELS) or saved.get('label_semantics') != model.V3_SEMANTICS
                    or saved.get('boundaries') != list(model.V3_BOUNDARIES)
                    or saved.get('cameras') != list(model.CAMERAS) or saved.get('joints') != list(model.JOINTS)):
                raise ValueError('Expected the v3 six-class, three-camera and twelve-joint phase model')

            def predict(images, state):
                features = model.encode(encoder, images, args.device).reshape(1, -1)
                return model.probabilities(saved, head, features, state[None], args.device)[0]

            images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in model.CAMERAS]
            for _ in range(3):
                before = time.monotonic()
                result = predict(images, np.zeros(12, dtype=np.float32))
            if result.shape != (6,) or not np.isfinite(result).all():
                raise ValueError('Phase model produced invalid warmup probabilities')
            warmup_ms = (time.monotonic() - before) * 1000
        send(dict(type='ready', protocol=1, labels=saved['labels'], label_semantics=saved['label_semantics'],
                  cameras=saved['cameras'], joints=saved['joints'], device=args.device, cpu_threads=2,
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), warmup_latency_ms=warmup_ms))
        while True:
            line = sys.stdin.buffer.readline(1_000_001)
            if not line:
                return 0
            if len(line) > 1_000_000 or not line.endswith(b'\n'):
                raise ValueError('Oversized or incomplete phase-agent request')
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get('id')) is not int:
                raise ValueError('Invalid phase-agent request')
            state = np.asarray(request['state'], dtype=np.float32)
            if state.shape != (12,) or not np.isfinite(state).all():
                raise ValueError('Expected twelve finite measured joint positions')
            images = []
            for cam in model.CAMERAS:
                raw = base64.b64decode(request['images'][cam], validate=True)
                if len(raw) != 224 * 224 * 3:
                    raise ValueError(f'{cam}: expected 224x224 RGB image')
                images.append(np.frombuffer(raw, dtype=np.uint8).reshape(224, 224, 3))
            with contextlib.redirect_stdout(sys.stderr):
                result = predict(images, state)
            send(dict(type='prediction', id=request['id'], probabilities=result.tolist()))
    except Exception as exc:
        send(dict(type='error', error=f'{type(exc).__name__}: {exc}'))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
