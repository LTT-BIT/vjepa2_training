"""Local JSONL bridge. NPZ contains rgb[3,16,256,256,3], joints[16,58], joint_names.

Loads the encoder/head once. Does not construct 30Hz histories or send robot commands.
Request timestamps are Unix seconds. Use a separate process from OpenPI/robot control.
"""
import argparse
import json
import sys
import numpy as np
from .tomato_agent_inference import PROTOCOL, TomatoAgent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pretrained-dir', required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--max-age-s', type=float, default=2.)
    a = p.parse_args()
    agent = TomatoAgent(a.checkpoint, a.pretrained_dir, a.device)
    print(json.dumps(dict(protocol=PROTOCOL, status='ready', model_version=agent.model_version)), flush=True)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get('operation') == 'health':
                response = dict(protocol=PROTOCOL, status='ready', model_version=agent.model_version)
            else:
                if request.get('protocol') != PROTOCOL:
                    raise ValueError('Unsupported protocol')
                with np.load(request['observation_npz'], allow_pickle=False) as data:
                    response = agent.predict(data['rgb'], data['joints'], data['joint_names'].tolist(),
                                             observation_id=request['observation_id'], session_id=request['session_id'],
                                             timestamp=request['timestamp'], max_age_s=a.max_age_s)
        except Exception as exc:
            response = dict(protocol=PROTOCOL, valid=False, error=str(exc),
                            observation_id=request.get('observation_id') if isinstance(request, dict) else None)
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
