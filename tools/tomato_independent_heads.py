"""Two independently optimized heads with one common frozen encoder and normalization."""
from pathlib import Path
import torch
from torch import nn
from . import tomato_phase_model as model, tomato_dataset as ds


class IndependentHeads(nn.Module):
    prediction_mode = 'joint'

    def __init__(self, current, future):
        super().__init__()
        self.current = current
        self.future = future

    def forward(self, features):
        return dict(current_logits=self.current(features), future_logits=self.future(features))


def validate_pair(current, future):
    if current['prediction_mode'] != 'current_only' or future['prediction_mode'] != 'future_only':
        raise ValueError('Independent bundle requires separately trained current_only/future_only')
    for key in ('contract', 'split_sha256', 'joint_names', 'dimension', 'pretrained', 'feature_version', 'sampler_version'):
        if current[key] != future[key]:
            raise ValueError('Independent head input identity mismatch: ' + key)
    for key in ('mean', 'std'):
        if not torch.equal(current[key], future[key]):
            raise ValueError('Independent heads must share exact train-only normalization: ' + key)


def pack(current_path, future_path, output):
    current, _ = model.load_checkpoint(current_path)
    future, _ = model.load_checkpoint(future_path)
    validate_pair(current, future)
    payload = dict(architecture='independent_dual_v1', format_version=3,
                   current=current, future=future,
                   sources={name:dict(path=str(path), sha256=ds.digest(path))
                            for name,path in [('current',current_path),('future',future_path)]})
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return ds.digest(output)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('architecture') != 'independent_dual_v1':
        return model.load_checkpoint_payload(payload, device=device)
    if payload.get('format_version') != 3:
        raise ValueError('Unsupported independent checkpoint version')
    current, ch = model.load_checkpoint_payload(payload['current'], device=device)
    future, fh = model.load_checkpoint_payload(payload['future'], device=device)
    validate_pair(current, future)
    prediction = ds.prediction_contract({**future['prediction'], 'prediction_mode':'joint'})
    saved = {**current, **prediction, 'prediction':prediction,
             'architecture':payload['architecture'], 'sources':payload['sources'],
             'validation':dict(current=current['validation'],future=future['validation']),
             'head_training':'independent; no shared trainable parameters'}
    return saved, IndependentHeads(ch, fh).to(device).eval()
