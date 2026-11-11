"""Trainable experimental readout, UNTRAINED until a separate campaign validates it.

Only measured state, visual history, time and ROI geometry enter this model.
No stage position, predictions, action commands, result labels or future truth.
"""
import numpy as np
import torch
from torch import nn


def reader_batch(snapshot, *, device='cpu'):
    if not snapshot.get('valid'):
        raise ValueError('Fresh memory snapshot required')
    rows = snapshot['rows']
    end = rows[-1]['timestamp']
    visual, positions, metadata = [], [], []
    for r in rows:
        visual.append(np.concatenate((r['global_visual'], r['local_visual'][None]), axis=0))
        positions.append(r['positions'].reshape(-1))
        w, h = r['source_size']
        boxes = r['boxes']/np.array([w, h, w, h])
        camera = np.eye(3)[('cam0', 'cam1', 'cam2').index(r['source_camera'])]
        metadata.append(np.concatenate((boxes.reshape(-1), r['roi_valid'].astype(float),
            r['times']-end, r['state_times']-r['times'], [r['timestamp']-end], camera)))
    return dict(visual=torch.tensor(np.asarray(visual)[None], dtype=torch.float32, device=device),
                positions=torch.tensor(np.asarray(positions)[None], dtype=torch.float32, device=device),
                metadata=torch.tensor(np.asarray(metadata)[None], dtype=torch.float32, device=device))


class RGBMemoryReader(nn.Module):
    """Latest-window query attends to bounded history; two four-class heads.

    State mean/std must be fitted on TRAIN sources only and saved in checkpoint.
    Whole-window ROI validity and source coordinates remain explicit metadata.
    """
    def __init__(self, state_mean, state_std, hidden=128):
        super().__init__()
        mean, std = (torch.as_tensor(v, dtype=torch.float32) for v in (state_mean, state_std))
        if (mean.shape != (58,) or std.shape != (58,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Training-fitted finite 58D mean and positive std required')
        if type(hidden) is not int or hidden < 4 or hidden % 4:
            raise ValueError('Hidden dimension must be positive and divisible by four')
        self.register_buffer('state_mean', mean)
        self.register_buffer('state_std', std)
        self.visual = nn.Sequential(nn.Linear(4096, hidden), nn.GELU())
        self.state = nn.Sequential(nn.Linear(928, hidden), nn.GELU())
        self.geometry = nn.Linear(116, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        self.current = nn.Linear(hidden, 4)
        self.future = nn.Linear(hidden, 4)

    def forward(self, visual, positions, metadata):
        b, n = positions.shape[:2]
        if (n < 1 or n > 64 or visual.shape != (b, n, 4, 1024)
                or positions.shape != (b, n, 928) or metadata.shape != (b, n, 116)):
            raise ValueError('Expected bounded aligned four-view/state/ROI history')
        if any(not torch.isfinite(x).all() for x in (visual, positions, metadata)):
            raise ValueError('Finite readout inputs required')
        q = positions.reshape(b, n, 16, 58)
        q = ((q-self.state_mean)/self.state_std).flatten(2)
        z = self.norm(self.visual(visual.flatten(2))+self.state(q)+self.geometry(metadata))
        # Only latest output is predicted: all keys are at or before its time.
        context, _ = self.attention(z[:, -1:], z, z, need_weights=False)
        fused = self.norm(z[:, -1]+context[:, 0])
        return dict(current_logits=self.current(fused), future_logits=self.future(fused))
