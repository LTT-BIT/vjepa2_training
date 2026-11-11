"""Tiny causal probability-history calibration; no stage truth enters inference."""
import numpy as np
import torch
from torch import nn


class ProbabilityHistory:
    dimension = 36

    def __init__(self):
        self.rows=[]
        self.last=None

    def update(self, current, future, timestamp):
        c,f=np.asarray(current,float),np.asarray(future,float)
        t=float(timestamp)
        if (not np.isfinite(t) or (self.last is not None and t<=self.last)
                or any(p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-4 for p in (c,f))):
            raise ValueError('Fresh timestamp and finite normalized four-class probabilities required')
        if self.last is not None and t-self.last>.75:self.rows=[]
        self.last=t
        self.rows.append((t,np.log(np.maximum(np.r_[c,f],1e-6))))
        self.rows=[r for r in self.rows if r[0]>=t-1.5-1e-8]
        values=[];masks=[]
        for lag in (0.,.5,1.,1.5):
            available=[r for r in self.rows if r[0]<=t-lag+1e-8]
            # Never use a newer frame to fill an older requested history slot.
            valid=bool(available and t-lag-available[-1][0]<.25)
            values.extend(available[-1][1] if valid else np.zeros(8))
            masks.append(float(valid))
        return np.asarray(values+masks,dtype=np.float32)


class ResidualCalibration(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.01))
        self.correction=nn.Linear(36,4)
        nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)

    def forward(self, history, future_log_probabilities):
        delta=self.correction((history-self.mean)/self.std)
        delta=delta-delta.mean(-1,keepdim=True)
        return future_log_probabilities+delta,delta
