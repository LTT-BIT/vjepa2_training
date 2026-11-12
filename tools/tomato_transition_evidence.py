"""Causal probability/observation change features and lightweight conditional readout.

No labels, boundary timestamps or true phase enter feature construction/inference.
The caller supplies the online maintained predecessor phase for a shadow score.
"""
import numpy as np
import torch
from torch import nn

LAGS=(.5,1.,1.5)
ARMS=('E0','E1','E2')


def past_indices(times, lag):
    t=np.asarray(times,float)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Strict finite observation timestamps required')
    return np.maximum(np.searchsorted(t,t-lag,side='right')-1,0)


def change_features(features,current,future,times,joint_names):
    x=np.asarray(features,np.float32);c=np.asarray(current,np.float32);f=np.asarray(future,np.float32);t=np.asarray(times,float)
    past_indices(t,0)
    if x.shape!=(len(t),4000) or c.shape!=(len(t),4) or f.shape!=c.shape or not all(np.isfinite(v).all() for v in [x,c,f]):
        raise ValueError('Aligned finite 4000D features and four-class probabilities required')
    if len(joint_names)!=58 or len(set(joint_names))!=58 or not all(n.endswith('.position') for n in joint_names):
        raise ValueError('58 unique named measured positions required')
    groups=[]
    for side in ('left','right'):
        for part in ('arm','hand'):
            indices=[i for i,n in enumerate(joint_names) if n.startswith(side+'_') and ('_fr3_' in n)==(part=='arm')]
            if len(indices)!=(7 if part=='arm' else 22):raise ValueError('Named arm/hand schema mismatch')
            groups.append(indices)
    visual=x[:,:3072].reshape(-1,3,1024);q=x[:,3072:].reshape(-1,16,58)[:,-1,:]
    prob=[c,f];obs=[];available=[]
    for lag in LAGS:
        idx=past_indices(t,lag);dt=t-t[idx];ok=dt>0;available.append(ok[:,None].astype(np.float32));prob += [c[idx],f[idx]]
        v0=visual[idx];dot=(visual*v0).sum(2);den=np.linalg.norm(visual,axis=2)*np.linalg.norm(v0,axis=2)
        cosine=np.clip(1-dot/np.maximum(den,1e-6),0,2);rel=np.linalg.norm(visual-v0,axis=2)/np.maximum(np.linalg.norm(v0,axis=2),1e-6)
        dq=q-q[idx];velocity=np.divide(dq,dt[:,None],out=np.zeros_like(dq),where=ok[:,None])
        speed=np.stack([np.sqrt(np.mean(velocity[:,g]**2,axis=1)) for g in groups],axis=1)
        obs += [cosine,rel,dq,velocity,speed]
    e0=np.concatenate(prob+available,axis=1).astype(np.float32)
    e1=np.concatenate(obs+available,axis=1).astype(np.float32)
    return dict(E0=e0,E1=e1,E2=np.concatenate([e0,e1],axis=1))


def training_examples(times,segments,fps):
    """Training/scoring-only labels; never imported by streaming feature logic."""
    from .tomato_state_recovery import CLASSES
    t=np.asarray(times);starts=np.array([s['start_frame']/fps for s in segments]);gt=np.searchsorted(starts,t,side='right')-1
    if len(segments)!=7:raise ValueError('Complete seven-position supervision required')
    from .tomato_dataset import LABELS
    if [LABELS.index(s['label']) for s in segments]!=list(CLASSES):raise ValueError('Unexpected label order')
    indices=[];phases=[];labels=[];weights=[]
    for k,b in enumerate(starts[1:]):
        for y in (0,1):
            ix=np.flatnonzero((gt==k+y)&(t>=b-3)&(t<=b+3))
            if len(ix):
                indices.extend(ix.tolist());phases.extend([k]*len(ix));labels.extend([y]*len(ix));weights.extend([1/len(ix)]*len(ix))
    return dict(index=np.array(indices,int),phase=np.array(phases,int),label=np.array(labels,np.float32),weight=np.array(weights,np.float32))


class EvidenceReadout(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.linear=nn.Linear(dimension,6)

    def forward(self,x,phase):
        return self.linear(x).gather(1,phase[:,None]).squeeze(1)

    def all_phases(self,x):
        return torch.sigmoid(self.linear(x))
