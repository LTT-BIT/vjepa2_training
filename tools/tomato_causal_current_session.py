"""Bounded causal Current readout for offline streaming validation.

Does not decide phases or write Prompt. No label/time-to-boundary inputs.
"""
from collections import deque
import numpy as np
import torch
from .tomato_causal_objective import CausalReadouts


class CausalCurrentSession:
    def __init__(self,net,mean,std,context=4,gap_reset_s=.6):
        self.net=net.eval();self.device=next(net.parameters()).device
        self.mean=np.asarray(mean,dtype=np.float32);self.std=np.asarray(std,dtype=np.float32)
        if self.mean.ndim!=1 or self.std.shape!=self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError('Invalid fixed training normalization')
        if context<1 or gap_reset_s<=0:raise ValueError('Invalid causal context')
        self.context=context;self.gap_reset_s=gap_reset_s;self.history=deque(maxlen=context);self.last=None

    @classmethod
    def load(cls,path,device='cpu'):
        s=torch.load(path,map_location='cpu',weights_only=True)
        if s.get('architecture')!='causal_current_readouts_grucell' or s.get('format_version')!=1:
            raise ValueError('Unsupported causal checkpoint')
        cfg=s['protocol'];net=CausalReadouts(len(s['mean']),cfg['hidden'],cfg['dropout']).to(device);net.load_state_dict(s['head'])
        return cls(net,s['mean'].numpy(),s['std'].numpy(),cfg['context'],cfg['gap_reset_s'])

    @torch.inference_mode()
    def update(self,feature,timestamp):
        try:
            x=np.asarray(feature,dtype=np.float32);t=float(timestamp)
            valid=x.shape==self.mean.shape and np.isfinite(x).all() and np.isfinite(t) and (self.last is None or t>self.last)
        except (ValueError,TypeError):valid=False
        if not valid:
            self.history.clear()
            return dict(valid=False,reason='invalid_or_reordered_observation',current_probabilities=None)
        if self.last is not None and t-self.last>self.gap_reset_s:self.history.clear()
        with np.errstate(over='ignore',invalid='ignore'):normalized=(x-self.mean)/self.std
        if not np.isfinite(normalized).all():
            self.history.clear()
            return dict(valid=False,reason='invalid_normalized_observation',current_probabilities=None)
        self.last=t;self.history.append((t,normalized))
        features=np.zeros((1,self.context,len(self.mean)),np.float32);mask=np.zeros((1,self.context),bool)
        features[0,-len(self.history):]=np.stack([v for _,v in self.history]);mask[0,-len(self.history):]=True
        p=self.net(torch.from_numpy(features).to(self.device),torch.from_numpy(mask).to(self.device))[0,-1].softmax(-1).cpu().numpy()
        if not np.isfinite(p).all():
            self.history.clear()
            return dict(valid=False,reason='nonfinite_readout',current_probabilities=None)
        return dict(valid=True,observation_timestamp=t,current_probabilities=p,context_observation_timestamps=[v for v,_ in self.history])
