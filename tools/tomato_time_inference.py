"""Optional additive hazard predictor; uses the SAME already-computed4000D features."""
from copy import deepcopy
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_transition_time import TransitionTimeHead,transition_fields


class TransitionTimePredictor:
    def __init__(self,checkpoint,base_model_version,device='cpu'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['identity']['version']!='tomato-transition-time-v1':raise ValueError('Unsupported time-head schema')
        if payload['identity']['original_classifiers']['sha256']!=base_model_version:raise ValueError('Time-head/classifier identity mismatch')
        self.base_model_version=base_model_version
        self.mean=payload['mean'].to(device);self.std=payload['std'].to(device)
        if self.mean.shape!=(4000,) or self.std.shape!=(4000,) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std<=0).any():raise ValueError('Invalid normalization')
        self.net=TransitionTimeHead().to(device);self.net.load_state_dict(payload['head']);self.net.eval();self.version=ds.digest(checkpoint);self.device=device
    def augment(self,response,features):
        if not response.get('valid') or response.get('model_version')!=self.base_model_version or not np.isfinite(response['input_timestamp']):raise ValueError('Invalid base response')
        x=np.asarray(features,dtype='float32')
        if x.shape!=(1,4000) or not np.isfinite(x).all():raise ValueError('Invalid causal feature input')
        with torch.inference_mode():z=self.net((torch.from_numpy(x).to(self.device)-self.mean)/self.std)[0].cpu()
        result=deepcopy(response);result.update(transition_fields(z,response['input_timestamp'],self.version));return result
