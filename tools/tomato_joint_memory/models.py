"""Latest-query, masked causal attention with a latest-anchor residual."""
import torch
from torch import nn
from tools.tomato_phase_model import joint_head_model

class TemporalJoint(nn.Module):
    prediction_mode='joint'
    def __init__(self,hidden=128,dropout=.3,aux_dim=0):
        super().__init__();self.hidden=hidden;self.aux_dim=aux_dim
        self.project=nn.Linear(4000,hidden);self.activation=nn.GELU()
        self.aux=nn.Linear(aux_dim,hidden,bias=False) if aux_dim else None
        self.relative_time=nn.Linear(3,hidden,bias=False)
        self.q=nn.Linear(hidden,hidden);self.k=nn.Linear(hidden,hidden);self.v=nn.Linear(hidden,hidden)
        self.context=nn.Linear(hidden,hidden);self.norm=nn.LayerNorm(hidden);self.drop=nn.Dropout(dropout)
        self.current=nn.Linear(hidden,4);self.future=nn.Linear(hidden,4)
    def forward(self,x,mask,dt,aux=None):
        z=self.activation(self.project(x))
        if self.aux is not None:z=z+self.aux(aux)
        time=torch.stack([dt/4,torch.sin(dt),torch.cos(dt)-1],dim=-1)
        keys=z+self.relative_time(time);b,n,h=z.shape;d=h//4
        q=self.q(z[:,-1]).reshape(b,4,1,d);k=self.k(keys).reshape(b,n,4,d).transpose(1,2);v=self.v(z).reshape(b,n,4,d).transpose(1,2)
        score=(q@k.transpose(-2,-1))/(d**.5);score=score.masked_fill(~mask[:,None,None,:],-torch.inf)
        value=(score.softmax(-1)@v).reshape(b,h)
        shared=self.drop(self.norm(z[:,-1]+self.context(value)))
        return dict(current_logits=self.current(shared),future_logits=self.future(shared))

def make_model(group,c):
    if group in ['B0','B0R','teacher']:return joint_head_model(58)
    return TemporalJoint(c['hidden'],c['dropout'],58 if group=='K1' else 9 if group=='P1' else 0)

def load_memory_checkpoint(path,c,expected_bindings,expected_sha256=None):
    from .contracts import sha,digest_json
    import numpy as np
    if expected_sha256 is not None and sha(path)!=expected_sha256:raise ValueError('Checkpoint SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')!='tomato-joint-memory-v1' or saved['bindings']!=expected_bindings:raise ValueError('Model/statistics/feature/protocol SHA mismatch')
    if 'stats' in saved and digest_json({k:np.asarray(v).tolist() for k,v in saved['stats'].items()})!=saved['bindings'].get('stats_sha256'):raise ValueError('Statistics SHA mismatch')
    model=make_model(saved['group'],c);model.load_state_dict(saved['state']);model.eval()
    return saved,model
