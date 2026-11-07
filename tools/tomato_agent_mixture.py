"""Low-cost inference mixtures; encoder runs once, component normalization must match."""
import torch
from torch import nn
from . import tomato_phase_model as model,tomato_temporal_heads as temporal,tomato_dataset as ds
from .tomato_agent_inference import TomatoAgent


class MixtureHead(nn.Module):
 prediction_mode='joint'
 def __init__(self,current,legacy,new,weight):
  super().__init__();self.current=current;self.legacy=legacy if weight<1 else None;self.new=new;self.weight=weight
 def forward(self,x):
  pnew=self.new(x[:,None,:])['future_logits'].softmax(-1)
  future=pnew if self.weight==1. else self.weight*pnew+(1-self.weight)*self.legacy(x).softmax(-1)
  return dict(current_logits=self.current(x),future_logits=future.clamp_min(1e-12).log())


def load_checkpoint(path,device='cpu'):
 data=torch.load(path,weights_only=True,map_location='cpu')
 if data.get('format_version')!=5 or data.get('architecture')!='tomato-agent-mixture-v1':raise ValueError('Invalid mixture')
 current,ch=model.load_checkpoint_payload(data['current'],device=device);legacy,fh=model.load_checkpoint_payload(data['legacy_future'],device=device);new=data['new_future']
 if current['prediction_mode']!='current_only' or legacy['prediction_mode']!='future_only':raise ValueError('Wrong component semantics')
 if new['format_version']!=4 or new['spec']['context']!=1 or new['spec']['dimension']!=4000:raise ValueError('Only compatible stateless future heads may be mixed')
 for other in [legacy,new]:
  if other['pretrained']!=current['pretrained'] or not torch.equal(other['mean'],current['mean']) or not torch.equal(other['std'],current['std']):raise ValueError('Mixture encoder/normalization differs')
 if new['input_schema']['joint_names']!=current['joint_names'] or new['spec']['horizon_s']!=legacy['future_horizon_s'] or legacy['split_sha256']!=current['split_sha256']:raise ValueError('Mixture input/time/split mismatch')
 if not 0<data['new_future_weight']<=1:raise ValueError('Invalid mixture weight')
 nh=temporal.TemporalHead(new['spec']);nh.load_state_dict(new['head'])
 head=MixtureHead(ch,fh,nh.to(device),data['new_future_weight']).to(device).eval()
 if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Nonfinite mixture weights')
 prediction=ds.prediction_contract({**legacy['prediction'],'prediction_mode':'joint'})
 saved={**current,**prediction,'prediction':prediction,'architecture':data['architecture'],'sources':data['sources']}
 return saved,head


class MixtureVideoAgent(TomatoAgent):
 def __init__(self,checkpoint,pretrained_dir,device='cuda'):
  self.saved,self.head=load_checkpoint(checkpoint,device)
  self.encoder,actual=model.load_encoder(pretrained_dir,device,42)
  if actual!=self.saved['pretrained']:raise ValueError('Encoder mismatch')
  self.model_version=ds.digest(checkpoint);self.joint_names=self.saved['joint_names'];self.device=device
