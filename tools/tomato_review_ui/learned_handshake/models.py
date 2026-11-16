import numpy as np
import torch
from torch import nn
from .contracts import DecisionProposal,LABELS,sha,read

class PolicyNet(nn.Module):
 def __init__(self,name,dimension=92,state_dim=48,hidden=32,dropout=.1):
  super().__init__();self.name=name
  if name=='L0':self.trunk=nn.Identity();self.action=nn.Linear(56,1);return
  if name=='L2':self.proj=nn.Linear(11,32);self.cell=nn.GRUCell(32,hidden);dimension=hidden+state_dim
  self.trunk=nn.Sequential(nn.Linear(dimension,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU())
  self.action=nn.Linear(32,1 if name=='L1a' else 4 if name=='L3a' else 5)
  if name=='L3a':self.time=nn.Linear(32,6)
 def forward(self,x,seq,state,mask):
  if self.name=='L0':z=torch.cat([seq[:,-1,:8],state],1)
  elif self.name=='L2':
   h=seq.new_zeros((len(seq),self.cell.hidden_size))
   for i in range(seq.shape[1]):h=torch.where(mask[:,i,None],self.cell(torch.nn.functional.gelu(self.proj(seq[:,i])),h),h)
   z=self.trunk(torch.cat([h,state],1))
  else:z=self.trunk(x)
  return dict(action=self.action(z),time=self.time(z) if self.name=='L3a' else None)

class LearnedPolicy:
 def __init__(self,name,model,stats,c,mapping='cost'):
  self.name=name;self.model=model.cpu().eval();self.stats=stats;self.c=c;self.mapping=mapping;self.diagnostics={}
 @torch.inference_mode()
 def predict(self,features):
  tensors={}
  for k in ['flat','sequence','state']:
   value=(features[k]-self.stats[k+'_mean'])/self.stats[k+'_std'];tensors[k]=torch.tensor(value[None],dtype=torch.float32)
  tensors['sequence'][0,~torch.tensor(features['mask'])]=0
  out=self.model(tensors['flat'],tensors['sequence'],tensors['state'],torch.tensor(features['mask'][None]))
  return {k:v[0].numpy() if v is not None else None for k,v in out.items()}
 def propose(self,o,features,core,task_policy):
  z=self.predict(features);a=z['action'];candidate=None;request=False
  if self.name in ['L0','L1a']:
   prob=float(1/(1+np.exp(-np.clip(a[0],-60,60))));i=int(np.argmax(o.future));legal=LABELS[i] in task_policy.get(core.active,[]) and o.future[i]>=.65 and o.future[i]>=o.future[LABELS.index(core.active)]
   candidate=LABELS[i] if legal else None;request=legal and prob>=.5;self.diagnostics=dict(gate_probability=prob)
  elif self.name=='L3a':
   pp=np.exp(a-a.max());pp/=pp.sum();tt=z['time'];tt=np.exp(tt-tt.max());tt/=tt.sum();i=int(pp.argmax());candidate=LABELS[i];prob=float(pp[i]);switch=float(tt@np.array(self.c['hazard_switch_cost']));hold=float(tt@np.array(self.c['hazard_hold_cost']))
   request=prob>=.5 and (switch<hold if self.mapping=='cost' else tt[2:5].sum()>=.5);self.diagnostics=dict(next_probabilities=pp.tolist(),time_probabilities=tt.tolist(),switch_cost=switch,hold_cost=hold)
  else:
   pp=np.exp(a-a.max());pp/=pp.sum();i=int(pp.argmax());candidate=LABELS[i-1] if i else None;request=bool(i);prob=float(pp[i]);self.diagnostics=dict(action_probabilities=pp.tolist())
  return DecisionProposal(o.observation_id,o.observed_at,o.available_at,candidate,bool(request),prob,self.name+('_mass' if self.mapping!='cost' else ''),o.model_version,o.policy_version)


def load_checkpoint(path,bindings,c):
 if sha(path)!=Path_sha(path):raise ValueError('Decision checkpoint SHA mismatch')
 saved=torch.load(path,map_location='cpu',weights_only=False)
 if saved['bindings']!=bindings or saved['schema']!=dict(flat=92,state=48,sequence=[4,11]):raise ValueError('Policy/schema/normalization/input identity mismatch')
 model=PolicyNet(saved['name'],dropout=c['dropout']);model.load_state_dict(saved['state']);model.eval()
 from .contracts import json_sha
 if json_sha({k:v.tolist() for k,v in saved['stats'].items()})!=saved['stats_sha256']:raise ValueError('Policy normalization SHA mismatch')
 return saved,model

def Path_sha(path):
 from pathlib import Path
 return (Path(path).parent/'checkpoint.sha256').read_text().strip()
