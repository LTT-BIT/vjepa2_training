import numpy as np
import torch
from torch import nn
from .contract import *
from .features import D,schema

class GateNet(nn.Module):
 def __init__(self,group,dropout=.1):
  super().__init__();self.group=group;flat=4*D+4
  if group=='G0':self.trunk=nn.Identity();self.gate=nn.Linear(flat,1)
  elif group=='G2':self.project=nn.Linear(D,32);self.cell=nn.GRUCell(32,32);self.trunk=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Dropout(dropout));self.gate=nn.Linear(32,1)
  else:self.trunk=nn.Sequential(nn.Linear(flat,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU());self.gate=nn.Linear(32,1)
  if group=='G3':self.hazard=nn.Linear(32,6)
 def forward(self,seq,mask):
  if self.group=='G2':
   h=seq.new_zeros(len(seq),32)
   for i in range(4):h=torch.where(mask[:,i,None],self.cell(torch.nn.functional.gelu(self.project(seq[:,i])),h),h)
   z=self.trunk(h)
  else:z=self.trunk(torch.cat([seq.flatten(1),mask.float()],1))
  return dict(gate=self.gate(z)[:,0],hazard=self.hazard(z) if self.group=='G3' else None)

class GatePolicy:
 def __init__(self,group,model,stats,version):self.group=group;self.name=group;self.model=model.cpu().eval();self.stats=stats;self.version=version;self.diagnostics={}
 @torch.inference_mode()
 def decide(self,o,feature,candidate,core=None):
  if candidate.stage is None:
   self.diagnostics=dict(model_called=False);return GateDecision(None,'HOLD',0.,o.observation_id,o.observed_at,o.available_at,self.version)
  seq=(feature['sequence']-self.stats['mean'])/self.stats['std'];seq=seq.copy();seq[~feature['mask']]=0
  z=self.model(torch.tensor(seq[None]),torch.tensor(feature['mask'][None]));score=float(torch.sigmoid(z['gate'][0]));self.diagnostics=dict(model_called=True,gate_probability=score)
  if z['hazard'] is not None:self.diagnostics['hazard_probabilities']=z['hazard'][0].softmax(0).tolist()
  return GateDecision(candidate.stage,'REQUEST' if score>=.5 else 'HOLD',score,o.observation_id,o.observed_at,o.available_at,self.version)

class R1Policy:
 name='R1';version='fixed-R1-v2'
 def __init__(self):
  from ..learned_handshake.rule_policies import RulePolicy
  self.rule=RulePolicy('R1');self.diagnostics={}
 def decide(self,o,feature,candidate,core):
  p=self.rule.propose(o,{},core,TASK_POLICY)
  if p.candidate_stage!=candidate.stage:raise ValueError('R1 provider mismatch')
  return GateDecision(candidate.stage,'REQUEST' if p.request_commit else 'HOLD',p.proposal_score,o.observation_id,o.observed_at,o.available_at,self.version)


def load_checkpoint(path,bindings,c):
 if sha(path)!=(Path(path).parent/'checkpoint.sha256').read_text().strip():raise ValueError('Checkpoint SHA mismatch')
 s=torch.load(path,map_location='cpu',weights_only=False)
 if s['format']!='candidate-gate-v3' or s['bindings']!=bindings or s['schema']!=schema():raise ValueError('Policy feature/input/protocol identity mismatch')
 if json_sha({k:v.tolist() for k,v in s['stats'].items()})!=s['stats_sha256']:raise ValueError('Train normalization changed')
 m=GateNet(s['group'],c['dropout']);m.load_state_dict(s['state']);m.eval();return s,m
