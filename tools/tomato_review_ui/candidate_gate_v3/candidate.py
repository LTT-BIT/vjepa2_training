"""One causal Future-top1 candidate; never requests, commits or sees labels."""
from dataclasses import dataclass
from .contract import LABELS

@dataclass(frozen=True)
class Candidate:
 stage:str|None
 observation_id:str
 observed_at:float
 available_at:float
 first_observed_at:float|None
 first_available_at:float|None
 count:int
 confidence:float
 reason:str
 token:int

class CandidateProvider:
 def __init__(self,theta=.65,ttl=.75):self.theta=theta;self.ttl=ttl;self.previous=None;self.last=None;self.active=None;self.serial=0
 def clear(self):self.previous=None
 def update(self,o,active,allowed):
  if self.last is not None and o.observed_at<=self.last:raise ValueError('Candidate requires a distinct chronological anchor')
  gap=self.last is not None and o.observed_at-self.last>self.ttl+1e-9
  if active!=self.active or gap:self.previous=None
  self.active=active;self.last=o.observed_at;i=max(range(4),key=lambda j:o.future[j]);f=LABELS[i];p=float(o.future[i]);reason='eligible';stage=f
  if not o.valid:reason='invalid';stage=None
  elif f==active:reason='equals_active';stage=None
  elif f not in allowed.get(active,[]):reason='task_mask';stage=None
  elif p<self.theta:reason='below_prepare';stage=None
  elif not 0<=o.available_at-o.observed_at<=self.ttl:reason='expired';stage=None
  if stage is None:self.previous=None;return Candidate(None,o.observation_id,o.observed_at,o.available_at,None,None,0,p,reason,self.serial)
  previous=self.previous
  if previous is None or previous.stage!=stage:self.serial+=1;count=1;first=o.observed_at;available=o.available_at
  else:count=previous.count+1;first=previous.first_observed_at;available=previous.first_available_at
  value=Candidate(stage,o.observation_id,o.observed_at,o.available_at,first,available,count,p,reason,self.serial);self.previous=value;return value
