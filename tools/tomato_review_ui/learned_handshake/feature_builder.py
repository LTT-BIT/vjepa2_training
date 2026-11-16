"""Strict inference whitelist. No label/annotation/dataframe imports."""
from collections import deque
import math
import numpy as np
from .contracts import LABELS,STATES,Observation

class FeatureBuilder:
 def __init__(self,session_id,model_version,policy_version,length=4,ttl=.75):
  self.session=session_id;self.model=model_version;self.policy=policy_version;self.length=length;self.ttl=ttl;self.history=deque(maxlen=length);self.last_arrival=None;self.ids=set()
 def observe(self,o):
  if (o.session_id,o.model_version,o.policy_version)!=(self.session,self.model,self.policy):raise ValueError('Identity mismatch')
  if not o.valid or not o.observation_id or o.observation_id in self.ids:raise ValueError('Invalid/duplicate observation')
  if not np.isfinite([o.observed_at,o.available_at]).all() or not 0<=o.available_at-o.observed_at<=self.ttl:raise ValueError('Stale/noncausal prediction')
  for p in [o.current,o.future]:
   if len(p)!=4 or not np.isfinite(p).all() or min(p)<0 or abs(sum(p)-1)>1e-6:raise ValueError('Invalid probabilities')
  if self.last_arrival is not None and o.available_at<self.last_arrival:raise ValueError('Arrival reversal')
  if self.history and o.observed_at<=self.history[-1].observed_at:raise ValueError('Duplicate/HELD/reversed anchor')
  if self.history and o.observed_at-self.history[-1].observed_at>self.ttl+1e-9:self.history.clear();self.ids.clear()
  self.history.append(o);self.ids.add(o.observation_id);self.last_arrival=o.available_at
  # Bounded duplicate index: old timestamps are independently rejected.
  self.ids={v.observation_id for v in self.history}
 def build(self,core,task_policy):
  if not self.history:raise ValueError('No fresh observation')
  o=self.history[-1];now=o.available_at;items=list(self.history);seq=np.zeros((self.length,11),np.float32);mask=np.zeros(self.length,bool)
  for pos,v in enumerate(items,start=self.length-len(items)):
   gap=0 if pos==self.length-len(items) else v.observed_at-items[pos-(self.length-len(items))-1].observed_at
   seq[pos]=[*v.current,*v.future,(v.observed_at-o.observed_at)/2,gap,1];mask[pos]=True
  cp=np.array(o.current);fp=np.array(o.future);prior=np.array([*items[-2].current,*items[-2].future]) if len(items)>1 else np.r_[cp,fp];slope=(np.r_[cp,fp]-prior)/(o.observed_at-items[-2].observed_at) if len(items)>1 else np.zeros(8)
  one=lambda label:np.array([float(label==s) for s in LABELS])
  pending=core.pending or {};legal=np.array([float(s in task_policy.get(core.active,[])) for s in LABELS]);candidate=LABELS[int(fp.argmax())]
  next_tick=math.ceil((now-1e-9)/.5)*.5
  state=np.r_[one(core.active),one(pending.get('stage')),one(candidate),[min(pending.get('count',0),16)/16,min(now-pending.get('first_available_at',now),5)/5,min(now-core.commit_at,20)/20 if core.commit_at is not None else 0,float(core.commit_at is not None)], [float(core.state==s) for s in STATES],[(now-o.observed_at)/.75,(next_tick-now)/.5],legal,[np.sort(cp)[-1]-np.sort(cp)[-2],np.sort(fp)[-1]-np.sort(fp)[-2],-sum(cp*np.log(cp.clip(1e-12))),-sum(fp*np.log(fp.clip(1e-12)))],slope].astype(np.float32)
  ci=LABELS.index(candidate);ai=LABELS.index(core.active)
  state=np.r_[state,[cp[ci],fp[ci],cp[ai],fp[ai],cp[ci]-cp[ai],fp[ci]-fp[ai],slope[ci],slope[ci+4]]].astype(np.float32)
  flat=np.r_[seq.flatten(),state].astype(np.float32)
  return dict(flat=flat,sequence=seq,mask=mask,state=state,active=core.active,policy_state=core.state,candidate=candidate,commit_tick=next_tick,observed_at=o.observed_at,available_at=now)


def observation(row,session,model,policy):
 # Only these fields are read even if callers attach complete GT tables.
 return Observation(session_id=session,observation_id=f'{session}:{row["frame_index"]}',model_version=model,policy_version=policy,observed_at=float(row['observed_at']),available_at=round(row['observed_at']+.09,9),current=tuple(row['heads']['current']['probabilities'][k] for k in LABELS),future=tuple(row['heads']['future']['probabilities'][k] for k in LABELS))
