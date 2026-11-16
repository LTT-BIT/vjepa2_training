"""Reuse original ACK/timer/planner; inject requests only, never directly set Active."""
import math
from ..handshake import Handshake,Evidence,LABELS
from .contracts import H0_CONFIG,TASK_POLICY

BLOCKED={'WAIT_CURRENT_ACK','WAIT_LATE_ACK','RECOVERY_REQUIRED'}
class ExecutionShell:
 def __init__(self,task_policy=None):self.core=Handshake(H0_CONFIG);self.task_policy=TASK_POLICY if task_policy is None else task_policy
 def ingest(self,o):
  self.last_envelope=o
  s=self.core;m=Evidence(o.observed_at,o.available_at,o.current,o.future,o.valid)
  gap=s.last is not None and m.available_at-s.last.available_at>.75+1e-9
  if gap:s.cancel(m.available_at,'stale_evidence')
  old_pending=s.pending;old_state=s.state
  if old_state not in BLOCKED:s.pending=None;s.state='READY'
  # ACK's future vector is intentionally neutralized so the production candidate
  # rule cannot create a hidden request. Restore real evidence immediately after.
  neutral=tuple(float(label==s.active) for label in LABELS)
  s.prediction(Evidence(m.observed_at,m.available_at,m.current,neutral,m.valid));s.last=m
  if old_pending is not None and not gap and m.valid and old_state not in BLOCKED and s.state=='READY':s.pending=old_pending;s.state=old_state
 def apply(self,p):
  s=self.core;finite=lambda v:type(v) in [int,float] and math.isfinite(v)
  t=max(s.last.available_at if s.last else 0.,p.available_at if finite(p.available_at) else 0.);reason=None
  if (p.observation_id,p.model_version,p.policy_version)!=(self.last_envelope.observation_id,self.last_envelope.model_version,self.last_envelope.policy_version):reason='proposal_identity_mismatch'
  elif type(p.request_commit) is not bool:reason='invalid_request_type'
  elif p.candidate_stage is not None and p.candidate_stage not in LABELS:reason='unknown_candidate'
  elif not finite(p.proposal_score) or not 0<=p.proposal_score<=1:reason='invalid_score'
  elif s.last is None or not finite(p.observed_at) or not finite(p.available_at) or p.available_at<s.last.available_at-1e-9 or abs(s.last.observed_at-p.observed_at)>1e-8 or t-s.last.available_at>.75+1e-9:reason='stale_proposal'
  elif s.state in BLOCKED:reason='shell_'+s.state
  elif p.candidate_stage==s.active:reason='candidate_equals_active'
  elif p.candidate_stage is not None and p.candidate_stage not in self.task_policy.get(s.active,[]):reason='task_mask'
  s.emit('policy_proposal',t,policy_name=p.reason,score=p.proposal_score if finite(p.proposal_score) else None,candidate=p.candidate_stage,requested=p.request_commit,requested_at=t,available_at=p.available_at if finite(p.available_at) else None,last_observation_id=p.observation_id,masked_reason=reason,task_policy=self.task_policy)
  if reason:
   if s.state not in BLOCKED:s.cancel(t,reason)
   return reason
  if p.candidate_stage is None:s.cancel(t,'policy_HOLD');return None
  if s.pending and s.pending['stage']!=p.candidate_stage:s.cancel(t,'candidate_changed')
  if s.pending is None:
   s.pending=dict(stage=p.candidate_stage,count=0,first_observed_at=p.observed_at,first_available_at=t,first_candidate_at=t,first_qualified_at=None,request_at=None,source_target_time=p.observed_at+1,relation='policy',catch_up=False);s.state='CANDIDATE_PENDING';s.emit('propose',t)
  q=s.pending;q['count']+=1;q['last_observed_at']=p.observed_at;q['last_confirm_available_at']=t;q['latest_confidence']=p.proposal_score
  if p.request_commit:
   if q['first_qualified_at'] is None:q['first_qualified_at']=t
   if q['request_at'] is None:q['request_at']=t;s.state='SWITCH_REQUESTED';s.emit('request',t,policy_score=p.proposal_score)
  elif s.state=='SWITCH_REQUESTED':q['request_at']=None;s.state='CANDIDATE_PENDING';s.emit('request_withdrawn',t)
  s.emit('support',t,observed_at=p.observed_at)
  return None
