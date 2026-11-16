"""Public candidate request adapter. Real Future is never replaced or neutralized."""
import math
from ..handshake import Evidence
from ..learned_handshake.shell_adapter import ExecutionShell,BLOCKED
from .contract import DecisionProposal

class CandidateRequestPort(ExecutionShell):
 def __init__(self,allowed=None):super().__init__(allowed);self.latest_candidate=None;self.decisions=set();self.last_plan=-1.
 def ingest(self,o):
  self.last_envelope=o
  return self.core.receive_evidence(Evidence(o.observed_at,o.available_at,o.current,o.future,o.valid))
 def bind_candidate(self,c):self.latest_candidate=c
 def apply_decision(self,d,name):
  c=self.latest_candidate;o=self.last_envelope;reason=None
  if d.observation_id in self.decisions:reason='duplicate_gate_decision'
  elif c is None or d.candidate_stage!=c.stage or d.observation_id!=c.observation_id:reason='candidate_mismatch'
  elif d.observed_at!=o.observed_at or d.available_at!=o.available_at:reason='decision_clock_mismatch'
  elif d.action not in ['HOLD','REQUEST']:reason='unknown_action'
  elif d.decision_policy_sha!=o.policy_version:reason='policy_identity_mismatch'
  if reason:
   if self.core.state not in BLOCKED:self.core.cancel(o.available_at,reason)
   self.core.emit('gate_rejected',o.available_at,reason=reason);return reason
  self.decisions={d.observation_id}
  p=DecisionProposal(d.observation_id,d.observed_at,d.available_at,d.candidate_stage,d.action=='REQUEST' and c.stage is not None,d.score,name,o.model_version,d.decision_policy_sha)
  return super().apply(p)
 def planning_tick(self,t):
  if not math.isfinite(t) or abs(t/.5-round(t/.5))>1e-9:raise ValueError('Not a protected planning tick')
  if t<=self.last_plan:return False
  self.last_plan=t;self.core.planning(t);return True
