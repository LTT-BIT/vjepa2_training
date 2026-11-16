"""Inference-only probability and own-decision history; no annotation imports."""
from collections import deque
import math
import numpy as np
from ..learned_handshake.feature_builder import FeatureBuilder as EnvelopeValidator
from .contract import LABELS,STATES,Observation

FIELDS=(['pC_'+s for s in LABELS]+['pF_'+s for s in LABELS]+['pc_candidate','pc_active','pf_candidate','pf_active','margin_c','margin_f','entropy_c','entropy_f']+['delta_c_'+s for s in LABELS]+['delta_f_'+s for s in LABELS]+['observation_gap','prediction_age','candidate_span','candidate_count','pending_span','pending_count','elapsed_commit','has_commit','until_plan']+['active_'+s for s in LABELS]+['candidate_'+s for s in LABELS]+['legal_'+s for s in LABELS]+['state_'+s for s in STATES]+['pending_'+s for s in LABELS]+['new_observation','gap_reset','valid','relative_available_time'])
D=len(FIELDS)

def schema():return dict(version='GateFeatureV1',fields=FIELDS,per_anchor=D,history=4,flat=4*D+4,mask='left pad false; past/current only',relative_time='available_at relative to newest /2, clipped[-2,0]')

class GateFeatures:
 def __init__(self,session,model,policy):self.validator=EnvelopeValidator(session,model,policy);self.rows=deque(maxlen=4);self.last_snapshot=None;self.gap=False
 def observe(self,o):
  prev=self.validator.history[-1] if self.validator.history else None;gap=prev is not None and o.observed_at-prev.observed_at>.75+1e-9
  self.validator.observe(o)
  if gap:self.rows.clear()
  self.gap=gap
 def snapshot(self,o,core,candidate,allowed):
  if self.last_snapshot==o.observation_id:raise ValueError('Cannot make a new feature row from HELD evidence')
  items=list(self.validator.history);p=np.asarray([*o.current,*o.future]);dp=p-np.asarray([*items[-2].current,*items[-2].future]) if len(items)>1 else np.zeros(8);cp,fp=p[:4],p[4:];active=LABELS.index(core.active);ci=LABELS.index(candidate.stage) if candidate.stage is not None else None;pending=core.pending or {};now=o.available_at
  one=lambda stage:[float(stage==s) for s in LABELS]
  gap=o.observed_at-items[-2].observed_at if len(items)>1 else 0.;tick=round(math.ceil((now-1e-9)/.5)*.5,9)
  probabilities=[cp[ci] if ci is not None else 0,cp[active],fp[ci] if ci is not None else 0,fp[active]]
  values=np.r_[p,probabilities,np.sort(cp)[-1]-np.sort(cp)[-2],np.sort(fp)[-1]-np.sort(fp)[-2],-sum(cp*np.log(cp.clip(1e-12))),-sum(fp*np.log(fp.clip(1e-12))),dp,[min(gap,5)/5,(now-o.observed_at)/.75,min(now-(candidate.first_available_at if candidate.first_available_at is not None else now),5)/5,min(candidate.count,16)/16,min(now-pending.get('first_available_at',now),5)/5,min(pending.get('count',0),16)/16,min(now-core.commit_at,20)/20 if core.commit_at is not None else 0,float(core.commit_at is not None),(tick-now)/.5],one(core.active),one(candidate.stage),[float(s in allowed.get(core.active,[])) for s in LABELS],[float(core.state==s) for s in STATES],one(pending.get('stage')),[1.,float(self.gap),1.]].astype(np.float32)
  if values.shape!=(D-1,):raise ValueError('Feature schema dimension mismatch')
  self.rows.append((now,values));self.last_snapshot=o.observation_id;seq=np.zeros((4,D),np.float32);mask=np.zeros(4,bool)
  for j,(t,v) in enumerate(self.rows,start=4-len(self.rows)):seq[j]=np.r_[v,max(-2,min(0,(t-now)/2))];mask[j]=True
  return dict(sequence=seq,mask=mask,active=core.active,policy_state=core.state,candidate=candidate.stage,candidate_reason=candidate.reason,candidate_count=candidate.count,candidate_first_available=candidate.first_available_at,commit_tick=tick,observed_at=o.observed_at,available_at=now,observation_id=o.observation_id,eligible=candidate.stage is not None and core.state in ['READY','CANDIDATE_PENDING','SWITCH_REQUESTED'])


def make_observation(row,session,policy):
 return Observation(session_id=row.get('session_id',session),observation_id=row.get('observation_id',f'{session}:{row["frame_index"]}'),model_version=row.get('model_version','B0-seed42'),policy_version=row.get('policy_version',policy),observed_at=float(row['observed_at']),available_at=float(row.get('available_at',round(row['observed_at']+.09,9))),current=tuple(row['heads']['current']['probabilities'][k] for k in LABELS),future=tuple(row['heads']['future']['probabilities'][k] for k in LABELS),valid=row.get('valid',True))
