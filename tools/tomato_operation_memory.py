"""Independent outcome evidence layered over unchanged TaskMemory.

Stage predictions and commands cannot confirm a physical result. No hardware I/O.
"""
from dataclasses import dataclass, asdict, replace
from copy import deepcopy
import math
from .tomato_task_memory import TaskMemory

EVENTS=('right_grasp_started','truss_stably_lifted','left_pick_started','tomato_detached','truss_returned','adjustment_ended','right_regrasp_started','second_grasp_verified')
KINDS=('observation','stage_prediction','executed_command','verified_operation_result')


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    session_id: str
    episode_id: str
    observation_id: str
    timestamp: float
    event_type: str
    kind: str
    source: str
    probability: float | None
    verdict: str = 'unknown'
    status: str = 'candidate'
    valid_until: float | None = None
    support_ids: tuple = ()
    reviewers: tuple = ()
    independent_label: bool = False
    qualified_verifier: bool = False
    reason: str = 'insufficient_evidence'
    version: int = 1

    def __post_init__(self):
        if self.kind not in KINDS or self.event_type not in EVENTS:raise ValueError('Unknown evidence kind/event')
        if not all(isinstance(x,str) and x for x in (self.evidence_id,self.session_id,self.episode_id,self.observation_id,self.source)):raise ValueError('Missing evidence identity')
        if not math.isfinite(self.timestamp) or self.valid_until is not None and (not math.isfinite(self.valid_until) or self.valid_until<self.timestamp):raise ValueError('Invalid evidence time')
        if self.probability is not None and (not math.isfinite(self.probability) or not 0<=self.probability<=1):raise ValueError('Invalid probability')
        if self.verdict not in ('positive','negative','unknown') or self.status not in ('candidate','confirmed','rejected','expired','uncertain'):raise ValueError('Invalid verdict/status')


class OperationMemory:
    """Compact evidence augmentation, never owns or overwrites B0 decision state."""
    protocol='tomato-operation-memory-v1'
    def __init__(self,session_id,episode_id,model_version,*,limit=256,trusted_verifiers=()):
        self.stage=TaskMemory(session_id,model_version,mode='M1')
        self.session_id,self.episode_id=session_id,episode_id
        self.trusted_verifiers=frozenset(trusted_verifiers)
        self.limit=limit;self.records={};self.confirmed={};self.conflicts=[];self.audit=[];self.version=0;self.clock=None

    def _note(self,event,reason):
        self.version+=1;self.audit.append(dict(version=self.version,evidence_id=event.evidence_id,status=event.status,reason=reason,time=self.clock))
        self.audit=self.audit[-self.limit:]

    def append(self,e,*,now):
        if (e.session_id!=self.session_id or e.episode_id!=self.episode_id or not math.isfinite(now)
            or e.timestamp>now or self.clock is not None and now<self.clock):raise ValueError('Identity/time mismatch')
        if e.evidence_id in self.records or e.evidence_id in self.confirmed:
            old=self.records.get(e.evidence_id,self.confirmed.get(e.evidence_id))
            if replace(old,status=e.status,version=e.version,reason=e.reason)!=e:raise ValueError('Conflicting duplicate evidence; submit explicit revision')
            return False
        if e.status not in ('candidate','uncertain'):raise ValueError('Append candidate first; confirmation is a separate validated operation')
        if len(self.records)>=self.limit:
            old_key=next((k for k,v in self.records.items() if v.status in ('rejected','expired')),None)
            if old_key is None:raise ValueError('Evidence capacity exhausted; explicit archive/reset required')
            del self.records[old_key]
        self.clock=now;self.records[e.evidence_id]=e;self._note(e,'append_evidence')
        if e.kind=='verified_operation_result' and e.independent_label and len(set(e.reviewers))>=2 and e.verdict=='negative':
            disputed=[v.evidence_id for v in self.confirmed.values() if v.event_type==e.event_type and v.timestamp==e.timestamp]
            if disputed:self.conflicts.append(dict(event_type=e.event_type,ids=[e.evidence_id]+disputed,time=now));self.conflicts=self.conflicts[-self.limit:]
        return True

    def update_evidence(self,key,*,now,**changes):
        if set(changes)-{'probability','verdict','reviewers','independent_label','qualified_verifier','support_ids','reason','valid_until'}:raise ValueError('Immutable evidence identity')
        e=self.records[key]
        if e.status not in ('candidate','uncertain'):raise ValueError('Do not rewrite terminal evidence')
        if now<max(e.timestamp,self.clock or now):raise ValueError('Out of order revision')
        self.clock=now;self.records[key]=replace(e,**changes,version=e.version+1);self._note(self.records[key],'update_evidence')

    def confirm(self,key,*,now):
        e=self.records[key]
        if now<max(e.timestamp,self.clock or now) or e.valid_until is not None and now>e.valid_until:raise ValueError('Expired/out of order evidence')
        if e.status not in ('candidate','uncertain'):raise ValueError('Duplicate confirmation')
        human=e.independent_label and len(set(e.reviewers))>=2 and e.source=='independent_operation_review'
        trained=e.qualified_verifier and e.source in self.trusted_verifiers and e.source.startswith('outcome_verifier:') and e.probability is not None and e.probability>=.9 and len(set(e.support_ids))>=2
        if e.kind!='verified_operation_result' or e.verdict!='positive' or not (human or trained):raise ValueError('Independent physical-result evidence required, never stage/command proxy')
        self.clock=now
        conflicts=[v for v in self.records.values() if v.kind=='verified_operation_result' and v.independent_label and len(set(v.reviewers))>=2 and v.evidence_id!=key and v.event_type==e.event_type and v.timestamp==e.timestamp and v.verdict!=e.verdict and v.verdict!='unknown']
        if conflicts:
            self.conflicts.append(dict(event_type=e.event_type,ids=[key]+[v.evidence_id for v in conflicts],time=now));self.conflicts=self.conflicts[-self.limit:];self.records[key]=replace(e,status='uncertain',reason='contradictory_result_evidence',version=e.version+1);self._note(self.records[key],'conflict_hold_unknown');return False
        self.records[key]=replace(e,status='confirmed',reason='independent_result_evidence',version=e.version+1);self.confirmed[key]=self.records[key];self._note(self.records[key],'confirmed_operation_result');return True

    def reject(self,key,*,now,reason='review_rejected'):
        return self._terminal(key,'rejected',now,reason)

    def _terminal(self,key,status,now,reason):
        e=self.records[key]
        if e.status not in ('candidate','uncertain') or now<max(e.timestamp,self.clock or now):raise ValueError('Immutable terminal/out of order evidence')
        self.clock=now;self.records[key]=replace(e,status=status,reason=reason,version=e.version+1);self._note(self.records[key],reason)

    def expire(self,now):
        if not math.isfinite(now) or self.clock is not None and now<self.clock:raise ValueError('Invalid clock')
        self.clock=now
        for key,e in list(self.records.items()):
            if e.status in ('candidate','uncertain') and e.valid_until is not None and now>e.valid_until:self._terminal(key,'expired',now,'evidence_expired')

    def snapshot(self,*,now):
        if self.clock is not None and now<self.clock:raise ValueError('Snapshot cannot view future memory from the past')
        disputed={i for c in self.conflicts for i in c['ids']}
        facts=[dict(**asdict(e),disputed=e.evidence_id in disputed) for e in self.confirmed.values() if e.timestamp<=now]
        state_candidates=[e for e in facts if not e['disputed'] and e['valid_until'] is not None and now<=e['valid_until'] and e['event_type'] in ('truss_stably_lifted','second_grasp_verified','truss_returned')]
        latest=max(state_candidates,key=lambda e:e['timestamp']) if state_candidates else None
        object_state=dict(truss_support='plate' if latest['event_type']=='truss_returned' else 'right_hand',evidence_id=latest['evidence_id'],evidence_timestamp=latest['timestamp'],expires_at=latest['valid_until']) if latest else 'unknown'
        commands=[asdict(e) for e in self.records.values() if e.kind=='executed_command']
        active=[asdict(e) for e in self.records.values() if e.status in ('candidate','uncertain') and (e.valid_until is None or now<=e.valid_until)]
        return deepcopy(dict(protocol=self.protocol,session_id=self.session_id,episode_id=self.episode_id,observation_id=self.stage.short[-1].get('observation_id') if self.stage.short else None,observation_timestamp=self.stage.last_valid,confirmed_phase=self.stage.phase,current_probabilities=self.stage.short[-1].get('current') if self.stage.short else None,command_history=commands,command_history_status='externally_reported_only' if commands else 'not_observed',transition_events=self.stage.confirmed,verified_results=facts,unconfirmed_results=active,object_state=object_state,operation_progress={e:('verified' if any(f['event_type']==e and not f['disputed'] for f in facts) else 'unknown') for e in EVENTS},picked_count=None,progress=self.stage.view()['progress'],conflicts=self.conflicts[-32:],memory_version=self.version,last_update_reason=self.audit[-1]['reason'] if self.audit else 'initialized',suggests_prompt_update=False))

    def reset(self,session_id,episode_id,model_version):
        if session_id==self.session_id:raise ValueError('Reset requires a new session identity')
        self.__init__(session_id,episode_id,model_version,limit=self.limit,trusted_verifiers=self.trusted_verifiers)


def public_memory(memory,*,observation_id,timestamp,valid_for_s=1.,mode='predicted',now=None):
    if memory.stage.last_valid is not None and memory.stage.last_valid>timestamp:raise ValueError('Past-observation export would leak later task state')
    snap=memory.snapshot(now=timestamp if now is None else now)
    # Canonical facts retain provenance. Historical operation truth does not
    # expire merely because the latest observation does; current object state is unknown.
    facts=[dict(event_id=e['evidence_id'],event_type=e['event_type'],event_time=e['timestamp'],evidence_time=e['timestamp'],source=e['source'],independent=True) for e in snap['verified_results'] if not e['disputed'] and e['timestamp']<=timestamp]
    object_state=snap['object_state']
    if isinstance(object_state,dict) and object_state['evidence_timestamp']>timestamp:object_state='unknown'
    return dict(protocol='tomato-memory-input-v1',mode=mode,session_id=memory.session_id,episode_id=memory.episode_id,observation_id=observation_id,input_timestamp=timestamp,valid_until=timestamp+valid_for_s,memory_version=memory.version,phase_index=memory.stage.phase,verified_facts=facts,uncertain=['operation_results_unknown'] if not facts else [],picked_count=None,object_state=object_state,current_goal=memory.stage.view()['confirmed_phase_name'])



def oracle_memory_at(labels,*,session_id,episode_id,source_recording_id,observation_id,timestamp,fps,phase_index):
    """Offline Oracle of independent *past* outcomes, not a seven-phase Oracle.

    phase_index must come from the online B0 decision, exactly as predicted mode.
    Human results are released only at evidence_end_frame, never at segment onset.
    """
    from .tomato_outcome_verifier import validate_label
    facts=[]
    for row in labels:
        validate_label(row)
        if row['episode_id']!=episode_id:continue
        if row['source_recording_id']!=source_recording_id:raise ValueError('Oracle source mismatch')
        if row['verdict']=='positive' and row['evidence_end_frame']/fps<=timestamp:
            facts.append(dict(event_id=row['case_id'],event_type=row['event_type'],event_time=row['start_frame']/fps,evidence_time=row['evidence_end_frame']/fps,source='independent_operation_review',independent=True))
    return dict(protocol='tomato-memory-input-v1',mode='oracle',session_id=session_id,episode_id=episode_id,observation_id=observation_id,input_timestamp=timestamp,valid_until=timestamp+1.,memory_version=len(facts),phase_index=phase_index,verified_facts=facts,uncertain=['operation_results_unknown'] if not facts else [],picked_count=None,object_state='unknown')
