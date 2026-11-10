"""Bounded event-sourced task memory. No encoder, GT, robot or Prompt write API.

Recognition, decision, command and physical outcome are distinct event scopes.
Only the first two have evidence in the recorded classifier replay.
"""
from collections import deque
from dataclasses import asdict, dataclass, field
import math
from copy import deepcopy
import numpy as np
from .tomato_state_recovery import PHASES, CLASSES

LABELS = ('pickup', 'harvest', 'return', 'adjust')
STATUSES = ('candidate', 'confirmed', 'rejected', 'expired', 'uncertain')


@dataclass(frozen=True)
class Observation:
    session_id: str
    sequence: int
    observation_id: str
    timestamp: float
    arrival_timestamp: float
    current: tuple
    future: tuple
    model_version: str
    labels: tuple = LABELS
    positions: tuple | None = None
    state_timestamp: float | None = None
    joint_names: tuple | None = None


@dataclass
class EventRecord:
    event_id: str
    event_type: str
    event_time: float
    updated_at: float
    from_phase: int | None
    to_phase: int | None
    prediction_source: str
    confidence: float | None
    support_observation_ids: list
    status: str
    reason: str
    version: int
    scope: str = 'decision'
    physical_success: str = 'unobserved'
    confirmation_time: float | None = None
    details: dict = field(default_factory=dict)
    revisions: list = field(default_factory=list)


class TaskMemory:
    schema_version = 'tomato-task-memory-v1'

    def __init__(self, session_id, model_version, *, mode='M1', short_size=16,
                 long_size=256, joint_names=None, sink=None, generation=0):
        if mode not in ('M1','M2') or not session_id or not model_version:
            raise ValueError('Explicit session/model identity and memory mode required')
        if short_size < 2 or long_size < 16:
            raise ValueError('Insufficient bounded memory capacity')
        if joint_names is not None and (len(joint_names)!=58 or len(set(joint_names))!=58):
            raise ValueError('Require exact duplicate-free 58D joint schema')
        self.session_id,self.model_version,self.mode=session_id,model_version,mode
        self.generation=generation
        if joint_names is not None and sum('fr3' in n for n in joint_names)!=14:
            raise ValueError('Named state must identify 14 arm and 44 hand joints')
        self.joint_names=tuple(joint_names) if joint_names is not None else None
        self.short=deque(maxlen=short_size);self.events=deque(maxlen=long_size)
        self.confirmed=[]  # At most six ordered transition records, independent of FIFO audit.
        self.pending=None;self.advisory=None
        self.phase=0;self.version=0;self.serial=0;self.last_valid=None
        self.last_sequence=-1;self.clock=None;self.count=0;self.recognized=None
        self.flags=set();self.conflict_run=0;self.clear_run=0;self.last_q=None
        self.sink=sink;self.motion=None;self.rejected_inputs=0
        self._new('task_initialized',0.,None,0,'confirmed','known_start_from_task_protocol',[],None,'protocol')
        self._new('phase_enter',0.,None,0,'confirmed','initial_position_not_action_success',[],None,'protocol')

    def reset(self, session_id, model_version=None):
        old_id=self.session_id
        self._new('task_reset',self.clock or 0.,self.phase,None,'confirmed','explicit_reset',[],None,'protocol')
        self.__init__(session_id,model_version or self.model_version,mode=self.mode,
                      short_size=self.short.maxlen,long_size=self.events.maxlen,
                      joint_names=self.joint_names,sink=self.sink,generation=self.generation+1)
        return dict(previous_session=old_id,session_id=session_id)

    def _publish(self,event):
        if self.sink is not None:self.sink(asdict(event))

    def _new(self,kind,t,previous,dest,status,reason,ids,confidence,source,scope='decision',details=None):
        self.version+=1;self.serial+=1
        e=EventRecord(f'{self.session_id}:g{self.generation}:e{self.serial}',kind,float(t),float(self.clock if self.clock is not None else t),previous,dest,source,confidence,list(ids),status,reason,self.version,scope)
        e.details=dict(details or {})
        e.revisions.append(dict(status=status,version=self.version,reason=reason,time=e.updated_at))
        self.events.append(e);self._publish(e);return e

    def _status(self,e,status,reason):
        if e.status not in ('candidate','uncertain'):
            raise ValueError('Terminal event cannot be rewritten')
        self.version+=1;e.status=status;e.version=self.version;e.updated_at=float(self.clock)
        e.reason=reason;e.revisions.append(dict(status=status,version=e.version,reason=reason,time=e.updated_at));self._publish(e)

    def poll(self,now):
        """Missing-observation tick expires evidence, never advances task position."""
        if not math.isfinite(now) or self.clock is not None and now<self.clock:
            return False
        self.clock=float(now)
        for name in ('pending','advisory'):
            event=getattr(self,name)
            if event is not None and now-event.event_time>1.5:
                self._status(event,'expired','candidate_ttl_exceeded');setattr(self,name,None)
                if name=='pending':self.count=0
        if self.last_valid is not None and now-self.last_valid>.75:
            self.flags.add('observation_gap');self.count=0
        return True

    def _validate(self,o):
        try:
            if o.session_id!=self.session_id or o.model_version!=self.model_version:return 'identity_mismatch'
            if type(o.sequence) is not int or o.sequence<=self.last_sequence:return 'duplicate_or_old_sequence'
            if o.observation_id!=f'{self.session_id}:{o.sequence}':return 'noncanonical_observation_id'
            if tuple(o.labels)!=LABELS:return 'probability_order_mismatch'
            if not np.isfinite([o.timestamp,o.arrival_timestamp]).all():return 'nonfinite_timestamp'
            if not 0<=o.arrival_timestamp-o.timestamp<=1.:return 'stale_or_future_dated'
            if self.last_valid is not None and o.timestamp<=self.last_valid:return 'out_of_order_timestamp'
            if self.clock is not None and o.arrival_timestamp<self.clock:return 'arrival_clock_reversal'
            for p in (o.current,o.future):
                a=np.asarray(p,float)
                if a.shape!=(4,) or not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-4:return 'invalid_probabilities'
            if o.positions is not None:
                q=np.asarray(o.positions,float)
                if self.joint_names is None or tuple(o.joint_names or ())!=self.joint_names:return 'joint_schema_mismatch'
                if q.shape!=(58,) or not np.isfinite(q).all():return 'invalid_state'
                if o.state_timestamp is None or not np.isfinite(o.state_timestamp) or not 0<=o.timestamp-o.state_timestamp<=.5:return 'state_not_causal_or_stale'
        except (ValueError,TypeError,AttributeError):return 'invalid_envelope'
        return None

    def _reject(self,o,reason):
        # Wrong-session messages cannot expire or rewrite this session's memory.
        if reason!='identity_mismatch':
            now=getattr(o,'arrival_timestamp',None)
            if isinstance(now,(float,int)) and math.isfinite(now):self.poll(max(now,self.clock or now))
            if self.pending is not None:
                self._status(self.pending,'expired','invalid_observation_breaks_confirmation');self.pending=None
            self.count=0;self.flags.add(reason)
        self.rejected_inputs+=1
        self._new('observation_rejected',self.clock or 0.,self.phase,None,'rejected',reason,[],None,'input_validator','transport')
        stamp=getattr(o,'timestamp',None)
        stamp=float(stamp) if isinstance(stamp,(int,float)) and math.isfinite(stamp) else None
        self.short.append(dict(valid=False,reason=reason,timestamp=stamp))
        return self.view(valid=False)

    def ledger_phase(self):
        phase=0
        for e in self.confirmed:
            if e['from_phase']!=phase or e['to_phase']!=phase+1 or e['status']!='confirmed':
                raise RuntimeError('Corrupted confirmed event prefix; never silently resynchronize')
            phase=e['to_phase']
        return phase

    def _candidate(self,o,c):
        if self.ledger_phase()!=self.phase:
            raise RuntimeError('Phase/event ledger divergence')
        target=self.ledger_phase()+1
        qualifies=target<7 and int(c.argmax())==CLASSES[target] and c[CLASSES[target]]>=.8
        if not qualifies:
            if self.pending is not None:
                self._status(self.pending,'rejected','current_evidence_not_continuous');self.pending=None
            self.count=0;return False
        if self.pending is None:
            self.pending=self._new('transition',o.timestamp,self.phase,target,'candidate','current_evidence_started',[o.observation_id],float(c.max()),self.model_version)
            self.count=1
        else:
            self.pending.support_observation_ids.append(o.observation_id)
            self.pending.support_observation_ids=self.pending.support_observation_ids[-8:]
            self.pending.confidence=min(self.pending.confidence,float(c.max()));self.count+=1
        return self.count>=2

    def _commit(self,dest,o,source):
        if dest!=self.phase+1:raise ValueError('No unobserved jumps or repeated-class reset')
        if self.pending is None:
            # M1 records what B0 decided even when its streak spans a gap.
            self.pending=self._new('transition',o.timestamp,self.phase,dest,'candidate','external_decision_without_local_streak',[o.observation_id],float(max(o.current)),source)
            self.flags.add('external_confirmation_without_fresh_local_streak')
        event=self.pending;event.prediction_source=source;event.confirmation_time=float(o.timestamp);self._status(event,'confirmed','b0_decision_observed' if self.mode=='M1' else 'fresh_current_k2_confirmed')
        event.prediction_source=source
        # Confirmation time is the latest evidence timestamp, not candidate onset.
        self._new('phase_exit',o.timestamp,self.phase,dest,'confirmed','decision_exit_not_physical_completion',event.support_observation_ids,event.confidence,source)
        self.phase=dest;self.confirmed.append(asdict(event));self.pending=None;self.count=0
        self._new('phase_enter',o.timestamp,dest-1,dest,'confirmed','ordered_decision_enter',event.support_observation_ids,event.confidence,source)
        if self.advisory is not None:
            self._status(self.advisory,'expired','stage_context_changed');self.advisory=None

    def observe(self,o,*,b0_phase=None):
        """b0_phase is an immutable decision snapshot for M1; never a GT position."""
        reason=self._validate(o)
        if reason:return self._reject(o,reason)
        self.poll(o.arrival_timestamp)
        if self.last_valid is not None and o.timestamp-self.last_valid>.75 and self.pending is not None:
            self._status(self.pending,'expired','fresh_observation_gap');self.pending=None;self.count=0
        c=np.array(o.current,float,copy=True);f=np.array(o.future,float,copy=True)
        # Transport alarms describe the current stream; their immutable events
        # remain in the audit after healthy observations resume.
        self.flags.intersection_update({'history_current_conflict','external_confirmation_without_fresh_local_streak','invalid_b0_snapshot'})
        self.motion=None
        if o.positions is not None:
            q=np.asarray(o.positions,float)
            if self.last_q is not None and o.state_timestamp>self.last_q[0] and o.state_timestamp-self.last_q[0]<=.75:
                v=(q-self.last_q[1])/(o.state_timestamp-self.last_q[0]);arms=np.array(['fr3' in n for n in self.joint_names])
                self.motion=dict(interval_s=float(o.state_timestamp-self.last_q[0]),arm_velocity_rms=float(np.sqrt(np.mean(v[arms]**2))),hand_velocity_rms=float(np.sqrt(np.mean(v[~arms]**2))),source='causal_observed_interval_difference_not_success')
            self.last_q=(o.state_timestamp,q.copy())
        self.short.append(dict(observation_id=o.observation_id,timestamp=o.timestamp,arrival_timestamp=o.arrival_timestamp,age_s=o.arrival_timestamp-o.timestamp,valid=True,current=c.tolist(),future=f.tolist(),motion=self.motion))
        self.last_valid=float(o.timestamp);self.last_sequence=o.sequence
        k=int(c.argmax())
        if c[k]<.8:self.flags.add('uncertain_current')
        elif k!=self.recognized:
            self._new('model_stage_recognized',o.timestamp,None,None,'confirmed','four_class_prediction_only',[o.observation_id],float(c[k]),self.model_version,'recognition',dict(class_id=k,class_name=LABELS[k]));self.recognized=k
        ready=self._candidate(o,c)
        if self.mode=='M1':
            if type(b0_phase) is not int or b0_phase not in (self.phase,min(6,self.phase+1)):
                self.flags.add('invalid_b0_snapshot')
            elif b0_phase>self.phase:self._commit(b0_phase,o,'B0_current_08_k2')
        elif ready:self._commit(self.phase+1,o,'memory_fresh_current_08_k2')
        incompatible=c[k]>=.8 and k!=CLASSES[self.phase] and (self.phase==6 or k!=CLASSES[self.phase+1])
        self.conflict_run=self.conflict_run+1 if incompatible else 0
        self.clear_run=self.clear_run+1 if not incompatible and c[k]>=.8 else 0
        if self.conflict_run==2:
            self.flags.add('history_current_conflict')
            self._new('phase_conflict',o.timestamp,self.phase,None,'uncertain','strong_class_conflicts_with_ordered_history',[row['observation_id'] for row in list(self.short)[-2:] if row.get('valid')],float(c[k]),self.model_version)
        if self.clear_run==2 and 'history_current_conflict' in self.flags:
            self.flags.remove('history_current_conflict');self._new('conflict_resolved',o.timestamp,self.phase,None,'confirmed','compatible_observations_returned_history_preserved',[o.observation_id],float(c[k]),self.model_version)
        target=self.phase+1
        future_sign=target<7 and int(f.argmax())==CLASSES[target] and f[CLASSES[target]]>=.5
        if future_sign and self.advisory is None:
            self.advisory=self._new('future_transition_advisory',o.timestamp,self.phase,target,'candidate','future_prediction_never_current_fact',[o.observation_id],float(f.max()),self.model_version,'prediction')
        elif not future_sign and self.advisory is not None:
            self._status(self.advisory,'rejected','future_evidence_withdrawn');self.advisory=None
        return self.view()

    def view(self,valid=True):
        return dict(schema_version=self.schema_version,session_id=self.session_id,model_version=self.model_version,valid=valid,memory_version=self.version,confirmed_phase=self.phase,confirmed_phase_name=PHASES[self.phase],candidate_phase=self.pending.to_phase if self.pending else None,pending_event_id=self.pending.event_id if self.pending else None,current_candidate=dict(event_id=self.pending.event_id,status=self.pending.status,confidence=self.pending.confidence,evidence_time=self.pending.event_time,support_observation_ids=list(self.pending.support_observation_ids)) if self.pending else None,future_candidate=self.advisory.to_phase if self.advisory else None,last_valid_timestamp=self.last_valid,progress=dict(ordered_transition_count=len(self.confirmed),decision_fraction=self.phase/6.,physical_completion='unknown',task_complete=False),uncertainty=sorted(self.flags),suggests_prompt_update=False,mode=self.mode,short_samples=len(self.short),recent_events=len(self.events),confirmed_event_ids=[e['event_id'] for e in self.confirmed],history_entered_positions=[0]+[e['to_phase'] for e in self.confirmed],motion=deepcopy(self.motion))

    def snapshot(self):
        return deepcopy(dict(**self.view(),short_term=list(self.short),long_term=[asdict(e) for e in self.events],confirmed_transitions=list(self.confirmed),pending=asdict(self.pending) if self.pending else None,advisory=asdict(self.advisory) if self.advisory else None))
