"""Experimental visual/joint/event cooperation over the verified B0 ledger.

No online integration, policy calls, robot I/O or inferred physical success.
Historical seven-position experiment; shared in-repository implementation.
Use tools.tomato_rgb_memory for the current state + local RGB research path.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from tools.tomato_operation_memory import OperationMemory
from tools.tomato_task_memory import Observation
from tools.tomato_state_recovery import B0State, PHASES

GOALS = (
    'acquire_truss', 'detach_fruit_and_transfer', 'place_and_release_truss',
    'reposition_on_plate', 'reacquire_after_adjustment',
    'detach_fruit_and_transfer_again', 'place_remaining_object',
)
EXPECTED = (
    'truss_stably_lifted', 'tomato_detached', 'truss_returned',
    'adjustment_ended', 'second_grasp_verified', 'tomato_detached', 'truss_returned',
)
CONTEXT = ((0,), (0,1), (1,2), (2,3), (2,3,4), (4,5), (5,6))


@dataclass(frozen=True)
class Frame:
    session_id: str
    episode_id: str
    observation_id: str
    timestamp: float
    arrival_timestamp: float
    visual_timestamp: float
    state_timestamp: float
    visual: object                 # [3,1024], same frozen encoder/pooling contract
    positions: object              # exact named58D measured positions
    current: object                # four-class probabilities
    future: object


class CooperativeMemory:
    """A bounded feature reader plus trusted event ledger; B0 remains untouched.

    Different goals can retrieve different context for identical open-hand
    observations. No hand-openness classifier or success heuristic is invented.
    """
    protocol = 'tomato-cooperative-memory-experiment-v1'

    def __init__(self, session_id, episode_id, model_version, joint_names,
                 encoder_signature, *, recent_size=16, anchor_size=12,
                 trusted_verifiers=()):
        names = tuple(joint_names)
        if (len(names) != 58 or len(set(names)) != 58
                or sum('fr3' in n for n in names) != 14):
            raise ValueError('Exact named14 arm +44 hand schema required')
        if not all(isinstance(x,str) and x for x in (session_id,episode_id,model_version,encoder_signature)):
            raise ValueError('Explicit session/episode/model/encoder identities required')
        if recent_size < 2 or anchor_size < 7:
            raise ValueError('Insufficient bounded context capacity')
        self.session_id, self.episode_id = session_id, episode_id
        self.model_version, self.encoder_signature = model_version, encoder_signature
        self.joint_names = names
        self.recent_size, self.anchor_size = recent_size, anchor_size
        self.recent = deque(maxlen=recent_size)
        self.anchors = deque(maxlen=anchor_size)
        self.ledger = OperationMemory(session_id, episode_id, model_version,
                                      trusted_verifiers=trusted_verifiers)
        self.b0 = B0State()
        self.sequence = 0
        self.last = None
        self.outcomes = {}          # key -> online goal at evidence time, not GT position
        self._verifiers = tuple(trusted_verifiers)

    def observe(self, frame, *, encoder_signature):
        if (frame.session_id != self.session_id or frame.episode_id != self.episode_id
                or encoder_signature != self.encoder_signature):
            raise ValueError('Cross-session/episode/encoder input rejected')
        times = (frame.timestamp, frame.arrival_timestamp, frame.visual_timestamp, frame.state_timestamp)
        if (not all(math.isfinite(t) for t in times)
                or not 0 <= frame.arrival_timestamp-frame.timestamp <= 1.
                or not 0 <= frame.timestamp-frame.visual_timestamp <= .05
                or not 0 <= frame.timestamp-frame.state_timestamp <= .05
                or self.ledger.clock is not None and frame.arrival_timestamp < self.ledger.clock
                or self.last is not None and frame.timestamp <= self.last):
            raise ValueError('Fresh aligned causal observations required')
        v = np.asarray(frame.visual, np.float32)
        q = np.asarray(frame.positions, np.float32)
        c, f = (np.asarray(p,np.float64) for p in (frame.current,frame.future))
        if v.shape != (3,1024) or q.shape != (58,) or not np.isfinite(v).all() or not np.isfinite(q).all():
            raise ValueError('Expected finite three-view1024D + named58D state')
        if any(p.shape!=(4,) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-4 for p in (c,f)):
            raise ValueError('Current/Future four-class distributions required')
        if not frame.observation_id or any(r['observation_id']==frame.observation_id for r in self.recent):
            raise ValueError('Unique observation identity required')
        gap = self.last is not None and frame.timestamp-self.last > .75
        if gap:
            self.recent.clear()     # Never represent stale features as fresh motion.
        previous = self.recent[-1] if self.recent else None
        velocity = ((q-previous['positions'])/(frame.timestamp-previous['time'])
                    if previous is not None else None)
        decision = self.b0.update(c,f,frame.timestamp)  # Existing B0 rules verbatim.
        row = dict(time=float(frame.timestamp),observation_id=frame.observation_id,
                   phase=decision['phase'],visual=v.copy(),positions=q.copy(),
                   velocity=None if velocity is None else velocity.copy(),
                   current=c.copy(),future=f.copy())
        self.recent.append(row)
        if self.last is None or decision['changed']:
            self.anchors.append(dict(kind='stage_decision',goal_phase=decision['phase'],
                                     physical_success='unknown',frame=deepcopy(row)))
        self.ledger.stage.observe(Observation(
            self.session_id,self.sequence,frame.observation_id,frame.timestamp,
            frame.arrival_timestamp,tuple(c),tuple(f),self.model_version,
            positions=tuple(q),state_timestamp=frame.state_timestamp,joint_names=self.joint_names),
            b0_phase=decision['phase'])
        self.ledger.expire(frame.arrival_timestamp)
        self.sequence += 1
        self.last = frame.timestamp
        return dict(decision=deepcopy(decision),decision_unchanged=True,
                    physical_success_inferred=False,gap=gap)

    def add_result(self, evidence, *, available_at, now):
        """Only independent human/registered-verifier evidence can affect result status.

        Goal association uses the saved ONLINE position at evidence time. It
        never accepts a human seven-position label or reuses a first-harvest
        outcome as proof of the second harvest.
        """
        # This experimental adapter uses first-confirmable time as the ledger
        # timestamp. It must not backdate a result to action/segment onset.
        if not math.isfinite(available_at) or available_at!=evidence.timestamp or self.last is None or available_at>self.last:
            raise ValueError('Evidence timestamp must be its available first-confirmable time')
        frames = [r for r in self.recent if r['time'] <= evidence.timestamp]
        if not frames or evidence.timestamp > self.last:
            raise ValueError('Evidence needs an available causal feature frame')
        human = (evidence.independent_label and evidence.source=='independent_operation_review'
                 and len(set(evidence.reviewers)) >= 2)
        trained = (evidence.source in self._verifiers and evidence.qualified_verifier
                   and evidence.source.startswith('outcome_verifier:')
                   and evidence.probability is not None and evidence.probability>=.9
                   and len(set(evidence.support_ids))>=2)
        if evidence.kind!='verified_operation_result' or not (human or trained):
            raise ValueError('Stage, commands and feature changes are not result evidence')
        if evidence.valid_until is not None and now>evidence.valid_until:
            raise ValueError('Expired evidence cannot update current goal')
        if evidence.evidence_id in self.outcomes:
            return False
        row = frames[-1]
        self.ledger.append(evidence,now=now)
        confirmed = self.ledger.confirm(evidence.evidence_id,now=now) if evidence.verdict=='positive' else False
        self.outcomes[evidence.evidence_id] = row['phase']
        self.outcomes = {k:v for k,v in self.outcomes.items()
                         if k in self.ledger.records or k in self.ledger.confirmed}
        self.anchors.append(dict(kind='independent_result',evidence_id=evidence.evidence_id,
                                 goal_phase=row['phase'],confirmed=confirmed,frame=deepcopy(row)))
        return confirmed

    def read(self, *, now):
        """Fresh multimodal tensor + goal-relevant anchors; advisory only."""
        if self.last is None:
            return dict(valid=False,reason='no_observation',suggests_prompt_update=False)
        if not math.isfinite(now) or not 0 <= now-self.last <= 1.:
            return dict(valid=False,reason='stale_or_future_query',suggests_prompt_update=False)
        phase = self.b0.phase
        snap = self.ledger.snapshot(now=now)
        disputed = {i for conflict in snap['conflicts'] for i in conflict['ids']}
        eligible = [r for key,r in self.ledger.records.items()
                    if self.outcomes.get(key)==phase and r.event_type==EXPECTED[phase]
                    and r.timestamp<=self.last and key not in disputed
                    and (r.valid_until is None or now<=r.valid_until)]
        good = any(r.status=='confirmed' and r.verdict=='positive' for r in eligible)
        failed = any(r.verdict=='negative' for r in eligible)
        status = 'unknown' if good and failed else 'verified' if good else 'failure_evidence' if failed else 'unknown'
        current = self.recent[-1]
        anchors = []
        for a in self.anchors:
            if a['goal_phase'] not in CONTEXT[phase]:
                continue
            a = deepcopy(a)
            a['historical_only'] = a['goal_phase'] != phase
            if a['kind']=='independent_result':
                e=self.ledger.records.get(a['evidence_id'])
                a['evidence_currently_valid'] = bool(e and e.evidence_id not in disputed
                    and (e.valid_until is None or now<=e.valid_until))
            anchors.append(a)
        differences = []
        for a in anchors:
            x,y = current['visual'],a['frame']['visual']
            denom = np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
            distance = np.divide((x*y).sum(1),denom,out=np.zeros(3),where=denom>0)
            differences.append(dict(kind=a['kind'],time=a['frame']['time'],
                                    visual_cosine_distance=[float(1-distance[i]) if denom[i]>0 else None for i in range(3)],
                                    meaning='uncalibrated feature change, not physical success'))
        return deepcopy(dict(protocol=self.protocol,valid=True,session_id=self.session_id,
            episode_id=self.episode_id,observation_id=current['observation_id'],timestamp=self.last,
            encoder_signature=self.encoder_signature,joint_names=self.joint_names,
            phase=phase,phase_name=PHASES[phase],goal=GOALS[phase],expected_result=EXPECTED[phase],
            result_status=status,object_state=snap['object_state'],picked_count=None,
            recent=list(self.recent),context_phases=CONTEXT[phase],goal_anchors=anchors,anchor_comparison=differences,
            guidance='request_transition_review' if status=='verified' else 'request_retry_review' if status=='failure_evidence' else 'need_result_evidence',
            guidance_is_advisory=True,hand_open_semantics='not inferred from posture alone',
            suggests_prompt_update=False,robot_action=None,learned_memory_reader=False))

    def reset(self, session_id, episode_id):
        if session_id==self.session_id:
            raise ValueError('New session identity required')
        self.__init__(session_id,episode_id,self.model_version,self.joint_names,self.encoder_signature,
                      recent_size=self.recent_size,anchor_size=self.anchor_size,trusted_verifiers=self._verifiers)
