"""Pure online state machine for an offline experiment. No ground-truth fields."""
from dataclasses import dataclass, asdict
import math

LABELS = ('pickup', 'harvest', 'return', 'adjust')
EDGES = dict(zip(LABELS, LABELS[1:] + LABELS[:1]))


@dataclass(frozen=True, slots=True)
class PolicyInput:
    episode_id: str
    source_recording_id: str
    anchor_frame: int
    observed_at: float
    available_at: float
    model_name: str
    future_probabilities: tuple
    joint_current_probabilities: tuple | None = None
    horizon_s: float = 1.

    def __post_init__(self):
        for p in [self.future_probabilities, self.joint_current_probabilities]:
            if p is not None and (len(p) != 4 or any(not math.isfinite(v) or v < 0 for v in p)
                                  or abs(sum(p)-1) > 1e-6):
                raise ValueError('Invalid four-class probabilities')
        if not all(math.isfinite(v) for v in [self.observed_at,self.available_at,self.horizon_s]):
            raise ValueError('Nonfinite time')
        if self.available_at < self.observed_at or self.horizon_s < 0:
            raise ValueError('Noncausal prediction')


class PromptPolicy:
    def __init__(self, config):
        self.cfg = dict(config)
        if config['id'] not in ['P0','P1','P2','P3','P4']:
            raise ValueError('Unknown policy')
        if not (0 <= config.get('theta',0) <= 1 and config.get('K',1) in [1,2]
                and config.get('d',0) >= 0 and config.get('prediction_ttl_s',.75) > 0):
            raise ValueError('Invalid policy parameters')
        self.reset(config.get('initial_prompt','pickup'))

    def reset(self, initial_prompt):
        if initial_prompt not in LABELS: raise ValueError('Unsupported initial prompt')
        self.active = dict(stage=initial_prompt,created_at=0.,last_confirmed_at=0.,
                           source_anchor=None,source_target_time=None,reason='startup_prior')
        self.pending=None;self.candidate=None;self.count=0;self.seen=set();self.last=None;self.confirmation_started=None

    def _event(self, kind, now, reason, **extra):
        return dict(kind=kind,time=round(now,9),reason=reason,active=dict(self.active),
                    pending=dict(self.pending) if self.pending else None,**extra)

    def _cancel(self, now, reason):
        events=[]
        if self.pending:
            canceled=dict(self.pending);self.pending=None
            events.append(self._event('cancel',now,reason,canceled=canceled))
        self.pending=None;self.candidate=None;self.count=0;self.confirmation_started=None
        return events

    def on_prediction(self, msg):
        now=msg.available_at;key=(msg.source_recording_id,msg.episode_id,msg.anchor_frame)
        if key in self.seen: raise ValueError('Duplicate anchor cannot confirm twice')
        if self.last and (msg.observed_at <= self.last.observed_at or msg.anchor_frame <= self.last.anchor_frame):
            raise ValueError('Out-of-order anchor')
        self.seen.add(key)
        expired=self.on_timer(now)
        self.last=msg
        p=msg.future_probabilities;idx=max(range(4),key=lambda i:p[i]);stage=LABELS[idx]
        events=expired+[self._event('prediction',now,'new_anchor',input=asdict(msg),candidate=stage,
                        source_target_time=msg.observed_at+msg.horizon_s)]
        if stage == self.active['stage']:
            self.active['last_confirmed_at']=now
            return events+self._cancel(now,'candidate_equals_active')
        if self.cfg.get('current_gate'):
            q=msg.joint_current_probabilities
            if q is None: raise ValueError('Current gate requires actual Joint Current probabilities')
            if LABELS[max(range(4),key=lambda i:q[i])] != self.active['stage']:
                return events+self._cancel(now,'current_disagrees_with_active')+[self._event('gate_reject',now,'current_disagrees_with_active')]
        policy=int(self.cfg['id'][1:])
        if policy>=1 and p[idx]<self.cfg.get('theta',.8):
            return events+self._cancel(now,'confidence_withdrawn')
        if policy>=2 and p[idx]-p[LABELS.index(self.active['stage'])]<self.cfg.get('margin',0):
            return events+self._cancel(now,'margin_withdrawn')
        if policy>=4 and EDGES[self.active['stage']]!=stage:
            return events+self._cancel(now,'illegal_edge')+[self._event('illegal_edge',now,'illegal_edge',candidate=stage)]
        if self.candidate!=stage:
            events+=self._cancel(now,'candidate_changed')
            self.candidate=stage;self.count=0;self.confirmation_started=now;self.confirmation_observed_at=msg.observed_at
        self.count+=1
        needed=self.cfg.get('K',1) if policy>=2 else 1
        if self.count < needed:
            return events+[self._event('confirming',now,'insufficient_distinct_anchors',count=self.count,candidate=stage)]
        if self.pending:
            self.pending.update(last_confirmed_at=now,source_anchor=msg.anchor_frame,
                source_observed_at=msg.observed_at,source_available_at=now,
                source_target_time=msg.observed_at+msg.horizon_s)
            return events+[self._event('pending_confirm',now,'new_anchor_support')]
        delay=self.cfg.get('d',0) if policy>=3 else 0.
        self.pending=dict(stage=stage,created_at=now,last_confirmed_at=now,source_anchor=msg.anchor_frame,
            source_observed_at=msg.observed_at,source_available_at=now,
            source_target_time=msg.observed_at+msg.horizon_s,reason='confirmed_candidate',
            release_at=round(now+delay,9),request_observed_at=msg.observed_at,
            request_available_at=now,confirmation_start_at=self.confirmation_started,
            first_observed_at=self.confirmation_observed_at,first_available_at=self.confirmation_started)
        return events+[self._event('pending',now,'confirmed_candidate')]

    def on_timer(self, now):
        # Arrival at the exact TTL is accepted first; expiry is strict > TTL.
        if self.last and now-self.last.available_at > self.cfg.get('prediction_ttl_s',.75)+1e-9:
            events=self._cancel(now,'prediction_expired')
            self.last=None
            return events+[self._event('expire',now,'prediction_expired')]
        return []

    def on_planning_boundary(self, now):
        events=self.on_timer(now)
        if self.pending and now+1e-9>=self.pending['release_at']:
            p=dict(self.pending);old=self.active['stage'];self.pending=None
            self.active={k:p[k] for k in ['stage','created_at','last_confirmed_at','source_anchor','source_target_time','reason']}
            self.active['created_at']=now;self.active['reason']='planning_commit'
            self.candidate=None;self.count=0
            events.append(self._event('commit',now,'planning_commit',from_stage=old,to_stage=p['stage'],request=p))
        return events

    def finalize(self, now):
        return self._cancel(now,'input_stream_closed')+[self._event('stream_end',now,'data_stopped_not_FINISH')]
