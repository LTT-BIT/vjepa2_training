"""Prediction -> timer -> planning; no policy/feature input contains annotations."""
import heapq
from dataclasses import asdict
from ..handshake import Handshake,Evidence,replay as original_replay
from ..learned_handshake.replay import result as finish
from .contract import *
from .candidate import CandidateProvider
from .features import GateFeatures,make_observation
from .port import CandidateRequestPort


def reject(core,o,reason):
 # Only a bad message for this already bound stream can withdraw its evidence.
 if o.valid is False:
  core.cancel(o.available_at,'invalid_observation');core.late_count=0;core.late_last=None
 core.emit('input_rejected',o.available_at,reason=reason)


def run(rows,duration,policy=None,session='episode',allowed=None,collect=True,blackout_after_commit=False):
 allowed=TASK_POLICY if allowed is None else allowed;name='H0' if policy is None else policy.name;version='H0-original' if policy is None else policy.version
 core=Handshake(H0_CONFIG) if policy is None else None;port=CandidateRequestPort(allowed) if policy is not None else None;core=core if core is not None else port.core
 provider=CandidateProvider();features=GateFeatures(session,'B0-seed42',version);queue=[];records=[];first_blackout=None;blackout_end=None;last_frame=None;last_request_source=None;commit_audit=[]
 for i,row in enumerate(rows):
  o=make_observation(row,session,version)
  if not __import__('math').isfinite(o.available_at):raise ValueError('Nonfinite arrival clock')
  heapq.heappush(queue,(o.available_at,0,i,'prediction',o));heapq.heappush(queue,(round(o.available_at+.75000001,9),1,i,'timer',None))
 for i in range(int(duration/.5)+1):
  if i*.5<duration:heapq.heappush(queue,(i*.5,2,i,'planning',None))
 # A pure logging callback at the H0 candidate boundary captures pre-decision
 # own state while the default public prediction path and events stay intact.
 current={}
 default=core.propose_default_candidate
 def capture_default(m):
  current['feature']=features.snapshot(current['o'],core,current['candidate'],allowed)
  return default(m)
 if policy is None:core.propose_default_candidate=capture_default
 while queue:
  t,_,n,kind,o=heapq.heappop(queue)
  if t>=duration:continue
  before=core.version
  if kind=='prediction':
   if blackout_end is not None and t<blackout_end:continue
   try:features.observe(o)
   except (ValueError,TypeError) as exc:
    reject(core,o,str(exc));continue
   # receive_evidence handles ACK but never chooses a candidate for a gate policy.
   if policy is None:
    candidate=provider.update(o,core.active,allowed);current.update(o=o,candidate=candidate);current.pop('feature',None);core.prediction(Evidence(o.observed_at,o.available_at,o.current,o.future,o.valid))
    f=current.get('feature')
    if f is None:f=features.snapshot(o,core,candidate,allowed)
    decision=None;diagnostics={};reason=None
   else:
    port.ingest(o);candidate=provider.update(o,core.active,allowed);port.bind_candidate(candidate);f=features.snapshot(o,core,candidate,allowed)
    try:decision=policy.decide(o,f,candidate,core)
    except Exception as exc:
     core.emit('policy_failure_HOLD',t,reason=str(exc));decision=GateDecision(candidate.stage,'HOLD',0.,o.observation_id,o.observed_at,o.available_at,version)
    reason=port.apply_decision(decision,policy.name);diagnostics=dict(getattr(policy,'diagnostics',{}))
   if core.state=='SWITCH_REQUESTED':last_request_source=dict(session_id=o.session_id,observation_id=o.observation_id,observed_at=o.observed_at,available_at=o.available_at)
   if collect:records.append(dict(feature=f,candidate=asdict(candidate),decision=asdict(decision) if decision else None,diagnostics=diagnostics,blocked_reason=reason))
  elif kind=='timer':core.timer(t)
  elif port is not None:port.planning_tick(t)
  else:core.planning(t)
  if before!=core.version:
   commit_audit.append(dict(time=t,expected_session=session,request_source=last_request_source,version=core.version,planning_tick=kind=='planning',illegal=core.active not in allowed.get(core.events[-1]['from_stage'],[])))
   heapq.heappush(queue,(round(core.deadline,9),1,n+100000,'timer',None));heapq.heappush(queue,(round(core.hard_deadline,9),1,n+200000,'timer',None))
   if blackout_after_commit and first_blackout is None:first_blackout=t;blackout_end=t+5.5
 r=finish(core,duration,records)
 if policy is None:r['summary']['late_ack_required_support']=2
 r['commit_audit']=commit_audit;r['task_policy']=allowed;r['stress_blackout_first_commit']=first_blackout;return r


def pure_h0(rows,duration):return original_replay(rows,H0_CONFIG,duration)
