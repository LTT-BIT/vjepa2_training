"""Causal queue. Driver alone knows duration; policy never receives GT or endpoint."""
import heapq
from .contracts import H0_CONFIG,TASK_POLICY
from .feature_builder import FeatureBuilder,observation
from .shell_adapter import ExecutionShell
from ..handshake import replay as original_replay


def h0(rows,duration):return original_replay(rows,H0_CONFIG,duration)


def result(core,duration,records):
 intervals=[];start=0.;stage='pickup';version=0;pending=[];opened=None
 for e in core.events:
  if e['kind']=='commit':intervals.append(dict(start_s=start,end_s=e['time'],stage=stage,prompt_version=version));start=e['time'];stage=e['active_stage'];version=e['prompt_version']
  if e['kind']=='propose':opened=dict(start_s=e['time'],stage=e['pending']['stage'])
  if e['kind'] in ['commit','cancel'] and opened:pending.append(dict(**opened,end_s=e['time']));opened=None
 intervals.append(dict(start_s=start,end_s=duration,stage=stage,prompt_version=version))
 if opened:pending.append(dict(**opened,end_s=duration))
 return dict(events=core.events,active_intervals=[i for i in intervals if i['end_s']>i['start_s']],pending_intervals=pending,records=records,config=H0_CONFIG,summary=dict(commits=core.version,acks=sum(e['kind']=='ack' for e in core.events),late_acks=sum(e['kind']=='late_ack' for e in core.events),timeouts=sum(e['kind']=='timeout' for e in core.events),hard_timeouts=sum(e['kind']=='hard_timeout' for e in core.events),state=core.state,external_recoveries=0,finish='not_supported'))


def run(rows,duration,policy,session='episode',task_policy=None,collect=True):
 task_policy=TASK_POLICY if task_policy is None else task_policy;pv=getattr(policy,'version',policy.name+'-v1');shell=ExecutionShell(task_policy);s=shell.core;builder=FeatureBuilder(session,'B0-seed42',pv);queue=[];records=[]
 for i,row in enumerate(rows):
  o=observation(row,session,'B0-seed42',pv);heapq.heappush(queue,(o.available_at,0,i,'prediction',o));heapq.heappush(queue,(round(o.available_at+.75000001,9),1,i,'timer',None))
 i=0
 while i*.5<duration:heapq.heappush(queue,(i*.5,2,i,'planning',None));i+=1
 while queue:
  t,_,n,kind,o=heapq.heappop(queue)
  if t>=duration:continue
  old=s.version
  if kind=='prediction':
   try:builder.observe(o)
   except ValueError as exc:s.emit('input_rejected',t,reason=str(exc));continue
   shell.ingest(o);features=builder.build(s,task_policy)
   try:proposal=policy.propose(o,features,s,task_policy)
   except Exception as exc:
    s.emit('policy_failure_HOLD',t,reason=str(exc));s.cancel(t,'policy_failure');continue
   if collect:records.append(dict(features=features,proposal=proposal,diagnostics=dict(getattr(policy,'diagnostics',{}))))
   shell.apply(proposal)
  elif kind=='planning':s.planning(t)
  else:s.timer(t)
  if old!=s.version:
   heapq.heappush(queue,(round(s.deadline,9),1,n+10000,'timer',None));heapq.heappush(queue,(round(s.hard_deadline,9),1,n+20000,'timer',None))
 return result(s,duration,records)
