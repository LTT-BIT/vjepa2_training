"""Same frozen event matcher; independent execution invariants and decomposition."""
import numpy as np
from ..learned_handshake.evaluate import score as old_score,aggregate,bootstrap,csv_rows
from .contract import *
from .targets import label


def score(e,r):
 result=old_score(e,r);events=r['events'];last=0.;longest=0.;begin=None
 for ev in events:
  blocked=ev['policy_state'] in ['WAIT_CURRENT_ACK','WAIT_LATE_ACK','RECOVERY_REQUIRED']
  if blocked and begin is None:begin=ev['time']
  if not blocked and begin is not None:longest=max(longest,ev['time']-begin);begin=None
 if begin is not None:longest=max(longest,e['length']/e['fps']-begin)
 commits=[v for v in events if v['kind']=='commit'];bad_clock=sum(abs(v['time']/.5-round(v['time']/.5))>1e-9 for v in commits);versions=[v['prompt_version'] for v in commits]
 if bad_clock or versions!=list(range(1,len(commits)+1)):raise ValueError('Execution protocol violated')
 cross=sum(a['request_source'] is None or a['request_source']['session_id']!=a['expected_session'] for a in r['commit_audit'])
 if cross:raise ValueError('Cross-session/missing request source commit')
 result.update(non_planning_commit=bad_clock,cross_session_commit=cross,longest_blocked_s=longest,gate_candidate_ticks=sum(x['candidate']['stage'] is not None for x in r['records']),request_eligible_ticks=sum(x['feature']['eligible'] for x in r['records']),gate_holds=sum(x.get('decision') is not None and x['candidate']['stage'] is not None and x['decision']['action']=='HOLD' for x in r['records']),candidate_absent=sum(x['candidate']['stage'] is None for x in r['records']))
 return result


def coverage(episodes,rollouts):
 result=[]
 for e in episodes:
  m=e['metadata'];fps=m['fps'];segments=m['annotation']['segments'];rows=e['rows'];records=rollouts[m['episode_id']]['records']
  for j,(a,b) in enumerate(zip(segments,segments[1:])):
   boundary=b['start_frame']/fps;oracle=[];own=[];blocked=[]
   for row in rows:
    available=round(row['observed_at']+.09,9);tick=round(np.ceil((available-1e-9)/.5)*.5,9);f=max(LABELS,key=lambda s:row['heads']['future']['probabilities'][s]);prob=row['heads']['future']['probabilities'][f]
    if -.2-1e-9<=tick-boundary<=.5+1e-9 and f==b['label'] and f in TASK_POLICY.get(a['label'],[]) and prob>=.65:oracle.append(dict(observed_at=row['observed_at'],available_at=available,tick=tick,probability=prob))
   for rec in records:
    f=rec['feature'];target=label(m,f)
    if target['event']==j and f['candidate']==b['label'] and target['delta_s'] is not None and -.5-1e-9<=target['delta_s']<=.2+1e-9:
     (own if f['eligible'] else blocked).append(f['available_at'])
   result.append(dict(episode_id=m['episode_id'],source_recording_id=m['source_recording_id'],boundary_index=j,from_stage=a['label'],to_stage=b['label'],boundary_time=boundary,oracle_candidates=oracle,oracle_covered=bool(oracle),h0_executable_covered=bool(own),h0_blocked_covered=bool(blocked),h0_executable_at=own,h0_blocked_at=blocked))
 return dict(events=result,events_total=len(result),oracle_covered=sum(r['oracle_covered'] for r in result),h0_executable_covered=sum(r['h0_executable_covered'] for r in result),h0_blocked_covered=sum(r['h0_blocked_covered'] for r in result),note='GT-aligned Active is oracle candidate-supply diagnostic only; H0-state coverage is conditional, not an upper bound for all learned own states; NEVER inference input')
