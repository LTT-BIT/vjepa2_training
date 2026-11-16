import csv
from collections import Counter
import numpy as np
from ..handshake_experiment import score as original_score
from .contracts import LABELS,write


def score(metadata,replay):
 r=original_score(metadata,replay);events=replay['events'];duration=metadata['length']/metadata['fps'];recovery=waiting=0.
 for i,e in enumerate(events):
  end=events[i+1]['time'] if i+1<len(events) else duration
  if e['policy_state']=='RECOVERY_REQUIRED':recovery+=max(0,end-e['time'])
  if e['policy_state'] in ['WAIT_CURRENT_ACK','WAIT_LATE_ACK']:waiting+=max(0,end-e['time'])
 commits=[e for e in events if e['kind']=='commit'];illegal=sum(e['to_stage'] not in replay.get('task_policy',{'pickup':['harvest'],'harvest':['return'],'return':['adjust'],'adjust':['pickup']}).get(e['from_stage'],[]) for e in commits)
 proposals=[e for e in events if e['kind']=='policy_proposal'];masks=Counter(e['masked_reason'] for e in proposals if e['masked_reason'])
 r.update(duration_s=duration,wrong_fraction=r['wrong_active_s']/duration,recovery_required_s=recovery,ack_wait_s=waiting,illegal_actual_commits=illegal,policy_proposals=len(proposals),policy_requests=sum(e['requested'] for e in proposals),masked_reasons=dict(masks),task_mask_rejections=masks['task_mask'],hold_fraction=sum(not e['requested'] for e in proposals)/len(proposals) if proposals else None)
 return r


def aggregate(reports):
 keys=['matched','eligible','missed','strict_hits','unmatched_commits','early_ge05','late_gt05','wrong_active_s','exact_order','hard_timeouts','timeouts','acks','late_acks','short_return_missed','recovery_required_s','ack_wait_s','illegal_actual_commits','policy_proposals','policy_requests','task_mask_rejections']
 result={k:sum(r.get(k,0) for r in reports) for k in keys};result.update(episodes=len(reports),mean_wrong_fraction=float(np.mean([r['wrong_fraction'] for r in reports])),mean_wrong_seconds=float(np.mean([r['wrong_active_s'] for r in reports])),by_source=reports)
 return result


def csv_rows(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap(reports,baseline,n=5000,seed=20261112):
 if [r['source_recording_id'] for r in reports]!=[r['source_recording_id'] for r in baseline]:raise ValueError('Source pairing')
 rng=np.random.default_rng(seed);weights=rng.multinomial(len(reports),np.ones(len(reports))/len(reports),size=n);out={}
 for key in ['strict_hits','wrong_active_s','early_ge05','missed','unmatched_commits','hard_timeouts']:
  delta=np.array([a[key]-b[key] for a,b in zip(reports,baseline)],float);out[key]=dict(delta_total=float(delta.sum()),ci95=np.percentile(weights@delta,[2.5,97.5]).tolist(),positive_sources=int((delta>1e-7).sum()),negative_sources=int((delta< -1e-7).sum()),unchanged_sources=int((abs(delta)<=1e-7).sum()),worst_delta=float(delta.max()),best_delta=float(delta.min()))
 return out
