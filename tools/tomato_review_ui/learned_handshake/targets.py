"""GT is permitted only here and in scorers, never in policy feature construction."""
import numpy as np
from .contracts import LABELS,TASK_POLICY
from .shell_adapter import BLOCKED


def label(metadata,feature,boundary_shift=0.):
 seg=metadata['annotation']['segments'];fps=metadata['fps'];t=feature['commit_tick'];active=feature['active']
 starts=[s['start_frame']/fps+(boundary_shift if i else 0) for i,s in enumerate(seg)];pos=max(0,min(len(seg)-1,int(np.searchsorted(starts,t,side='right')-1)))
 target=-1;boundary=None;event=-1;catchup=False
 if seg[pos]['label']==active:
  if pos+1<len(seg):target=LABELS.index(seg[pos+1]['label']);boundary=starts[pos+1];event=pos
 elif pos>0 and seg[pos-1]['label']==active and seg[pos]['label'] in TASK_POLICY.get(active,[]):
  target=LABELS.index(seg[pos]['label']);boundary=starts[pos];event=pos-1;catchup=True
 if target>=0 and LABELS[target] not in TASK_POLICY.get(active,[]):target=-1;boundary=None;event=-1
 censored=(seg[pos]['label']==active and pos+1==len(seg));supervision_valid=t<metadata.get('length',seg[-1]['end_frame'])/fps-1e-9
 delta=boundary-t if boundary is not None else None;time_bin=-100 if censored or not supervision_valid else 5;valid_time=not censored and supervision_valid
 if delta is not None:
  if delta>1:time_bin=0
  elif delta>.5:time_bin=1
  elif delta>0:time_bin=2
  elif delta>=-.2-1e-9:time_bin=3
  elif delta>=-1-1e-9:time_bin=4
  else:time_bin=-100;valid_time=False  # late catchup still gets action supervision; not mislabeled F
 request=target>=0 and ((delta is not None and -.5-1e-9<=delta<=.2+1e-9) or catchup)
 if feature['policy_state'] in BLOCKED:request=False
 action=target+1 if request else 0
 cp=feature['sequence'][-1,:4];fp=feature['sequence'][-1,4:8];candidate=int(fp.argmax());legal=LABELS[candidate] in TASK_POLICY.get(active,[]) and fp[candidate]>=.65 and fp[candidate]>=fp[LABELS.index(active)]
 gate=float(request and legal and candidate==target)
 if not supervision_valid:action=-100;gate=-100.;target=-1;time_bin=-100;valid_time=False
 return dict(supervision_valid=supervision_valid,right_censored=censored,action=action,gate=gate,next_stage=target,time_bin=time_bin,event=event,late_catchup=catchup,delta_s=delta,valid_time=valid_time,candidate_eligible=legal)


def weights(source_ids,labels):
 # Every source has unit mass, every adjacent event/tail equal mass; each event's
 # occupied timing stratum equal mass. Stable A negatives cannot drown its boundary.
 out=np.zeros(len(labels),np.float32)
 for source in sorted(set(source_ids)):
  ids=[i for i,s in enumerate(source_ids) if s==source];events={labels[i]['event'] for i in ids}
  for event in events:
   event_ids=[i for i in ids if labels[i]['event']==event];buckets={labels[i]['time_bin'] for i in event_ids}
   for bucket in buckets:
    keep=[i for i in event_ids if labels[i]['time_bin']==bucket];out[keep]=1/(len(events)*len(buckets)*len(keep))
 return out/out.mean()
