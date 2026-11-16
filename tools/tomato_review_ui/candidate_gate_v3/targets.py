"""Supervision/scoring only: candidate-specific timing with right censoring."""
import numpy as np
from .contract import LABELS

UPPER=np.array([-.5,0.,.2,.5,1.,np.inf])

def label(metadata,f,shift=0.):
 seg=metadata['annotation']['segments'];fps=metadata['fps'];tick=f['commit_tick'];starts=[s['start_frame']/fps+(shift if i else 0) for i,s in enumerate(seg)];end=metadata['length']/fps
 eligible=bool(f['eligible'] and tick<end-1e-9);pos=min(len(seg)-1,max(0,int(np.searchsorted(starts,tick,side='right')-1)));active=f['active'];candidate=f['candidate'];event=-1;boundary=None;censored=False;reason='wrong_active_or_candidate'
 if candidate is not None:
  if seg[pos]['label']==active:
   if pos+1<len(seg) and seg[pos+1]['label']==candidate:event=pos;boundary=starts[pos+1];reason='next_adjacent'
   elif pos+1==len(seg):censored=True;reason='right_censored'
  elif pos>0 and seg[pos-1]['label']==active and seg[pos]['label']==candidate:event=pos-1;boundary=starts[pos];reason='late_adjacent_catchup'
 delta=boundary-tick if boundary is not None else None
 positive=boundary is not None and (-.5-1e-9<=delta<=.2+1e-9 or reason=='late_adjacent_catchup')
 allowed=np.zeros(6,bool);observed=False
 if boundary is not None:
  bucket=0 if round(delta,9)<-.5 else 1+int(np.searchsorted(UPPER[1:],round(delta,9),side='left'));allowed[bucket]=True;observed=True
 elif censored:
  # Unknown b is after the last observed frame. Sum all bins compatible with
  # b-tick > end-tick; a partly overlapping bin stays allowed (coarse censoring).
  allowed=UPPER>max(0,end-tick)
 else:bucket=-1
 return dict(eligible=eligible,gate_valid=not censored or end-tick>=.2-1e-9,gate=float(positive),event=event,delta_s=delta,observed=observed,censored=censored,hazard_allowed=allowed,hazard_valid=bool(allowed.any()),hazard_bucket=int(np.flatnonzero(allowed)[0]) if observed else -1,reason=reason)


def sample_weights(sources,labels):
 out=np.zeros(len(labels),np.float32)
 for source in set(sources):
  indices=[i for i,s in enumerate(sources) if s==source];events={labels[i]['event'] for i in indices}
  for ev in events:
   ids=[i for i in indices if labels[i]['event']==ev];strata={labels[i]['hazard_bucket'] if labels[i]['observed'] else -2 if labels[i]['censored'] else -1 for i in ids}
   for bucket in strata:
    keep=[i for i in ids if (labels[i]['hazard_bucket'] if labels[i]['observed'] else -2 if labels[i]['censored'] else -1)==bucket];out[keep]=1/(len(events)*len(strata)*len(keep))
 return out/out.mean()
