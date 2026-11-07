"""Audit actual unchanged OpenPI warning semantics, separate from classifier change events."""
import sys
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds,tomato_metrics as metrics
ROOT=Path('runs/tomato/temporal_final_20261108');PRIOR=Path('runs/tomato/optimization_20261108')


def evaluate(name,seed,horizon):
 from openpi.subtasks.agent_bridge import PredictionPromptProvider
 version=ds.read_json(PRIOR/'evaluations'/f'{name}_s{seed}.json')['checkpoint_sha256']
 m=ds.read_json(PRIOR/'prepared_h10/manifest.json');rows=[e for e in m['episodes'] if e['split']=='val'];z=np.load(PRIOR/'evaluations'/f'{name}_s{seed}_val_logits.npz')
 probs={h:metrics.probabilities(z[h+'_logits']) for h in ['current','future']};provider=PredictionPromptProvider();offset=0;reports={};hits=0;total=0;false_windows=0;warning_windows=0;negative_windows=0;duration=0.;leads=[];false_rising=0;rising_total=0;false_categories=dict(late_after_boundary=0,premature_beyond_horizon=0,wrong_progress_or_missing_target=0)
 for e in rows:
  session=f'{name}-{seed}-{e["episode_id"]}';provider.reset(session);segments=metrics.continuous_phases(e['annotation']['segments']);boundaries=[dict(phase=i,time=s['start_frame']/e['fps']) for i,s in enumerate(segments) if i]
  times=np.array(e['anchors'])/e['fps'];valid=np.array(e['anchors'])+round(horizon*e['fps'])<e['length'];signals=[];events=[];previous=None
  for i,t in enumerate(times):
   idx=offset+i;oid=str(i);base=1000000.+float(t)
   response=dict(protocol='tomato-stage-prediction-v1',labels=list(ds.LABELS),valid=True,error=None,observation_id=oid,session_id=session,
    model_version=version,prediction_mode='joint',future_horizon_s=horizon,input_timestamp=base,result_timestamp=base+.04,
    current_logits=z['current_logits'][idx].astype(float).tolist(),future_logits=z['future_logits'][idx].astype(float).tolist(),
    current_probabilities=probs['current'][idx].tolist(),future_probabilities=probs['future'][idx].tolist())
   context=provider.accept_prediction(response,observation_id=oid,now=base+.05)
   key=context.phase_index+1 if context.future_warning is not None else None
   if valid[i]:
    actual=next((b['phase'] for b in boundaries if t<b['time']<=t+horizon),None)
    negative_windows+=int(actual is None)
    if key is not None:
     signals.append(dict(time=float(t),phase=key,correct=key==actual));warning_windows+=1;false_windows+=int(key!=actual)
     if key!=actual:
      target=next((b for b in boundaries if b['phase']==key),None)
      category='wrong_progress_or_missing_target' if target is None else ('late_after_boundary' if t>=target['time'] else ('premature_beyond_horizon' if target['time']>t+horizon else 'wrong_progress_or_missing_target'))
      false_categories[category]+=1
     if key!=previous:events.append(dict(time=float(t),phase=key,correct=key==actual))
    previous=key
  matches=[]
  for b in boundaries:
   eligible=[s for s in signals if s['phase']==b['phase'] and b['time']-horizon<=s['time']<b['time']]
   lead=b['time']-eligible[0]['time'] if eligible else None
   matches.append(dict(**b,detected=lead is not None,lead_s=lead));hits+=int(lead is not None);total+=1
   if lead is not None:leads.append(lead)
  false_rising+=sum(not e['correct'] for e in events);rising_total+=len(events);duration+=float(valid.sum())*.5
  reports[e['episode_id']]=dict(boundaries=matches,warning_windows=len(signals),false_warning_windows=sum(not s['correct'] for s in signals),warning_rising_events=events)
  offset+=len(times)
 return dict(name=name,seed=seed,horizon_s=horizon,boundaries=total,any_correct_warning_hits=hits,any_correct_warning_recall=hits/total,
  miss_rate=1-hits/total,conditional_lead_p50_s=float(np.median(leads)),warning_windows=warning_windows,false_warning_windows=false_windows,
  false_warning_window_categories=false_categories,false_warning_window_fraction=false_windows/warning_windows if warning_windows else 0.,false_warning_rising_events=false_rising,
  warning_rising_events=rising_total,false_warning_rising_per_minute=false_rising/(duration/60),per_episode=reports,
  definition='Actual provider level warning: threshold.8, expected next ordered phase, no Future confirmation; Current needs2 confirmations. Boundary recall means any correctly targeted level warning within horizon. Overlong warnings accumulate false windows; rising-event count is reported separately. Input-time basis.')


def run():
 sys.path.insert(0,'/home/user/tomato-openpi/src')
 results=[evaluate(n,s,h) for n,h in [('independent_h15',1.5),('joint',1.)] for s in [42,43,44]]
 ds.write_json(ROOT/'actual_openpi_warning_audit.json',dict(results=results,selection_unchanged=True,
  notes='Different metric from raw four-class accepted-change events/K2. No seven-class probability or future label enters provider. Offline recorded logits, no robot/RTC.'))
 print({n:float(np.mean([r['any_correct_warning_recall'] for r in results if r['name']==n])) for n in ['joint','independent_h15']})

if __name__=='__main__':run()
