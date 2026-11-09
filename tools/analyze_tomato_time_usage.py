"""No-training event-level time-head diagnostics and non-veto timing/ranking ablations."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds,tomato_boundary_metrics as bm
from .evaluate_tomato_reviewed_boundaries import revise_episode,match_events
from .tomato_transition_time import survival_targets
from .train_tomato_transition_time import summary
SOURCE=Path('runs/tomato/transition_time_20261109');ROOT=Path('runs/tomato/time_usage_analysis_20261109')


def distribution(x):
 a=np.asarray(x);return dict(n=len(a),mean=float(a.mean()) if len(a) else None,quantiles=np.percentile(a,[0,25,50,75,100]).tolist() if len(a) else [])


def rank_metrics(score,y):
 s=np.asarray(score);y=np.asarray(y,bool);pos=s[y];neg=s[~y]
 auc=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None
 order=np.argsort(-s,kind='stable');sy=y[order];ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(s)-1];tp=np.cumsum(sy)[ends];precision=tp/(ends+1);recall=tp/max(1,sy.sum());ap=float(np.sum(np.diff(np.r_[0.,recall])*precision)) if sy.any() else None
 return dict(auroc=auc,average_precision=ap,curve=[dict(rank=int(i+1),precision=float(p),recall=float(r),score=float(s[order[i]])) for i,p,r in zip(ends,precision,recall)])


def calibration(p,y):
 p=np.asarray(p);y=np.asarray(y);bins=[]
 for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
  k=(p>=lo)&(p<hi if hi<.999 else p<=1)
  if k.any():bins.append(dict(lo=float(lo),hi=float(hi),n=int(k.sum()),predicted=float(p[k].mean()),observed=float(y[k].mean())))
 return dict(samples=len(p),brier=float(np.mean((p-y)**2)),mean_predicted=float(p.mean()),observed_rate=float(y.mean()),ece=sum(b['n']*abs(b['predicted']-b['observed']) for b in bins)/len(p),bins=bins)


def run():
 if (ROOT/'analysis.json').exists():raise FileExistsError('Analysis output already frozen')
 ROOT.mkdir(exist_ok=True,parents=True);old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');by={e['episode_id']:e for e in old['episodes']};proposals=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals'];identity=ds.read_json(SOURCE/'data_identity.json');rows=[]
 for rec in identity['episodes']:
  if rec['split']!='val':continue
  e=revise_episode(by[rec['episode_id']],proposals.get(rec['episode_id'],[]));f=np.load(rec['feature_path']);rows.append(dict(episode=e,times=f['times'],anchors=f['anchors'],features=f['features'],targets=survival_targets(e['annotation']['segments'],f['anchors'],e['fps'],e['length'])))
 out=dict(seeds={},all_event_rows=[],soft_uses={},provenance=dict(source=str(SOURCE),test_read=False,training=False,original_labels_changed=False,baseline_label_proposal_sha256=ds.digest('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')));candidates=[]
 for seed in [42,43,44]:
  r=ds.read_json(SOURCE/f'seed{seed}/result.json');operating=r['external_validation_thresholds_are_ablation_not_selection'][str(r['selected_threshold'])];eventrows=[];warningrows=[];cal=defaultdict(lambda:[[],[]]);remaining=defaultdict(list);sweep={};timing=[]
  for row in rows:
   e=row['episode'];eid=e['episode_id'];f=np.load(SOURCE/f'seed{seed}'/(eid+'.npz'));ts=f['times'];mass=f['transition_probabilities'];risk=1-mass[:,-1];truth=bm.boundaries(dict(episode=e));b0=operating['per_episode']['B0'][eid];b1=operating['per_episode']['B1_gate'][eid];b2=operating['per_episode']['B2'][eid]
   for ordinal,(a,b,c) in enumerate(zip(b0['boundaries'],b1['boundaries'],b2['boundaries']),1):
    i=int(np.argmin(abs(ts-a['event_time']))) if a['event_time'] is not None else None
    er=dict(seed=seed,episode_id=eid,ordinal=ordinal,transition=ds.LABELS[a['previous']]+'->'+ds.LABELS[a['label']],boundary_s=a['time'],B0_hit=a['detected'],B1_hit=b['detected'],B2_hit=c['detected'],B0_event_time=a['event_time'],B1_event_time=b['event_time'],B2_event_time=c['event_time'],risk_at_B0_hit=float(risk[i]) if i is not None else None)
    eventrows.append(er)
   credited={b['event_time'] for b in b0['boundaries'] if b['detected']}
   for ev in b0['events']:
    i=int(np.argmin(abs(ts-ev['time'])));correct=ev['time'] in credited;key=ds.LABELS[ev['previous']]+'->'+ds.LABELS[ev['label']];p=mass[i,:4];pred_dt=float(p@np.array([.125,.375,.75,1.25])/max(p.sum(),1e-9));wr=dict(seed=seed,episode_id=eid,transition=key,time=ev['time'],correct=correct,risk=float(risk[i]),future_confidence=float(f['future_probabilities'][i,ev['label']]),predicted_dt_s=pred_dt,distribution=mass[i].tolist());warningrows.append(wr)
    if correct:
     b=next(b for b in b0['boundaries'] if b['event_time']==ev['time']);timing.append(dict(transition=key,actual_remaining_s=b['time']-ev['time'],predicted_remaining_s=pred_dt,abs_error_s=abs(pred_dt-(b['time']-ev['time'])),fixed15_abs_error_s=abs(1.5-(b['time']-ev['time']))))
   t=row['targets'];valid=t['event_observed']|(t['followup']>=1.5);y=t['event_observed']&(t['time_to_event']<=1.5)
   for i in np.flatnonzero(valid):
    nowclass=ds.label_at_frame(e['annotation']['segments'],int(row['anchors'][i]));future=next((b for b in truth if b['time']>ts[i]+1e-9),None);key=ds.LABELS[nowclass]+'->'+ds.LABELS[future['label']] if future else 'right_censored_or_terminal';cal[key][0].append(float(risk[i]));cal[key][1].append(int(y[i]))
    if t['event_observed'][i]:
     dt=t['time_to_event'][i];binname=next((str(hi) for hi in [.25,.5,1.,1.5,3.,10.,float('inf')] if dt<=hi),'inf');remaining[(key,binname)].append(float(risk[i]))
  for threshold in [.1,.2,.3,.5,.7,.9]:
   per={}
   for row in rows:
    eid=row['episode']['episode_id'];f=np.load(SOURCE/f'seed{seed}'/(eid+'.npz'));ev=[e for e in operating['per_episode']['B0'][eid]['events'] if 1-f['transition_probabilities'][np.argmin(abs(f['times']-e['time'])),-1]>=threshold];matched,false=match_events(ev,bm.boundaries(dict(episode=row['episode'])),'future',1.5);per[eid]=dict(boundaries=matched,false_events=false,duration_s=operating['per_episode']['B0'][eid]['duration_s'])
   sweep[str(threshold)]=summary(per)
  scores=[w['risk'] for w in warningrows];y=[w['correct'] for w in warningrows];bytype={}
  for typ in sorted({w['transition'] for w in warningrows}):
   a=[w for w in warningrows if w['transition']==typ];bytype[typ]=dict(correct=distribution([w['risk'] for w in a if w['correct']]),wrong=distribution([w['risk'] for w in a if not w['correct']]))
  sr=dict(B0_hit_B1_miss=[x for x in eventrows if x['B0_hit'] and not x['B1_hit']],B1_hit_B2_miss=[x for x in eventrows if x['B1_hit'] and not x['B2_hit']],correct_warning_risk=distribution([w['risk'] for w in warningrows if w['correct']]),wrong_warning_risk=distribution([w['risk'] for w in warningrows if not w['correct']]),by_transition_warning_risk=bytype,window_calibration={k:calibration(*v) for k,v in cal.items()},risk_by_remaining_time=[dict(transition=k[0],remaining_upper_s=k[1],**distribution(v)) for k,v in remaining.items()],ranking=dict(time_risk=rank_metrics(scores,y),future_confidence=rank_metrics([w['future_confidence'] for w in warningrows],y),geometric_confidence=rank_metrics([np.sqrt(w['risk']*w['future_confidence']) for w in warningrows],y)),threshold_sweep_not_selection=sweep,matched_warning_timing=dict(samples=len(timing),mae_s=float(np.mean([w['abs_error_s'] for w in timing])),fixed15_mae_s=float(np.mean([w['fixed15_abs_error_s'] for w in timing]))),timing_rows=timing)
  out['seeds'][str(seed)]=sr;out['all_event_rows']+=eventrows;candidates+=warningrows
  # Non-veto strategies preserve the same event times and sets; no sorting of the
  # online timeline, only score/interval metadata changes.
  out['soft_uses'][str(seed)]={k:dict(**operating['summaries']['B0'],events_identical=True,warning_time_unchanged=True) for k in ['time_annotation_only','rank_metadata_only']}
 ds.write_json(ROOT/'analysis.json',out);ds.write_json(ROOT/'candidate_warnings.json',candidates)
 # All eight difficult events: complete probability and named state/visual traces.
 names=old['contract']['joint_names'];arm=[i for i,n in enumerate(names) if n.startswith('right_fr3')];hand=[i for i,n in enumerate(names) if n.startswith('right_') and not n.startswith('right_fr3')];difficult=[]
 for row in rows:
  e=row['episode'];ts,q=ds.read_state(e,names,'arms-and-hands');velocity=np.gradient(q,ts,axis=0);f=np.load(SOURCE/'seed42'/(e['episode_id']+'.npz'))
  for ordinal,b in enumerate(bm.boundaries(dict(episode=e)),1):
   if b['previous']!=3 or b['label']!=0:continue
   folder=ROOT/'adjust_cases'/f"{e['episode_id']}_b{ordinal}";folder.mkdir(parents=True);keep=(ts>=b['time']-4)&(ts<=b['time']+3);pk=(row['times']>=b['time']-4)&(row['times']<=b['time']+3);event=next(x for x in out['all_event_rows'] if x['seed']==42 and x['episode_id']==e['episode_id'] and x['ordinal']==ordinal)
   np.savez_compressed(folder/'traces.npz',state_times=ts[keep],right_arm_positions=q[keep][:,arm],right_arm_velocities=velocity[keep][:,arm],right_hand_positions=q[keep][:,hand],right_hand_velocities=velocity[keep][:,hand],prediction_times=f['times'][pk],current=f['current_probabilities'][pk],future=f['future_probabilities'][pk],time_distribution=f['transition_probabilities'][pk],pooled_camera_features=row['features'][pk,:3072].reshape(-1,3,1024))
   same=[x for x in ds.read_json(SOURCE/'seed42/result.json')['external_validation_thresholds_are_ablation_not_selection']['0.5']['per_episode']['B0'][e['episode_id']]['events'] if x['previous']==3 and x['label']==0];nearest=min(same,key=lambda x:abs(x['time']-b['time'])) if same else None
   event={**event,'nearest_warning_offset_s':nearest['time']-b['time'] if nearest else None,'camera_keys':e['camera_keys'],'right_arm_joint_names':[names[i] for i in arm],'right_hand_joint_names':[names[i] for i in hand],'traces':str(folder/'traces.npz'),'source_videos':e['videos'],'camera_role_note':'All three camera keys retained; right-wrist role is not established by metadata.'};ds.write_json(folder/'case.json',event);difficult.append(event)
 assert len(difficult)==8;ds.write_json(ROOT/'adjust_cases.json',difficult)
 print('seed42',dict(B0_hit_B1_miss=len(out['seeds']['42']['B0_hit_B1_miss']),B1_hit_B2_miss=len(out['seeds']['42']['B1_hit_B2_miss']),correct=out['seeds']['42']['correct_warning_risk'],wrong=out['seeds']['42']['wrong_warning_risk'],timing=out['seeds']['42']['matched_warning_timing']),flush=True)



def attach_time_metadata(events,times,masses):
    """Non-veto augmentation: preserve every event and its original timestamp/order."""
    result=[]
    for event in events:
        i=int(np.argmin(abs(np.asarray(times)-event['time'])));p=np.asarray(masses[i]);conditional=p[:4]/max(p[:4].sum(),1e-9);j=int(conditional.argmax());edges=[0.,.25,.5,1.,1.5]
        result.append(dict(**event,time_risk=float(1-p[-1]),time_distribution=p.tolist(),conditional_remaining_if_within15_s=float(conditional@np.array([.125,.375,.75,1.25])),conditional_interval_if_within15_s=[event['time']+edges[j],event['time']+edges[j+1]],modal_interval_s=([event['time']+1.5,None] if int(p.argmax())==4 else [event['time']+edges[int(p.argmax())],event['time']+edges[int(p.argmax())+1]]),semantics='Timing/confidence metadata only; not calibrated probability that this warning is correct; never veto or advance Prompt'))
    return result

if __name__=='__main__':run()
