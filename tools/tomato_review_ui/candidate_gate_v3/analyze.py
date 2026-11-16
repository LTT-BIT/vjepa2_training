"""Offline diagnostics only. No annotation access from inference modules."""
from collections import Counter
import numpy as np

from .contract import *
from .targets import label
from .score import bootstrap


def clean(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 return v


def calibration(y,p,bins=10):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if not len(y):return dict(n=0)
 idx=np.minimum((p*bins).astype(int),bins-1);ece=0.;rows=[]
 for b in range(bins):
  keep=idx==b
  if keep.any():
   confidence=float(p[keep].mean());frequency=float(y[keep].mean());n=int(keep.sum());ece+=n/len(y)*abs(confidence-frequency);rows.append(dict(bin=b,n=n,confidence=confidence,frequency=frequency))
 pred=p>=.5;tp=float(y[pred].sum());order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1];ct=np.cumsum(ys)[ends];ap=float(np.sum(np.diff(np.r_[0,ct])*ct/(ends+1))/y.sum()) if y.sum() else None
 return dict(n=len(y),positives=int(y.sum()),brier=float(np.mean((y-p)**2)),ece=float(ece),precision=tp/max(int(pred.sum()),1),recall=tp/max(float(y.sum()),1),pr_auc=ap,bins=rows)


def gate_metrics(episodes,rollouts):
 y=[];p=[];hz=[];hb=[];censored_nll=[];bytime={};counts=Counter()
 for e in episodes:
  for r in rollouts[e['metadata']['episode_id']]['records']:
   f=r['feature'];target=label(e['metadata'],f);counts[target['reason']]+=1
   if not target['eligible'] or 'gate_probability' not in r['diagnostics']:continue
   prob=r['diagnostics']['gate_probability'];bucket=target['hazard_bucket'];key=str(bucket) if target['observed'] else 'censored' if target['censored'] else 'wrong_state'
   if target['gate_valid']:
    y.append(target['gate']);p.append(prob);bytime.setdefault(key,([],[]));bytime[key][0].append(target['gate']);bytime[key][1].append(prob)
   if 'hazard_probabilities' in r['diagnostics']:
    h=np.array(r['diagnostics']['hazard_probabilities']);allowed=target['hazard_allowed']
    if target['hazard_valid']:censored_nll.append(-np.log(max(1e-12,h[allowed].sum())))
    if target['observed']:hz.append(h);hb.append(bucket)
 result=dict(gate=calibration(y,p),gate_time_strata={k:calibration(*v) for k,v in bytime.items()},label_reasons=dict(counts))
 if hz:
  h=np.array(hz);b=np.array(hb);pred=h.argmax(1);conf=np.zeros((6,6),int)
  for a,c in zip(b,pred):conf[a,c]+=1
  result['hazard']=dict(observed_n=len(b),observed_nll=float(-np.log(h[np.arange(len(b)),b].clip(1e-12)).mean()),observed_multiclass_brier=float(np.square(h-np.eye(6)[b]).sum(1).mean()),top1_calibration=calibration((pred==b).astype(int),h.max(1)),bin_accuracy=float((pred==b).mean()),confusion_true_rows_pred_columns=conf.tolist(),mean_absolute_bin_error=float(abs(pred-b).mean()),censor_compatible_nll=float(np.mean(censored_nll)),note='Six coarse bins include unbounded tails: no fabricated continuous time-MAE; bin-MAE only. Censored NLL sums all compatible bins.')
 return result


def teacher_shift(parts,base_train,base_val):
 result={}
 for split,field,roll in [('train','oof',base_train),('val','rows',base_val)]:
  heads={h:dict(y=[],p=[],boundary_p=[]) for h in ['current','future']};disagree=[]
  for e in parts[split]:
   m=e['metadata'];seg=m['annotation']['segments'];bounds=np.array([s['start_frame']/m['fps'] for s in seg[1:]]);starts=np.array([s['start_frame']/m['fps'] for s in seg]);end=m['length']/m['fps']
   for r in e[field]:
    probs=[]
    for h,offset in [('current',0),('future',1.)]:
     t=r['observed_at']+offset;p=np.array([r['heads'][h]['probabilities'][s] for s in LABELS]);probs.append(p.argmax())
     if t>=end:continue
     y=LABELS.index(seg[np.searchsorted(starts,t,side='right')-1]['label']);heads[h]['y'].append(y);heads[h]['p'].append(p)
     if len(bounds) and abs(bounds-t).min()<=1:heads[h]['boundary_p'].append(p[y])
    disagree.append(probs[0]!=probs[1])
  stats={}
  for h,v in heads.items():
   p=np.array(v['p']);y=np.array(v['y']);pred=p.argmax(1);stats[h]=dict(n=len(y),accuracy=float((y==pred).mean()),nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),brier=float(((p-np.eye(4)[y])**2).sum(1).mean()),confidence_quantiles=np.quantile(p.max(1),[0,.1,.25,.5,.75,.9,1]).tolist(),correct_class_quantiles=np.quantile(p[np.arange(len(y)),y],[0,.1,.5,.9,1]).tolist(),ece=calibration((pred==y).astype(int),p.max(1))['ece'],boundary_correct_class_quantiles=np.quantile(v['boundary_p'],[0,.1,.5,.9,1]).tolist())
  recs=[r for rr in roll.values() for r in rr['records']];result[split]=dict(teacher='held-source OOF' if split=='train' else 'full75 B0 frozen',heads=stats,current_future_disagreement=float(np.mean(disagree)),candidate_fraction=float(np.mean([r['candidate']['stage'] is not None for r in recs])),executable_fraction=float(np.mean([r['feature']['eligible'] for r in recs])))
 result['interpretation']='OOF vs full-B0 descriptive shift, unequal source counts; no val calibration fitted, no threshold adjustment. Membership/statistic/probability identities in oof_teacher_manifest.json.'
 return result


def mismatch(ours,base):
 wrong=0.;n=0;unseen=0;ack=0;pending=0
 seen={(r['feature']['active'],r['feature']['policy_state'],r['feature']['candidate']) for rr in base.values() for r in rr['records']}
 for eid,rr in ours.items():
  br={r['feature']['observation_id']:r for r in base[eid]['records']}
  for r in rr['records']:
   f=r['feature'];n+=1;unseen+=(f['active'],f['policy_state'],f['candidate']) not in seen
   b=br[f['observation_id']]['feature'];ack+=f['policy_state']!=b['policy_state'];pending+=f['candidate_count']!=b['candidate_count']
  for a in rr['active_intervals']:
   for b in base[eid]['active_intervals']:
    if a['stage']!=b['stage']:wrong+=max(0,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
 return dict(active_shift_seconds=wrong,decision_ticks=n,unseen_state_ticks=unseen,unseen_fraction=unseen/max(n,1),ack_state_mismatches=ack,candidate_support_mismatches=pending,definition='unseen tuple(active,policy_state,candidate) against union H0 train tuples; Active shift exact interval intersections')


def select(summary,c):
 base=summary['H0'];audit={};eligible=[]
 for group in c['groups']:
  rows=[summary[f'{group}_s{s}'] for s in c['seeds']];passes=[]
  for r in rows:
   passes.append(r['illegal_actual_commits']==r['cross_session_commit']==r['non_planning_commit']==0 and all(r[k]<=base[k]+1e-7 for k in ['early_ge05','hard_timeouts','missed','unmatched_commits','wrong_active_s']) and r['strict_hits']>=base['strict_hits'] and r['exact_order']>=base['exact_order'])
  ms=float(np.mean([r['strict_hits'] for r in rows]));mw=float(np.mean([r['wrong_active_s'] for r in rows]));benefit=ms>=25 or mw<=.9*base['wrong_active_s'];audit[group]=dict(seed_passes=passes,all_three_pass=all(passes),mean_strict=ms,mean_wrong_s=mw,worst_early=max(r['early_ge05'] for r in rows),worst_hard=max(r['hard_timeouts'] for r in rows),benefit=benefit,qualifies=all(passes) and benefit)
  if all(passes) and benefit:eligible.append(group)
 eligible.sort(key=lambda g:(-audit[g]['mean_strict'],audit[g]['worst_early'],audit[g]['worst_hard'],audit[g]['mean_wrong_s'],{'G0':0,'G1':1,'G2':2,'G3':3}[g]))
 return dict(selected=eligible[0]+'_s42' if eligible else 'KEEP_H0',audit=audit,representative_seed=42,no_test_used=True,benefit_rule='mean strict>=25 OR mean wrong<=.9*H0; all three seeds independently pass every non-degradation rule',ranking='mean strict desc, worst early, worst hard, mean wrong, fixed capacity tie order')


def comparisons(summary,c):
 out={};h=summary['H0']['by_source']
 for g in c['groups']:
  rows=[summary[f'{g}_s{s}'] for s in c['seeds']];mean=[]
  for i in range(len(h)):
   r=dict(h[i]);r.update({k:float(np.mean([x['by_source'][i][k] for x in rows])) for k in ['strict_hits','wrong_active_s','early_ge05','missed','unmatched_commits','hard_timeouts']});mean.append(r)
  out[g+'_vs_H0_seedmean']=bootstrap(mean,h,c['bootstrap_replicates'],c['bootstrap_seed'])
  for s in c['seeds']:out[f'{g}_s{s}_vs_H0']=bootstrap(summary[f'{g}_s{s}']['by_source'],h,c['bootstrap_replicates'],c['bootstrap_seed'])
 for a,b in [('G0','G1'),('G1','G2'),('G1','G3')]:
  ar=[];br=[]
  for i in range(len(h)):
   for group,target in [(a,ar),(b,br)]:
    r=dict(h[i]);r.update({k:float(np.mean([summary[f'{group}_s{s}']['by_source'][i][k] for s in c['seeds']])) for k in ['strict_hits','wrong_active_s','early_ge05','missed','unmatched_commits','hard_timeouts']});target.append(r)
  out[b+'_minus_'+a+'_sourcepaired_seedmean']=bootstrap(br,ar,c['bootstrap_replicates'],c['bootstrap_seed'])
 return out


def temporal_strata(metadata,replays,decisions):
 """Offline boundary±.5/1s and stable>1s integrals; no policy changes."""
 output={}
 for name,rolls in replays.items():
  sums={k:dict(duration_s=0.,wrong_active_s=0.) for k in ['boundary_pm05','boundary_pm1','stable_gt1']};wait=Counter();yps={}
  for eid,r in rolls.items():
   m=metadata[eid];duration=m['length']/m['fps'];segments=m['annotation']['segments'];bounds=np.array([g['start_frame']/m['fps'] for g in segments[1:]])
   cuts=sorted(set([0.,duration]+[max(0.,min(duration,t+w)) for t in bounds for w in [-1.,-.5,0.,.5,1.]]+[a['start_s'] for a in r['active_intervals']]+[a['end_s'] for a in r['active_intervals']]))
   for a,b in zip(cuts,cuts[1:]):
    mid=(a+b)/2;distance=float(abs(bounds-mid).min()) if len(bounds) else float('inf');active=next(x['stage'] for x in r['active_intervals'] if x['start_s']<=mid<x['end_s']);gt=next(g['label'] for g in segments if g['start_frame']/m['fps']<=mid<g['end_frame']/m['fps']);wrong=active!=gt
    for key,keep in [('boundary_pm05',distance<=.5),('boundary_pm1',distance<=1),('stable_gt1',distance>1)]:
     if keep:sums[key]['duration_s']+=b-a;sums[key]['wrong_active_s']+=(b-a)*wrong
   for i,e in enumerate(r['events']):
    end=r['events'][i+1]['time'] if i+1<len(r['events']) else duration;wait[e['policy_state']]+=max(0,end-e['time'])
   for rec in decisions.get(name,{}).get(eid,[]):
    t=label(m,rec['feature'])
    if not t['eligible'] or not t['gate_valid'] or 'gate_probability' not in rec['diagnostics']:continue
    dt=abs(bounds-rec['feature']['commit_tick']).min() if len(bounds) else 1e10;key='boundary_pm05' if dt<=.5 else 'boundary_05to1' if dt<=1 else 'stable_gt1';ys,ps=yps.setdefault(key,([],[]));ys.append(t['gate']);ps.append(rec['diagnostics']['gate_probability'])
  output[name]=dict(active_time=sums,pending_ack_state_seconds=dict(wait),gate_time_strata={key:calibration(*v) for key,v in yps.items()})
 return output
