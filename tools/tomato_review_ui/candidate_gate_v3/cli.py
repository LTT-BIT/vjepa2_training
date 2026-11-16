"""Reproducible candidate-only handshake ablation; immutable phase outputs."""
import argparse,gzip,json,time,traceback,sys
from pathlib import Path
import numpy as np
import torch
from .contract import *
from .identity import check,load,verify
from .features import schema
from .replay import run,pure_h0
from .policy import R1Policy,GatePolicy,load_checkpoint
from .score import score,aggregate,csv_rows,coverage
from .data import dataset,statistics,aggregate as combine,audit
from .train import fit
from .analyze import clean,teacher_shift,mismatch,select,comparisons,gate_metrics


def seal(c,out):
 source={str(p.relative_to(REPO)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
 p=dict(version='candidate-gate-v3-20261113',config=c,source_sha256=source,production_api=read(out/'public_interface_change.json'),clock=H0_CONFIG,task_policy=TASK_POLICY,candidate='at most one Future top1 != own Active, legal external graph, p>=.65; no .8 prefilter',schema=schema(),architectures=dict(G0='240->1 explicitL2',G1='240->64 GELU Drop.1 ->32 GELU ->1',G2='per-anchor59->32 GELU, masked causalGRUCell32 over4,32->32GELU Drop.1->1',G3='exact G1 trunk/gate plus32->6 auxiliary only'),target=dict(commit_window=[-.2,.5],adjacent_late_catchup='positive while GT current=candidate and previousGT=own Active; no GT state reset; wrong pairing HOLD',no_candidate='HOLD, masked from gate loss',end='tick>=duration excluded; final segment right-censored boundary; hazard sums compatible bins with upper edge>end-tick; main gate BCE MASKED when remaining<.2s because positive anticipatory window extends beyond observation; no fabricated boundary/FINISH',hazard_bins='delta=b-nextactualtick: (-inf,-.5),[-.5,0],(0,.2],(.2,.5],(.5,1],(1,inf)',gate_class_weight='weighted train neg/pos capped10',weights='equal source, equal event, equal occupied time stratum; shared label function',hazard_loss='same per-sample weights; conditional mean over valid hazard rows, NLL of summed compatible probabilities; lambda.25'),earlystop='same H0-state val main weighted gate BCE, patience10, min_delta1e-8; no val Active epoch/version search',aggregation=dict(trigger_any12='train-only unseen(active,state,candidate)>5% OR accumulated exact Active mismatch>10s',rounds=1,if_triggered='all12 reinitialize same seeds; v1 mandatory; eachtrain baseN+ownN resampledseed, fixed equal per-epoch updates; common normalization pooled base+all12 own train, no val/test',if_not='all12v0; no selective aggregation'),selection=dict(each_seed='illegal/cross/nonplanning0; early<=4, hard<=3, missed<=11, extra<=11, strict>=23, exactorder>=9, wrong<=98.233333333333; all3 seeds',benefit='groupmean strict>=25 OR wrong<=.9*H0',rank='meanstrict desc,worstearly,worsthard,meanwrong,capacitytieG0/G1/G2/G3',representative_seed=42),test='lock winner before one test evaluation: H0 and winnerseed42 only; KEEP_H0 means H0 only; historically exposed, not blind',stress='all6 seed42 fixed source-index faults or causalfirstcommit, fixed clocks; stress.py SHA binds schedules; not tuning',optional='K1 NOT_RUN; no new Encoder/visual memory training')
 lock(out/'protocol.lock.json',p);write(out/'feature_schema.json',schema());return dict(protocol_sha256=sha(out/'protocol.lock.json'),input_identity_sha256=sha(out/'input_identity.json'),oof_sha256=sha(out/'oof_teacher_manifest.json'))


def save(out,name,split,e,r,save_records=True):
 m=e['metadata'];eid=m['episode_id'];p=out/'replays'/name/split/(eid+'.json')
 if p.exists():raise FileExistsError(str(p))
 sr=score(m,r);write(p,clean(dict(replay={k:v for k,v in r.items() if k!='records'},metrics=sr)))
 if save_records:
  path=out/'decisions'/name/split/(eid+'.jsonl.gz');path.parent.mkdir(parents=True,exist_ok=True)
  with gzip.open(path,'wt') as f:
   for rec in r['records']:f.write(json.dumps(clean(rec),ensure_ascii=False,separators=(',',':'))+'\n')
 return sr


def evaluate(episodes,policy,out,name,split,use_oof=False,save_records=True):
 reports=[];rolls={}
 for e in episodes:
  m=e['metadata'];r=run(e['oof'] if use_oof else e['rows'],m['length']/m['fps'],policy,m['episode_id']);reports.append(save(out,name,split,e,r,save_records));rolls[m['episode_id']]=r
 a=aggregate(reports)
 for key in ['cross_session_commit','non_planning_commit','gate_candidate_ticks','request_eligible_ticks','gate_holds','candidate_absent']:a[key]=sum(r[key] for r in reports)
 folder=out/'evaluations'/name/split;write(folder/'metrics.json',a);csv_rows(folder/'episode_scores.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in reports]);csv_rows(folder/'transition_scores.csv',[dict(episode_id=r['episode_id'],**v) for r in reports for v in r['transitions']])
 return a,rolls


def build(c,out):
 parts={s:[] for s in ['train','val','test']}
 for e in load(c):parts[e['metadata']['split']].append(e)
 htrain={};hval={}
 for split,target,field in [('train',htrain,'oof'),('val',hval,'rows')]:
  for e in parts[split]:
   m=e['metadata'];target[m['episode_id']]=run(e[field],m['length']/m['fps'],session=m['episode_id'])
 td=dataset(parts['train'],htrain);vd=dataset(parts['val'],hval);stats=statistics([td])
 if not (out/'train_labels_audit.json').exists():
  sensitivity={}
  for delta in [-.1,-1/30,1/30,.1]:
   shifted=dataset(parts['train'],htrain,delta);sensitivity[str(delta)]=dict(gate_changes=sum(a['gate']!=b['gate'] for a,b in zip(td['labels'],shifted['labels'])),hazard_changes=sum(a['hazard_bucket']!=b['hazard_bucket'] for a,b in zip(td['labels'],shifted['labels'])))
  sources=[]
  for s in read(out/'split.lock.json')['train']:
   ids=[i for i,x in enumerate(td['sources']) if x==s];sources.append(dict(source=s,eligible=len(ids),positives=sum(td['labels'][i]['gate'] for i in ids)))
  examples=[]
  # Deterministic audit includes hard buckets and censored tails, no annotation edits.
  for bucket in [-1,0,1,2,3,4,5]:
   ids=[i for i,t in enumerate(td['labels']) if t['hazard_bucket']==bucket]
   for i in ids[:3]:
    eid,ri=td['keys'][i];rec=htrain[eid]['records'][ri];examples.append(dict(source=td['sources'][i],episode=eid,feature_state={k:v for k,v in rec['feature'].items() if k not in ['sequence','mask']},candidate=rec['candidate'],target=td['labels'][i]))
  write(out/'train_labels_audit.json',clean(dict(train=audit(td),val=audit(vd),per_source=sources,by_transition=dict(__import__('collections').Counter((htrain[eid]['records'][ri]['feature']['active']+'->'+htrain[eid]['records'][ri]['feature']['candidate']) for eid,ri in td['keys'])),sensitivity=sensitivity,deterministic_label_audit=examples)))
  write(out/'teacher_shift_report.json',teacher_shift(parts,htrain,hval));write(out/'candidate_coverage_oracle_diagnostic.json',coverage(parts['val'],hval))
 return parts,td,vd,stats,htrain,hval


def load_policy(record,c,bindings):
 s,m=load_checkpoint(record['path'],bindings,c);return GatePolicy(s['group'],m,s['stats'],record['manifest']['checkpoint_sha256'])


def execute(c,out,smoke=False):
 bindings=seal(c,out);parts,td,vd,stats,base,hval=build(c,out)
 if smoke:
  for g in c['groups']:fit(g,42,td,vd,stats,c,out/'smoke'/g,bindings,True)
  write(out/'smoke_receipt.json',dict(groups=c['groups'],epochs=2,seed=42,finite_losses=True,reloaded=True,no_test_scored=True));print('SMOKE_PASS',flush=True);return
 if not (out/'smoke_receipt.json').exists():raise ValueError('Run smoke before main')
 if (out/'model_index.json').exists():raise FileExistsError('Main models already present; refuse overwrite')
 models={};own_data={};diagnostics={}
 for g in c['groups']:
  for seed in c['seeds']:
   name=f'{g}_s{seed}';folder=out/'decision_models'/g/f'seed{seed}'/'v0';policy,meta=fit(g,seed,td,vd,stats,c,folder,bindings);ours={}
   for e in parts['train']:
    m=e['metadata'];r=run(e['oof'],m['length']/m['fps'],policy,m['episode_id']);ours[m['episode_id']]=r;save(out,name,'train_v0_oof',e,r)
   diagnostics[name]=mismatch(ours,base);own_data[name]=dataset(parts['train'],ours);models[name]=dict(path=str(folder/'best.pt'),manifest=meta,version='v0');write(out/'train_rollout_diagnostic.partial.json',diagnostics);print('TRAIN_STATE',name,diagnostics[name],flush=True)
 trigger=any(d['unseen_fraction']>c['aggregate_unseen_fraction'] or d['active_shift_seconds']>c['aggregate_active_shift_seconds'] for d in diagnostics.values());lock(out/'aggregation.lock.json',dict(triggered=trigger,scope='all12',diagnostics=diagnostics,val_Active_used=False,rounds=int(trigger),final_version='v1' if trigger else 'v0'))
 if trigger:
  stats=statistics([td]+list(own_data.values()))
  for g in c['groups']:
   for seed in c['seeds']:
    name=f'{g}_s{seed}';dt=combine(td,own_data[name],seed);folder=out/'decision_models'/g/f'seed{seed}'/'v1';policy,meta=fit(g,seed,dt,vd,stats,c,folder,bindings);models[name]=dict(path=str(folder/'best.pt'),manifest=meta,version='v1',own_sampling=dt['own_sampling']);print('AGGREGATED',name,flush=True)
 write(out/'model_index.json',models);del own_data
 summary={};summary['H0'],hr=evaluate(parts['val'],None,out,'H0','val');summary['R1'],_=evaluate(parts['val'],R1Policy(),out,'R1','val')
 h=summary['H0']
 expected=dict(strict_hits=23,eligible=54,early_ge05=4,missed=11,unmatched_commits=11,hard_timeouts=3,exact_order=9)
 if any(h[k]!=v for k,v in expected.items()) or abs(h['wrong_active_s']-98.23333333333335)>1e-6:raise ValueError('Historical H0 mismatch')
 for name,record in models.items():
  policy=load_policy(record,c,bindings);summary[name],rolls=evaluate(parts['val'],policy,out,name,'val');write(out/'evaluations'/name/'val/gate_metrics.json',gate_metrics(parts['val'],rolls));final_train,train_roll=evaluate(parts['train'],policy,out,name,'train_final_oof',True);write(out/'evaluations'/name/'train_final_oof/gate_metrics.json',gate_metrics(parts['train'],train_roll));write(out/'summary/val.partial.json',summary);print('VAL',name,{k:summary[name][k] for k in ['strict_hits','wrong_active_s','early_ge05','missed','hard_timeouts']},flush=True)
 lock(out/'selection.lock.json',select(summary,c));selection=read(out/'selection.lock.json');lock(out/'test_lock_receipt.json',dict(protocol_sha256=bindings['protocol_sha256'],selection_sha256=sha(out/'selection.lock.json'),val_sha256=json_sha(summary),selected=selection['selected'],historically_exposed=True,test_source_ids=read(out/'split.lock.json')['test']))
 historical={};historical['H0'],_=evaluate(parts['test'],None,out,'H0','test')
 if selection['selected']!='KEEP_H0':
  n=selection['selected'];historical[n],_=evaluate(parts['test'],load_policy(models[n],c,bindings),out,n,'test')
 write(out/'summary/val_overall.json',summary);write(out/'summary/historical_test_locked.json',historical);write(out/'source_bootstrap.json',comparisons(summary,c))
 if c['run_stress']:
  from .stress import execute as stress
  policies={'H0':None,'R1':R1Policy()};policies.update({g:load_policy(models[g+'_s42'],c,bindings) for g in c['groups']});stress(parts,policies,out,save)
 write(out/'run_complete.json',dict(models=12,initial_models=12,aggregation_models=12 if trigger else 0,selected=selection['selected'],new_encoder_calls=0,new_teachers=0,test_scored=list(historical),k1='NOT_RUN optional'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/tomato_candidate_gate_v3_20261113.yaml');p.add_argument('--output-dir',required=True,type=Path);m=p.add_mutually_exclusive_group(required=True)
 for flag in ['check-only','smoke','run','verify','report-only']:m.add_argument('--'+flag,action='store_true')
 a=p.parse_args();c=config(a.config);out=a.output_dir.resolve()
 for k in ['baseline_root','memory_experiment','reference_experiment']:
  if out==Path(c[k]) or out.is_relative_to(Path(c[k])):raise ValueError('Output within protected inputs')
 out.mkdir(parents=True,exist_ok=True)
 try:
  if a.check_only:
   if (out/'input_identity.json').exists():raise FileExistsError('P0 exists; refusing overwrite')
   check(c,out)
  elif a.verify:
   verify(out)
   from .verification import verify_run
   verify_run(c,out)
  elif a.report_only:
   from .report import generate
   generate(c,out)
  else:execute(c,out,a.smoke)
 except Exception:
  write(out/f'failure_{time.time_ns()}.json',dict(command=sys.argv,traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
