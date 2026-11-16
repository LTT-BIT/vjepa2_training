"""Independent experiment: check-only, phase-a, smoke, run, verify, report-only."""
import argparse,dataclasses,json,time,traceback,sys
from pathlib import Path
import numpy as np
import torch
from .contracts import *
from .identity import check,load,verify_sources
from .rule_policies import RulePolicy
from .replay import h0,run
from .evaluate import score,aggregate,csv_rows,bootstrap
from .targets import label
from .train import dataset,combine,fit
from .models import LearnedPolicy,load_checkpoint


def clean(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(v) for v in value]
 return value


def seal_protocol(c,out):
 sources={str(p.relative_to(REPO)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
 p=dict(version='learned-handshake-v2',config=c,source_sha256=sources,clock=H0_CONFIG,task_policy=TASK_POLICY,feature_schema=dict(flat=92,state=48,sequence=[4,11],no_absolute_episode_time=True),architecture=dict(L0='linear current+delta+state 56->1',L1a='92->64->32->1',L1b='92->64->32->5',L2='11->32 masked GRUCell32 + state48 ->64->32->5',L3a='same L1b trunk, next4 + time6'),targets=dict(request_window_commit_minus_boundary=[-.2,.5],time_bins='A delta>1; B(.5,1]; C(0,.5]; D[-.2,0]; E[-1,-.2); F no valid target; delta<-1 late-catchup masked time but positive action while GT current=legal candidate',end_tail='right-censored next boundary: mask next/time, retain HOLD only while actual commit tick remains annotated; exclude supervision beyond recording; never physical FINISH',weights='equal source, equal event, equal occupied time stratum; sqrt weighted class mass; BCE pos_weight from train capped10'),rules=dict(R1='prepare>=.65, commitFuture>=.95 K1; .8<=p<.95 K2 distinct anchors',R2='prepareFuture>=.65 and legal; request pF>=.8 OR pC(candidate)>=.8; Current never veto',R3='NOT_RUN: referenced V1 fixed threshold unavailable; no invented replacement'),aggregation=dict(rounds_max=1,trigger='train-only Active time disagreement from H0 OOF >.10',action='combine H0+own train trajectories equally by source/event/time strata; reinitialize same seed; final v1 used unconditionally when triggered; no Active-based checkpoint search'),validation='fixed H0-state supervised loss early stopping; final own-trajectory Active risk ordering only after checkpoints fixed',selection=dict(non_degrade_seeds=2,illegal_actual_max=0,hard_timeouts_delta_max=0,early_ge05_delta_max=0,missed_delta_max=0,extra_unmatched_delta_max=0,order='strict hits descending then wrong seconds ascending then exact order descending; qualifying learns >=2/3 seeds, select only risk-passing seed; report worst seed; deterministic rules same constraints; must improve strict or wrong to replace H0'),conditional_groups=dict(L3b='NOT_RUN_BUDGET initial minimum matrix',length8='NOT_RUN_BUDGET',K1='only fixed H0 and selected val policy after main lock, use existing K1 seed42 Future with B0 Current, no retrain'),test='H0 + unique validation-selected checkpoint only; test exposed historically; no test tuning',hazard_mapping=dict(switch_cost=c['hazard_switch_cost'],hold_cost=c['hazard_hold_cost'],request='next_max>=.5 AND expected switch cost<hold cost',ablation='same weights CDE mass>=.5, val-only diagnostic, not selection eligible'),smoke='seed42 two epochs, no architecture choice')
 lock(out/'protocol.lock.json',p);write(out/'feature_schema.json',p['feature_schema'])
 return dict(protocol_sha256=sha(out/'protocol.lock.json'),input_identity_sha256=sha(out/'input_identity.json'),oof_sha256=sha(out/'oof_teacher_manifest.json'))


def saved_replay(out,name,split,e,r):
 eid=e['metadata']['episode_id'];p=out/'replays'/name/split/(eid+'.json')
 if p.exists():raise FileExistsError('Replay exists: '+str(p))
 sr=score(e['metadata'],r);write(p,dict(replay=clean({k:v for k,v in r.items() if k!='records'}),metrics=sr))
 if r.get('records'):
  pred=out/'predictions'/name/split/(eid+'.jsonl');pred.parent.mkdir(parents=True,exist_ok=True)
  with pred.open('w') as f:
   for record in r['records']:f.write(json.dumps(clean(record),ensure_ascii=False,separators=(',',':'))+'\n')
 return sr


def evaluate_set(episodes,policy,out,name,split,use_oof=False):
 rr=[];rollouts={}
 for e in episodes:
  rows=e['oof'] if use_oof else e['rows'];eid=e['metadata']['episode_id'];duration=e['metadata']['length']/e['metadata']['fps'];r=h0(rows,duration) if policy is None else run(rows,duration,policy,session=eid)
  rr.append(saved_replay(out,name,split,e,r));rollouts[eid]=r
 return aggregate(rr),rollouts


def h0_states(episodes,use_oof):
 out={}
 for e in episodes:
  eid=e['metadata']['episode_id'];rows=e['oof'] if use_oof else e['rows'];duration=e['metadata']['length']/e['metadata']['fps'];r=run(rows,duration,RulePolicy('H0'),session=eid);baseline=h0(rows,duration)
  if r['active_intervals']!=baseline['active_intervals']:raise ValueError('Execution shell H0 behavior parity failed '+eid)
  out[eid]=r
 return out


def phase_a(c,out):
 entries=load(c);parts={s:[e for e in entries if e['metadata']['split']==s] for s in ['train','val','test']};seal_protocol(c,out)
 totals={};parity=[]
 for e in parts['train']+parts['val']:
  eid=e['metadata']['episode_id'];ref=read(Path(c['memory_experiment'])/'evaluations/B0-frozen'/e['metadata']['split']/'handshake/C0'/(eid+'.json'))['replay'];actual=h0(e['rows'],e['metadata']['length']/e['metadata']['fps'])
  if actual!=ref:raise ValueError('Historical H0 events/config/intervals mismatch '+eid)
  parity.append(eid)
 h,rr=evaluate_set(parts['val'],None,out,'H0','val');totals['H0']=h
 if h['strict_hits']!=23 or abs(h['wrong_active_s']-98.23333333333335)>1e-6 or h['eligible']!=54:raise ValueError('H0 expected val metrics differ')
 write(out/'h0_parity.json',dict(exact_episode_replays=len(parity),episode_ids=parity,val_strict_hits=h['strict_hits'],val_eligible=h['eligible'],val_wrong_seconds=h['wrong_active_s'],full_events_and_active_equal=True,test_scored=False))
 for name in ['R1','R2']:totals[name],_=evaluate_set(parts['val'],RulePolicy(name),out,name,'val')
 write(out/'summary/rules_val.json',totals);print('PHASE_A_PASS',[(k,r['strict_hits'],r['wrong_active_s']) for k,r in totals.items()],flush=True)


def build_training(c,out):
 entries=load(c);parts={s:[e for e in entries if e['metadata']['split']==s] for s in ['train','val','test']};trainroll=h0_states(parts['train'],True);valroll=h0_states(parts['val'],False);td=dataset(parts['train'],trainroll);vd=dataset(parts['val'],valroll)
 audit=dict(samples=len(td['labels']),train_sources=75,action_counts=np.bincount([r['action'] for r in td['labels']],minlength=5).tolist(),time_counts={str(k):sum(r['time_bin']==k for r in td['labels']) for k in range(-100,6) if k==-100 or k>=0},positive_gate=sum(r['gate'] for r in td['labels']),late_catchup=sum(r['late_catchup'] for r in td['labels']),behavior_policy='H0 on OOF',sensitivity={})
 for delta in [-.1,-1/30,1/30,.1]:
  changed=dataset(parts['train'],trainroll,delta)['labels'];audit['sensitivity'][str(delta)]=dict(action_changes=sum(a['action']!=b['action'] for a,b in zip(td['labels'],changed)),time_changes=sum(a['time_bin']!=b['time_bin'] for a,b in zip(td['labels'],changed)))
 write(out/'train_labels_audit.json',audit)
 return parts,td,vd,trainroll


def mismatch(ours,base):
 wrong=duration=0.
 for eid,rr in ours.items():
  for a in rr['active_intervals']:
   duration+=a['end_s']-a['start_s']
   for b in base[eid]['active_intervals']:
    if a['stage']!=b['stage']:wrong+=max(0,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
 return wrong/duration


def self_rollout(episodes,policy):
 return {e['metadata']['episode_id']:run(e['oof'],e['metadata']['length']/e['metadata']['fps'],policy,session=e['metadata']['episode_id']) for e in episodes}


def select(summary,c):
 base=summary['H0'];risk=lambda r:r['illegal_actual_commits']==0 and r['hard_timeouts']<=base['hard_timeouts'] and r['early_ge05']<=base['early_ge05'] and r['missed']<=base['missed'] and r['unmatched_commits']<=base['unmatched_commits']
 audit={};candidates=[]
 for name in ['R1','R2']:
  r=summary[name];ok=risk(r);audit[name]=dict(pass_risk=ok)
  if ok:candidates.append((name,r))
 for group in c['policies']:
  rows=[summary[f'{group}_s{seed}'] for seed in c['seeds']];good=[risk(r) for r in rows];audit[group]=dict(non_degrade_seeds=sum(good),pass_risk=sum(good)>=c['min_non_degrading_seeds'],requirement='at least2/3 risk-passing seeds; final checkpoint must itself pass')
  if sum(good)>=c['min_non_degrading_seeds']:
   # Risk-equal group, choose median-ranked seed by val strict/wrong/order, not test.
   names=[f'{group}_s{seed}' for seed in c['seeds']];ranked=sorted([(n,r) for n,r,ok in zip(names,rows,good) if ok],key=lambda v:(-v[1]['strict_hits'],v[1]['wrong_active_s'],-v[1]['exact_order'],v[0]));candidates.append(ranked[len(ranked)//2])
 candidates=[v for v in candidates if v[1]['strict_hits']>base['strict_hits'] or (v[1]['strict_hits']==base['strict_hits'] and v[1]['wrong_active_s']<base['wrong_active_s']-1e-7)]
 winner=min(candidates,key=lambda v:(-v[1]['strict_hits'],v[1]['wrong_active_s'],-v[1]['exact_order'],v[0]))[0] if candidates else 'KEEP_H0'
 return dict(selected=winner,audit=audit,criteria='fixed val risk -> strict hits -> wrong seconds -> order; median seed within qualifying group',no_test_used=True)


def execute(c,out,smoke=False):
 bindings=seal_protocol(c,out);parts,td,vd,base_states=build_training(c,out)
 if smoke:
  for group in c['policies']:fit(group,42,td,vd,c,out/'smoke'/group,bindings,True)
  write(out/'smoke_receipt.json',dict(groups=c['policies'],seed=42,epochs=2,test_scored=False));return
 if not (out/'smoke_receipt.json').exists():raise ValueError('Run --smoke first')
 summary=read(out/'summary/rules_val.json');model_meta={}
 for group in c['policies']:
  for seed in c['seeds']:
   name=f'{group}_s{seed}';folder=out/'decision_models'/group/f'seed{seed}';policy,meta=fit(group,seed,td,vd,c,folder/'v0',bindings)
   ours=self_rollout(parts['train'],policy);disagreement=mismatch(ours,base_states);version='v0'
   if disagreement>c['aggregation_threshold']:
    own=dataset(parts['train'],ours);aggregate_data=combine(td,own);policy,meta=fit(group,seed,aggregate_data,vd,c,folder/'v1',bindings);version='v1'
   write(folder/'aggregation.json',dict(train_only=True,active_time_disagreement=disagreement,threshold=.1,rounds=int(version=='v1'),chosen_version=version,v1_used_without_comparing_val_Active=True))
   model_meta[name]=dict(path=str(folder/version/'best_policy.pt'),bindings=bindings,manifest=meta,aggregation=read(folder/'aggregation.json'))
   summary[name],_=evaluate_set(parts['val'],policy,out,name,'val');final_train,_=evaluate_set(parts['train'],policy,out,name,'train_oof',True);write(folder/'train_rollout_metrics.json',final_train)
   if group=='L3a':
    policy.mapping='mass';summary[name+'_mass'],_=evaluate_set(parts['val'],policy,out,name+'_mass','val');policy.mapping='cost'
   write(out/'summary/val.partial.json',summary);write(out/'model_index.json',model_meta);print('VAL',name,summary[name]['strict_hits'],summary[name]['wrong_active_s'],flush=True)
 decision=select(summary,c);write(out/'selection.lock.json',decision)
 receipt=dict(protocol_sha256=bindings['protocol_sha256'],selected=decision['selected'],val_sha256=json_sha(summary),model_sha256={n:m['manifest']['checkpoint_sha256'] for n,m in model_meta.items()},test_source_ids=read(out/'split.lock.json')['test'],historically_exposed=True)
 lock(out/'test_lock_receipt.json',receipt)
 historical={};historical['H0'],_=evaluate_set(parts['test'],None,out,'H0','test')
 selected=decision['selected']
 if selected!='KEEP_H0':
  if selected in ['R1','R2']:policy=RulePolicy(selected)
  else:
   record=model_meta[selected];saved,model=load_checkpoint(record['path'],bindings,c);policy=LearnedPolicy(saved['name'],model,saved['stats'],c);policy.version=record['manifest']['checkpoint_sha256']
  historical[selected],_=evaluate_set(parts['test'],policy,out,selected,'test')
 write(out/'summary/val_overall.json',summary);write(out/'summary/historical_test_locked.json',historical);write(out/'run_complete.json',dict(models=15,seed_count=3,new_encoder_calls=0,new_joint_teachers_trained=0,selected=selected))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/tomato_learned_handshake_v2_20261112.yaml');p.add_argument('--output-dir',required=True,type=Path);m=p.add_mutually_exclusive_group(required=True)
 for flag in ['check-only','phase-a','smoke','run','verify','report-only']:m.add_argument('--'+flag,action='store_true')
 a=p.parse_args();c=config(a.config);out=a.output_dir.resolve()
 for protected in [Path(c['baseline_root']).resolve(),Path(c['memory_experiment']).resolve()]:
  if out==protected or out.is_relative_to(protected):raise ValueError('Output inside protected input')
 out.mkdir(parents=True,exist_ok=True)
 try:
  if a.check_only:
   if (out/'input_identity.json').exists():raise FileExistsError('Identity exists')
   check(c,out)
  elif a.verify:write(out/'integrity.json',dict(unchanged=True,protected_files=verify_sources(read(out/'protected_sha256.json'))))
  elif a.phase_a:phase_a(c,out)
  elif a.report_only:
   from .report import generate
   generate(c,out)
  else:execute(c,out,a.smoke)
 except Exception:
  write(out/f'failure_{time.time_ns()}.json',dict(command=sys.argv,traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
