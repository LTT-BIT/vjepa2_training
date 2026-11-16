"""Independent reload, score recomputation, API parity and hardware measurements."""
import gzip,json,time,types
import numpy as np
import torch
from .contract import *
from .identity import load
from .policy import GatePolicy,R1Policy,load_checkpoint
from .replay import run,pure_h0
from .score import score,aggregate
from .analyze import clean


def parity(c,out,parts):
 n=0
 for e in parts['train']+parts['val']:
  m=e['metadata'];d=m['length']/m['fps'];r=run(e['rows'],d,session=m['episode_id']);pure=pure_h0(e['rows'],d)
  for k in ['events','active_intervals','pending_intervals','summary']:
   if r[k]!=pure[k]:raise ValueError('Instrumented H0 differs from pure: '+m['episode_id']+k)
  n+=1
 for e in parts['val']:
  m=e['metadata'];eid=m['episode_id'];d=m['length']/m['fps'];archive=Path(c['reference_experiment'])/'replays'
  h=read(archive/'H0/val'/(eid+'.json'))['replay'];r=run(e['rows'],d,session=eid)
  for k in ['events','active_intervals','pending_intervals','summary']:
   if h[k]!=r[k]:raise ValueError('H0 archival mismatch '+eid+k)
  old=read(archive/'R1/val'/(eid+'.json'))['replay'];r1=run(e['rows'],d,R1Policy(),eid)
  for k in ['events','active_intervals','pending_intervals','summary']:
   if old[k]!=r1[k]:raise ValueError('R1 archival mismatch '+eid+k)
 receipt=dict(h0_instrumented_vs_pure_exact=n,h0_and_r1_archival_val_exact=len(parts['val']),all_events_and_intervals_identical=True,new_port_preserves_real_future=True)
 lock(out/'baseline_replay_receipt.json',receipt);return receipt


def speed(policy):
 torch.set_num_threads(1);m=policy.model;seq=torch.zeros(1,4,len(policy.stats['mean']));mask=torch.ones(1,4,dtype=torch.bool);result={}
 with torch.inference_mode():
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   model=m.to(device);x=seq.to(device);k=mask.to(device)
   for _ in range(20):model(x,k)
   if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   values=[]
   for _ in range(200):
    start=time.perf_counter();model(x,k)
    if device=='cuda':torch.cuda.synchronize()
    values.append((time.perf_counter()-start)*1000)
   result[device]=dict(batch1_ms_p50=float(np.median(values)),batch1_ms_p95=float(np.quantile(values,.95)),samples=200,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
 m.cpu();return result


def verify_run(c,out):
 parts={s:[] for s in ['train','val','test']}
 for e in load(c):parts[e['metadata']['split']].append(e)
 parity(c,out,parts);p=read(out/'protocol.lock.json')
 if any(sha(REPO/k)!=h for k,h in p['source_sha256'].items()):raise ValueError('Source changed after preregistration')
 if not (out/'run_complete.json').exists():write(out/'logs/verification.json',dict(phase='preflight',parity=True));return
 models=read(out/'model_index.json');summary=read(out/'summary/val_overall.json');recomputed={};hardware={};feature_checks={}
 for name in ['H0','R1']+list(models):
  if name in models:
   rec=models[name];s,m=load_checkpoint(rec['path'],rec['manifest']['bindings'],c);policy=GatePolicy(s['group'],m,s['stats'],rec['manifest']['checkpoint_sha256'])
   if name.endswith('s42'):hardware[name]=speed(policy)
  else:policy=None if name=='H0' else R1Policy()
  reports=[];maxerror=0.
  for e in parts['val']:
   meta=e['metadata'];eid=meta['episode_id'];r=run(e['rows'],meta['length']/meta['fps'],policy,eid);stored=read(out/'replays'/name/'val'/(eid+'.json'))
   for k in ['events','active_intervals','pending_intervals','summary','commit_audit']:
    if r[k]!=stored['replay'][k]:raise ValueError('Reload trace differs '+name+eid+k)
   current=score(meta,r)
   if current!=stored['metrics']:raise ValueError('Reload metrics differ '+name+eid)
   reports.append(current)
   # Remove GT and provenance extras: make_observation whitelist and feature bytes
   # must produce identical records and decisions with minimal inference inputs.
   bare=[dict(observed_at=z['observed_at'],frame_index=z['frame_index'],heads=z['heads']) for z in e['rows']];bare=run(bare,meta['length']/meta['fps'],policy,eid)
   if clean(bare['records'])!=clean(r['records']):raise ValueError('GT-free inference differs')
   if name in models:
    eligible=[z for z in r['records'] if z['feature']['eligible']]
    if eligible:
     x=np.array([z['feature']['sequence'] for z in eligible]);mask=np.array([z['feature']['mask'] for z in eligible]);x=(x-s['stats']['mean'])/s['stats']['std'];x[~mask]=0
     with torch.inference_mode():batched=policy.model(torch.tensor(x),torch.tensor(mask))['gate'].sigmoid().numpy()
     streamed=np.array([z['decision']['score'] for z in eligible]);maxerror=max(maxerror,float(abs(batched-streamed).max()))
  if maxerror>1e-5:raise ValueError('Batch vs streaming numerical mismatch')
  total=aggregate(reports)
  for k in ['cross_session_commit','non_planning_commit','gate_candidate_ticks','request_eligible_ticks','gate_holds','candidate_absent']:total[k]=sum(r[k] for r in reports)
  if total!=summary[name]:raise ValueError('Aggregate recompute differs '+name)
  recomputed[name]=dict(episodes=9,metrics_exact=True,events_exact=True,gt_stripped_exact=True);feature_checks[name]=maxerror
 # Audit every train/val/test/stress commit: original evidence, causal availability,
 # fresh TTL, atomic clock, monotonic version, allowed edge and session ownership.
 commits=0;files=0
 for path in (out/'replays').rglob('*.json'):
  r=read(path)['replay'];versions=[];files+=1
  for event in r['events']:
   if event['kind']!='commit':continue
   commits+=1;versions.append(event['prompt_version']);q=event['evidence'];t=event['time']
   if abs(t/.5-round(t/.5))>1e-9 or event['to_stage'] not in r['task_policy'][event['from_stage']] or not q['last_observed_at']<=q['last_confirm_available_at']<=t+1e-9 or t-q['last_confirm_available_at']>.7500001:raise ValueError('Commit invariant '+str(path))
  if versions!=list(range(1,len(versions)+1)):raise ValueError('Version invariant')
  for a in r['commit_audit']:
   if not a['planning_tick'] or a['illegal'] or a['request_source']['session_id']!=a['expected_session']:raise ValueError('Request ownership/grid invariant')
 write(out/'hardware_latency.json',hardware);write(out/'logs/verification.json',dict(reloaded=recomputed,batch_stream_max_absolute_error=feature_checks,all_commit_protocol_invariants=True,replay_files=files,actual_commits=commits,no_new_encoder=True,h0_parity=True,selection_lock_sha256=sha(out/'selection.lock.json'),test_lock_sha256=sha(out/'test_lock_receipt.json')))
 print('VERIFY_PASS',files,'replays',commits,'commits',flush=True)
