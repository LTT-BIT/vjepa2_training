"""Staged, validation-only temporal experiments with immutable inputs and complete provenance."""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_boundary_metrics as bm
from . import train_tomato_vjepa_phase as core,tomato_temporal_heads as tm
from .tomato_runtime import initialize_runtime,hardware_report
ROOT=Path('runs/tomato/temporal_20261108')
PRIOR=Path('runs/tomato/optimization_20261108')
REUSE_ROOT=None
RULE=dict(version=1,split='original train/val only; historical test excluded from prediction/search',
 main_horizon_s=1.5,reason='Earlier locked validation experiment favored 1.5s anticipation; keep 1s original baseline separately.',
 selection_epoch='Fixed (Current F1 + .5 Future F1)/1.5, then lower NLL; independent of optimized loss weights.',
 pilot_seed=42,final_seeds=[42,43,44],context_anchors=4,epochs=100,patience=15,batch_size=128,
 pilot_groups=dict(loss=['loss11','loss12','focal'],temporal=['gru','tcn'],fusion=['gated','velocity','split']),
 pilot_filter='Current F1 at least baseline seed42 minus .005; one candidate/group by future recall - .1 false/min + .01 Current F1 at .5/K2. Boundary auxiliary advances if same guard and better rank or AP improves .02.',
 final_rank='Feasible first; then mean future recall - .1 false/min + .01 currentF1 at .5/K2; fixed seed42 representative.',
 reliable_gates=dict(current_f1_drop_max=.005,current_recall_min=.9,current_delay_p95_max=1.5,
 future_recall_min=.8,future_false_per_min_max=1.),
 fallback='Do not replace stable original Joint seed42 without reliable gates. Research winner may be exported.',
 cv='Top research candidate vs matched-horizon MLP: three source-grouped folds on train+val86 only, seed42 diagnostic. No test membership or data.',
 no_cartesian_search=True)


def lock(path,value):
 if path.exists():
  if ds.read_json(path)!=value:raise ValueError('Immutable experiment changed: '+str(path))
 else:ds.write_json(path,value)


def freeze():
 prior={str(p):ds.digest(p) for base in [Path('runs/tomato/campaign94_20261107'),PRIOR]+([REUSE_ROOT] if REUSE_ROOT else []) for p in sorted(base.rglob('*')) if p.is_file()}
 lock(ROOT/'frozen_prior.json',prior)
 return dict(passed=True,files=len(prior),sha256=ds.digest(ROOT/'frozen_prior.json'))


def prepare():
 ROOT.mkdir(parents=True,exist_ok=True);freeze();lock(ROOT/'protocol.json',RULE)
 cfg=ds.load_config('configs/tomato_campaign94.yaml');manifest=core.load_locked_manifest(cfg)
 directory=ROOT/'prepared';directory.mkdir(exist_ok=True)
 for n in ['manifest.json','split.json','validation_receipt.json','validation_lock.json','feature_index.json']:lock(directory/n,ds.read_json(Path(cfg['output_dir'])/n))
 entries=core.load_training_arrays({**cfg,'output_dir':str(directory)},manifest)
 # No test features enter any optimizer, prediction, early stopping or cross-validation.
 return cfg,manifest,{s:entries[s] for s in ('train','val')}


def binary_scores(prob,y):
 p=np.asarray(prob);y=np.asarray(y,dtype=bool);order=np.argsort(-p,kind='stable');sy=y[order];sp=p[order]
 ends=np.r_[np.flatnonzero(np.diff(sp)!=0),len(sp)-1];tp=np.cumsum(sy)[ends];fp=ends+1-tp
 recall=tp/max(1,int(y.sum()));precision=tp/(tp+fp)
 ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
 tpr=np.r_[0.,recall,1.];fpr=np.r_[0.,fp/max(1,int((~y).sum())),1.]
 return dict(samples=len(y),positives=int(y.sum()),prevalence=float(y.mean()),average_precision=ap,
             roc_auc=float(np.trapz(tpr,fpr)),brier=float(np.mean((p-y)**2)),
             thresholds={str(t):dict(precision=float(y[p>=t].mean()) if np.any(p>=t) else 0.,recall=float((p[y]>=t).mean()),
               false_positive_rate=float((p[~y]>=t).mean())) for t in [.3,.5,.7,.9]})


def evaluate(outputs,entries,horizon):
 result={};n=sum(len(e['features']) for e in entries)
 for h in ['current','future']:
  y=np.concatenate([r[h+'_labels'] for r in entries]);keep=np.ones(n,bool) if h=='current' else np.concatenate([r['future_valid_mask'] for r in entries]);z=outputs[h+'_logits']
  result[h]=dict(overall=metrics.scores(z[keep],y[keep]),distance=bm.distance_report(entries,z,h),
   events={f'{p}/{k}':bm.event_report(entries,z,h,horizon,p,k) for p in [.5,.65,.8,.9] for k in [1,2]})
 y=np.concatenate([tm.boundary_targets(r['episode'],r['anchors'],horizon)[0] for r in entries]);valid=np.concatenate([r['future_valid_mask'] for r in entries])
 proxy=1-(metrics.probabilities(outputs['current_logits'])*metrics.probabilities(outputs['future_logits'])).sum(1)
 result['boundary_proxy']=binary_scores(proxy[valid],y[valid])
 if 'boundary_logits' in outputs:
  prob=1/(1+np.exp(-np.clip(outputs['boundary_logits'],-50,50)))
  result['boundary_aux']=binary_scores(prob[valid],y[valid])
 return result


def retarget(entries,horizon):
 return [{**r,**ds.build_phase_targets(r['anchors'],r['episode']['annotation']['segments'],r['episode']['length'],r['episode']['fps'],horizon)} for r in entries]


def train_one(name,spec,seed,entries,manifest,config,subdir='experiments'):
 out=ROOT/subdir/f'{name}_s{seed}';out.mkdir(parents=True,exist_ok=True)
 source={str(p):ds.digest(p) for p in [Path(__file__),Path(tm.__file__),Path(bm.__file__)]}
 identity=dict(spec=spec,seed=seed,sources=source,split={s:[r['episode']['source_recording_id'] for r in rows] for s,rows in entries.items()},
  data_manifest_sha256=ds.digest(ROOT/'prepared/manifest.json'),feature_index_sha256=ds.digest(ROOT/'prepared/feature_index.json'),
  git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),epochs=100,patience=15,learning_rate=3e-4,weight_decay=.01,batch_size=128)
 if REUSE_ROOT is not None:
  old=REUSE_ROOT/subdir/f'{name}_s{seed}'
  if (old/'result.json').exists():
   previous=ds.read_json(old/'identity.json')
   # Explicit read-only reuse of completed artifacts, never resume a changed-code optimizer.
   for key in identity:
    if key not in ('sources','git_sha') and previous[key]!=identity[key]:raise ValueError('Reuse data/recipe identity changed: '+key)
   result=ds.read_json(old/'result.json');assert ds.digest(old/'best.pt')==result['checkpoint_sha256']
   lock(ROOT/'reuse_receipts'/f'{name}_s{seed}.json',dict(original_identity_sha256=ds.digest(old/'identity.json'),
        result_sha256=ds.digest(old/'result.json'),checkpoint_sha256=result['checkpoint_sha256'],source_directory=str(old),
        reason='Completed frozen run reused read-only after position-whitelist bug fix; no changed-code resume. Architecture, data, targets and recipe unchanged.'))
   return result
 lock(out/'identity.json',identity)
 if (out/'result.json').exists():
  result=ds.read_json(out/'result.json');assert ds.digest(out/'best.pt')==result['checkpoint_sha256'];return result
 initialize_runtime(dict(device='cuda',seed=seed));torch.cuda.reset_peak_memory_stats()
 data={s:retarget(rows,spec['horizon_s']) for s,rows in entries.items()}
 raw=np.concatenate([tm.append_velocity(r) if spec['architecture']=='velocity' else r['features'] for r in data['train']])
 mean=raw.mean(0,dtype=np.float64).astype(np.float32);std=raw.std(0,dtype=np.float64).clip(.01).astype(np.float32)
 assert raw.shape[1]==spec['dimension']
 normalized={s:tm.normalized_features(rows,mean,std,spec) for s,rows in data.items()}
 x,ids=normalized['train'];xg=torch.from_numpy(x).cuda();vg=torch.from_numpy(np.concatenate([r['future_valid_mask'] for r in data['train']])).cuda()
 yc=np.concatenate([r['current_labels'] for r in data['train']]);yf=np.concatenate([r['future_labels'] for r in data['train']]);valid=vg.cpu().numpy()
 yg=[torch.from_numpy(v).cuda() for v in [yc,yf]]
 weights=[torch.from_numpy(core.fit_class_weights(v)[0]).cuda() for v in [yc,yf[valid]]]
 by=np.concatenate([tm.boundary_targets(r['episode'],r['anchors'],spec['horizon_s'])[0] for r in data['train']])
 bpos=float(np.sqrt((1-by[valid].mean())/by[valid].mean()));byg=torch.from_numpy(by).cuda()
 head=tm.TemporalHead(spec).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01)
 rng=np.random.default_rng(seed);best=(-1.,-float('inf'));stale=0;history=[];start_epoch=1;began=time.perf_counter()
 if (out/'last_state.pt').exists():
  state=torch.load(out/'last_state.pt',weights_only=True,map_location='cpu');assert state['identity']==identity
  head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);rng.bit_generator.state=state['numpy_rng']
  best=tuple(state['best']);stale=state['stale'];history=state['history'];start_epoch=state['epoch']+1
 for epoch in range(start_epoch,101):
  if stale>=15:break
  head.train();losses=[];ep=time.perf_counter()
  batches=core.training_batches(yf,valid,len(yc),128,rng,'joint')
  for batch in batches:
   ix=torch.as_tensor(batch,device='cuda');outputs=head(xg[torch.as_tensor(ids[batch],device='cuda')]);parts=[]
   for j,h in enumerate(['current','future']):
    keep=torch.ones(len(ix),device='cuda',dtype=torch.bool) if h=='current' else vg[ix];z=outputs[h+'_logits'][keep];y=yg[j][ix][keep]
    losses_ce=nn.functional.cross_entropy(z,y,weight=weights[j],reduction='none')
    if spec.get('focal_gamma',0):losses_ce*=((1-z.softmax(1).gather(1,y[:,None]).squeeze(1))**spec['focal_gamma'])
    parts.append(losses_ce.sum()/weights[j][y].sum())
   loss=(parts[0]+spec['lambda_future']*parts[1])/(1+spec['lambda_future'])
   if spec.get('boundary_weight',0):
    keep=vg[ix];aux=nn.functional.binary_cross_entropy_with_logits(outputs['boundary_logits'][keep],byg[ix][keep],pos_weight=torch.tensor(bpos,device='cuda'))
    loss+=spec['boundary_weight']*aux
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
  prediction=tm.predict(head,*normalized['val']);val={}
  for h in ['current','future']:
   labels=np.concatenate([r[h+'_labels'] for r in data['val']]);keep=np.ones(len(labels),bool) if h=='current' else np.concatenate([r['future_valid_mask'] for r in data['val']]);val[h]=metrics.scores(prediction[h+'_logits'][keep],labels[keep])
  key=((val['current']['macro_f1']+.5*val['future']['macro_f1'])/1.5,-(val['current']['nll']+.5*val['future']['nll'])/1.5)
  row=dict(epoch=epoch,loss=float(np.mean(losses)),val=val,selection_score=key[0],elapsed_s=time.perf_counter()-ep);history.append(row)
  if key>best:
   best=key;stale=0
   payload=dict(format_version=4,architecture_version='tomato-causal-head-v1',spec=spec,training_seed=seed,
    mean=torch.from_numpy(mean),std=torch.from_numpy(std),head={k:v.detach().cpu() for k,v in head.state_dict().items()},
    provenance=identity,pretrained=manifest['pretrained'],feature_version='three-view-mean-rgb-imagenet-v1',best_epoch=epoch,validation=val,
    boundary_pos_weight=bpos,normalization_split='train only',
    input_schema=dict(label_order=list(ds.LABELS),joint_names=manifest['contract']['joint_names'],arms_indices=list(range(14)),hands_indices=list(range(14,58)),
     visual_dimension=3072,state_history_shape=[16,58],feature_hz=2,context=spec['context'],causal=True,feature_sampler_version=config.get('feature_sampler_version',ds.SAMPLER_VERSION),
     history_sample_hz=config.get('history_sample_hz',10),history_contract=config.get('history_contract')))
   torch.save(payload,out/'best.pt')
  else:stale+=1
  torch.save(dict(identity=identity,epoch=epoch,head=head.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state,best=list(best),stale=stale,history=history),out/'last_state.pt')
  (out/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
 saved,head=tm.load_checkpoint(out/'best.pt',device='cuda');prediction=tm.predict(head,*normalized['val']);evaluation=evaluate(prediction,data['val'],spec['horizon_s'])
 np.savez_compressed(out/'val_logits.npz',**prediction)
 # Head timing includes history gather on device, excludes common encoder and transport.
 timings=[];head.eval()
 with torch.inference_mode():
  for i in range(110):
   torch.cuda.synchronize();t=time.perf_counter();head(xg[torch.as_tensor(ids[i%len(ids):i%len(ids)+1],device='cuda')]);torch.cuda.synchronize()
   if i>=10:timings.append((time.perf_counter()-t)*1000)
 result=dict(name=name,seed=seed,spec=spec,checkpoint=str(out/'best.pt'),checkpoint_sha256=ds.digest(out/'best.pt'),best_epoch=saved['best_epoch'],
  epochs=len(history),training_s=sum(v['elapsed_s'] for v in history),wall_s=time.perf_counter()-began,
  parameters=sum(p.numel() for p in head.parameters()),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),runtime=hardware_report(),
  head_latency_ms=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95)),max=float(max(timings))),evaluation=evaluation,
  split_sha256=ds.fingerprint(identity['split']),feature_identity_sha256=identity['feature_index_sha256'],test_predicted=False)
 ds.write_json(out/'result.json',result)
 print('DONE',subdir,name,seed,'F1',round(evaluation['current']['overall']['macro_f1'],4),round(evaluation['future']['overall']['macro_f1'],4),'forecast',evaluation['future']['events']['0.5/2']['recall'],'sec',round(result['training_s'],1),flush=True)
 del xg,vg,head,opt;torch.cuda.empty_cache();return result


def rank(r):
 e=r['evaluation'];f=e['future']['events']['0.5/2']
 return f['recall']-.1*f['false_events_per_minute']+.01*e['current']['overall']['macro_f1']


def run():
 cfg,manifest,entries=prepare();specbase=dict(architecture='mlp',context=1,dimension=4000,horizon_s=1.5,lambda_future=.5,boundary_weight=0.,focal_gamma=0.)
 specs={name:{**specbase,**change} for name,change in dict(mlp_h15={},loss11=dict(lambda_future=1.),loss12=dict(lambda_future=2.),focal=dict(focal_gamma=2.),
 boundary_aux=dict(boundary_weight=.2),gru=dict(architecture='gru',context=4),tcn=dict(architecture='tcn',context=4),
 gated=dict(architecture='gated'),velocity=dict(architecture='velocity',dimension=4058),split=dict(architecture='split'),
 vision_only=dict(architecture='vision_only'),state_only=dict(architecture='state_only')).items()}
 lock(ROOT/'recipes.json',specs);pilots={}
 for name in ['mlp_h15','loss11','loss12','focal','boundary_aux','gru','tcn','gated','velocity','split','vision_only','state_only']:
  pilots[name]=train_one(name,specs[name],42,entries,manifest,cfg)
 base=pilots['mlp_h15'];floor=base['evaluation']['current']['overall']['macro_f1']-.005;final=['mlp_h15']
 for group,names in RULE['pilot_groups'].items():
  eligible=[n for n in names if pilots[n]['evaluation']['current']['overall']['macro_f1']>=floor]
  if eligible:final.append(max(eligible,key=lambda n:(rank(pilots[n]),n)))
 aux=pilots['boundary_aux']
 if aux['evaluation']['current']['overall']['macro_f1']>=floor and (rank(aux)>rank(base) or aux['evaluation']['boundary_aux']['average_precision']>base['evaluation']['boundary_proxy']['average_precision']+.02):final.append('boundary_aux')
 lock(ROOT/'pilot_selection.json',dict(finalists=final,pilot_rank={n:rank(r) for n,r in pilots.items()},current_f1_floor=floor,test_used=False))
 results=list(pilots.values())
 for name in final:
  for seed in [43,44]:results.append(train_one(name,specs[name],seed,entries,manifest,cfg))
 groups={n:[r for r in results if r['name']==n] for n in final};baseline=np.mean([r['evaluation']['current']['overall']['macro_f1'] for r in groups['mlp_h15']]);candidates=[]
 for name,rs in groups.items():
  current=np.mean([r['evaluation']['current']['overall']['macro_f1'] for r in rs]);cr=np.mean([r['evaluation']['current']['events']['0.5/2']['recall'] for r in rs]);delay=np.mean([r['evaluation']['current']['events']['0.5/2']['delay_p95_s'] if r['evaluation']['current']['events']['0.5/2']['delay_p95_s'] is not None else 99 for r in rs]);fr=np.mean([r['evaluation']['future']['events']['0.5/2']['recall'] for r in rs]);ff=np.mean([r['evaluation']['future']['events']['0.5/2']['false_events_per_minute'] for r in rs])
  candidates.append(dict(name=name,mean_rank=float(np.mean([rank(r) for r in rs])),feasible=bool(current>=baseline-.005 and cr>=.9 and delay<=1.5 and fr>=.8 and ff<=1),current_f1=float(current),current_recall=float(cr),current_p95_delay_s=float(delay),future_recall=float(fr),future_false_per_min=float(ff)))
 candidates.sort(key=lambda v:(v['feasible'],v['mean_rank'],v['name']),reverse=True);winner=candidates[0]['name']
 selection=dict(winner=winner,representative_seed=42,reliable_approved=candidates[0]['feasible'],candidates=candidates,
  checkpoint=groups[winner][0]['checkpoint'],checkpoint_sha256=groups[winner][0]['checkpoint_sha256'],protocol_sha256=ds.digest(ROOT/'protocol.json'),
  stable_default='runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt',test_used=False)
 lock(ROOT/'selection.json',selection);print('SELECT',json.dumps(selection),flush=True)
 # Diagnostic group CV only for top final architecture vs same-horizon MLP, no test sources.
 allrows=entries['train']+entries['val'];sources=sorted(r['episode']['source_recording_id'] for r in allrows);np.random.default_rng(20261108).shuffle(sources)
 folds=[set(v.tolist()) for v in np.array_split(np.array(sources),3)];cv=[]
 for fold,validation in enumerate(folds):
  split={s:[r for r in allrows if (r['episode']['source_recording_id'] in validation)==(s=='val')] for s in ['train','val']}
  assert not {r['episode']['source_recording_id'] for r in split['train']}&validation
  for name in sorted(set(['mlp_h15',winner])):cv.append(train_one(name+f'_fold{fold}',specs[name],42,split,manifest,cfg,'group_cv'))
 ds.write_json(ROOT/'results_index.json',dict(experiments=[dict(name=r['name'],seed=r['seed'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in results],cv=[dict(name=r['name'],checkpoint=r['checkpoint']) for r in cv]))
 ds.write_json(ROOT/'freeze_verification.json',freeze());print('ALL STAGES COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));parser.add_argument('--reuse-root');a=parser.parse_args();ROOT=Path(a.output_root);REUSE_ROOT=Path(a.reuse_root) if a.reuse_root else None;run()
