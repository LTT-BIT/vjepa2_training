"""Event/source-weighted supervision with at most one locked train-only aggregation."""
import time,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tools.tomato_runtime import initialize_runtime
from .contracts import write,sha,json_sha
from .models import PolicyNet,LearnedPolicy,load_checkpoint
from .targets import label,weights


def dataset(episodes,rollouts,shift=0.):
 features=[];labels=[];sources=[]
 for e in episodes:
  for record in rollouts[e['metadata']['episode_id']]['records']:
   f=record['features'];target=label(e['metadata'],f,shift)
   if not target['supervision_valid']:continue
   features.append(f);labels.append(target);sources.append(e['metadata']['source_recording_id'])
 return dict(flat=np.array([f['flat'] for f in features]),sequence=np.array([f['sequence'] for f in features]),state=np.array([f['state'] for f in features]),mask=np.array([f['mask'] for f in features]),labels=labels,sources=sources,weights=weights(sources,labels))


def combine(a,b):
 out={k:np.concatenate([a[k],b[k]]) for k in ['flat','sequence','state','mask']};out['labels']=a['labels']+b['labels'];out['sources']=a['sources']+b['sources'];out['weights']=weights(out['sources'],out['labels']);return out


def fit_stats(data):
 stats={}
 for k in ['flat','sequence','state']:
  values=data[k][data['mask']] if k=='sequence' else data[k]
  stats[k+'_mean']=values.mean(0,dtype=np.float64).astype(np.float32);stats[k+'_std']=values.std(0,dtype=np.float64).clip(.05).astype(np.float32)
 return stats


def tensors(data,stats,device):
 out={k:torch.from_numpy((data[k]-stats[k+'_mean'])/stats[k+'_std']).to(device) for k in ['flat','sequence','state']};out['mask']=torch.from_numpy(data['mask']).to(device);out['sequence'][~out['mask']]=0
 out['weight']=torch.tensor(data['weights'],device=device)
 for k in ['action','gate','next_stage','time_bin']:out[k]=torch.tensor([r[k] for r in data['labels']],device=device,dtype=torch.float32 if k=='gate' else torch.long)
 return out


def class_weight(y,w,n):
 mass=np.bincount(y[y>=0],weights=w[y>=0],minlength=n);out=np.sqrt(mass.sum()/np.maximum(mass,1e-6));out[mass==0]=0;return (out/out[out>0].mean()).astype(np.float32)


def loss(model,t,ids,cw,pos_weight,regularize=True):
 z=model(t['flat'][ids],t['sequence'][ids],t['state'][ids],t['mask'][ids]);w=t['weight'][ids]
 def avg(v,mask=None):
  ww=w if mask is None else w[mask]
  return (v*ww).sum()/ww.sum().clamp(min=1e-9)
 if model.name in ['L0','L1a']:
  v=nn.functional.binary_cross_entropy_with_logits(z['action'][:,0],t['gate'][ids],pos_weight=pos_weight,reduction='none');return avg(v)+(.5*model.l2_coefficient*model.action.weight.square().sum() if model.name=='L0' and regularize else 0.)
 if model.name=='L3a':
  nxt=t['next_stage'][ids];tm=t['time_bin'][ids];valid=nxt>=0;vt=tm>=0
  a=avg(nn.functional.cross_entropy(z['action'][valid],nxt[valid],weight=cw['next_stage'],reduction='none'),valid) if valid.any() else z['action'].sum()*0
  b=avg(nn.functional.cross_entropy(z['time'][vt],tm[vt],weight=cw['time_bin'],reduction='none'),vt) if vt.any() else z['time'].sum()*0
  return a+b
 return avg(nn.functional.cross_entropy(z['action'],t['action'][ids],weight=cw['action'],reduction='none'))


def fit(name,seed,train,val,c,folder,bindings,smoke=False):
 folder=Path(folder)
 if folder.exists():raise FileExistsError('Refusing training overwrite '+str(folder))
 folder.mkdir(parents=True);initialize_runtime(dict(device=c['device'],seed=seed));torch.set_num_threads(1)
 stats=fit_stats(train);tt=tensors(train,stats,c['device']);vv=tensors(val,stats,c['device']);model=PolicyNet(name,dropout=c['dropout']).to(c['device']);model.l2_coefficient=c['weight_decay'];opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0. if name=='L0' else c['weight_decay']);rng=np.random.default_rng(seed)
 cw={k:torch.tensor(class_weight(np.array([r[k] for r in train['labels']]),train['weights'],n),device=c['device']) for k,n in [('action',5),('next_stage',4),('time_bin',6)]}
 positive=sum(w*r['gate'] for w,r in zip(train['weights'],train['labels']));negative=float(train['weights'].sum())-positive;pw=torch.tensor(min(10,negative/max(positive,1e-9)),device=c['device'])
 write(folder/'train_stats.json',{k:v.tolist() for k,v in stats.items()});statsha=json_sha({k:v.tolist() for k,v in stats.items()});best=float('inf');stale=0;updates=0;start=time.perf_counter();order_hash=[]
 for epoch in range(1,(2 if smoke else c['epochs'])+1):
  model.train();order=rng.permutation(len(train['labels']));order_hash.append(json_sha(order.tolist()));trainloss=[];gradmax=0.
  for a in range(0,len(order),c['batch_size']):
   ids=torch.as_tensor(order[a:a+c['batch_size']],device=c['device']);opt.zero_grad(set_to_none=True);v=loss(model,tt,ids,cw,pw);v.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'],error_if_nonfinite=True);gradmax=max(gradmax,float(gn));opt.step();updates+=1;trainloss.append(float(v.detach()))
  model.eval()
  with torch.no_grad():validation=float(loss(model,vv,slice(None),cw,pw,regularize=False))
  row=dict(epoch=epoch,train_loss=float(np.mean(trainloss)),val_supervised_loss=validation,max_preclip_gradient_norm=gradmax,updates=updates)
  with (folder/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if validation<best-1e-8:
   best=validation;stale=0;torch.save(dict(format='tomato-learned-handshake-v2',name=name,seed=seed,best_epoch=epoch,schema=dict(flat=92,state=48,sequence=[4,11]),bindings=bindings,stats=stats,stats_sha256=statsha,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),folder/'best_policy.pt')
  else:stale+=1
  if epoch%10==0:print(name,seed,epoch,'val_loss',validation,flush=True)
  if stale>=c['patience']:break
 (folder/'checkpoint.sha256').write_text(sha(folder/'best_policy.pt')+'\n');saved,model=load_checkpoint(folder/'best_policy.pt',bindings,c)
 meta=dict(regularization='explicit 0.5*0.01*sum(nonbias W^2), optimizer decay0' if name=='L0' else 'AdamW decoupled decay .01',name=name,seed=seed,best_epoch=saved['best_epoch'],epochs=epoch,updates=updates,train_samples=len(train['labels']),val_samples=len(val['labels']),parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-start,checkpoint_sha256=sha(folder/'best_policy.pt'),train_stats_sha256=statsha,batch_order_sha256=order_hash,positive_gate=int(sum(r['gate'] for r in train['labels'])),positive_action=int(sum(r['action']>0 for r in train['labels'])),class_weights={k:v.cpu().tolist() for k,v in cw.items()},pos_weight=float(pw),bindings=bindings)
 write(folder/'manifest.json',meta)
 del tt,vv
 if c['device'].startswith('cuda'):torch.cuda.empty_cache()
 policy=LearnedPolicy(name,model,stats,c);policy.version=meta['checkpoint_sha256']
 return policy,meta
