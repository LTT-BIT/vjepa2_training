import time,json
from pathlib import Path
import numpy as np
import torch
from tools.tomato_runtime import initialize_runtime,hardware_report
from .contract import *
from .features import schema
from .policy import GateNet,GatePolicy,load_checkpoint
from .data import audit


def tensors(data,stats,device):
 seq=(data['sequence']-stats['mean'])/stats['std'];seq[~data['mask']]=0
 return dict(sequence=torch.from_numpy(seq).to(device),mask=torch.from_numpy(data['mask']).to(device),weight=torch.tensor(data['weights'],device=device),gate=torch.tensor([r['gate'] for r in data['labels']],device=device),gate_valid=torch.tensor([r.get('gate_valid',True) for r in data['labels']],device=device),allowed=torch.tensor(np.array([r['hazard_allowed'] for r in data['labels']]),device=device),hazard_valid=torch.tensor([r['hazard_valid'] for r in data['labels']],device=device))


def objective(model,t,ids,pos_weight,c,regularize=False):
 z=model(t['sequence'][ids],t['mask'][ids]);w=t['weight'][ids];gate=torch.nn.functional.binary_cross_entropy_with_logits(z['gate'],t['gate'][ids],pos_weight=pos_weight,reduction='none');wg=w*t.get('gate_valid',torch.ones_like(t['gate'],dtype=torch.bool))[ids];bce=(wg*gate).sum()/wg.sum().clamp_min(1e-9)
 hz=bce*0
 if model.group=='G3':
  valid=t['hazard_valid'][ids]
  if valid.any():
   logp=z['hazard'][valid].log_softmax(1);allowed=t['allowed'][ids][valid];nll=-torch.logsumexp(logp.masked_fill(~allowed,-torch.inf),dim=1);hz=(nll*w[valid]).sum()/w[valid].sum()
 total=bce+c['hazard_lambda']*hz
 if model.group=='G0' and regularize:total=total+.5*c['explicit_l2']*model.gate.weight.square().sum()
 return total,bce,hz


def fit(group,seed,train,val,stats,c,out,bindings,smoke=False):
 out=Path(out)
 if out.exists():raise FileExistsError('Training output exists: '+str(out))
 out.mkdir(parents=True);initialize_runtime(dict(seed=seed,device=c['device']));torch.set_num_threads(1);tt=tensors(train,stats,c['device']);vv=tensors(val,stats,c['device']);model=GateNet(group,c['dropout']).to(c['device']);opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0 if group=='G0' else c['weight_decay'])
 valid=np.array([r['gate_valid'] for r in train['labels']]);positive=float((train['weights']*valid*np.array([r['gate'] for r in train['labels']])).sum());negative=float((train['weights']*valid).sum())-positive;pw=torch.tensor(min(10,negative/max(positive,1e-9)),device=c['device']);rng=np.random.default_rng(seed);best=float('inf');stale=0;updates=0;orders=[];start=time.perf_counter()
 stats_sha=json_sha({k:v.tolist() for k,v in stats.items()});write(out/'normalization.json',{k:v.tolist() for k,v in stats.items()});write(out/'config.json',dict(config=c,group=group,seed=seed,bindings=bindings,stats_sha256=stats_sha))
 if c['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,(2 if smoke else c['epochs'])+1):
  model.train();order=rng.permutation(len(train['labels']));orders.append(json_sha(order.tolist()));values=[];max_grad=0.
  for start_id in range(0,len(order),c['batch_size']):
   ids=torch.as_tensor(order[start_id:start_id+c['batch_size']],device=c['device']);opt.zero_grad(set_to_none=True);total,bce,hz=objective(model,tt,ids,pw,c,True)
   if not torch.isfinite(total):raise ValueError('Nonfinite gate/hazard objective')
   total.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'],error_if_nonfinite=True);max_grad=max(max_grad,float(gn));opt.step();updates+=1;values.append([float(total.detach()),float(bce.detach()),float(hz.detach())])
  model.eval()
  with torch.inference_mode():vtotal,vbce,vhz=objective(model,vv,slice(None),pw,c)
  row=dict(epoch=epoch,updates=updates,train_total=float(np.mean(values,axis=0)[0]),train_bce=float(np.mean(values,axis=0)[1]),train_hazard=float(np.mean(values,axis=0)[2]),val_gate_bce=float(vbce),val_hazard_nll=float(vhz),val_total=float(vtotal),max_preclip_grad=max_grad)
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  key=float(vbce)  # identical main-gate validation criterion for all four groups
  if key<best-1e-8:
   best=key;stale=0;torch.save(dict(format='candidate-gate-v3',group=group,seed=seed,best_epoch=epoch,bindings=bindings,schema=schema(),stats=stats,stats_sha256=stats_sha,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),out/'best.pt')
  else:stale+=1
  if epoch%10==0:print(group,seed,epoch,'val_BCE',key,flush=True)
  if stale>=c['patience']:break
 (out/'checkpoint.sha256').write_text(sha(out/'best.pt')+'\n');saved,model=load_checkpoint(out/'best.pt',bindings,c);peak=torch.cuda.max_memory_allocated() if c['device'].startswith('cuda') else 0
 meta=dict(group=group,seed=seed,best_epoch=saved['best_epoch'],epochs=epoch,updates=updates,updates_per_epoch=int(np.ceil(len(train['labels'])/c['batch_size'])),parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=peak,stats_sha256=stats_sha,batch_order_sha256=orders,train_audit=audit(train),val_audit=audit(val),pos_weight=float(pw),checkpoint_sha256=sha(out/'best.pt'),bindings=bindings,hardware=hardware_report(),regularization='explicit .5*.01*||W||², bias excluded, optimizer decay0' if group=='G0' else 'AdamW decoupled .01',early_stopping='val main gate weighted BCE only')
 write(out/'manifest.json',meta);del tt,vv
 if c['device'].startswith('cuda'):torch.cuda.empty_cache()
 return GatePolicy(group,model,stats,meta['checkpoint_sha256']),meta
