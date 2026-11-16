"""Matched training: original Joint objective/batching and source-isolated histories."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from tools.tomato_phase_model import supervision_loss
from tools.train_tomato_vjepa_phase import training_batches,fit_class_weights,validation_selection
from tools.tomato_metrics import scores
from tools.tomato_runtime import initialize_runtime,hardware_report
from .contracts import write,read,sha,digest_json
from .cache_reader import history_indices,causal_motion,past_beliefs
from .models import make_model,load_memory_checkpoint

class TensorData:
    def __init__(self,entries,c,group,stats=None,beliefs=None):
        self.entries=entries;self.group=group;self.device=c['device'];self.n=c['memory_lengths'].get(group,1);self.slices=[]
        x=np.concatenate([e['features'] for e in entries]);self.mean=x.mean(0,dtype=np.float64).astype(np.float32) if stats is None else stats['mean']
        self.std=x.std(0,dtype=np.float64).clip(.01).astype(np.float32) if stats is None else stats['std']
        self.stats={'mean':self.mean,'std':self.std};aux=[];hist=[];mask=[];dt=[];offset=0
        for e in entries:
            ids,ms,tt=history_indices(e['times'],self.n,c['gap_reset_s']);hist.append(np.where(ms,ids+offset,-1));mask.append(ms);dt.append(tt)
            if group=='K1':aux.append(causal_motion(e['features'],e['times'],c['gap_reset_s']))
            if group=='P1':aux.append(past_beliefs(beliefs[e['episode']['episode_id']],e['times'],c['gap_reset_s']))
            self.slices.append((offset,offset+len(ids)));offset+=len(ids)
        self.x=torch.from_numpy((x-self.mean)/self.std).to(self.device)
        self.ids=torch.from_numpy(np.concatenate(hist)).to(self.device);self.mask=torch.from_numpy(np.concatenate(mask)).to(self.device);self.dt=torch.from_numpy(np.concatenate(dt)).to(self.device)
        self.aux=None
        if aux:
            a=np.concatenate(aux)
            if group=='K1':
                am=a.mean(0,dtype=np.float64).astype(np.float32) if stats is None else stats['aux_mean'];ast=a.std(0,dtype=np.float64).clip(.01).astype(np.float32) if stats is None else stats['aux_std']
                a=(a-am)/ast;self.stats.update(aux_mean=am,aux_std=ast)
            self.aux=torch.from_numpy(a).to(self.device)
        self.yc_np=np.concatenate([e['current_labels'] for e in entries]);self.yf_np=np.concatenate([e['future_labels'] for e in entries]);self.valid_np=np.concatenate([e['future_valid_mask'] for e in entries])
        self.yc=torch.from_numpy(self.yc_np).to(self.device);self.yf=torch.from_numpy(self.yf_np).to(self.device);self.valid=torch.from_numpy(self.valid_np).to(self.device)
    def forward(self,model,ii):
        if self.group in ['B0','B0R','teacher']:return model(self.x[ii])
        ids=self.ids[ii];mask=self.mask[ii];xx=self.x[ids.clamp(min=0)].masked_fill(~mask[:,:,None],0)
        aux=self.aux[ids.clamp(min=0)].masked_fill(~mask[:,:,None],0) if self.aux is not None else None
        return model(xx,mask,self.dt[ii],aux)
    @torch.inference_mode()
    def predict(self,model,batch=512):
        model.eval();out={h:[] for h in ['current','future']}
        for start in range(0,len(self.x),batch):
            z=self.forward(model,slice(start,start+batch))
            for h in out:out[h].append(z[h+'_logits'].cpu().numpy())
        return {h:np.concatenate(v) for h,v in out.items()}

def statistics_identity(stats):return digest_json({k:np.asarray(v).tolist() for k,v in stats.items()})

def fit(entries,val,c,group,seed,out,bindings,beliefs=None,smoke=False):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Refusing existing training directory {out}')
    out.mkdir(parents=True);runtime=initialize_runtime(dict(seed=seed,device=c['device']));torch.set_num_threads(4)
    train=TensorData(entries,c,group,beliefs=beliefs);validation=TensorData(val,c,group,train.stats,beliefs)
    wc,cc=fit_class_weights(train.yc_np);wf,fc=fit_class_weights(train.yf_np[train.valid_np]);weights={h:torch.from_numpy(w).to(c['device']) for h,w in [('current',wc),('future',wf)]}
    model=make_model(group,c).to(c['device']);opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);rng=np.random.default_rng(seed)
    stats_sha=statistics_identity(train.stats);train_bindings={**bindings,'stats_sha256':stats_sha}
    write(out/'train_stats.json',{k:v.tolist() for k,v in train.stats.items()});write(out/'source_membership.json',dict(train=[e['episode']['source_recording_id'] for e in entries],val=[e['episode']['source_recording_id'] for e in val]))
    start=time.perf_counter();best=(-1.,-float('inf'));stale=0;updates=0;peak=0
    if c['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    order_hash=[]
    for epoch in range(1,(2 if smoke else c['epochs'])+1):
        model.train();losses=[];batchloss={h:[] for h in weights}
        batches=training_batches(train.yf_np,train.valid_np,len(train.x),c['batch_size'],rng,'joint')
        order_hash.append(digest_json([ids.tolist() for ids in batches]))
        for ids in batches:
            ids=torch.as_tensor(ids,device=c['device']);opt.zero_grad(set_to_none=True)
            total,parts=supervision_loss(train.forward(model,ids),train.yc[ids],train.yf[ids],train.valid[ids],weights,dict(prediction_mode='joint',lambda_current=c['lambda_current'],lambda_future=c['lambda_future'],future_horizon_s=1.))
            total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm'],error_if_nonfinite=True);opt.step();updates+=1
            losses.append(float(total.detach()));[batchloss[h].append(float(parts[h].detach())) for h in weights]
        z=validation.predict(model);sc=scores(z['current'],validation.yc_np);sf=scores(z['future'][validation.valid_np],validation.yf_np[validation.valid_np]);f,nll=validation_selection(sc,sf,dict(prediction_mode='joint',lambda_current=1.,lambda_future=.5))
        row=dict(epoch=epoch,updates=updates,batches=len(batches),max_actual_batch_size=max(map(len,batches)),training_loss=float(np.mean(losses)),current_ce=float(np.mean(batchloss['current'])),future_ce=float(np.mean(batchloss['future'])),val_current=sc['macro_f1'],val_future=sf['macro_f1'],selection=f,nll=nll)
        with (out/'training_metrics.jsonl').open('a') as ff:
            import json
            ff.write(json.dumps(row)+'\n')
        if (f,-nll)>best:
            best=(f,-nll);stale=0
            torch.save(dict(format='tomato-joint-memory-v1',group=group,seed=seed,epoch=epoch,bindings=train_bindings,state=model.cpu().state_dict(),stats=train.stats),out/'best_head.pt');model.to(c['device'])
        else:stale+=1
        if epoch%10==0:print(group,seed,'epoch',epoch,'val',sc['macro_f1'],sf['macro_f1'],flush=True)
        if stale>=c['patience']:break
    saved,model=load_memory_checkpoint(out/'best_head.pt',c,train_bindings);model=model.to(c['device'])
    if c['device'].startswith('cuda'):peak=torch.cuda.max_memory_allocated()
    meta=dict(group=group,seed=seed,best_epoch=saved['epoch'],epochs=epoch,updates=updates,updates_per_epoch=len(batches),batch_order_sha256=order_hash,parameters=sum(p.numel() for p in model.parameters()),training_s=time.perf_counter()-start,peak_allocated_bytes=peak,checkpoint_sha256=sha(out/'best_head.pt'),bindings=train_bindings,class_counts=dict(current=cc.tolist(),future=fc.tolist()),class_weights=dict(current=wc.tolist(),future=wf.tolist()),hardware=hardware_report(),encoder_new_calls=0,smoke=smoke)
    write(out/'manifest.json',meta);(out/'checkpoint.sha256').write_text(meta['checkpoint_sha256']+'\n')
    result={}
    for split,data in [('train',train),('val',validation)]:
        zz=data.predict(model)
        for e,(a,b) in zip(data.entries,data.slices):result[e['episode']['episode_id']]={h:v[a:b] for h,v in zz.items()}
    del train,validation
    if c['device'].startswith('cuda'):torch.cuda.empty_cache()
    return model,saved,meta,result

def predict_entries(model,saved,entries,c,group,beliefs=None):
    data=TensorData(entries,c,group,saved['stats'],beliefs);zz=data.predict(model)
    return {e['episode']['episode_id']:{h:v[a:b] for h,v in zz.items()} for e,(a,b) in zip(entries,data.slices)}
