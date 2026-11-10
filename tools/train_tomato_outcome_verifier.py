"""Small independent-label A/B/C/D trainer. Unknown targets are never negatives."""
from pathlib import Path
import argparse
import time
import numpy as np
from . import tomato_dataset as ds
from .tomato_outcome_verifier import validate_label,training_readiness,causal_features,build_model,masked_bce
from .tomato_operation_memory import EVENTS


def examples(manifest,identity,labels,mode):
    by={e['episode_id']:e for e in manifest['episodes']};cache={r['episode_id']:r for r in identity['episodes']};parts={'train':[],'val':[]}
    for eid in sorted({r['episode_id'] for r in labels}):
        e=by[eid]
        if e['split'] not in parts:raise ValueError('Historical test not eligible for supervision or selection')
        if ds.digest(cache[eid]['feature_path'])!=cache[eid]['feature_sha256']:raise ValueError('Feature cache identity changed')
        z=np.load(cache[eid]['feature_path']);anchors=z['anchors'];times=z['times'];features=z['features'];target=np.zeros((len(times),8),np.float32);known=np.zeros_like(target,bool)
        for row in [validate_label(r) for r in labels if r['episode_id']==eid]:
            if row['source_recording_id']!=e['source_recording_id'] or row['end_frame']>e['length'] or row['annotation_sha256']!=ds.digest(e['annotation']['path']):raise ValueError('Independent label/source mismatch')
            if row['verdict']=='unknown':continue
            keep=(anchors>=row['start_frame'])&(anchors<row['end_frame'])&(anchors>=row['evidence_end_frame']);k=EVENTS.index(row['event_type']);value=float(row['verdict']=='positive')
            if np.any(known[keep,k]&(target[keep,k]!=value)):raise ValueError('Conflicting operation labels')
            known[keep,k]=True;target[keep,k]=value
        x=np.array([causal_features(v,features[i-1] if i else None,mode=mode,timestamp=float(times[i]),previous_timestamp=float(times[i-1]) if i else None) for i,v in enumerate(features)])
        keep=known.any(1);parts[e['split']].append((x[keep],target[keep],known[keep]))
    return {s:tuple(np.concatenate([r[i] for r in rows]) for i in range(3)) if rows else None for s,rows in parts.items()}


def run(labels_path,out,mode,seed=42,epochs=10):
    if out.exists():raise FileExistsError(out)
    labels=ds.read_json(labels_path)['approved'];ready=training_readiness(labels);out.mkdir(parents=True)
    ds.write_json(out/'readiness.json',ready)
    if not ready['can_train']:
        ds.write_json(out/'result.json',dict(status='not_trained',reason='insufficient_independent_outcome_labels',real_labels=len(labels),test_read=False));print('NOT TRAINED: independent outcome labels insufficient');return
    import torch
    manifest=ds.read_json('runs/tomato/refresh101_20261109_v2/finalized_exports/refresh101/dataset_manifest.json');identity=ds.read_json('runs/tomato/refresh101_20261109_v2/data_identity.json');data=examples(manifest,identity,labels,mode)
    if any(v is None or len(v[0])==0 for v in data.values()):raise ValueError('Both grouped train and val supervision required')
    torch.manual_seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu';x,y,mask=data['train'];mean=x.mean(0);std=x.std(0).clip(.01);vx,vy,vm=data['val']
    model=build_model(mode).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3);rng=np.random.default_rng(seed);history=[];start=time.perf_counter();best=float('inf')
    for epoch in range(epochs):
        model.train();losses=[]
        for ix in np.array_split(rng.permutation(len(x)),max(1,(len(x)+255)//256)):
            opt.zero_grad();loss=masked_bce(model(torch.tensor((x[ix]-mean)/std,device=device)),torch.tensor(y[ix],device=device),torch.tensor(mask[ix],device=device));loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.inference_mode():vl=masked_bce(model(torch.tensor((vx-mean)/std,device=device)),torch.tensor(vy,device=device),torch.tensor(vm,device=device)).item()
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=vl))
        if vl<best:
            best=vl;torch.save(dict(format='tomato-outcome-verifier-v1',mode=mode,state_dict={k:v.cpu() for k,v in model.state_dict().items()},mean=torch.from_numpy(mean),std=torch.from_numpy(std),independent_labels=True,qualified=False,label_sha256=ds.digest(labels_path),seed=seed),out/'best.pt')
    ds.write_json(out/'result.json',dict(status='research_training_complete_not_qualified',history=history,elapsed_s=time.perf_counter()-start,device=device,mode=mode,seed=seed,test_read=False,physical_success_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=list('ABCD'),required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();run(a.labels,a.out,a.mode,a.seed)
