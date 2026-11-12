"""Grouped, matched causal objective comparison; never score historical test."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .tomato_causal_objective import CausalReadouts,contexts,supervised_loss,overlap_loss
from .train_tomato_vjepa_phase import fit_class_weights
from .train_tomato_boundary_supervision import evaluate

CONFIG=Path('configs/tomato_causal_objective_20261112/protocol.json')


def pack(rows,cfg):
    raw=[];ys=[];ix=[];prev=[];offset=0
    for r in rows:
        c=contexts(r['times'],cfg['context'],cfg['gap_reset_s'])
        p=np.full_like(c,-1)
        for i in range(1,len(c)):
            if r['times'][i]-r['times'][i-1]<=cfg['gap_reset_s']:p[i]=c[i-1]
        ix.append(np.where(c<0,-1,c+offset));prev.append(np.where(p<0,-1,p+offset))
        raw.append(r['features']);ys.append(r['labels']);offset+=len(c)
    return np.concatenate(raw),np.concatenate(ys),np.concatenate(ix),np.concatenate(prev)


def fit(rows,method,cfg,seed,path,identity):
    initialize_runtime(dict(device='cuda',seed=seed))
    raw,y,indices,previous=pack(rows,cfg)
    mean=raw.mean(0,dtype=np.float64).astype('float32');std=raw.std(0,dtype=np.float64).clip(.01).astype('float32')
    x=torch.from_numpy(np.concatenate([np.zeros((1,raw.shape[1]),np.float32),(raw-mean)/std])).cuda()
    yg=torch.from_numpy(np.r_[0,y]).cuda();ix=torch.from_numpy(indices).cuda();pr=torch.from_numpy(previous).cuda()
    cw,_=fit_class_weights(y);cw=torch.from_numpy(cw).cuda()
    net=CausalReadouts(raw.shape[1],cfg['hidden'],cfg['dropout']).cuda()
    initial=ds.fingerprint({k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()})
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);losses=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    for epoch in range(cfg['epochs']):
        epoch_loss=[]
        for batch in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/cfg['batch_size']))):
            b=torch.as_tensor(batch,device='cuda');ids=ix[b];net.train();opt.zero_grad(set_to_none=True)
            z=net(x[ids+1],ids>=0)
            loss=supervised_loss(z,yg[ids+1],ids>=0,cw,method!='T0_last',cfg['earlier_readout_weight'])
            if method=='T2_overlap':
                net.eval();prior=pr[b]
                loss=loss+cfg['overlap_kl_weight']*overlap_loss(net(x[ids+1],ids>=0),net(x[prior+1],prior>=0),ids,prior)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    info=dict(training_s=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),initialization_sha256=initial,loss=losses,parameters=sum(p.numel() for p in net.parameters()))
    torch.save(dict(format_version=1,architecture='causal_current_readouts_grucell',method=method,seed=seed,head={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=torch.from_numpy(mean),std=torch.from_numpy(std),class_weights=cw.cpu(),training_source_ids=[r['episode']['source_recording_id'] for r in rows],identity=identity,protocol=cfg,training=info),path)
    info['checkpoint_sha256']=ds.digest(path)
    return net,mean,std,info


@torch.inference_mode()
def predict(net,mean,std,rows,cfg):
    net.eval();out={};latencies=[]
    for r in rows:
        raw=r['features'];x=torch.from_numpy(np.concatenate([np.zeros((1,raw.shape[1]),np.float32),(raw-mean)/std])).cuda()
        ix=torch.from_numpy(contexts(r['times'],cfg['context'],cfg['gap_reset_s'])).cuda();chunks=[]
        for ids in ix.split(cfg['batch_size']):chunks.append(net(x[ids+1],ids>=0)[:,-1].softmax(-1).cpu().numpy())
        out[r['episode']['episode_id']]=np.concatenate(chunks)
        # Readout-only device-resident single-window latency, NOT encoder end-to-end.
        for ids in ix[:min(20,len(ix))]:
            ids=ids[None];torch.cuda.synchronize();t=time.perf_counter();net(x[ids+1],ids>=0);torch.cuda.synchronize();latencies.append((time.perf_counter()-t)*1000)
    return out,dict(readout_only_ms_p50=float(np.median(latencies)),readout_only_ms_p95=float(np.percentile(latencies,95)),readout_only_ms_max=float(max(latencies)),scope='device-resident 4-observation head only; excludes encoder, decode and network')


def run(out):
    if out.exists():raise FileExistsError('Preserve prior artifacts')
    cfg=ds.read_json(CONFIG);lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))
    manifest=ds.read_json(lock['manifest']);by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']}
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train'])
    rows=[];inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256']}
    for e in manifest['episodes']:
        if e['split']=='test':continue
        p=Path(by[e['episode_id']]['feature_path']);assert ds.digest(p)==by[e['episode_id']]['feature_sha256'];inputs[str(p)]=ds.digest(p)
        z=np.load(p);assert np.array_equal(z['anchors'],e['anchors']);assert np.all(z['input_indices']<=z['anchors'][:,None])
        labels=ds.build_phase_targets(z['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5)['current_labels']
        rows.append(dict(episode=e,features=z['features'],labels=labels,times=z['times']))
    pool=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val']
    ids=sorted(r['episode']['source_recording_id'] for r in pool);assert len(set(ids))==84
    rng=np.random.default_rng(cfg['fold_seed']);folds=[[ids[i] for i in part] for part in np.array_split(rng.permutation(len(ids)),cfg['folds'])]
    out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'folds.json',folds)
    sources=[Path(__file__),Path('tools/tomato_causal_objective.py')]
    for p in sources:(out/p.name).write_bytes(p.read_bytes())
    identity=dict(manifest_sha256=lock['manifest_sha256'],code_sha256={str(p):ds.digest(p) for p in sources},historical_test_scored=False,labels_changed=False,joint_names=manifest['contract']['joint_names'])
    results={};start=time.perf_counter()
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();results[str(seed)]={}
        for method in cfg['methods']:
            oof={};logs=[]
            for k,held in enumerate(folds):
                train=[r for r in pool if r['episode']['source_recording_id'] not in held];outer=[r for r in pool if r['episode']['source_recording_id'] in held]
                net,mean,std,info=fit(train,method,cfg,seed,folder/f'{method}_fold{k}.pt',identity)
                pred,lat=predict(net,mean,std,outer,cfg);oof.update(pred);logs.append(dict(**info,latency=lat));del net;torch.cuda.empty_cache();print('FOLD',seed,method,k,round(info['training_s'],2),flush=True)
            evaluations={name:evaluate([r for r in pool if condition(r)],oof,seed) for name,condition in [('oof84',lambda r:True),('oof10_base_unseen_exposed',lambda r:r['episode']['source_recording_id'] not in seen)]}
            net,mean,std,info=fit(pool,method,cfg,seed,folder/f'{method}_final84.pt',identity);dp,lat=predict(net,mean,std,dev,cfg)
            evaluations['development8_exposed']=evaluate(dev,dp,seed)
            for r in rows:
                eid=r['episode']['episode_id'];np.savez_compressed(folder/f'{method}_{eid}.npz',current=oof[eid] if eid in oof else dp[eid],times=r['times'])
            results[str(seed)][method]=dict(evaluation=evaluations,fold_training=logs,final_training=info,latency=lat)
            ds.write_json(folder/f'{method}_result.json',results[str(seed)][method]);del net;torch.cuda.empty_cache();print('RESULT',seed,method,{k:v['summary']['prompt_counts'] for k,v in evaluations.items()},flush=True)
        inits=[[q['initialization_sha256'] for q in results[str(seed)][m]['fold_training']]+[results[str(seed)][m]['final_training']['initialization_sha256']] for m in cfg['methods']];assert inits[0]==inits[1]==inits[2]
    ds.write_json(out/'results.json',results);ds.write_json(out/'receipt.json',dict(inputs=inputs,identity=identity,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_s=time.perf_counter()-start,model_fits=54,encoder_calls=0,default_changed=False));verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
