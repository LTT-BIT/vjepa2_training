"""Matched Current-head training: sole factor is boundary sample weighting."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .train_tomato_vjepa_phase import fit_class_weights
from .evaluate_tomato_disagreement import score,summarize

CONFIG=Path('configs/tomato_boundary_supervision_20261112/protocol.json')
BASE=Path('runs/tomato/disagreement_20261112_v2')


def boundary_weights(times,segments,fps,radius=1.,weight=3.):
    b=np.array([s['start_frame']/fps for s in segments[1:]])
    return np.where(np.min(abs(np.asarray(times)[:,None]-b[None,:]),axis=1)<=radius,weight,1.).astype('float32')


def fit(rows,method,cfg,seed,out,identity):
    initialize_runtime(dict(device='cuda',seed=seed));raw=np.concatenate([r['features'] for r in rows]);y=np.concatenate([r['labels'] for r in rows]);bw=np.concatenate([r['weights'] if method=='C1_boundary_weighted' else np.ones(len(r['labels']),np.float32) for r in rows]);mean=raw.mean(0,dtype=np.float64).astype('float32');std=raw.std(0,dtype=np.float64).clip(.01).astype('float32');cw,_=fit_class_weights(y)
    x=torch.from_numpy((raw-mean)/std).cuda();yg=torch.from_numpy(y).cuda();wg=torch.from_numpy(bw).cuda();cg=torch.from_numpy(cw).cuda();net=pm.head_model(58).cuda();initial=ds.fingerprint({k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()});opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(seed);history=[];torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(cfg['epochs']):
        net.train();losses=[];order=rng.permutation(len(y))
        for batch in np.array_split(order,max(1,int(np.ceil(len(y)/cfg['batch_size'])))):
            ix=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True);z=net(x[ix]);loss=(torch.nn.functional.cross_entropy(z,yg[ix],weight=cg,reduction='none')*wg[ix]).sum()/(cg[yg[ix]]*wg[ix]).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss))
        history.append(float(np.mean(losses)))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;net.eval();torch.save(dict(format_version=1,architecture='current_mlp_boundary_ablation',method=method,seed=seed,epochs=cfg['epochs'],head={k:v.detach().cpu() for k,v in net.state_dict().items()},mean=torch.from_numpy(mean),std=torch.from_numpy(std),class_weights=torch.from_numpy(cw),training_source_ids=[r['episode']['source_recording_id'] for r in rows],identity=identity,protocol=cfg,initialization_sha256=initial),out)
    return net,mean,std,dict(training_s=elapsed,initialization_sha256=initial,checkpoint_sha256=ds.digest(out),loss=history,parameters=sum(p.numel() for p in net.parameters()))


@torch.inference_mode()
def predict(net,mean,std,rows):
    return {r['episode']['episode_id']:metrics.probabilities(pm.predict_logits(net,r['features'],mean,std)) for r in rows}


def evaluate(rows,pred,seed):
    per={};logits=[];ys=[];events=[]
    for r in rows:
        e=r['episode'];eid=e['episode_id'];z=np.load(BASE/f'seed{seed}'/f'{eid}_input.npz');c=pred[eid];s,trace=score(e,z['times'],z['anchors'],c,z['future'],'B0');per[eid]=s;logits.append(np.log(np.maximum(c,1e-30)));ys.append(r['labels']);events+= [dict(episode_id=eid,**x) for x in s['prompt']['events']]
    summary=summarize(per);summary['classification']=metrics.scores(np.concatenate(logits),np.concatenate(ys));summary['within_delay']={str(t):sum(x['offset_s'] is not None and 0<=x['offset_s']<=t+1e-9 for x in events) for t in [1.,1.5,2.]}
    return dict(summary=summary,per_episode=per,events=events)


def run(out):
    if out.exists():raise FileExistsError('Preserve prior runs')
    cfg=ds.read_json(CONFIG);lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));manifest=ds.read_json(lock['manifest']);by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);rows=[];inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256']}
    for e in manifest['episodes']:
        if e['split']=='test':continue
        p=Path(by[e['episode_id']]['feature_path']);assert ds.digest(p)==by[e['episode_id']]['feature_sha256'];inputs[str(p)]=ds.digest(p);z=np.load(p);assert np.array_equal(z['anchors'],e['anchors']);gt=ds.build_phase_targets(z['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5)['current_labels'];rows.append(dict(episode=e,features=z['features'],labels=gt,weights=boundary_weights(z['times'],e['annotation']['segments'],e['fps'],cfg['boundary_radius_s'],cfg['boundary_weight'])))
    pool=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val'];ids=sorted(r['episode']['source_recording_id'] for r in pool);rng=np.random.default_rng(cfg['fold_seed']);folds=[[ids[i] for i in part] for part in np.array_split(rng.permutation(len(ids)),cfg['folds'])];out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'folds.json',folds);(out/'executed_runner.py').write_bytes(Path(__file__).read_bytes());results={};started=time.perf_counter();identity=dict(manifest_sha256=lock['manifest_sha256'],joint_names=manifest['contract']['joint_names'],code_sha256=ds.digest(__file__),historical_test_scored=False,labels_changed=False)
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();results[str(seed)]={}
        for method in cfg['methods']:
            oof={};logs=[]
            for k,held in enumerate(folds):
                fitrows=[r for r in pool if r['episode']['source_recording_id'] not in held];outer=[r for r in pool if r['episode']['source_recording_id'] in held];assert not {r['episode']['source_recording_id'] for r in fitrows}&set(held)
                net,mean,std,info=fit(fitrows,method,cfg,seed,folder/f'{method}_fold{k}.pt',identity);oof.update(predict(net,mean,std,outer));logs.append(info);del net;torch.cuda.empty_cache();print('FOLD',seed,method,k,'seconds',round(info['training_s'],2),flush=True)
            evaluations={co:evaluate([r for r in pool if condition(r)],oof,seed) for co,condition in [('oof84',lambda r:True),('oof10_base_unseen_exposed',lambda r:r['episode']['source_recording_id'] not in seen)]}
            net,mean,std,info=fit(pool,method,cfg,seed,folder/f'{method}_final84.pt',identity);dp=predict(net,mean,std,dev);evaluations['development8_exposed']=evaluate(dev,dp,seed)
            for r in rows:
                eid=r['episode']['episode_id'];np.savez_compressed(folder/f'{method}_{eid}.npz',current=oof[eid] if eid in oof else dp[eid])
            results[str(seed)][method]=dict(evaluation=evaluations,fold_training=logs,final_training=info);ds.write_json(folder/f'{method}_result.json',results[str(seed)][method]);del net;torch.cuda.empty_cache();print('RESULT',seed,method,{k:v['summary']['prompt_counts'] for k,v in evaluations.items()},flush=True)
        a,b=[results[str(seed)][m] for m in cfg['methods']];assert [q['initialization_sha256'] for q in a['fold_training']]==[q['initialization_sha256'] for q in b['fold_training']] and a['final_training']['initialization_sha256']==b['final_training']['initialization_sha256']
    ds.write_json(out/'results.json',results);ds.write_json(out/'receipt.json',dict(inputs=inputs,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256=ds.digest(__file__),wall_s=time.perf_counter()-started,model_fits=36,encoder_calls=0,default_changed=False,memory_enabled=False));verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
