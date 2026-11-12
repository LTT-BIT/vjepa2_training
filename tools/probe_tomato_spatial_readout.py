"""Train-only12 recording frozen spatial token pilot, no ROI or memory."""
import argparse
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .tomato_spatial_readout import spatial_pool,SpatialReadout

CONFIG=Path('configs/tomato_spatial_readout_20261112/protocol.json')


def prepare(out):
    cfg=ds.read_json(CONFIG);lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));m=ds.read_json(lock['manifest']);old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};rng=np.random.default_rng(cfg['selection_seed']);chosen=[]
    for yes,count in [(True,8),(False,4)]:
        pool=sorted([e for e in m['episodes'] if e['split']=='train' and (e['source_recording_id'] in seen)==yes],key=lambda e:e['source_recording_id']);chosen += [pool[i] for i in rng.permutation(len(pool))[:count]]
    ids=[e['source_recording_id'] for e in chosen];assert len(set(ids))==12;ds.write_json(out/'selection.json',dict(source_ids=ids,episode_ids=[e['episode_id'] for e in chosen],scope='training only, fixed before extracting pixels'))
    initialize_runtime(dict(device='cuda',seed=42));encoder,provenance=pm.load_encoder(Path('pretrained/vjepa2-vitl-fpc64-256'),device='cuda');inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256']};rows=[];timing=[]
    for e in chosen:
        pm.assert_source_unchanged(e);p=Path(by[e['episode_id']]['feature_path']);assert ds.digest(p)==by[e['episode_id']]['feature_sha256'];inputs[str(p)]=ds.digest(p);z=np.load(p);anchors=z['anchors'];idx=sorted(set(int(np.argmin(abs(anchors/e['fps']-(s['start_frame']/e['fps']+offset)))) for s in e['annotation']['segments'][1:] for offset in cfg['sample_offsets_s']));hist=z['input_indices'][idx];assert np.all(hist<=anchors[idx,None]);views,_=pm.decode_selected(e,np.unique(hist));_,q=ds.read_state(e,m['contract']['joint_names']);tokens=[];errors=[]
        for j,h in enumerate(hist):
            rgb=np.array([[v[int(i)] for i in h] for v in views],dtype=np.uint8);parameter=next(encoder.parameters());torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                full=encoder(pixel_values_videos=pm.preprocess(rgb,parameter.device,parameter.dtype)).last_hidden_state;region=spatial_pool(full);torch.cuda.synchronize();timing.append((time.perf_counter()-start)*1000);tokens.append(region.cpu().numpy());errors.append(float((region.mean(1)-full.float().mean(1)).abs().max()))
        tensor=np.array(tokens,np.float16);targets=ds.build_phase_targets(anchors[idx],e['annotation']['segments'],e['length'],e['fps'],1.5)['current_labels'];path=out/f'{e["episode_id"]}_tokens.npz';np.savez_compressed(path,tokens=tensor,positions=q[hist],anchors=anchors[idx],input_indices=hist,labels=targets)
        rows.append(dict(episode=e,tokens=tensor.astype('float32'),positions=q[hist],labels=targets));ds.write_json(out/f'{e["episode_id"]}_receipt.json',dict(source_recording_id=e['source_recording_id'],input_sha256=e['input_sha256'],cache_sha256=ds.digest(path),causal_indices=hist.tolist(),original_mean_vs_regions_max_error=max(errors)));print('ENCODED',e['episode_id'],len(idx),flush=True);del views
    del encoder;torch.cuda.empty_cache();ds.write_json(out/'extraction.json',dict(inputs=inputs,encoder=provenance,windows=sum(len(r['labels']) for r in rows),encoder_call_ms=dict(p50=float(np.median(timing)),p95=float(np.percentile(timing,95)),maximum=float(max(timing))),scope='encoder only including preprocess/device transfer; no decode/readout/transport'))
    return rows,cfg


def fit(train,method,cfg,seed,path):
    initialize_runtime(dict(device='cuda',seed=seed));v=np.concatenate([r['tokens'] for r in train]);q=np.concatenate([r['positions'] for r in train]);y=np.concatenate([r['labels'] for r in train]);vm=v.mean(2).mean(0);vs=v.mean(2).std(0).clip(.1);qm=q.mean((0,1));qs=q.std((0,1)).clip(.01);vg=torch.from_numpy((v-vm[None,:,None,:])/vs[None,:,None,:]).cuda();qg=torch.from_numpy((q-qm)/qs).cuda();yg=torch.from_numpy(y).cuda();net=SpatialReadout().cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);initial=ds.fingerprint({k:x.detach().cpu().numpy().tolist() for k,x in net.state_dict().items()});start=time.perf_counter()
    for _ in range(cfg['epochs']):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(vg,qg,method),yg);loss.backward();opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;net.eval();torch.save(dict(head={k:x.detach().cpu() for k,x in net.state_dict().items()},visual_mean=torch.from_numpy(vm),visual_std=torch.from_numpy(vs),state_mean=torch.from_numpy(qm),state_std=torch.from_numpy(qs),method=method,seed=seed,protocol=cfg,training_source_ids=[r['episode']['source_recording_id'] for r in train],scope='training_only_group_pilot_not_deployable'),path)
    return net,(vm,vs,qm,qs),dict(training_s=elapsed,initialization_sha256=initial,checkpoint_sha256=ds.digest(path),parameters=sum(p.numel() for p in net.parameters()))


@torch.inference_mode()
def predict(net,norm,outer,method):
    vm,vs,qm,qs=norm;result={}
    for r in outer:
        v=torch.from_numpy((r['tokens']-vm[None,:,None,:])/vs[None,:,None,:]).cuda();q=torch.from_numpy((r['positions']-qm)/qs).cuda();result[r['episode']['episode_id']]=net(v,q,method).cpu().numpy()
    return result


def run(out):
    if out.exists():raise FileExistsError('New directory required')
    out.mkdir(parents=True);start=time.perf_counter();rows,cfg=prepare(out);ds.write_json(out/'protocol.json',cfg);folds=np.array_split(np.random.default_rng(cfg['selection_seed']).permutation(12),3);ds.write_json(out/'folds.json',[[rows[i]['episode']['source_recording_id'] for i in ix] for ix in folds]);result={}
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();result[str(seed)]={}
        for method in cfg['methods']:
            pred={};logs=[]
            for k,held in enumerate(folds):
                fitrows=[r for i,r in enumerate(rows) if i not in held];outer=[rows[i] for i in held];net,norm,info=fit(fitrows,method,cfg,seed,folder/f'{method}_fold{k}.pt');pred.update(predict(net,norm,outer,method));logs.append(info);del net;torch.cuda.empty_cache()
            ys=np.concatenate([r['labels'] for r in rows]);zz=np.concatenate([pred[r['episode']['episode_id']] for r in rows]);result[str(seed)][method]=dict(classification=metrics.scores(zz,ys),folds=logs,per_episode={r['episode']['episode_id']:metrics.scores(pred[r['episode']['episode_id']],r['labels']) for r in rows});ds.write_json(folder/f'{method}_result.json',result[str(seed)][method]);print('RESULT',seed,method,result[str(seed)][method]['classification']['macro_f1'],flush=True)
        a,b=[result[str(seed)][x] for x in cfg['methods']];assert [r['initialization_sha256'] for r in a['folds']]==[r['initialization_sha256'] for r in b['folds']]
    ds.write_json(out/'results.json',result);ds.write_json(out/'receipt.json',dict(wall_s=time.perf_counter()-start,model_fits=18,source_sha256={p:ds.digest(p) for p in ['tools/probe_tomato_spatial_readout.py','tools/tomato_spatial_readout.py']},git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_recordings=12,development_used=False,historical_test_used=False,roi_used=False,memory_enabled=False,default_changed=False));verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
