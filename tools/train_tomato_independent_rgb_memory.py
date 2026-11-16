"""Matched independent Current/Future training for three/four/local-history inputs."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds
from . import train_tomato_rgb_memory as base
from .tomato_rgb_memory_training import history_length,memory_scope
from .tomato_independent_rgb_memory import SingleRGBReader,IndependentRGBReader,ARCHITECTURE
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .train_tomato_vjepa_phase import fit_class_weights
from .train_tomato_local_only_memory import verify_stream

METHODS=['G0_global_current','G1_local_current','G3_local_only_history8']


def fit_single(rows,method,target,seed,cfg,mean,std,path,identity):
    initialize_runtime(dict(device='cuda',seed=seed))
    bank=base.bank(rows,history_length(method));labels=bank[target];valid=labels>=0
    weight,_=fit_class_weights(labels[valid].cpu().numpy());weight=torch.from_numpy(weight).cuda()
    net=SingleRGBReader(mean,std,cfg['hidden'],memory_scope(method),target).cuda()
    initial=ds.fingerprint({k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()})
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);history=[];updates=0
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(cfg['epochs']):
        net.train();losses=[]
        for ids in np.array_split(rng.permutation(len(labels)),int(np.ceil(len(labels)/cfg['batch_size']))):
            ids=torch.as_tensor(ids,device='cuda');ids=ids[valid[ids]]
            if not len(ids):continue
            optimizer.zero_grad(set_to_none=True)
            logits=net(**base.features(bank,ids,method))
            loss=torch.nn.functional.cross_entropy(logits,labels[ids],weight=weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1;losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    torch.cuda.synchronize()
    info=dict(training_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in net.parameters()),initialization_sha256=initial,
        loss=history,updates=updates,target=target,valid_training_windows=int(valid.sum()))
    torch.save(dict(format_version=1,architecture='old100_single_rgb_memory_v1',target=target,
        method=method,head={k:v.detach().cpu() for k,v in net.state_dict().items()},
        state_mean=torch.from_numpy(mean),state_std=torch.from_numpy(std),class_weights=weight.cpu(),
        optimizer_state=optimizer.state_dict(),rng_state=rng.bit_generator.state,
        torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
        protocol=cfg,identity=identity,training=info),path)
    info['checkpoint_sha256']=ds.digest(path)
    del bank
    return net,info


def run(cache,out):
    if out.exists():raise FileExistsError('New output required')
    oldcfg=ds.read_json(base.CONFIG);cfg=deepcopy(oldcfg)
    cfg.update(version='old100-independent-readouts-r11-v1',methods=METHODS,
        head_training='independent entire readouts, separate optimizers; encoder frozen/shared',
        current_loss_weight=None,future_loss_weight=None,
        selection='30 fixed epochs per branch; no validation checkpoint/threshold selection')
    lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest'])
    rows,index,mean,std=base.load_rows(cache,oldcfg,manifest)
    train=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val']
    assert not {r['episode']['source_recording_id'] for r in train}&{r['episode']['source_recording_id'] for r in dev}
    out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'cache_binding.json',index)
    sources=[Path(__file__),Path(base.__file__),Path('tools/tomato_independent_rgb_memory.py'),
        Path('tools/tomato_rgb_memory_model.py'),Path('tools/tomato_rgb_memory_training.py'),
        Path('tools/tomato_rgb_memory_inference.py')]
    identity=dict(pretrained=lock['pretrained'],manifest_sha256=lock['manifest_sha256'],
        cache_signature=index['signature'],cache_index_sha256=ds.digest(cache/'index.json'),
        statistics_sha256=index['statistics_sha256'],joint_names=index['joint_names'],
        source_sha256={str(p):ds.digest(p) for p in sources})
    for p in sources:(out/p.name).write_bytes(p.read_bytes())
    results={};verification=[]
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();results[str(seed)]={};initial={}
        for method in METHODS:
            branches={};infos={}
            for target in ['current','future']:
                branches[target],infos[target]=fit_single(train,method,target,seed,cfg,mean,std,
                    folder/f'{method}_{target}.pt',identity)
                initial.setdefault(target,[]).append(infos[target]['initialization_sha256'])
                print('TRAINED',seed,method,target,round(infos[target]['training_s'],2),flush=True)
            net=IndependentRGBReader(branches['current'],branches['future']).cuda().eval()
            info=dict(parameters=sum(p.numel() for p in net.parameters()),
                training_s=sum(v['training_s'] for v in infos.values()),branches=infos)
            cp=folder/f'{method}.pt'
            torch.save(dict(format_version=1,architecture=ARCHITECTURE,method=method,seed=seed,
                model_version=f'independent-{method}-s{seed}-{ds.fingerprint(cfg)[:12]}-{index["signature"][:12]}',
                head={k:v.detach().cpu() for k,v in net.state_dict().items()},
                state_mean=torch.from_numpy(mean),state_std=torch.from_numpy(std),
                training_sources=[r['episode']['source_recording_id'] for r in train],
                protocol=cfg,identity=identity,training=info,
                input_schema=dict(visual=[4,1024],positions=[16,58],metadata=116,
                    joint_names=index['joint_names'],labels=list(ds.LABELS),history=history_length(method),
                    memory_scope=memory_scope(method),language_prompt=False,stage_oracle=False,
                    head_training='independent')),cp)
            info['checkpoint_sha256']=ds.digest(cp)
            pred=base.predict(net,dev,method,cfg);evaluation=base.evaluate(dev,pred)
            results[str(seed)][method]=dict(training=info,checkpoint_path=str(cp.resolve()),evaluation=evaluation)
            for r in dev:
                eid=r['episode']['episode_id']
                np.savez_compressed(folder/f'{method}_{eid}.npz',**pred[eid],times=r['times'],anchors=r['anchors'])
            ds.write_json(folder/f'{method}_result.json',results[str(seed)][method])
            ds.write_json(out/'results.json',results)
            print('SCORED',seed,method,evaluation['summary']['current']['macro_f1'],evaluation['summary']['future']['macro_f1'],evaluation['summary']['prompt_counts'],flush=True)
            del net,branches;torch.cuda.empty_cache()
            verification.append(dict(seed=seed,method=method,checkpoint_sha256=ds.digest(cp),
                **verify_stream(cp,dev,index,pred)))
            ds.write_json(out/'verification.json',dict(models=verification,
                latency_scope='cached features to two independent readouts; no encoder/ROI/network',
                historical_test_evaluated=False))
        assert all(len(set(values))==1 for values in initial.values())
    ds.write_json(out/'receipt.json',dict(identity=identity,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        trained_branches=18,paired_bundles=9,encoder_unfrozen=False,historical_test_evaluated=False,
        default_changed=False,robot_calls=0,matched_initializations_by_target=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.cache,a.out)
