"""A/B/C ablation: reuse A/B, train only fourth-view history C with equal capacity."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds
from . import train_tomato_rgb_memory as training
from .tomato_rgb_memory_inference import RGBMemorySession
from .tomato_frozen_protocol import verify

METHOD='G3_local_only_history8'


def verify_stream(cp, dev, index, pred):
    latency=[];errors=[];observations=0
    for r in dev:
        eid=r['episode']['episode_id']
        session=RGBMemorySession(cp,session_id=eid,episode_id=eid,device='cuda',offline_replay=True)
        actual={k:[] for k in ['current','future']}
        for i,t in enumerate(r['times']):
            torch.cuda.synchronize();start=time.perf_counter()
            x=session.observe(r['visual'][i],r['positions'][i].reshape(16,58),r['metadata'][i],
                timestamp=float(t),arrival_timestamp=float(t)+.04,
                session_id=eid,episode_id=eid,cache_signature=index['signature'])
            torch.cuda.synchronize();latency.append((time.perf_counter()-start)*1000)
            assert x['valid'] and not x['suggests_prompt_update']
            for k in actual:actual[k].append(x[k+'_probabilities'])
            for _,v,q,_ in list(session.rows)[:-1]:
                assert not v[:3].any()
                np.testing.assert_array_equal(q,np.tile(session.saved['state_mean'].numpy(),16))
            observations+=1
        for k in actual:
            v=np.stack(actual[k]);np.testing.assert_allclose(v,pred[eid][k],atol=5e-6,rtol=1e-4)
            errors.append(float(np.max(np.abs(v-pred[eid][k]))))
    return dict(observations=observations,stream_batch_max_abs_error=max(errors),
        old_global_and_state_discarded=True,
        head_only_latency_ms=dict(p50=float(np.median(latency)),p95=float(np.percentile(latency,95)),max=float(max(latency))))


def run(cache,reference,out):
    if out.exists():raise FileExistsError('Preserve historical runs; new directory required')
    oldcfg=ds.read_json(reference/'protocol.json')
    cfg=deepcopy(oldcfg);cfg.update(version='old100-r11-only-local-history-v1',
        methods=['G0_global_current','G1_local_current',METHOD],
        memory_scope='local_only',new_training_methods=[METHOD],
        comparison='A three global current; B four current; C three global current plus local history8')
    lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest'])
    rows,index,mean,std=training.load_rows(cache,oldcfg,manifest)
    train=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val']
    assert not {r['episode']['source_recording_id'] for r in train}&{r['episode']['source_recording_id'] for r in dev}
    out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'cache_binding.json',index)
    sources=[Path(__file__),Path(training.__file__),Path('tools/tomato_rgb_memory_model.py'),
        Path('tools/tomato_rgb_memory_training.py'),Path('tools/tomato_rgb_memory_inference.py')]
    identity=dict(pretrained=lock['pretrained'],manifest_sha256=lock['manifest_sha256'],
        cache_signature=index['signature'],cache_index_sha256=ds.digest(cache/'index.json'),
        statistics_sha256=index['statistics_sha256'],joint_names=index['joint_names'],
        source_sha256={str(p):ds.digest(p) for p in sources})
    for p in sources:(out/p.name).write_bytes(p.read_bytes())
    old=ds.read_json(reference/'results.json');results={};inputs={};verification=[]
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();results[str(seed)]={}
        for method in ['G0_global_current','G1_local_current']:
            cp=reference/f'seed{seed}'/f'{method}.pt'
            assert ds.digest(cp)==old[str(seed)][method]['training']['checkpoint_sha256']
            inputs[str(cp)]=ds.digest(cp)
            saved=torch.load(cp,map_location='cpu',weights_only=True)
            assert saved['protocol']==oldcfg
            # Re-evaluate saved probabilities on exactly the same labels and grid.
            pred={}
            for r in dev:
                eid=r['episode']['episode_id'];p=reference/f'seed{seed}'/f'{method}_{eid}.npz'
                inputs[str(p)]=ds.digest(p);z=np.load(p)
                assert np.array_equal(z['anchors'],r['anchors'])
                pred[eid]=dict(current=z['current'],future=z['future'])
            ev=training.evaluate(dev,pred)
            assert ev['summary']['prompt_counts']==old[str(seed)][method]['evaluation']['summary']['prompt_counts']
            results[str(seed)][method]=dict(training=old[str(seed)][method]['training'],
                checkpoint_path=str(cp.resolve()),reused=True,evaluation=ev)
        cp=folder/f'{METHOD}.pt'
        net,info=training.fit(train,METHOD,seed,cfg,mean,std,cp,identity)
        assert info['initialization_sha256']==old[str(seed)]['G1_local_current']['training']['initialization_sha256']
        assert info['parameters']==old[str(seed)]['G1_local_current']['training']['parameters']
        pred=training.predict(net,dev,METHOD,cfg);ev=training.evaluate(dev,pred)
        for r in dev:
            eid=r['episode']['episode_id']
            np.savez_compressed(folder/f'{METHOD}_{eid}.npz',**pred[eid],times=r['times'],anchors=r['anchors'])
        results[str(seed)][METHOD]=dict(training=info,checkpoint_path=str(cp.resolve()),reused=False,evaluation=ev)
        ds.write_json(folder/'result.json',results[str(seed)])
        print('TRAINED',seed,'current',ev['summary']['current']['macro_f1'],'future',ev['summary']['future']['macro_f1'],'prompt',ev['summary']['prompt_counts'],flush=True)
        del net;torch.cuda.empty_cache()
        verification.append(dict(seed=seed,checkpoint_sha256=ds.digest(cp),**verify_stream(cp,dev,index,pred)))
        ds.write_json(out/'verification.json',dict(models=verification,latency_scope='frozen feature bank to probabilities, no encoder/network/ROI',online_roi_validated=False))
    ds.write_json(out/'results.json',results)
    ds.write_json(out/'receipt.json',dict(identity=identity,inputs=inputs,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reference=str(reference.resolve()),reference_results_sha256=ds.digest(reference/'results.json'),
        trained_models=3,reused_models=6,encoder_unfrozen=False,
        historical_test_evaluated=False,default_changed=False,robot_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.cache,a.reference,a.out)
