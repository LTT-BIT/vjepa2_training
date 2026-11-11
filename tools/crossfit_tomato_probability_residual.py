"""Nested record-group calibration using only10 recordings unseen by frozen B0."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics
from .tomato_runtime import initialize_runtime
from .tomato_independent_heads import load_checkpoint
from .tomato_probability_residual import ResidualCalibration
from .tomato_residual_event_audit import nested_folds
from .train_tomato_probability_residual import fit,predict,arrays,history_features,score
from .evaluate_tomato_complete100_history import freeze_dataset,SOURCE
from .tomato_state_recovery import B0State

CONFIG=Path('configs/tomato_residual_audit_20261111/protocol.json')
S1=Path('runs/tomato/probability_residual_unseen_20261111_v1')


def fixed_fit(rows,epochs,cfg,seed):
    initialize_runtime(dict(device='cuda',seed=seed));a=arrays(rows)
    mean=a['history'].mean(0,dtype=np.float64).astype('float32');std=a['history'].std(0,dtype=np.float64).clip(.01).astype('float32')
    net=ResidualCalibration(mean,std).cuda();opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);rng=np.random.default_rng(seed)
    x=torch.from_numpy(a['history']).cuda();base=torch.from_numpy(np.log(np.maximum(a['future'],1e-6)).astype('float32')).cuda();y=torch.from_numpy(a['labels']).long().cuda()
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/cfg['batch_size'])))):
            ix=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True);z,delta=net(x[ix],base[ix]);loss=torch.nn.functional.cross_entropy(z,y[ix])+cfg['residual_penalty']*delta.square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return net.eval()


def row_for(e,current,future):
    targets=ds.build_phase_targets(np.array(e['anchors']),e['annotation']['segments'],e['length'],e['fps'],1.5)
    return dict(episode=e,current=current,future=future,times=np.array(e['anchors'])/e['fps'],valid=targets['future_valid_mask'],labels=targets['future_labels'],history=history_features(current,future,np.array(e['anchors'])/e['fps']))


def run(out):
    if out.exists():raise FileExistsError('New run required')
    cfg=ds.read_json(CONFIG);m,lock=freeze_dataset();old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json')
    oldtrain=set(old['splits']['train']);pool=[e for e in m['episodes'] if e['split']=='train' and e['source_recording_id'] not in oldtrain]
    assert len(pool)==10 and not {e['source_recording_id'] for e in pool}&set(old['splits']['val']+old['splits']['test'])
    folds=nested_folds([e['source_recording_id'] for e in pool],cfg['fold_seed']);out.mkdir(parents=True)
    ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'folds.json',folds);(out/'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};cache={};inputs={str(CONFIG):ds.digest(CONFIG)}
    for e in pool:
        r=by[e['episode_id']];assert ds.digest(r['feature_path'])==r['feature_sha256'];inputs[r['feature_path']]=r['feature_sha256'];z=np.load(r['feature_path'])
        assert np.array_equal(z['anchors'],e['anchors']) and np.allclose(z['times'],np.array(e['anchors'])/e['fps']);cache[e['episode_id']]=z['features']
    results={};start=time.perf_counter()
    for seed in cfg['seeds']:
        initialize_runtime(dict(device='cuda',seed=seed));folder=out/f'seed{seed}';folder.mkdir();torch.cuda.reset_peak_memory_stats()
        source=ds.read_json(SOURCE/f'seed{seed}/result.json')['identity']['original_classifiers'];assert ds.digest(source['path'])==source['sha256'];inputs[source['path']]=source['sha256']
        saved,head=load_checkpoint(source['path'],'cuda');assert saved['split_sha256']==old['split_sha256'];rows=[]
        for e in pool:
            z=pm.predict_heads(head,cache[e['episode_id']],saved['mean'],saved['std']);rows.append(row_for(e,*[metrics.probabilities(z[k+'_logits']) for k in ('current','future')]))
        del head;torch.cuda.empty_cache();oof={};selected=[];foldlog=[];nested_seconds=0.
        for k,f in enumerate(folds):
            parts={name:[r for r in rows if r['episode']['source_recording_id'] in ids] for name,ids in f.items()}
            net,info=fit(parts['fit'],parts['selection'],cfg,seed);selected.append(info['selected_epoch']);nested_seconds+=info['training_s'];del net;torch.cuda.empty_cache()
            net=fixed_fit(parts['refit'],info['selected_epoch'],cfg,seed);p,lat=predict(net,parts['outer']);assert not set(p)&set(oof);oof.update(p)
            torch.save(dict(state_dict=net.cpu().state_dict(),fold=f,selected_epoch=info['selected_epoch'],seed=seed,base_model=source),folder/f'fold{k}.pt')
            foldlog.append(dict(fold=k,split=f,selection=info));del net;torch.cuda.empty_cache()
        assert len(oof)==10
        oof0,ev0=score(rows,{r['episode']['episode_id']:r['future'] for r in rows});oof2,ev2=score(rows,oof)
        epoch=int(np.median(selected));net=fixed_fit(rows,epoch,cfg,seed)
        # Development observations are loaded only after selection and final fit.
        dev=[]
        for e in m['episodes']:
            if e['split']!='val':continue
            path=S1/f'seed{seed}'/(e['episode_id']+'.npz');inputs[str(path)]=ds.digest(path);z=np.load(path);assert np.allclose(z['times'],np.array(e['anchors'])/e['fps']);dev.append(row_for(e,z['current'],z['original_future']))
        pred,latency=predict(net,dev);a,ae=score(dev,{r['episode']['episode_id']:r['future'] for r in dev});b,be=score(dev,pred)
        for r in dev:
            eid=r['episode']['episode_id'];raw=B0State();cal=B0State()
            for i,t in enumerate(r['times']):
                u=raw.update(r['current'][i],r['future'][i],float(t));v=cal.update(r['current'][i],pred[eid][i],float(t));assert (u['phase'],u['changed'])==(v['phase'],v['changed'])
            np.savez_compressed(folder/(eid+'.npz'),times=r['times'],current=r['current'],original_future=r['future'],calibrated_future=pred[eid])
        for r in rows:
            eid=r['episode']['episode_id'];np.savez_compressed(folder/('oof_'+eid+'.npz'),times=r['times'],current=r['current'],original_future=r['future'],calibrated_future=oof[eid])
        checkpoint=folder/'residual.pt';torch.save(dict(format_version=1,architecture='nested_causal_probability_residual_36x4',state_dict=net.cpu().state_dict(),seed=seed,selected_epoch=epoch,base_model=source,protocol=cfg,manifest_sha256=lock['manifest_sha256'],fold_sha256=ds.fingerprint(folds),schema=dict(class_order=ds.LABELS,lags_s=[0,.5,1.,1.5],input='Current/Future probabilities; no Oracle',output='Future1.5s only')),checkpoint)
        results[str(seed)]=dict(oof=dict(S0=oof0,S2=oof2),development=dict(S0=a,S2=b),folds=foldlog,final_epoch=epoch,latency_ms=latency,checkpoint_sha256=ds.digest(checkpoint),prompt_identical=True,nested_selection_training_s=nested_seconds)
        ds.write_json(folder/'events.json',dict(oof=dict(S0=ev0,S2=ev2),development=dict(S0=ae,S2=be)));ds.write_json(folder/'result.json',results[str(seed)])
        print('SEED',seed,'selected',selected,'final',epoch,'OOF',oof0['events']['hits'],oof2['events']['hits'],'dev',a['events']['hits'],b['events']['hits'],'FP',a['events']['false_events'],b['events']['false_events'],flush=True)
        del net;torch.cuda.empty_cache()
    gate=True
    for cohort in ('oof','development'):
        pairs=[(r[cohort]['S0']['events'],r[cohort]['S2']['events']) for r in results.values()]
        gate &= all(b['hits']>=a['hits'] and b['false_events']<=a['false_events'] for a,b in pairs) and sum(b['hits']>a['hits'] for a,b in pairs)>=2
        gate &= np.mean([b['recall'] for a,b in pairs])>np.mean([a['recall'] for a,b in pairs]) and np.mean([b['false_per_minute'] for a,b in pairs])<=np.mean([a['false_per_minute'] for a,b in pairs])
    ds.write_json(out/'summary.json',dict(seeds=results,candidate_gate_passed=bool(gate),wall_s=time.perf_counter()-start,encoder_calls=0,B0_retrained=False,historical_test_arrays_read=False,production_changed=False))
    ds.write_json(out/'receipt.json',dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dataset_lock=lock,input_sha256=inputs,
                                        source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/train_tomato_probability_residual.py'),Path('tools/tomato_probability_residual.py'),Path('tools/tomato_residual_event_audit.py')]},pool_source_ids=[e['source_recording_id'] for e in pool]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
