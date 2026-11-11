"""One predeclared 148-parameter causal calibration experiment; frozen B0."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as pm, tomato_boundary_metrics as bm
from .tomato_independent_heads import load_checkpoint
from .tomato_probability_residual import ProbabilityHistory, ResidualCalibration
from .tomato_state_recovery import B0State, StableLabel
from .tomato_runtime import initialize_runtime
from .evaluate_tomato_complete100_history import freeze_dataset, SOURCE
from .evaluate_tomato_reviewed_boundaries import match_events
from .evaluate_tomato_causal_filter import aggregate

CONFIG=Path('configs/tomato_probability_residual_20261111/protocol.json')


def history_features(c,f,ts):
    h=ProbabilityHistory()
    return np.stack([h.update(a,b,float(t)) for a,b,t in zip(c,f,ts)])


def score(rows,pred):
    per={};zz=[];yy=[]
    for r in rows:
        e=r['episode'];eid=e['episode_id'];valid=r['valid'];ts=r['times'][valid]
        prob=pred[eid];assert prob.shape==r['future'].shape
        events=[];rule=StableLabel()
        for p,t in zip(prob[valid],ts):
            event=rule.update(p,float(t))
            if event:events.append(event)
        truth=bm.boundaries(dict(episode=e));matched,false=match_events(events,truth,'future',1.5)
        per[eid]=dict(boundaries=matched,false_events=false,events=events,duration_s=float(ts[-1]-ts[0]+.5))
        zz.append(np.log(np.maximum(prob[valid],1e-30)));yy.append(r['labels'][valid])
    return dict(events=aggregate(per),classification=metrics.scores(np.concatenate(zz),np.concatenate(yy))),per


def arrays(rows):
    return {k:np.concatenate([r[k][r['valid']] for r in rows]) for k in ('history','future','labels')}


def fit(inner,holdout,cfg,seed):
    initialize_runtime(dict(device='cuda',seed=seed))
    a,b=arrays(inner),arrays(holdout)
    mean=a['history'].mean(0,dtype=np.float64).astype('float32');std=a['history'].std(0,dtype=np.float64).clip(.01).astype('float32')
    model=ResidualCalibration(mean,std).cuda()
    tensors=[]
    for data in (a,b):
        tensors.append((torch.from_numpy(data['history']).cuda(),torch.from_numpy(np.log(np.maximum(data['future'],1e-6)).astype('float32')).cuda(),torch.from_numpy(data['labels']).long().cuda()))
    x,base,y=tensors[0];vx,vbase,vy=tensors[1]
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);rng=np.random.default_rng(seed)
    with torch.no_grad():initial=float(torch.nn.functional.cross_entropy(model(vx,vbase)[0],vy))
    best=initial;selected=0;state=copy.deepcopy(model.state_dict());stale=0;history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[]
        order=rng.permutation(len(y))
        for batch in np.array_split(order,max(1,int(np.ceil(len(order)/cfg['batch_size'])))):
            ix=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True)
            z,delta=model(x[ix],base[ix]);loss=torch.nn.functional.cross_entropy(z,y[ix])+cfg['residual_penalty']*delta.square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss))
        model.eval()
        with torch.no_grad():nll=float(torch.nn.functional.cross_entropy(model(vx,vbase)[0],vy))
        history.append(dict(epoch=epoch,train_objective=float(np.mean(losses)),internal_nll=nll))
        if nll<best-1e-8:best=nll;selected=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=cfg['patience']:break
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    model.load_state_dict(state);model.eval()
    return model,dict(initial_nll=initial,best_nll=best,selected_epoch=selected,executed_epochs=len(history),training_s=elapsed,history=history,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()))


@torch.inference_mode()
def predict(model,rows):
    result={};latency=[]
    for r in rows:
        x=torch.from_numpy(r['history']).cuda();base=torch.from_numpy(np.log(np.maximum(r['future'],1e-6)).astype('float32')).cuda()
        z,_=model(x,base);result[r['episode']['episode_id']]=torch.softmax(z,dim=-1).cpu().numpy()
    # Real single-observation resident-GPU calibration timing, excluding encoder.
    for i in range(110):
        torch.cuda.synchronize();start=time.perf_counter_ns();model(x[:1],base[:1]);torch.cuda.synchronize()
        if i>=10:latency.append((time.perf_counter_ns()-start)/1e6)
    return result,dict(p50=float(np.percentile(latency,50)),p95=float(np.percentile(latency,95)),maximum=max(latency),scope='Single residual head, resident GPU input, no encoder/capture/transfer')


def run(out,config=CONFIG):
    if out.exists():raise FileExistsError('New output directory required')
    cfg=ds.read_json(config);manifest,lock=freeze_dataset();out.mkdir(parents=True)
    identity=ds.read_json('runs/tomato/complete100_20261111/data_identity.json');by={r['episode_id']:r for r in identity['episodes']}
    training=sorted([e for e in manifest['episodes'] if e['split']=='train'],key=lambda e:e['source_recording_id'])
    if cfg.get('internal_split_strategy')=='b0_unseen_training_recordings':
        old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json')
        oldtrain={e['source_recording_id'] for e in old['episodes'] if e['split']=='train'}
        holdids={e['source_recording_id'] for e in training if e['source_recording_id'] not in oldtrain}
        assert len(holdids)==cfg['internal_holdout_recordings']==10
    else:
        order=np.random.default_rng(cfg['internal_split_seed']).permutation(len(training));holdids={training[i]['source_recording_id'] for i in order[:cfg['internal_holdout_recordings']]}
    split=dict(inner=[e['source_recording_id'] for e in training if e['source_recording_id'] not in holdids],holdout=sorted(holdids),development=[e['source_recording_id'] for e in manifest['episodes'] if e['split']=='val'])
    assert len(split['inner'])==84-cfg['internal_holdout_recordings'] and not set(split['inner'])&holdids
    ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'split.json',split)
    inputs={str(config):ds.digest(config)};cache={}
    if cfg.get('internal_split_strategy')=='b0_unseen_training_recordings':
        inputs['runs/tomato/campaign94_20261107/prepared/manifest.json']=ds.digest('runs/tomato/campaign94_20261107/prepared/manifest.json')
    for e in training:
        r=by[e['episode_id']];path=Path(r['feature_path']);assert ds.digest(path)==r['feature_sha256'];inputs[str(path)]=r['feature_sha256']
        z=np.load(path);assert np.array_equal(z['anchors'],e['anchors']) and np.allclose(z['times'],np.array(e['anchors'])/e['fps'])
        cache[e['episode_id']]=np.asarray(z['features'],np.float32)
    results={}
    for seed in cfg['seeds']:
        initialize_runtime(dict(device='cuda',seed=seed));folder=out/f'seed{seed}';folder.mkdir()
        modelid=ds.read_json(SOURCE/f'seed{seed}'/'result.json')['identity']['original_classifiers'];assert ds.digest(modelid['path'])==modelid['sha256'];inputs[modelid['path']]=modelid['sha256']
        saved,head=load_checkpoint(modelid['path'],'cuda');rows=[]
        if cfg.get('internal_split_strategy')=='b0_unseen_training_recordings':
            assert saved['split_sha256']==old['split_sha256']
        for e in manifest['episodes']:
            if e['split']=='test':continue
            eid=e['episode_id'];ts=np.asarray(e['anchors'])/e['fps']
            if e['split']=='train':
                z=pm.predict_heads(head,cache[eid],saved['mean'],saved['std']);c,f=[metrics.probabilities(z[k+'_logits']) for k in ('current','future')]
            else:
                path=SOURCE/f'seed{seed}'/(eid+'.npz');inputs[str(path)]=ds.digest(path);z=np.load(path);c,f=z['current_probabilities'],z['future_probabilities'];assert np.allclose(ts,z['times'])
            targets=ds.build_phase_targets(np.array(e['anchors']),e['annotation']['segments'],e['length'],e['fps'],1.5)
            rows.append(dict(episode=e,current=c,future=f,times=ts,valid=targets['future_valid_mask'],labels=targets['future_labels'],history=history_features(c,f,ts)))
        del head;torch.cuda.empty_cache()
        inner=[r for r in rows if r['episode']['split']=='train' and r['episode']['source_recording_id'] not in holdids]
        hold=[r for r in rows if r['episode']['source_recording_id'] in holdids];dev=[r for r in rows if r['episode']['split']=='val']
        net,training_result=fit(inner,hold,cfg,seed);pred,latency=predict(net,hold+dev)
        comparisons={};event_rows={}
        for name,part in [('internal_holdout',hold),('development8',dev)]:
            a,ap=score(part,{r['episode']['episode_id']:r['future'] for r in part});b,bp=score(part,pred)
            comparisons[name]=dict(B0=a,residual=b);event_rows[name]=dict(B0=ap,residual=bp)
        frozen=ds.read_json(f'runs/tomato/complete100_history_20261111_final/seed{seed}_per_episode.json')['none']
        for r in dev:
            eid=r['episode']['episode_id'];assert event_rows['development8']['B0'][eid]['events']==frozen[eid]['warnings']['events']
            raw,cal=B0State(),B0State()
            for i,t in enumerate(r['times']):
                a=raw.update(r['current'][i],r['future'][i],float(t));b=cal.update(r['current'][i],pred[eid][i],float(t))
                assert a['phase']==b['phase']==frozen[eid]['trace'][i]['phase'] and a['changed']==b['changed']
            np.savez_compressed(folder/(eid+'.npz'),times=r['times'],current=r['current'],original_future=r['future'],calibrated_future=pred[eid])
        checkpoint=folder/'residual.pt';torch.save(dict(format_version=1,architecture='causal_probability_residual_36x4',state_dict=net.cpu().state_dict(),base_model=modelid,
                                                       protocol=cfg,data_manifest_sha256=lock['manifest_sha256'],split_sha256=ds.fingerprint(split),seed=seed,selected_epoch=training_result['selected_epoch'],
                                                       schema=dict(class_order=ds.LABELS,history_lags_s=cfg['history_lags_s'],dimension=36,probability_floor=1e-6,input='causal Current/Future probabilities only',output='future1.5s logits; Current unchanged')),checkpoint)
        results[str(seed)]=dict(comparisons=comparisons,training=training_result,latency_ms=latency,checkpoint=str(checkpoint),checkpoint_sha256=ds.digest(checkpoint),prompt_and_position_identical=True)
        ds.write_json(folder/'events.json',event_rows);ds.write_json(folder/'result.json',results[str(seed)])
        print('SEED',seed,'epochs',training_result['selected_epoch'],'dev',comparisons['development8'],flush=True)
        del net;torch.cuda.empty_cache()
    means={name:{metric:float(np.mean([r['comparisons']['development8'][name]['events'][metric] for r in results.values()])) for metric in ['recall','false_per_minute']} for name in ['B0','residual']}
    a,b=results['42']['comparisons']['development8']['B0']['events'],results['42']['comparisons']['development8']['residual']['events']
    passed=means['residual']['recall']>means['B0']['recall'] and means['residual']['false_per_minute']<=means['B0']['false_per_minute'] and b['hits']>a['hits'] and b['false_events']<=a['false_events']
    ds.write_json(out/'summary.json',dict(seeds=results,means=means,candidate_gate_passed=passed,production_changed=False,encoder_calls=0,historical_test_arrays_read=False,labels_modified=False))
    ds.write_json(out/'receipt.json',dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dataset_lock=lock,inputs=inputs,
                                       source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_probability_residual.py')]},device=torch.cuda.get_device_name()))
    print('COMPLETE',means,'candidate gate',passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args();run(a.out,a.config)
