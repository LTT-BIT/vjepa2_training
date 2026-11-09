"""Three-seed hazard training, immutable feature reuse and causal offline ablations."""
import argparse,copy,subprocess,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .tomato_runtime import initialize_runtime,hardware_report
from .tomato_independent_heads import load_checkpoint
from .tomato_transition_time import TransitionTimeHead,survival_targets,hazard_nll,event_distribution,PredictiveTransitionDecision,CLASSES
from .evaluate_tomato_reviewed_boundaries import revise_episode,match_events

CFG=Path('configs/tomato_time_transition_20261109/experiment.json')
OLD=Path('runs/tomato/campaign94_20261107/prepared')
REFERENCE=Path('docs/validation/tomato_reviewed_boundaries_20261109')


def load_entries(out):
    manifest=ds.read_json(OLD/'manifest.json');index=ds.read_json(OLD/'feature_index.json');paths={e['episode_id']:e['path'] for e in index['episodes']}
    approved=ds.read_json('configs/tomato_boundary_review_20261109/annotation_revision_provenance.json')
    excluded=ds.read_json('configs/tomato_curated91_20261109/quality_exclusions.json');bad={e['source_recording_id'] for e in excluded['excluded_episodes']}
    proposals=ds.read_json(REFERENCE/'proposals.json')['truth_proposals'];entries={'train':[],'val':[]};receipts=[];changed=[]
    contract={**manifest['contract'],'sampler_version':manifest['sampler_version']}
    for e in manifest['episodes']:
        if e['split']=='test' or e['source_recording_id'] in bad:continue
        pm.assert_source_unchanged(e)
        actual=ds.digest(e['annotation']['path']);ee=copy.deepcopy(e)
        if actual!=e['annotation']['sha256']:
            assert e['episode_id']==approved['episode_id'] and actual==approved['corrected_annotation_sha256'],'Unexpected annotation revision'
            ee['annotation']['segments']=ds.read_json(e['annotation']['path'])['segments'];ee['annotation']['sha256']=actual;ee['annotation']['revision']=ds.read_json(e['annotation']['path'])['revision']
            changed.append(dict(episode_id=e['episode_id'],old=e['annotation']['sha256'],new=actual,used_for_training=True))
        # Feature identity explicitly excludes annotations; load_cache verifies the whole
        # source/timestamp/joint/history contract. Targets are rebuilt below.
        data=pm.load_cache(paths[e['episode_id']],e,contract,manifest['pretrained'])
        if e['split']=='val':ee=revise_episode(ee,proposals.get(e['episode_id'],[]))
        targets=survival_targets(ee['annotation']['segments'],data['anchors'],ee['fps'],ee['length'])
        entries[e['split']].append(dict(episode=ee,x=data['features'],times=data['times'],anchors=data['anchors'],targets=targets))
        receipts.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],source_annotation_sha256=actual,feature_path=paths[e['episode_id']],feature_sha256=ds.digest(paths[e['episode_id']]),supervision_sha256=ds.fingerprint(dict(segments=ee['annotation']['segments'],anchors=data['anchors'].tolist())),windows=len(data['anchors'])))
        print('VERIFIED',e['split'],e['episode_id'],flush=True)
    assert {s:len(v) for s,v in entries.items()}=={'train':74,'val':9}
    ds.write_json(out/'data_identity.json',dict(old_manifest_sha256=ds.digest(OLD/'manifest.json'),feature_index_sha256=ds.digest(OLD/'feature_index.json'),original_split_sha256=manifest['split_sha256'],approved_training_revision=changed,provisional_validation_proposals_sha256=ds.digest(REFERENCE/'proposals.json'),episodes=receipts,test_features_read=False))
    return entries,manifest


def arrays(entries,mean=None,std=None):
    raw=np.concatenate([e['x'] for e in entries])
    if mean is None:mean=raw.mean(0,dtype=np.float64).astype('float32');std=raw.std(0,dtype=np.float64).clip(.01).astype('float32')
    y=np.concatenate([r['targets']['labels'] for r in entries]);mask=np.concatenate([r['targets']['mask'] for r in entries])
    return ((raw-mean)/std).astype('float32'),y,mask,mean,std


def train_model(train,val,seed,cfg,epochs=None):
    initialize_runtime(dict(device='cuda',seed=seed));x,y,mask,mean,std=arrays(train)
    a=torch.from_numpy(x).cuda();b=torch.from_numpy(y).cuda();c=torch.from_numpy(mask).cuda()
    valid=np.flatnonzero(mask.sum(1)>0);net=TransitionTimeHead(x.shape[1]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    if val:
        vx,vy,vm,_,_=arrays(val,mean,std);v=[torch.from_numpy(z).cuda() for z in (vx,vy,vm)]
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;stale=0;history=[];began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,(epochs or cfg['max_epochs'])+1):
        net.train();losses=[];order=rng.permutation(valid)
        for i in range(0,len(order),cfg['batch_size']):
            ix=torch.as_tensor(order[i:i+cfg['batch_size']],device='cuda');opt.zero_grad(set_to_none=True);loss=hazard_nll(net(a[ix]),b[ix],c[ix]);loss.backward();opt.step();losses.append(loss.item())
        net.eval()
        with torch.inference_mode():vn=hazard_nll(net(v[0]),v[1],v[2]).item() if val else None
        history.append(dict(epoch=epoch,train_nll=float(np.mean(losses)),internal_nll=vn))
        if val:
            if vn<best-1e-6:best=vn;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=cfg['patience']:break
    elapsed=time.perf_counter()-began
    return net,mean,std,dict(best_epoch=best_epoch,epochs=len(history),training_s=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),history=history)


def hazard_predict(net,rows,mean,std):
    net.eval();out={}
    with torch.inference_mode():
        for r in rows:
            x=torch.from_numpy(((r['x']-mean)/std).astype('float32')).cuda();z=net(x);out[r['episode']['episode_id']]=dict(logits=z.cpu().numpy(),masses=event_distribution(z).cpu().numpy())
    return out


def cf_predict(rows,seed):
    path=Path(f'runs/tomato/optimization_20261108/bundles/independent_h15_s{seed}.pt');saved,heads=load_checkpoint(path,'cuda');out={}
    for r in rows:
        logits=pm.predict_heads(heads,r['x'],saved['mean'],saved['std']);out[r['episode']['episode_id']]={h:metrics.probabilities(logits[h+'_logits']) for h in ('current','future')}
    return out,dict(path=str(path),sha256=ds.digest(path),historical_cf_norm=True),saved,heads


def summary(per):
    bs=[b for v in per.values() for b in v['boundaries']];hits=[b for b in bs if b['detected']];fp=sum(len(v['false_events']) for v in per.values());mins=sum(v['duration_s'] for v in per.values())/60
    recall=len(hits)/len(bs);precision=len(hits)/max(1,len(hits)+fp);by=defaultdict(lambda:dict(total=0,detected=0))
    for b in bs:v=by[ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]];v['total']+=1;v['detected']+=int(b['detected'])
    lead=[-b['offset_s'] for b in hits]
    return dict(total=len(bs),detected=len(hits),missed=len(bs)-len(hits),precision=precision,recall=recall,f1=2*precision*recall/max(1e-9,precision+recall),false_events=fp,false_per_minute=fp/mins,minutes=mins,lead_p50_s=float(np.median(lead)) if lead else None,lead_p05_s=float(np.percentile(lead,5)) if lead else None,by_transition=dict(by))


def replay(rows,cf,ht,threshold):
    per={k:{} for k in ['B0','B1_gate','B2']};decisions=[]
    for r in rows:
        e=r['episode'];eid=e['episode_id'];times=r['times'];truth=bm.boundaries(dict(episode=e));p=cf[eid];mass=ht[eid]['masses'];valid=r['anchors']+45<e['length'];dur=(times[valid][-1]-times[valid][0]+.5)
        b0=bm.change_events(p['future'][valid],times[valid],.5,2)
        # Single-factor B1 diagnostic: gate existing warnings using hazard, keeping
        # classifier predictions, timing and confirmation count identical to B0.
        b1=[b for b in b0 if 1-mass[np.argmin(abs(times-b['time'])),-1]>=threshold]
        d=PredictiveTransitionDecision(threshold=threshold);events=[];trace=[]
        for i,t in enumerate(times):
            if not valid[i]:continue
            prior=d.phase;v=d.update(p['current'][i],p['future'][i],mass[i],float(t),float(t)+.01,transition_timestamp=float(t));trace.append(dict(time=float(t),**v))
            if v.get('warning') and prior<6:events.append(dict(time=float(t),previous=CLASSES[prior],label=CLASSES[prior+1]))
        for k,ev in [('B0',b0),('B1_gate',b1),('B2',events)]:
            matched,false=match_events(ev,truth,'future',1.5);common,_=match_events(ev,truth,'future',1.)
            per[k][eid]=dict(boundaries=matched,false_events=false,events=ev,duration_s=float(dur),common_1s_hits=sum(x['detected'] for x in common))
        # Confirmation must be scored separately from warnings, under current window.
        changes=[];prev=0
        for t in trace:
            if t.get('suggest_switch'):
                changes.append(dict(time=t['time'],previous=CLASSES[prev],label=CLASSES[t['phase_index']]));prev=t['phase_index']
        confirmed,false=match_events(changes,truth,'current',1.5)
        decisions.append(dict(episode_id=eid,confirmed=confirmed,false_confirmations=false,trace=trace))
    return dict(summaries={k:summary(v) for k,v in per.items()},per_episode=per,decision_replay=decisions)


def time_quality(rows,pred):
    losses=[];briers=[];errors=[];exact=[];weight=[]
    for r in rows:
        eid=r['episode']['episode_id'];t=r['targets'];z=torch.from_numpy(pred[eid]['logits']);y=torch.from_numpy(t['labels']);mask=torch.from_numpy(t['mask']);n=int((t['mask'].sum(1)>0).sum())
        losses.append(hazard_nll(z,y,mask).item());weight.append(n);p=pred[eid]['masses'];valid=t['event_observed']|(t['followup']>=1.5);yy=(t['event_observed']&(t['time_to_event']<=1.5)).astype(float)
        briers.extend(((1-p[valid,-1]-yy[valid])**2).tolist())
        hit=t['event_observed']&(t['time_to_event']<=1.5)
        if hit.any():
            dt=t['time_to_event'][hit];pp=p[hit,:4];est=(pp@np.array([.125,.375,.75,1.25]))/pp.sum(1).clip(1e-9)
            errors.extend((est-dt).tolist());exact.extend((pp.argmax(1)==np.searchsorted(np.array([.25,.5,1.,1.5]),dt-1e-9)).tolist())
    return dict(censored_nll=float(np.average(losses,weights=weight)),horizon_brier=float(np.mean(briers)),observed_near_event_samples=len(errors),near_event_conditional_mae_s=float(np.mean(np.abs(errors))),near_event_signed_error_p50_s=float(np.median(errors)),near_event_bin_accuracy=float(np.mean(exact)),timing_error_scope='conditional within1.5s only; tail has no finite midpoint; no imputed censor labels')


def run(out):
    if out.exists():raise FileExistsError('New output directory required')
    cfg=ds.read_json(CFG);out.mkdir(parents=True);initialize_runtime(dict(device='cuda',seed=42))
    frozen={str(p):ds.digest(p) for p in [CFG,OLD/'manifest.json',OLD/'feature_index.json',REFERENCE/'results.json',REFERENCE/'proposals.json',Path('configs/tomato_curated91_20261109/split.json')]}
    ds.write_json(out/'protocol.json',dict(config=cfg,input_hashes=frozen,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_transition_time.py')]},hardware=hardware_report()))
    (out/'executed_train_tomato_transition_time.py').write_bytes(Path(__file__).read_bytes())
    (out/'executed_tomato_transition_time.py').write_bytes(Path('tools/tomato_transition_time.py').read_bytes())
    entries,manifest=load_entries(out);train=entries['train'];val=entries['val'];order=np.random.default_rng(cfg['internal_split_seed']).permutation(len(train));internal=[train[i] for i in order[:10]];inner=[train[i] for i in order[10:]]
    ds.write_json(out/'internal_split.json',dict(train=[r['episode']['source_recording_id'] for r in inner],holdout=[r['episode']['source_recording_id'] for r in internal],external=[r['episode']['source_recording_id'] for r in val]))
    results=[]
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();net,mean,std,select=train_model(inner,internal,seed,cfg)
        # Refit inner model for the internally selected epoch before threshold calibration;
        # external validation has never been used by an optimizer or selector.
        net,mean,std,innerfit=train_model(inner,None,seed,cfg,epochs=select['best_epoch'])
        icf,_,_,_=cf_predict(internal,seed);ih=hazard_predict(net,internal,mean,std);candidates={}
        for threshold in cfg['thresholds']:
            candidates[str(threshold)]=replay(internal,icf,ih,threshold)['summaries']
        feasible=[t for t,v in candidates.items() if v['B2']['false_per_minute']<=v['B0']['false_per_minute']]
        chosen=max(feasible,key=lambda t:(candidates[t]['B2']['recall'],-candidates[t]['B2']['false_per_minute'])) if feasible else min(candidates,key=lambda t:candidates[t]['B2']['false_per_minute'])
        net,mean,std,final=train_model(train,None,seed,cfg,epochs=select['best_epoch']);torch.cuda.empty_cache();net.eval()
        hp=hazard_predict(net,val,mean,std);cf,source,saved,heads=cf_predict(val,seed)
        identity=dict(version='tomato-transition-time-v1',seed=seed,config_sha256=ds.digest(CFG),data_identity_sha256=ds.digest(out/'data_identity.json'),input_schema=manifest['contract'],pretrained=manifest['pretrained'],original_classifiers=source,label_version='train approved correction; val14 provisional proposals',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256=ds.digest('tools/tomato_transition_time.py'))
        torch.save(dict(head=net.cpu().state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),identity=identity,epochs=select['best_epoch'],optimizer_scope='time head only',bin_upper_s=[.25,.5,1.,1.5]),folder/'hazard.pt');net.cuda()
        timings=[];base_timings=[];torch.cuda.reset_peak_memory_stats()
        x=val[0]['x'][:1];hx=torch.from_numpy(((x-mean)/std).astype('float32')).cuda();cx=torch.from_numpy(((x-saved['mean'].numpy())/saved['std'].numpy()).astype('float32')).cuda()
        with torch.inference_mode():
            for i in range(110):
                torch.cuda.synchronize();t=time.perf_counter();heads(cx);torch.cuda.synchronize();a=time.perf_counter();net(hx);torch.cuda.synchronize();b=time.perf_counter()
                if i>=10:base_timings.append((a-t)*1000);timings.append((b-a)*1000)
        valruns={str(t):replay(val,cf,hp,t) for t in cfg['thresholds']}
        classification={}
        for head in ['current','future']:
            probs=[];labels=[]
            for r in val:
                e=r['episode'];a=r['anchors'];valid=np.ones(len(a),bool) if head=='current' else a+45<e['length'];probs.append(cf[e['episode_id']][head][valid]);labels.extend(ds.label_at_frame(e['annotation']['segments'],int(i)+(45 if head=='future' else 0)) for i in a[valid])
            classification[head]=metrics.scores(np.log(np.concatenate(probs).clip(1e-30)),np.asarray(labels))
        if seed==42:
            ref=ds.read_json(REFERENCE/'results.json')['independent_h15/reviewed_truth14_provisional'];b0=valruns[chosen]['summaries']['B0']
            assert b0['detected']==39 and b0['false_events']==13
            assert abs(classification['current']['macro_f1']-ref['classification']['current']['macro_f1'])<1e-8
        for r in val:
            eid=r['episode']['episode_id'];np.savez_compressed(folder/(eid+'.npz'),times=r['times'],anchors=r['anchors'],current_probabilities=cf[eid]['current'],future_probabilities=cf[eid]['future'],hazard_logits=hp[eid]['logits'],transition_probabilities=hp[eid]['masses'])
        result=dict(seed=seed,identity=identity,checkpoint_sha256=ds.digest(folder/'hazard.pt'),internal_epoch_selection=select,internal_refit=innerfit,final_training=final,internal_threshold_results=candidates,selected_threshold=float(chosen),external_validation_thresholds_are_ablation_not_selection=valruns,classification=classification,time_quality=time_quality(val,hp),head_latency_ms=dict(hazard_p50=float(np.median(timings)),hazard_p95=float(np.percentile(timings,95)),hazard_max=float(max(timings)),original_cf_p95=float(np.percentile(base_timings,95))),benchmark_peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in net.parameters()))
        ds.write_json(folder/'result.json',result);results.append(result)
        print('SEED',seed,'epoch',select['best_epoch'],'threshold',chosen,'RESULT',valruns[chosen]['summaries'],'TTT',result['time_quality'],flush=True)
        del net,heads;torch.cuda.empty_cache()
    ds.write_json(out/'summary.json',dict(results=[dict(seed=r['seed'],threshold=r['selected_threshold'],classification=r['classification'],time_quality=r['time_quality'],latency=r['head_latency_ms'],summaries=r['external_validation_thresholds_are_ablation_not_selection'][str(r['selected_threshold'])]['summaries']) for r in results]))
    assert all(ds.digest(p)==sha for p,sha in frozen.items())
    ds.write_json(out/'integrity.json',dict(passed=True,input_hashes_unchanged=True,test_features_read=False,encoder_calls=0,source_annotations_modified=False,remote_pushed=False,training_stages=9,seeds=[42,43,44]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
