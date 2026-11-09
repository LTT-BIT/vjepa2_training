"""One-shot frozen-model evaluation of newly uploaded independent recordings."""
from pathlib import Path
import copy,time
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .tomato_runtime import initialize_runtime
from .tomato_independent_heads import load_checkpoint
from .evaluate_tomato_reviewed_boundaries import match_events
from .train_tomato_transition_time import replay as hazard_replay
from .tomato_transition_time import TransitionTimeHead,event_distribution,survival_targets
ROOT=Path('runs/tomato/independent10_20261109')
DATA=Path('/home/user/datasets/tomato_independent10_20261109')


def summarize(per,head):
    vals=[x[head] for x in per.values()];bs=[b for v in vals for b in v['boundaries']];hits=[b for b in bs if b['detected']];fp=sum(len(v['false_events']) for v in vals);minutes=sum(v['duration_s'] for v in vals)/60
    precision=len(hits)/max(1,len(hits)+fp);recall=len(hits)/len(bs);types={}
    for b in bs:
        key=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']];v=types.setdefault(key,dict(total=0,detected=0));v['total']+=1;v['detected']+=int(b['detected'])
    return dict(total=len(bs),detected=len(hits),recall=recall,precision=precision,f1=2*precision*recall/max(1e-9,precision+recall),false_events=fp,false_per_minute=fp/minutes,evaluated_minutes=minutes,offset_p50_s=float(np.median([b['offset_s'] for b in hits])) if hits else None,offset_p95_s=float(np.percentile([b['offset_s'] for b in hits],95)) if hits else None,by_transition=types)


def run():
    initialize_runtime(dict(device='cuda',seed=42));protocol=ds.read_json(ROOT/'locked_protocol.json');inventory=ds.read_json(ROOT/'remote_inventory.json')
    if (ROOT/'results.json').exists():raise FileExistsError('Test already evaluated; do not re-search or overwrite')
    for v in protocol['model_sha256'].values():assert ds.digest(v['path'])==v['sha256']
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');oldids={e['source_recording_id'] for e in old['episodes']};oldhashes={h for e in old['episodes'] for h in e['input_sha256'].values()};contract={**old['contract'],'sampler_version':old['sampler_version']}
    entries=[];rejected=[];datahashes={};began=time.perf_counter()
    for item in inventory['records']:
        try:
            e=ds.load_episode_info(DATA/item['package'],item['episode_index']);assert e['source_recording_id'] not in oldids
            ann=ds.load_reviewed_coarse(DATA/item['annotation'],e,False);hashes={str(Path(p).relative_to(e['dataset_root'])):ds.digest(p) for p in [e['parquet'],*e['videos']]}
            assert not oldhashes&set(hashes.values()),'Old content reused under new recording ID'
            assert all(h not in datahashes for h in hashes.values()),'Repeated content inside new set'
            datahashes.update({h:e['source_recording_id'] for h in hashes.values()})
            media=ds.validate_source_and_media(e,contract['joint_names'],contract['joint_mode']);t,q=ds.read_state(e,contract['joint_names'],contract['joint_mode']);a=ds.iter_phase_anchors(t)
            e.update(annotation=ann,input_sha256=hashes,anchors=a.tolist(),split='independent_test',media_validation=media)
            entries.append(dict(episode=e,times=t[a],anchors=a,targets=survival_targets(ann['segments'],a,e['fps'],e['length'])))
            print('VALIDATED',item['package'],item['episode_index'],e['length'],len(a),flush=True)
        except (ValueError,AssertionError,OSError,KeyError) as exc:rejected.append(dict(item=item,error=str(exc)))
    ds.write_json(ROOT/'validation.json',dict(accepted=len(entries),rejected=rejected,seconds=time.perf_counter()-began,full_media_validation=True,source_and_content_no_overlap=True))
    if rejected or len(entries)!=10:raise ValueError('Independent test input validation failed; inspect rejections without relaxing checks')
    ds.write_json(ROOT/'manifest.json',dict(episodes=[r['episode'] for r in entries],contract=contract,pretrained=old['pretrained'],role='independent_test_only',source_protocol_sha256=ds.digest(ROOT/'locked_protocol.json')))
    encoder,provenance=pm.load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda',42);assert provenance==old['pretrained'];cache=ROOT/'features';cache.mkdir(exist_ok=True);extraction=[]
    for r in entries:
        path,perf=pm.extract_episode(encoder,r['episode'],contract,provenance,cache,batch_windows=4);d=pm.load_cache(path,r['episode'],contract,provenance);r['x']=d['features'];extraction.append(dict(episode_id=r['episode']['episode_id'],path=str(path),sha256=ds.digest(path),performance=perf));print('FEATURES',r['episode']['episode_id'],len(r['x']),flush=True)
    del encoder;torch.cuda.empty_cache();ds.write_json(ROOT/'extraction.json',extraction)
    results={};cf_primary=None
    for name,source in protocol['model_sha256'].items():
        if name=='hazard_s42':continue
        saved,head=load_checkpoint(source['path'],'cuda');horizon=saved['future_horizon_s'];per={};buffers={h:[[],[]] for h in ['current','future']};cf={}
        for r in entries:
            e=r['episode'];eid=e['episode_id'];a=r['anchors'];ts=r['times'];truth=bm.boundaries(dict(episode=e));z=pm.predict_heads(head,r['x'],saved['mean'],saved['std']);per[eid]={};cf[eid]={}
            for h in ['current','future']:
                p=metrics.probabilities(z[h+'_logits']);cf[eid][h]=p;valid=np.ones(len(a),bool) if h=='current' else a+round(horizon*e['fps'])<e['length'];ev=bm.change_events(p[valid],ts[valid],.5,2);matched,false=match_events(ev,truth,h,horizon);common,_=match_events(ev,truth,h,1.)
                y=np.array([ds.label_at_frame(e['annotation']['segments'],int(i)+(round(horizon*e['fps']) if h=='future' else 0)) for i in a[valid]]);buffers[h][0].append(z[h+'_logits'][valid]);buffers[h][1].append(y)
                per[eid][h]=dict(boundaries=matched,false_events=false,events=ev,duration_s=float(ts[valid][-1]-ts[valid][0]+.5),classification=metrics.scores(z[h+'_logits'][valid],y),common_1s_hits=sum(b['detected'] for b in common))
            np.savez_compressed(ROOT/(name+'_'+eid+'.npz'),anchors=a,times=ts,**z)
        results[name]=dict(horizon_s=horizon,classification={h:metrics.scores(np.concatenate(v[0]),np.concatenate(v[1])) for h,v in buffers.items()},events={h:summarize(per,h) for h in buffers},per_episode=per,model=source)
        if name==protocol['primary']:cf_primary=cf
        print('MODEL',name,'current',results[name]['classification']['current']['macro_f1'],'future',results[name]['classification']['future']['macro_f1'],'events',results[name]['events']['future'],flush=True)
        del head;torch.cuda.empty_cache()
    # Fixed diagnostic of the failed new time head. Never selects a new threshold/model.
    saved=torch.load(protocol['model_sha256']['hazard_s42']['path'],weights_only=True,map_location='cpu');net=TransitionTimeHead().cuda();net.load_state_dict(saved['head']);net.eval();hp={}
    with torch.inference_mode():
        for r in entries:
            x=torch.from_numpy(r['x']).cuda();z=net((x-saved['mean'].cuda())/saved['std'].cuda());hp[r['episode']['episode_id']]=dict(logits=z.cpu().numpy(),masses=event_distribution(z).cpu().numpy())
    diagnostic=hazard_replay(entries,cf_primary,hp,.5);ds.write_json(ROOT/'time_head_diagnostic.json',diagnostic)
    # Bootstrap recordings for primary recall, preserving all repeated cycles in a source.
    rng=np.random.default_rng(42);per=results[protocol['primary']]['per_episode'];ids=list(per);draw=[]
    for _ in range(3000):
        sample=[ids[i] for i in rng.integers(0,10,10)];bs=[b for eid in sample for b in per[eid]['future']['boundaries']];draw.append(sum(b['detected'] for b in bs)/len(bs))
    ds.write_json(ROOT/'results.json',results);ds.write_json(ROOT/'uncertainty.json',dict(primary_future_recall_recording_bootstrap_ci95=np.percentile(draw,[2.5,97.5]).tolist(),recordings=10,draws=3000,scope='same scene/operator may limit independence beyond recording'))
    for source in protocol['model_sha256'].values():assert ds.digest(source['path'])==source['sha256']
    for f in inventory['files']:assert ds.digest(DATA/f['path'])==f['sha256']
    ds.write_json(ROOT/'integrity.json',dict(passed=True,training_calls=0,normalization_refitted=False,model_or_threshold_selected_on_new_test=False,source_or_checkpoint_changed=False,remote_pushed=False,independent_recordings=10))
    print('COMPLETE independent10 frozen evaluation',flush=True)

if __name__=='__main__':run()
