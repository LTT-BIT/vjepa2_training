"""Data-only independent-head refresh: clean74 control vs new84; no test inference."""
import copy,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .tomato_runtime import initialize_runtime
from .train_tomato_vjepa_phase import fit_class_weights,training_batches
from .tomato_independent_heads import pack,load_checkpoint
from .evaluate_tomato_reviewed_boundaries import revise_episode,match_events
from .train_tomato_transition_time import summary
ROOT=Path('runs/tomato/refresh101_20261109_v2');CFG=Path('configs/tomato_refresh101_20261109/experiment.json')


def dataset_membership(old_splits,new_ids):
    out={s:list(v) for s,v in old_splits.items()}
    if set(new_ids)&set(sum(out.values(),[])):raise ValueError('New source overlaps old split')
    out['train']=sorted(out['train']+list(new_ids))
    assert not(set(out['train'])&set(out['val'])) and not(set(out['train'])&set(out['test']))
    return out


def prepare():
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');new=ds.read_json('runs/tomato/independent10_20261109/manifest.json');inventory=ds.read_json('runs/tomato/time_usage_analysis_20261109/server101_inventory.json')['records'];server={r['source_recording_id']:r for r in inventory};oldindex=ds.read_json('runs/tomato/campaign94_20261107/prepared/feature_index.json');paths={x['episode_id']:x['path'] for x in oldindex['episodes']};paths.update({x['episode_id']:x['path'] for x in ds.read_json('runs/tomato/independent10_20261109/extraction.json')})
    splits=ds.read_json('configs/tomato_curated91_20261109/split.json');newids={e['source_recording_id'] for e in new['episodes']};combined=dataset_membership(splits,newids);assert {s:len(v) for s,v in combined.items()}=={'train':84,'val':9,'test':8}
    approved=ds.read_json('configs/tomato_boundary_review_20261109/annotation_revision_provenance.json');props=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals'];all_entries=[];manifest_episodes=[];identity=[]
    for original in old['episodes']+new['episodes']:
        sid=original['source_recording_id']
        if sid not in server:continue
        e=copy.deepcopy(original);e['split']=next(s for s,ids in combined.items() if sid in ids);actual=ds.digest(e['annotation']['path']);assert actual==server[sid]['annotation_sha256'];assert e['input_sha256']==server[sid]['inputs']
        if actual!=e['annotation']['sha256']:
            assert e['episode_id']==approved['episode_id'] and actual==approved['corrected_annotation_sha256']
            ann=ds.read_json(e['annotation']['path']);e['annotation'].update(sha256=actual,segments=ann['segments'],revision=ann['revision'])
        if e['split']=='val':e=revise_episode(e,props.get(e['episode_id'],[]))
        manifest_episodes.append(e)
        if e['split']=='test':continue
        pm.assert_source_unchanged(original)
        d=pm.load_cache(paths[e['episode_id']],original,{**old['contract'],'sampler_version':old['sampler_version']},old['pretrained']);targets=ds.build_phase_targets(d['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5)
        all_entries.append(dict(episode=e,features=d['features'],times=d['times'],**targets))
        identity.append(dict(episode_id=e['episode_id'],source_recording_id=sid,split=e['split'],is_new10=sid in newids,feature_path=paths[e['episode_id']],feature_sha256=ds.digest(paths[e['episode_id']]),annotation_sha256=actual,annotation_revision=e['annotation']['revision'],effective_segments_sha256=ds.fingerprint(e['annotation']['segments']),supervision_sha256=ds.fingerprint(ds.serializable_targets(targets))))
        print('VERIFIED',e['split'],e['episode_id'],flush=True)
    assert len(manifest_episodes)==101 and len({e['source_recording_id'] for e in manifest_episodes})==101
    ds.write_json(ROOT/'data_identity.json',dict(episodes=identity,server_inventory_sha256=ds.digest('runs/tomato/time_usage_analysis_20261109/server101_inventory.json'),new10_exposed_test_promoted_to_train=True,archived_test_arrays_read=False))
    datasets={};manifests={}
    for cohort in ['clean91_control','refresh101']:
        members=combined if cohort=='refresh101' else splits;selected=[e for e in manifest_episodes if e['source_recording_id'] in set(sum(members.values(),[]))];rows=[r for r in all_entries if r['episode']['source_recording_id'] in set(sum(members.values(),[]))]
        mf={**old,'episodes':selected,'splits':members,'split_sha256':ds.fingerprint(members),'label_version':'approved training correction +14 provisional val boundary edits','data_identity_sha256':ds.digest(ROOT/'data_identity.json')}
        split_path=Path('configs/tomato_refresh101_20261109')/(cohort+'_split.json');ds.write_json(split_path,members)
        from .finalize_tomato_refresh_manifest import corrected_manifest
        mf=corrected_manifest(mf,split_path);manifests[cohort]=mf;datasets[cohort]={s:[r for r in rows if r['episode']['split']==s] for s in ['train','val']};folder=ROOT/cohort;folder.mkdir();ds.write_json(folder/'manifest.json',mf)
        ds.write_json(Path('configs/tomato_refresh101_20261109')/(cohort+'_split.json'),members)
    return datasets,manifests


def fit_head(rows,manifest,cohort,seed,mode,cfg):
    initialize_runtime(dict(device='cuda',seed=seed));folder=ROOT/cohort/f'{mode}_s{seed}';folder.mkdir();active='current' if mode=='current_only' else 'future'
    mf=copy.deepcopy(manifest);mf['contract']['seed']=seed;mf['prediction']=ds.prediction_contract(dict(prediction_mode=mode,future_horizon_s=1.5));ds.write_json(folder/'manifest.json',mf)
    arrays={}
    for s,parts in rows.items():arrays[s]={key:np.concatenate([r[key] for r in parts]) for key in ['features','current_labels','future_labels','future_valid_mask']}
    raw=arrays['train']['features'];mean=raw.mean(0,dtype=np.float64).astype('float32');std=raw.std(0,dtype=np.float64).clip(.01).astype('float32');x=torch.from_numpy(((raw-mean)/std).astype('float32')).cuda();vx=torch.from_numpy(((arrays['val']['features']-mean)/std).astype('float32')).cuda();y=arrays['train'][active+'_labels'];valid=arrays['train']['future_valid_mask'];vy=arrays['val'][active+'_labels'];vvalid=np.ones(len(vy),bool) if active=='current' else arrays['val']['future_valid_mask'];weights,_=fit_class_weights(y if active=='current' else y[valid]);w=torch.from_numpy(weights).cuda();yg=torch.from_numpy(y).cuda();head=pm.mode_head_model(58,mode).cuda();opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(seed);best=(-1.,-float('inf'));stale=0;history=[];began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg['epochs']+1):
        head.train();losses=[];batches=training_batches(arrays['train']['future_labels'],valid,len(raw),cfg['batch_size'],rng,mode)
        for batch in batches:
            ix=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(head(x[ix]),yg[ix],weight=w);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step();losses.append(loss.item())
        head.eval()
        with torch.inference_mode():z=head(vx).cpu().numpy()
        val=metrics.scores(z[vvalid],vy[vvalid]);key=(val['macro_f1'],-val['nll']);history.append(dict(epoch=epoch,train_ce=float(np.mean(losses)),validation=val))
        if key>best:
            best=key;stale=0;pm.save_checkpoint(folder/'best_head.pt',head,mean,std,mf,ds.digest(folder/'manifest.json'),epoch,{active:val})
        else:stale+=1
        if stale>=cfg['patience']:break
    elapsed=time.perf_counter()-began
    torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state),folder/'last_state.pt')
    result=dict(cohort=cohort,seed=seed,mode=mode,train_recordings=len(rows['train']),val_recordings=len(rows['val']),training_s=elapsed,epochs=len(history),history=history,checkpoint_sha256=ds.digest(folder/'best_head.pt'),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),normalization_sha256=ds.fingerprint(dict(mean=mean.tolist(),std=std.tolist())),code_sha256=ds.digest(__file__),data_identity_sha256=ds.digest(ROOT/'data_identity.json'))
    ds.write_json(folder/'result.json',result);print('TRAINED',cohort,seed,mode,'bestF1',best[0],'epochs',len(history),'seconds',elapsed,flush=True);del x,vx,head,opt;torch.cuda.empty_cache();return folder/'best_head.pt',result


def evaluate(bundle,rows):
    saved,head=load_checkpoint(bundle,'cuda');per={};buffer={h:[[],[]] for h in ['current','future']};cf={}
    for r in rows:
        e=r['episode'];eid=e['episode_id'];z=pm.predict_heads(head,r['features'],saved['mean'],saved['std']);truth=bm.boundaries(dict(episode=e));per[eid]={};cf[eid]=z
        for h in ['current','future']:
            valid=np.ones(len(r['anchors']),bool) if h=='current' else r['future_valid_mask'];p=metrics.probabilities(z[h+'_logits']);events=bm.change_events(p[valid],r['times'][valid],.5,2);matched,false=match_events(events,truth,h,1.5);common,_=match_events(events,truth,h,1.);y=r[h+'_labels'][valid];buffer[h][0].append(z[h+'_logits'][valid]);buffer[h][1].append(y);per[eid][h]=dict(boundaries=matched,false_events=false,events=events,duration_s=float(r['times'][valid][-1]-r['times'][valid][0]+.5),common_1s_hits=sum(b['detected'] for b in common),classification=metrics.scores(z[h+'_logits'][valid],y))
    ev=dict(classification={h:metrics.scores(np.concatenate(v[0]),np.concatenate(v[1])) for h,v in buffer.items()},events={h:summary({eid:v[h] for eid,v in per.items()}) for h in buffer},per_episode=per)
    del head;torch.cuda.empty_cache();return ev,cf


def run():
    if ROOT.exists():raise FileExistsError('New run required; never overwrite completed experiment')
    ROOT.mkdir(parents=True);cfg=ds.read_json(CFG);ds.write_json(ROOT/'protocol.json',dict(config=cfg,config_sha256=ds.digest(CFG),base_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),analysis_sha256=ds.digest('runs/tomato/time_usage_analysis_20261109/analysis.json'),source_sha256=ds.digest(__file__),no_test_inference=True))
    (ROOT/'executed_runner.py').write_bytes(Path(__file__).read_bytes());datasets,manifests=prepare();results=[]
    for cohort in cfg['cohorts']:
        for seed in cfg['seeds']:
            paths={};training=[]
            for mode in cfg['heads']:paths[mode],r=fit_head(datasets[cohort],manifests[cohort],cohort,seed,mode,cfg);training.append(r)
            bundle=ROOT/cohort/f'independent_h15_s{seed}.pt';sha=pack(paths['current_only'],paths['future_only'],bundle);ev,pred=evaluate(bundle,datasets[cohort]['val'])
            for eid,z in pred.items():np.savez_compressed(ROOT/cohort/(f'val_s{seed}_'+eid+'.npz'),**z)
            record=dict(cohort=cohort,seed=seed,bundle=str(bundle),bundle_sha256=sha,evaluation=ev,training_s=sum(r['training_s'] for r in training));results.append(record);ds.write_json(ROOT/cohort/f'evaluation_s{seed}.json',record)
            print('PAIR',cohort,seed,'F1',ev['classification']['current']['macro_f1'],ev['classification']['future']['macro_f1'],'future',ev['events']['future'],flush=True)
    aggregate={}
    for cohort in cfg['cohorts']:
        records=[r for r in results if r['cohort']==cohort];aggregate[cohort]={}
        for key in ['recall','false_per_minute','precision','f1']:
            values=[r['evaluation']['events']['future'][key] for r in records];aggregate[cohort][key]=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))
    gate=aggregate['refresh101']['recall']['mean']>aggregate['clean91_control']['recall']['mean'] and aggregate['refresh101']['false_per_minute']['mean']<=aggregate['clean91_control']['false_per_minute']['mean']
    ds.write_json(ROOT/'summary.json',dict(results=results,aggregate=aggregate,new_research_candidate_gate_passed=gate,stable_default_changed=False,independent_test_performed=False,new10_already_exposed_and_used_for_training=True,trained_classification_heads=12,encoder_calls=0,remote_push=False))
    print('COMPLETE',aggregate,'research gate',gate,flush=True)

if __name__=='__main__':run()
