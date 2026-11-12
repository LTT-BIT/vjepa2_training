"""Fresh per-observation checkpoint inference; no fit, test-set tuning or robot IO."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_metrics as metrics
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .tomato_independent_heads import load_checkpoint
from .tomato_causal_current_session import CausalCurrentSession
from .tomato_disagreement_policy import DisagreementPolicy
from .evaluate_tomato_disagreement import score,summarize

SOURCE=Path('runs/tomato/causal_objective_20261112_v1')
BASE=Path('runs/tomato/disagreement_20261112_v2')


@torch.inference_mode()
def run(out):
    if out.exists():raise FileExistsError('Preserve earlier tests')
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));manifest=ds.read_json(lock['manifest']);identity=ds.read_json('runs/tomato/complete100_20261111/data_identity.json');by={e['episode_id']:e for e in identity['episodes']}
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);episodes=[e for e in manifest['episodes'] if e['split']=='val' or e['split']=='train' and e['source_recording_id'] not in seen];assert len(episodes)==18
    folds=ds.read_json(SOURCE/'folds.json');out.mkdir(parents=True);inputs={lock['manifest']:lock['manifest_sha256']};results={};started=time.perf_counter();latencies={};maximum_error=dict(b0_current=0.,future=0.,t1_current=0.)
    for seed in [42,43,44]:
        initialize_runtime(dict(device='cuda',seed=seed));bp=Path(lock['models'][str(seed)]['path']);inputs[str(bp)]=ds.digest(bp);saved,b0=load_checkpoint(bp,'cuda');mean=saved['mean'].cuda();std=saved['std'].cuda();folder=out/f'seed{seed}';folder.mkdir();per={m:{} for m in ['B0','T1']};targets={};predictions={};latencies[str(seed)]={m:[] for m in ['B0_two_heads','T1_current']}
        for e in episodes:
            eid=e['episode_id'];cohort='development8_exposed' if e['split']=='val' else 'oof10_base_unseen_exposed';feature=Path(by[eid]['feature_path']);assert ds.digest(feature)==by[eid]['feature_sha256'];inputs[str(feature)]=ds.digest(feature);z=np.load(feature);assert np.array_equal(z['anchors'],e['anchors'])
            suffix='final84' if e['split']=='val' else 'fold'+str(next(k for k,held in enumerate(folds) if e['source_recording_id'] in held));path=SOURCE/f'seed{seed}'/f'T1_dense_{suffix}.pt';inputs[str(path)]=ds.digest(path);payload=torch.load(path,map_location='cpu',weights_only=True)
            if e['split']=='train':assert e['source_recording_id'] not in payload['training_source_ids']
            session=CausalCurrentSession.load(path,'cuda');policies={m:DisagreementPolicy('B0') for m in per};traces={m:[] for m in per};cp=[];fp=[];tp=[];refpath=BASE/f'seed{seed}'/f'{eid}_input.npz';ref=np.load(refpath);inputs[str(refpath)]=ds.digest(refpath);np.testing.assert_array_equal(z['times'],ref['times']);np.testing.assert_array_equal(z['anchors'],ref['anchors'])
            for x,t in zip(z['features'],z['times']):
                torch.cuda.synchronize();a=time.perf_counter();p=b0((torch.from_numpy(x[None]).cuda()-mean)/std);c=p['current_logits'].softmax(-1)[0].cpu().numpy();f=p['future_logits'].softmax(-1)[0].cpu().numpy();torch.cuda.synchronize();latencies[str(seed)]['B0_two_heads'].append((time.perf_counter()-a)*1000)
                torch.cuda.synchronize();a=time.perf_counter();r=session.update(x,float(t));torch.cuda.synchronize();latencies[str(seed)]['T1_current'].append((time.perf_counter()-a)*1000);assert r['valid'] and max(r['context_observation_timestamps'])<=t
                tc=r['current_probabilities'];cp.append(c);fp.append(f);tp.append(tc)
                for method,current in [('B0',c),('T1',tc)]:traces[method].append(dict(**policies[method].update(current,f,float(t)),current_probabilities=current.tolist(),future_probabilities=f.tolist(),observation_timestamp=float(t)))
            cp,fp,tp=map(np.stack,[cp,fp,tp]);expected_path=SOURCE/f'seed{seed}'/f'T1_dense_{eid}.npz';expected=np.load(expected_path)['current'];inputs[str(expected_path)]=ds.digest(expected_path)
            for key,a,b in [('b0_current',cp,ref['current']),('future',fp,ref['future']),('t1_current',tp,expected)]:np.testing.assert_allclose(a,b,rtol=1e-4,atol=3e-6);maximum_error[key]=max(maximum_error[key],float(abs(a-b).max()))
            for method,current in [('B0',cp),('T1',tp)]:
                s,trace=score(e,z['times'],z['anchors'],current,fp,'B0');assert [(r['phase'],r['changed']) for r in trace]==[(r['phase'],r['changed']) for r in traces[method]];s['cohort']=cohort;per[method][eid]=s
                with (folder/f'{eid}_{method}.jsonl').open('x') as fh:
                    for row in traces[method]:fh.write(json.dumps(row,allow_nan=False)+'\n')
            targets[eid]=ds.build_phase_targets(z['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5);predictions[eid]=dict(B0=cp,T1=tp,future=fp);np.savez_compressed(folder/f'{eid}.npz',times=z['times'],anchors=z['anchors'],b0_current=cp,t1_current=tp,future=fp);print('STREAM',seed,cohort,eid,flush=True)
        summaries={}
        for co in ['development8_exposed','oof10_base_unseen_exposed']:
            summaries[co]={}
            for method in per:
                selected={eid:p for eid,p in per[method].items() if p['cohort']==co};s=summarize(selected);ev=[e for p in selected.values() for e in p['prompt']['events']];s['within_delay']={str(t):sum(x['offset_s'] is not None and 0<=x['offset_s']<=t+1e-9 for x in ev) for t in [1.,1.5,2.]};s['severe_early_events']=[dict(episode_id=eid,**event) for eid,p in selected.items() for event in p['prompt']['events'] if event['offset_s'] is not None and event['offset_s'] < -1.]
                y=np.concatenate([targets[eid]['current_labels'] for eid in selected]);p=np.concatenate([predictions[eid][method] for eid in selected]);s['current']=metrics.scores(np.log(np.maximum(p,1e-30)),y)
                y=np.concatenate([targets[eid]['future_labels'][targets[eid]['future_valid_mask']] for eid in selected]);p=np.concatenate([predictions[eid]['future'][targets[eid]['future_valid_mask']] for eid in selected]);s['future']=metrics.scores(np.log(np.maximum(p,1e-30)),y);s['warnings']['false_per_minute']=s['warnings']['false_events']/(sum(p['warnings']['duration_s'] for p in selected.values())/60);summaries[co][method]=s
        results[str(seed)]=dict(summary=summaries,per_episode=per);ds.write_json(folder/'results.json',results[str(seed)]);del b0,session;torch.cuda.empty_cache()
    bench={seed:{method:dict(p50=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(max(v)),samples=len(v)) for method,v in rows.items()} for seed,rows in latencies.items()}
    ds.write_json(out/'results.json',results);ds.write_json(out/'receipt.json',dict(inputs=inputs,source_sha256={p:ds.digest(p) for p in [__file__,'tools/tomato_causal_current_session.py']},git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_s=time.perf_counter()-started,recordings=18,seeds=[42,43,44],fresh_checkpoint_inference=True,encoder_calls=0,training=False,new_independent_data=False,historical_test_scored=False,default_changed=False,probability_max_absolute_error=maximum_error,head_only_latency_ms=bench,latency_scope='Measured host feature to probabilities; no raw video encoder, network or robot execution'))
    verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));print('DONE',maximum_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);run(p.parse_args().out)
