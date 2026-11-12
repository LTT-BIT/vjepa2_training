"""Predeclared no-training disagreement ablation. Historical test is never read."""
import argparse
from collections import Counter
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as pm
from .tomato_independent_heads import load_checkpoint
from .tomato_frozen_protocol import verify
from .tomato_disagreement_policy import ARMS, DisagreementPolicy
from .tomato_state_recovery import B0State
from .tomato_boundary_metrics import boundaries
from .evaluate_tomato_state_recovery import event_score, state_score, prompt_score
from .tomato_full_chain import cascade_diagnostics

PROTOCOL=Path('configs/tomato_disagreement_20261112/protocol.json')
LOCK=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json')


def score(e,times,anchors,c,f,arm):
    model=DisagreementPolicy(arm);trace=[];latency=[]
    # No labels or episode clocks enter model.update.
    for t,cp,fp in zip(times,c,f):
        start=time.perf_counter_ns();trace.append(model.update(cp,fp,float(t)));latency.append((time.perf_counter_ns()-start)/1e6)
    if arm in ('B0','advisory'):
        base=B0State()
        for row,t,cp,fp in zip(trace,times,c,f):
            r=base.update(cp,fp,float(t))
            assert row['phase']==r['phase'] and row['warning']==r['warning'], 'B0 parity changed'
    truth=boundaries(dict(episode=e));gt=np.searchsorted([b['time'] for b in truth],times,side='right');ph=np.array([r['phase'] for r in trace]);state=state_score(ph,gt,times)
    valid=anchors+45<e['length'];vt=times[valid]
    w=event_score([r['warning'] for r in trace if r['warning'] and r['time']<=vt[-1]],truth,float(vt[-1]-vt[0]+.5))
    prompt=prompt_score(trace,truth,times)
    cascade=cascade_diagnostics(state['persistent'],ph,gt,truth)
    result=dict(warnings=w,prompt=prompt,state=state,position_accuracy=float(np.mean(ph==gt)),
        cascade_affected_boundaries=sum(len(r['potential_downstream_ordinals']) for r in cascade),
        ordered_complete=([0]+[r['phase'] for r in trace if r['changed']])==list(range(7)),
        all_prompts_on_time=all(x['outcome']=='on_time' for x in prompt['events']),
        decisions=dict(Counter(r['reason'] for r in trace)),coordination_events=model.events,
        latency_ms=dict(p50=float(np.median(latency)),p95=float(np.percentile(latency,95)),max=float(max(latency))))
    return result,trace


def summarize(per):
    ps=[p for p in per.values()];prompts=[r for p in ps for r in p['prompt']['events']];bounds=[r for p in ps for r in p['warnings']['boundaries']]
    offsets=[r['offset_s'] for r in prompts if r['offset_s'] is not None]
    return dict(recordings=len(ps),warnings=dict(hits=sum(b['detected'] for b in bounds),total=len(bounds),false_events=sum(len(p['warnings']['false_events']) for p in ps)),
        prompt_counts=dict(Counter(r['outcome'] for r in prompts)),prompt_by_transition={k:dict(Counter(r['outcome'] for r in prompts if r['transition']==k)) for k in sorted({r['transition'] for r in prompts})},
        mismatch_s=sum(p['state']['mismatch_s'] for p in ps),persistent_mismatch_s=sum(p['state']['persistent_mismatch_s'] for p in ps),
        cascade_affected_boundaries=sum(p['cascade_affected_boundaries'] for p in ps),ordered_complete=sum(p['ordered_complete'] for p in ps),
        all_prompts_on_time=sum(p['all_prompts_on_time'] for p in ps),
        prompt_abs_error_s=float(np.mean(np.abs(offsets))) if offsets else None,
        latency_p95_episode_max_ms=max(p['latency_ms']['p95'] for p in ps),
        defer_once=sum(p['decisions'].get('bounded_future_conflict_wait_one_fresh_current',0) for p in ps))


def run(out):
    if out.exists():raise FileExistsError('Preserve historical output')
    cfg=ds.read_json(PROTOCOL);lock=verify(LOCK);m=ds.read_json(lock['manifest']);out.mkdir(parents=True)
    ds.write_json(out/'protocol.json',cfg)
    by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']}
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');oldtrain={e['source_recording_id'] for e in old['episodes'] if e['split']=='train'}
    inputs={str(PROTOCOL):ds.digest(PROTOCOL),str(LOCK):ds.digest(LOCK),lock['manifest']:ds.digest(lock['manifest'])};results={};deltas=[];gpu_start=time.perf_counter()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    from .tomato_runtime import initialize_runtime
    initialize_runtime(dict(device=device,seed=42))
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    # Never open historical test weights/features/predictions or labels for scoring.
    for seed,model in lock['models'].items():
        folder=out/f'seed{seed}';folder.mkdir();per={a:{} for a in ARMS};cohorts={};classification={};statuses=Counter()
        saved,head=load_checkpoint(model['path'],device);inputs[model['path']]=ds.digest(model['path'])
        for e in m['episodes']:
            if e['split']=='test':continue
            eid=e['episode_id'];cohort='development8' if e['split']=='val' else 'training74_seen' if e['source_recording_id'] in oldtrain else 'training10_b0_unseen_previously_explored'
            cohorts[eid]=cohort
            if e['split']=='val':
                path=Path(f'runs/tomato/transition_time_20261109/seed{seed}/{eid}.npz');z=np.load(path);c,f=z['current_probabilities'],z['future_probabilities'];times,anchors=z['times'],z['anchors']
                receipt=ds.read_json('runs/tomato/complete100_history_20261111_final/receipt.json')
                assert ds.digest(path)==receipt['inputs'][str(path)]
            else:
                path=Path(by[eid]['feature_path']);assert ds.digest(path)==by[eid]['feature_sha256'];z=np.load(path);anchors,times=z['anchors'],z['times']
                zpred=pm.predict_heads(head,z['features'],saved['mean'],saved['std']);c,f=[metrics.probabilities(zpred[k+'_logits']) for k in ('current','future')]
            inputs[str(path)]=ds.digest(path)
            assert np.array_equal(anchors,e['anchors']) and np.allclose(times,anchors/e['fps'])
            # Persist shared immutable probability inputs once, replay all arms on these same values.
            np.savez_compressed(folder/f'{eid}_input.npz',times=times,anchors=anchors,current=c,future=f)
            for ci,fi in zip(c.argmax(1),f.argmax(1)):statuses[f'{ds.LABELS[ci]}->{ds.LABELS[fi]}']+=1
            targets=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],e['fps'],1.5)
            classification[eid]={k:metrics.scores(np.log(np.maximum(p[v],1e-30)),targets[k+'_labels'][v]) for k,p,v in [('current',c,np.ones(len(times),bool)),('future',f,targets['future_valid_mask'])]}
            for arm in ARMS:
                per[arm][eid],trace=score(e,times,anchors,c,f,arm)
                with (folder/f'{eid}_{arm}.jsonl').open('x') as fh:
                    for row,cp,fp in zip(trace,c,f):fh.write(json.dumps(dict(**row,current_probabilities=cp.tolist(),future_probabilities=fp.tolist()),allow_nan=False)+'\n')
            assert per['B0'][eid]['prompt']==per['advisory'][eid]['prompt']
            assert all(per['B0'][eid]['warnings']==per[a][eid]['warnings'] for a in ARMS)
            for arm in ARMS[1:]:
                for b,p in zip(per['B0'][eid]['prompt']['events'],per[arm][eid]['prompt']['events']):
                    if b['arrival_s']!=p['arrival_s']:
                        deltas.append(dict(seed=int(seed),cohort=cohort,episode_id=eid,source_recording_id=e['source_recording_id'],arm=arm,ordinal=b['ordinal'],transition=b['transition'],baseline=b,candidate=p))
            print('REPLAY',seed,cohort,eid,flush=True)
        # Published seed42 anchor and all seed metrics prevent accidental baseline migration.
        summaries={co:{a:summarize({eid:p for eid,p in per[a].items() if cohorts[eid]==co}) for a in ARMS} for co in sorted(set(cohorts.values()))}
        if seed=='42':assert summaries['development8']['B0']['warnings']==dict(hits=37,total=48,false_events=13)
        results[seed]=dict(summary=summaries,current_future_argmax_pairs=dict(statuses))
        ds.write_json(folder/'per_episode.json',per);ds.write_json(folder/'classification_unchanged.json',classification)
        del head
        if device=='cuda':torch.cuda.empty_cache()
    ds.write_json(out/'summary.json',results);ds.write_json(out/'changed_prompt_events.json',deltas)
    ds.write_json(out/'receipt.json',dict(inputs=inputs,source_sha256={p:ds.digest(p) for p in ['tools/tomato_disagreement_policy.py','tools/evaluate_tomato_disagreement.py']},git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_hash_before_replay=inputs[str(PROTOCOL)],historical_test_scored=False,training=False,encoder_calls=0,real_latency_measured=False,
        device=device,wall_s=time.perf_counter()-gpu_start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        default_changed=False,labels_changed=False,threshold_search=False,interpretation='Exposed dev and previously explored B0-unseen training recordings; no independent generalization claim'))
    print('DONE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);run(parser.parse_args().out)
