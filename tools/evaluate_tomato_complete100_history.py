"""Frozen100 dataset lock, validation-only B0 replay and causal stage-history arms."""
from pathlib import Path
from collections import Counter
import argparse
import subprocess
import time
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_boundary_metrics as bm
from .select_tomato_complete_episodes import ACTIVE, ORDER
from .tomato_stage_history import HistoryReplay, oracle_prefix
from .evaluate_tomato_state_recovery import state_score, annotate_state, prompt_score, event_score, summarize

SOURCE=Path('runs/tomato/transition_time_20261109')


def distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),p10=float(np.percentile(a,10)),p50=float(np.median(a)),p95=float(np.percentile(a,95))) if len(a) else dict(n=0,p10=None,p50=None,p95=None)


def freeze_dataset():
    selection=ds.read_json('configs/tomato_complete100_20261111/selection.json')
    assert ds.digest(ACTIVE)==selection['manifest_sha256']
    m=ds.read_json(ACTIVE); identity=ds.read_json(ACTIVE.parent/'data_identity.json'); by={r['episode_id']:r for r in identity['episodes']}
    assert Counter(e['split'] for e in m['episodes'])==dict(train=84,val=8,test=8)
    assert len({e['source_recording_id'] for e in m['episodes']})==100
    assert len({e['episode_id'] for e in m['episodes']})==100
    assert ds.fingerprint(m['splits'])==m['split_sha256']
    assert all(set(ids)=={e['source_recording_id'] for e in m['episodes'] if e['split']==split} for split,ids in m['splits'].items())
    rows=[]
    for e in m['episodes']:
        assert [s['label'] for s in e['annotation']['segments']]==ORDER
        assert ds.digest(e['annotation']['path'])==e['annotation']['sha256']
        cache=by.get(e['episode_id'])
        if e['split']!='test':
            assert cache and cache['source_recording_id']==e['source_recording_id'] and cache['split']==e['split']
        rows.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],
                         dataset_root=e['dataset_root'],episode_index=e['episode_index'],frames=e['length'],fps=e['fps'],
                         annotation_path=e['annotation']['path'],annotation_sha256=e['annotation']['sha256'],
                         effective_segments_sha256=ds.fingerprint(e['annotation']['segments']),
                         feature_cache_sha256=cache['feature_sha256'] if cache else None,
                         cache_scope='registered train/val cache' if cache else 'historical test cache intentionally not loaded'))
    return m,dict(version='tomato-complete100-stage-history-v1',manifest=str(ACTIVE),manifest_sha256=ds.digest(ACTIVE),
                  data_identity_sha256=ds.digest(ACTIVE.parent/'data_identity.json'),split_sha256=m['split_sha256'],
                  membership_sha256=ds.fingerprint(rows),recordings=rows,split_counts=dict(Counter(e['split'] for e in m['episodes'])),
                  labels='Existing effective Subtask intervals, including historical provisional validation boundary edits; no new approval or revision',
                  historical_test_values_read=False)


def score_arm(e,z,mode):
    model=HistoryReplay(mode); trace=[]; warnings=[]; cpu=[]
    for i,t in enumerate(z['times']):
        history=oracle_prefix(e['annotation']['segments'],e['fps'],float(z['times'][i-1]) if i else None) if mode=='oracle' else None
        start=time.perf_counter_ns();r=model.update(z['current_probabilities'][i],z['future_probabilities'][i],float(t),oracle_history=history);cpu.append((time.perf_counter_ns()-start)/1e6)
        trace.append(r)
        if r['warning']:warnings.append(r['warning'])
    truth=bm.boundaries(dict(episode=e)); times=z['times'];ph=[r['phase'] for r in trace]
    gt=np.searchsorted([b['time'] for b in truth],times,side='right');valid=z['anchors']+45<e['length'];ts=times[valid]
    c=np.zeros((7,7),int);np.add.at(c,(gt,ph),1)
    result=dict(warnings=event_score([w for w in warnings if w['time']<=ts[-1]],truth,float(ts[-1]-ts[0]+.5)),
                prompt=prompt_score(trace,truth,times),state=annotate_state(state_score(ph,gt,times),truth,ph,times),
                position_accuracy=float(np.mean(gt==ph)),position_confusion=c.tolist(),history_corrections=model.corrections,
                raw_current_future_probabilities_unchanged=True,trace=trace)
    return result,cpu


def run(out):
    if out.exists(): raise FileExistsError('Unique output required')
    m,lock=freeze_dataset();out.mkdir(parents=True);ds.write_json(out/'dataset_lock.json',lock)
    episodes=[e for e in m['episodes'] if e['split']=='val'];results={};inputs={str(ACTIVE):ds.digest(ACTIVE)};timings=[];comparisons=[]
    for seed in (42,43,44):
        per={mode:{} for mode in ('none','predicted','oracle')};entries=[];logits={h:[] for h in ('current','future')};models=ds.read_json(SOURCE/f'seed{seed}/result.json')['identity']['original_classifiers']
        assert ds.digest(models['path'])==models['sha256'];inputs[models['path']]=models['sha256']
        for e in episodes:
            path=SOURCE/f'seed{seed}'/(e['episode_id']+'.npz');inputs[str(path)]=ds.digest(path);z=np.load(path)
            assert np.array_equal(z['anchors'],np.asarray(e['anchors'])) and np.allclose(z['times'],z['anchors']/e['fps'])
            for mode in per:
                scored,cpu=score_arm(e,z,mode);per[mode][e['episode_id']]=scored;timings+=cpu
            assert per['none'][e['episode_id']]['trace']==per['predicted'][e['episode_id']]['trace']
            targets=ds.build_phase_targets(z['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5)
            entries.append(dict(episode=e,anchors=z['anchors'],times=z['times'],**{k:v for k,v in targets.items() if k!='anchors'}))
            for h in logits:logits[h].append(np.log(np.maximum(z[h+'_probabilities'],1e-30)))
        heads={}
        for h,parts in logits.items():
            zz=np.concatenate(parts);valid=np.concatenate([r['future_valid_mask'] for r in entries]) if h=='future' else np.ones(len(zz),bool)
            y=np.concatenate([r[h+'_labels'] for r in entries]);dist=bm.distance_report(entries,zz,h)
            near={}
            for radius in (.25,.5,1.):
                buffers=[];labels=[];offset=0
                for r in entries:
                    n=len(r['times']);bound=np.array([b['time'] for b in bm.boundaries(r)]);mask=np.min(abs(r['times'][:,None]-bound[None,:]),axis=1)<=radius
                    if h=='future':mask&=r['future_valid_mask']
                    buffers.append(zz[offset:offset+n][mask]);labels.append(r[h+'_labels'][mask]);offset+=n
                a,b=np.concatenate(buffers),np.concatenate(labels);near[str(radius)]=metrics.scores(a,b) if len(b) else None
            ev=bm.event_report(entries,zz,h,1.5,.5,2)
            ev['precision']=ev['detected']/max(1,ev['detected']+ev['unmatched_events']);ev['f1']=2*ev['detected']/max(1,2*ev['detected']+ev['unmatched_events']+ev['boundary_count']-ev['detected'])
            heads[h]=dict(classification=metrics.scores(zz[valid],y[valid]),events=ev,distance=dist,near_boundaries_s=near)
        summaries={mode:summarize(v) for mode,v in per.items()}
        for mode,rows in per.items():
            s=summaries[mode];matrix=np.sum([v['position_confusion'] for v in rows.values()],axis=0)
            s['seven_position_accuracy']=float(matrix.trace()/matrix.sum());s['seven_position_confusion']=matrix.tolist();s['history_corrections']=sum(v['history_corrections'] for v in rows.values())
            ev=s['warnings'];ev['f1']=2*ev['hits']/max(1,ev['hits']+ev['total']+ev['false_events']);ev['missed']=ev['total']-ev['hits']
            ev['lead_s']=distribution([-b['offset_s'] for v in rows.values() for b in v['warnings']['boundaries'] if b['detected']])
            s['prompt_offset_s']=distribution([r['offset_s'] for v in rows.values() for r in v['prompt']['events'] if r['offset_s'] is not None])
        results[str(seed)]=dict(model=models,heads=heads,summary=summaries)
        for e in episodes:
            eid=e['episode_id']
            for b,p,o in zip(per['none'][eid]['prompt']['events'],per['predicted'][eid]['prompt']['events'],per['oracle'][eid]['prompt']['events']):
                comparisons.append(dict(seed=seed,episode_id=eid,source_recording_id=e['source_recording_id'],ordinal=b['ordinal'],
                                        boundary_s=b['boundary_s'],transition=b['transition'],B0=b,predicted_history=p,oracle_past_history=o))
        ds.write_json(out/f'seed{seed}_per_episode.json',per)
    ds.write_json(out/'results.json',results)
    aggregate={}
    for mode in ('none','predicted','oracle'):
        aggregate[mode]={k:dict(mean=float(np.mean(v)),std=float(np.std(v)),values=v) for k,v in {
            'seven_position_accuracy':[r['summary'][mode]['seven_position_accuracy'] for r in results.values()],
            'mismatch_s':[r['summary'][mode]['mismatch_s'] for r in results.values()],
            'on_time_prompt':[r['summary'][mode]['prompt_counts'].get('on_time',0) for r in results.values()],
            'warning_hits':[r['summary'][mode]['warnings']['hits'] for r in results.values()]}.items()}
    ds.write_json(out/'aggregate.json',aggregate)
    ds.write_json(out/'history_event_comparison.json',comparisons)
    oldroot=Path('runs/tomato/task_memory_20261110_final');old=ds.read_json(oldroot/'summary.json');oldcls=ds.read_json(oldroot/'classification.json')
    for name in ('summary.json','classification.json'):inputs[str(oldroot/name)]=ds.digest(oldroot/name)
    ds.write_json(out/'historical9_comparison.json',dict(old9={seed:dict(summary=old[seed]['normal']['M0'],classification=oldcls[seed]) for seed in results},
        new8={seed:dict(summary=r['summary']['none'],classification={h:v['classification'] for h,v in r['heads'].items()}) for seed,r in results.items()},
        reason='Only validation membership changes: excluded three-phase episode ca784332e44d979f99018a52 contributed2 correctly warned events; no model improvement/regression inferred from this comparison'))
    receipt=dict(code_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs,
                 label_version=lock['labels'],validation_recordings=8,total_transitions=48,training=False,encoder_calls=0,robot_calls=0,
                 predicted_arm_oracle_inputs=False,oracle_arm='Explicit unavailable-online diagnostic; truth only through previous observed timestamp; no current/future truth or segment ends supplied',
                 classification_heads_frozen=True,physical_outcome_information_added=False,extra_update_cpu_ms=distribution(timings),
                 metrics_scope='CPU cache replay, not full GPU latency; 3 existing checkpoint seeds, no new fit or threshold search')
    ds.write_json(out/'receipt.json',receipt)
    print('B0 new8',results['42']['summary']['none']);print('HISTORY',aggregate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
