"""Validation-only optimization; immutable old campaign; reuse all frozen features."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_phase_model as model, tomato_metrics as metrics
from . import train_tomato_vjepa_phase as core, train_tomato_campaign as campaign
from . import tomato_independent_heads as dual, tomato_boundary_metrics as boundary
from .tomato_runtime import initialize_runtime

ROOT=Path('runs/tomato/optimization_20261108')
OLD=Path('runs/tomato/campaign94_20261107/experiments')
SEEDS=(42,43,44)
POINTS=[(p,k) for p in (.5,.65,.8,.9) for k in (1,2)]
RULE=dict(version=1,selection_split='val',seeds=list(SEEDS),representative_seed=42,
          candidate_families=['joint','joint_transition','independent_h05','independent_h10','independent_h15'],
          horizons_s=[.5,1.,1.5],thresholds=[.5,.65,.8,.9],confirmations=[1,2],
          feasibility=dict(current_macro_f1_drop_max=.005,current_boundary_recall_min=.9,
                           current_delay_p95_s_max=1.5,current_false_events_per_minute_max=1.,
                           future_boundary_recall_min=.8,future_false_events_per_minute_max=1.),
          ranking=['mean_future_recall descending','mean_future_lead_p50 descending',
                   'mean_current_recall descending','mean_future_macro_f1 descending','lexicographic name/operating point'],
          aggregation='Macro average of three seed metrics; no best-seed selection; event recall aggregates boundaries, class F1 aggregates windows; per-episode diagnostics also saved.',
          fallback='If no candidate passes, retain existing Joint seed42 as stable interface default; best-ranked validation research candidate may be packaged but is NOT approved as reliable transition detector.',
          comparison='Baseline Joint seed42 evaluated at both fixed .8/2 and selected operating points; old test never predicted this round.',
          rule_locked_before_new_validation_evaluation=True)


def write_locked(path,value):
    if path.exists():
        if ds.read_json(path)!=value: raise ValueError('Locked artifact changed: '+str(path))
    else: ds.write_json(path,value)


def freeze_or_verify():
    if ROOT.resolve().is_relative_to(OLD.parent.resolve()):
        raise ValueError('Optimization outputs must be outside immutable old campaign')
    current={str(p):ds.digest(p) for p in sorted(OLD.parent.rglob('*')) if p.is_file()}
    path=ROOT/'freeze.json'
    if path.exists():
        if ds.read_json(path)['prior_artifacts']!=current:
            raise ValueError('Previously frozen campaign changed')
    else:
        ds.write_json(path,dict(prior_artifacts=current,
            policy='Freeze old test/campaign before new validation-only selection. Test data may be read for integrity audits, never new test prediction.'))
    return dict(passed=True,frozen_files=len(current),freeze_sha256=ds.digest(path))


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);freeze_or_verify();write_locked(ROOT/'selection_rule.json',RULE)
    cfg=ds.load_config('configs/tomato_campaign94.yaml')
    parent=core.load_locked_manifest(cfg)
    # Copy metadata into a derivative directory before recomputing labels; never write old run.
    out=ROOT/'prepared_h10';out.mkdir(parents=True,exist_ok=True)
    original=Path(cfg['output_dir'])
    for name in ('manifest.json','split.json','validation_receipt.json','validation_lock.json','feature_index.json'):
        write_locked(out/name,ds.read_json(original/name))
    cfg={**cfg,'output_dir':str(out.resolve())}
    entries=core.load_training_arrays(cfg,parent)
    arrays=campaign.concatenate(entries)
    return cfg,parent,entries,arrays


def audit(cfg,parent,entries,arrays):
    groups={s:set(parent['splits'][s]) for s in entries}
    assert not any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    smoke=ds.read_json('configs/tomato_campaign94/split_audit.json')['known_exploratory_sources']
    assert not groups['test']&set(smoke)
    x=arrays['train']['features'];mean=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64).clip(.01).astype(np.float32)
    checks={};windows={};sources={}
    for split,rows in entries.items():
        windows[split]=sum(len(r['times']) for r in rows)
        for r in rows:
            e=r['episode'];idx=r['input_indices']
            assert e['source_recording_id'] in groups[split]
            assert np.all(idx<=r['anchors'][:,None]) and np.all(idx>=0) and np.all(idx<e['length'])
            assert idx.shape==(len(r['anchors']),16)
            sources[e['episode_id']]=dict(split=split,source_recording_id=e['source_recording_id'],length=e['length'],
                                          class_counts=np.bincount(r['current_labels'],minlength=4).tolist())
    for name in ('current','future','joint','joint_transition'):
        for seed in SEEDS:
            path=OLD/f'{name}_s{seed}'/'best_head.pt';saved,_=model.load_checkpoint(path)
            assert np.array_equal(saved['mean'].numpy(),mean) and np.array_equal(saved['std'].numpy(),std)
            assert saved['split_sha256']==parent['split_sha256'] and saved['pretrained']==parent['pretrained']
            checks[f'{name}_s{seed}']=dict(train_statistics_exact_match=True,checkpoint_sha256=ds.digest(path),best_epoch=saved['best_epoch'])
    report=dict(passed=True,source_counts={s:len(v) for s,v in groups.items()},windows=windows,
                smoke_assignment={v:next(s for s,g in groups.items() if v in g) for v in smoke},
                source_overlap=False,history_uses_only_same_episode_past_frames=True,
                normalization='All 12 checkpoints bit-exact equal train-only feature mean/std; validation/test excluded',
                checks=checks,episodes=sources,split_sha256=parent['split_sha256'],
                limitations=['Original recording UUID is grouping authority; unknown duplicate exports or adjacent independently named recordings cannot be ruled out without acquisition logs/content deduplication.',
                             'Known pilot/smoke sources excluded from test; smoke-exposed validation is not a pristine independent sample.',
                             'Original test has already been inspected in preceding round; frozen for provenance and never used for this selection.'])
    ds.write_json(ROOT/'audit.json',report)
    print('AUDIT',report['source_counts'],report['windows'],flush=True)


def horizon_data(cfg,parent,entries,h):
    if h==1.: return cfg,parent,entries,campaign.concatenate(entries)
    tag='h05' if h==.5 else 'h15';out=ROOT/('prepared_'+tag);out.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(parent);p['prediction']=ds.prediction_contract({**cfg,'future_horizon_s':h})
    p['derived_horizon_from']=dict(manifest_sha256=ds.digest(Path(cfg['output_dir'])/'manifest.json'),input_features_unchanged=True)
    result={s:[] for s in entries}
    episode_map={}
    for s,rows in entries.items():
        for r in rows:
            e=copy.deepcopy(r['episode']);targets=ds.build_phase_targets(r['anchors'],e['annotation']['segments'],e['length'],e['fps'],h)
            e['future_offset_frames']=round(h*e['fps']);e['future_invalid_count']=int((~targets['future_valid_mask']).sum())
            e['future_class_counts']=np.bincount(targets['future_labels'][targets['future_valid_mask']],minlength=4).tolist()
            episode_map[e['episode_id']]=e
            result[s].append({**r,'episode':e,**targets,'prediction':p['prediction']})
    p['episodes']=[episode_map[e['episode_id']] for e in p['episodes']]
    for split, rows in result.items():
        mask=np.concatenate([r['future_valid_mask'] for r in rows])
        labels=np.concatenate([r['future_labels'] for r in rows])
        p['summary'][split].update(future_valid=int(mask.sum()),future_invalid=int((~mask).sum()),
                                   future_class_counts=np.bincount(labels[mask],minlength=4).tolist())
    write_locked(out/'manifest.json',p);ds.write_json(out/'split.json',p['splits']);core._create_validation_receipt(out,p)
    index=ds.read_json(Path(cfg['output_dir'])/'feature_index.json');index['manifest_sha256']=ds.digest(out/'manifest.json')
    ds.write_json(out/'feature_index.json',index)
    c={**cfg,'output_dir':str(out.resolve()),'future_horizon_s':h}
    return c,p,result,campaign.concatenate(result)


def summarized_event(v):
    return {k:x for k,x in v.items() if k not in ('per_episode','definition')}


def evaluate(name,seed,path,entries):
    saved,head=dual.load_checkpoint(path,device='cuda')
    val=entries['val'];x=np.concatenate([r['features'] for r in val]);outputs=model.predict_heads(head,x,saved['mean'],saved['std'])
    report=dict(name=name,seed=seed,checkpoint=str(path),checkpoint_sha256=ds.digest(path),horizon_s=saved['future_horizon_s'],heads={})
    for h in ('current','future'):
        z=outputs[h+'_logits']
        if z is None: continue
        y=np.concatenate([r[h+'_labels'] for r in val]);keep=np.ones(len(y),bool) if h=='current' else np.concatenate([r['future_valid_mask'] for r in val])
        er={f'{p}/{k}':boundary.event_report(val,z,h,saved['future_horizon_s'],p,k) for p,k in POINTS}
        report['heads'][h]=dict(overall=metrics.scores(z[keep],y[keep]),distance=boundary.distance_report(val,z,h),events=er)
    ds.write_json(ROOT/'evaluations'/f'{name}_s{seed}.json',report)
    np.savez_compressed(ROOT/'evaluations'/f'{name}_s{seed}_val_logits.npz',**{k:v for k,v in outputs.items() if v is not None})
    print('EVAL',name,seed,{h:round(v['overall']['macro_f1'],6) for h,v in report['heads'].items()},flush=True)
    return report


def choose(reports):
    baseline=[r for r in reports if r['name']=='joint']
    baseline_f1=np.mean([r['heads']['current']['overall']['macro_f1'] for r in baseline]);candidates=[]
    for name in RULE['candidate_families']:
        rows=[r for r in reports if r['name']==name]
        for cp,ck in POINTS:
            for fp,fk in POINTS:
                ce=[r['heads']['current']['events'][f'{cp}/{ck}'] for r in rows]
                fe=[r['heads']['future']['events'][f'{fp}/{fk}'] for r in rows]
                avg=lambda rs,k:float(np.mean([r[k] for r in rs])) if all(r[k] is not None for r in rs) else None
                v=dict(name=name,current_threshold=cp,current_confirmations=ck,future_threshold=fp,future_confirmations=fk,
                       current_f1=float(np.mean([r['heads']['current']['overall']['macro_f1'] for r in rows])),
                       future_f1=float(np.mean([r['heads']['future']['overall']['macro_f1'] for r in rows])),
                       current_recall=avg(ce,'recall'),current_delay_p95=avg(ce,'delay_p95_s'),current_false_per_min=avg(ce,'false_events_per_minute'),
                       future_recall=avg(fe,'recall'),future_lead_p50=avg(fe,'lead_p50_s'),future_false_per_min=avg(fe,'false_events_per_minute'))
                v['feasible']=bool(v['current_f1']>=baseline_f1-.005 and v['current_recall']>=.9 and v['current_delay_p95'] is not None and v['current_delay_p95']<=1.5 and v['current_false_per_min']<=1 and v['future_recall']>=.8 and v['future_false_per_min']<=1)
                candidates.append(v)
    rank=lambda v:(-v['future_recall'],-(v['future_lead_p50'] or 0.),-v['current_recall'],-v['future_f1'],v['name'],v['current_threshold'],v['current_confirmations'],v['future_threshold'],v['future_confirmations'])
    feasible=sorted([v for v in candidates if v['feasible']],key=rank)
    research=sorted(candidates,key=rank)[0];selected=feasible[0] if feasible else research
    r=next(r for r in reports if r['name']==selected['name'] and r['seed']==42)
    result=dict(rule_sha256=ds.digest(ROOT/'selection_rule.json'),reliable_transition_candidate_approved=bool(feasible),
                selected=selected,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
                stable_default_checkpoint=str(OLD/'joint_s42/best_head.pt') if not feasible else r['checkpoint'],
                candidate_operating_points=candidates,test_accessed=False)
    write_locked(ROOT/'selection.json',result)
    ds.write_json(ROOT/'comparison.json',[dict(name=r['name'],seed=r['seed'],horizon_s=r['horizon_s'],
        heads={h:dict(overall=v['overall'],fixed_operating_point=summarized_event(v['events']['0.8/2']),
                      selected_operating_point=summarized_event(v['events'][f"{selected[h+'_threshold']}/{selected[h+'_confirmations']}"])) for h,v in r['heads'].items()}) for r in reports])
    return result


def run():
    initialize_runtime(dict(device='cuda',seed=42));cfg,parent,entries,arrays=prepare();audit(cfg,parent,entries,arrays)
    reports=[];curves={}
    for name in ('current','future','joint','joint_transition'):
        for seed in SEEDS:
            folder=OLD/f'{name}_s{seed}'
            reports.append(evaluate(name,seed,folder/'best_head.pt',entries))
            hist=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
            curves[f'{name}_s{seed}']=dict(selected_epoch=ds.read_json(folder/'validation_result.json')['best_epoch'],
                epochs=[dict(epoch=r['epoch'],current_f1=r['val']['current']['macro_f1'] if r['val']['current'] else None,
                             future_f1=r['val']['future']['macro_f1'] if r['val']['future'] else None,score=r['selection_score']) for r in hist])
    ds.write_json(ROOT/'old_training_curves.json',curves)
    for h,tag in [(1.,'h10'),(.5,'h05'),(1.5,'h15')]:
        hc,hp,he,ha=horizon_data(cfg,parent,entries,h)
        for seed in SEEDS:
            future=OLD/f'future_s{seed}'/'best_head.pt'
            if h!=1.:
                directory=ROOT/'training'/f'future_{tag}_s{seed}'
                campaign.train_one(hc,hp,ha,dict(name='future_'+tag,prediction_mode='future_only',transition_weight=1.),seed,directory)
                future=directory/'best_head.pt'
                reports.append(evaluate('future_'+tag,seed,future,he))
            packed=ROOT/'bundles'/f'independent_{tag}_s{seed}.pt'
            if not packed.exists(): dual.pack(OLD/f'current_s{seed}'/'best_head.pt',future,packed)
            reports.append(evaluate('independent_'+tag,seed,packed,he))
    choice=choose(reports);ds.write_json(ROOT/'freeze_verification.json',freeze_or_verify());print('SELECT',json.dumps({k:v for k,v in choice.items() if k!='candidate_operating_points'}),flush=True)
    print('VALIDATION COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',default=str(ROOT),help='Use a fresh directory for a new experiment version')
    args=parser.parse_args();ROOT=Path(args.output_root)
    run()
