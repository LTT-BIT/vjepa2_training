"""Single predeclared causal Future EMA ablation. No training/threshold search."""
import argparse
import subprocess
import time
from collections import Counter
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds, tomato_boundary_metrics as bm
from .evaluate_tomato_complete100_history import freeze_dataset, SOURCE
from .evaluate_tomato_reviewed_boundaries import match_events
from .tomato_causal_probability_filter import CausalProbabilityFilter
from .tomato_state_recovery import B0State

CONFIG=Path('configs/tomato_causal_filter_20261111/protocol.json')


def aggregate(per):
    bounds=[b for r in per.values() for b in r['boundaries']]
    hit=sum(b['detected'] for b in bounds)
    false=sum(len(r['false_events']) for r in per.values())
    lead=[-b['offset_s'] for b in bounds if b['detected']]
    minutes=sum(r['duration_s'] for r in per.values())/60
    return dict(hits=hit,total=len(bounds),misses=len(bounds)-hit,recall=hit/len(bounds),
                precision=hit/max(1,hit+false),f1=2*hit/max(1,len(bounds)+hit+false),
                false_events=false,false_per_minute=false/minutes,
                lead_s={str(q):float(np.percentile(lead,q)) if lead else None for q in (10,50,95)},
                by_transition={k:dict(hits=sum(b['detected'] for b in bounds if (b['previous'],b['label'])==v),total=sum((b['previous'],b['label'])==v for b in bounds))
                               for k,v in [('pickup->harvest',(0,1)),('harvest->return',(1,2)),('return->adjust',(2,3)),('adjust->pickup',(3,0))]})


def run(out):
    if out.exists():raise FileExistsError('Use a fresh output directory')
    cfg=ds.read_json(CONFIG);manifest,lock=freeze_dataset()
    prior=ds.read_json('runs/tomato/complete100_history_20261111_final/receipt.json')
    out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg)
    results={};inputs={str(CONFIG):ds.digest(CONFIG)};cost=[];comparisons=[]
    for seed in cfg['seeds']:
        base=ds.read_json(f'runs/tomato/complete100_history_20261111_final/seed{seed}_per_episode.json')['none']
        per={name:{} for name in ('B0','causal_ema')}
        for e in manifest['episodes']:
            if e['split']!='val':continue
            eid=e['episode_id'];path=SOURCE/f'seed{seed}'/(eid+'.npz')
            assert ds.digest(path)==prior['inputs'][str(path)]
            inputs[str(path)]=ds.digest(path)
            z=np.load(path);ts=z['times'];anchors=z['anchors']
            assert np.allclose(ts,anchors/e['fps']) and e['fps']==30
            # Labels remain future artificial annotations, never future observations.
            targets=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],e['fps'],1.5)
            valid=anchors+45<e['length']
            assert np.array_equal(targets['future_valid_mask'],valid)
            assert all(targets['future_labels'][i]==ds.label_at_frame(e['annotation']['segments'],int(anchors[i])+45) for i in np.flatnonzero(valid))
            cutoff=float(ts[valid][-1]);truth=bm.boundaries(dict(episode=e))
            raw=B0State();smooth=B0State();filt=CausalProbabilityFilter(cfg['half_life_s'],cfg['max_gap_s'])
            events={k:[] for k in per};phases=[]
            for i,t in enumerate(ts):
                start=time.perf_counter_ns()
                p,reset=filt.update(z['future_probabilities'][i],float(t))
                cost.append((time.perf_counter_ns()-start)/1e6)
                if reset:smooth.warnings.break_streak()
                a=raw.update(z['current_probabilities'][i],z['future_probabilities'][i],float(t))
                b=smooth.update(z['current_probabilities'][i],p,float(t))
                assert a['phase']==b['phase'] and a['changed']==b['changed']
                phases.append(a['phase'])
                for name,row in [('B0',a),('causal_ema',b)]:
                    if row['warning'] and t<=cutoff:events[name].append(row['warning'])
            assert phases==[r['phase'] for r in base[eid]['trace']]
            assert events['B0']==base[eid]['warnings']['events']
            for name in per:
                matched,false=match_events(events[name],truth,'future',1.5)
                per[name][eid]=dict(boundaries=matched,false_events=false,events=events[name],duration_s=float(ts[valid][-1]-ts[valid][0]+.5))
            for ordinal,(a,b) in enumerate(zip(per['B0'][eid]['boundaries'],per['causal_ema'][eid]['boundaries']),1):
                comparisons.append(dict(seed=seed,episode_id=eid,source_recording_id=e['source_recording_id'],ordinal=ordinal,
                                        transition=ds.LABELS[a['previous']]+'->'+ds.LABELS[a['label']],boundary_s=a['time'],
                                        B0=a,candidate=b,status='retained' if a['detected'] and b['detected'] else 'recovered' if b['detected'] else 'lost' if a['detected'] else 'still_missed'))
        summaries={k:aggregate(v) for k,v in per.items()}
        a,b=summaries['B0'],summaries['causal_ema']
        changes=Counter(r['status'] for r in comparisons if r['seed']==seed)
        results[str(seed)]=dict(summary=summaries,changes=dict(changes),
                               passes_paired_screen=b['hits']>a['hits'] and b['false_events']<=a['false_events'],
                               prompt_and_position_traces_identical=True)
        ds.write_json(out/f'seed{seed}_events.json',per)
    summary=dict(seeds=results,all_seeds_pass=all(r['passes_paired_screen'] for r in results.values()),
                 filter_cpu_ms={str(q):float(np.percentile(cost,q)) for q in (50,95,100)},
                 latency_scope='Filter only, CPU replay. No encoder/GPU inference measured.',
                 new_model_trained=False,production_changed=False,labels_changed=False,test_arrays_read=False,
                 horizon_audit='1.5 seconds =45 original frames at30Hz; labels match Subtask(t+45); input observations unchanged.',
                 evidence_scope='Exposed development set exploratory ablation. Not independent generalization evidence.')
    receipt=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dataset_lock=lock,
                 input_sha256=inputs,source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_causal_probability_filter.py')]})
    ds.write_json(out/'summary.json',summary);ds.write_json(out/'event_comparisons.json',comparisons);ds.write_json(out/'receipt.json',receipt)
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
