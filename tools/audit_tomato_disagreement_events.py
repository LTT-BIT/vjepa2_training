"""Post-replay attribution only: labels never feed a policy or alter thresholds."""
import argparse
from collections import Counter
from pathlib import Path
import json
import numpy as np
from . import tomato_dataset as ds
from .tomato_state_recovery import CLASSES


def run(root):
    manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');rows=[];groups={}
    for seed in (42,43,44):
        folder=root/f'seed{seed}';per=ds.read_json(folder/'per_episode.json')
        for e in manifest['episodes']:
            if e['split']=='test':continue
            eid=e['episode_id'];z=np.load(folder/f'{eid}_input.npz');t,c,f=z['times'],z['current'],z['future']
            traces={a:[json.loads(s) for s in (folder/f'{eid}_{a}.jsonl').read_text().splitlines()] for a in per}
            for b in per['B0'][eid]['prompt']['events']:
                target=CLASSES[b['ordinal']];ft=None;conf=None;cp=None;idx=None
                if b['arrival_s'] is not None:
                    idx=int(np.flatnonzero(t==b['arrival_s'])[0]);ft=int(f[idx].argmax());conf=float(f[idx,ft]);cp=float(c[idx,target])
                follows=CLASSES[b['ordinal']+1] if b['ordinal']<6 else None
                relation='missing' if ft is None else 'same_target' if ft==target else 'lookahead_after_target' if ft==follows else 'opposing_or_other'
                # Earliest raw Current K2 after truth boundary: scorer-only attribution.
                q=(c.argmax(1)==target)&(c[:,target]>=.8);pairs=q&np.r_[False,q[:-1]]
                allowed=np.flatnonzero(pairs&(t>=b['boundary_s'])&(t<b['boundary_s']+2))
                raw=float(t[allowed[0]]) if len(allowed) else None
                phase_before=int(traces['B0'][idx]['previous_phase']) if idx is not None else None
                warnings=per['B0'][eid]['warnings'];w=warnings['boundaries'][b['ordinal']-1]
                matching=[v for v in warnings['events'] if v['previous']==w['previous'] and v['label']==w['label']]
                # Nearby class-pair events classified after the fact; no timing labels enter model.
                nearby=[dict(time=v['time'],offset_s=v['time']-b['boundary_s'],timing='too_early' if v['time']<b['boundary_s']-1.5 else 'strict_hit' if v['time']<b['boundary_s'] else 'at_or_after_boundary') for v in matching if b['boundary_s']-5<=v['time']<=b['boundary_s']+3]
                future_truth_index=int(np.searchsorted([s['start_frame']/e['fps'] for s in e['annotation']['segments']],b['arrival_s']+1.5,side='right')-1) if b['arrival_s'] is not None and b['arrival_s']+1.5<e['length']/e['fps'] else None
                future_truth_class=ds.LABELS.index(e['annotation']['segments'][future_truth_index]['label']) if future_truth_index is not None else None
                row=dict(seed=seed,episode_id=eid,source_recording_id=e['source_recording_id'],split=e['split'],ordinal=b['ordinal'],transition=b['transition'],boundary_s=b['boundary_s'],
                    baseline=b,current_target_probability_at_commit=cp,future_class_at_commit=ds.LABELS[ft] if ft is not None else None,
                    future_confidence_at_commit=conf,future_relation_at_commit=relation,
                    future_target_truth_class_at_commit=ds.LABELS[future_truth_class] if future_truth_class is not None else None,
                    future_wrong_for_own_horizon=ft!=future_truth_class if future_truth_class is not None else None,
                    online_phase_before_commit=phase_before,raw_current_k2_in_on_time_window=raw,
                    future_strict_detected=w['detected'],nearby_future_pair_events=nearby,
                    candidate_prompts={a:per[a][eid]['prompt']['events'][b['ordinal']-1] for a in per if a!='B0'})
                rows.append(row)
    for split in ['train','val']:
        subset=[r for r in rows if r['split']==split];early=[r for r in subset if r['baseline']['outcome']=='early'];bad=[r for r in subset if r['baseline']['outcome'] in ('late','missing')]
        groups[split]=dict(events=len(subset),early=len(early),early_future_relations=dict(Counter(r['future_relation_at_commit'] for r in early)),
            early_same_target_future_ge08=sum(r['future_relation_at_commit']=='same_target' and r['future_confidence_at_commit']>=.8 for r in early),
            early_future_wrong_for_own_horizon=sum(r['future_wrong_for_own_horizon'] is True for r in early),
            early_future_correct_for_own_horizon=sum(r['future_wrong_for_own_horizon'] is False for r in early),
            late_or_missing=len(bad),late_missing_without_raw_k2_in_window=sum(r['raw_current_k2_in_on_time_window'] is None for r in bad),
            developer_note='Aggregate over3 seeds, repeated recordings; not independent event replicates or causal proof')
    ds.write_json(root/'event_attribution.json',dict(summary=groups,events=rows,role='Scorer-only diagnostic; no policy promotion'))
    print(groups)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
