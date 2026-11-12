"""Read-only attribution/recording bootstrap and versioned evidence bundles."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_transition_evidence import change_features,training_examples
from .train_tomato_transition_evidence import binary_scores,BASE
from .tomato_frozen_protocol import verify


def report(root):
    results=ds.read_json(root/'results.json');manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');by={e['episode_id']:e for e in manifest['episodes']};old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))
    cohorts={'oof10_base_unseen_exposed':[e for e in by.values() if e['split']=='train' and e['source_recording_id'] not in seen], 'development8_exposed':[e for e in by.values() if e['split']=='val']};bootstrap={};veto={};comparisons=[]
    for co,episodes in cohorts.items():
        buffers={};ids=[e['episode_id'] for e in episodes]
        for seed in (42,43,44):
            for e in episodes:
                eid=e['episode_id'];times=np.asarray(e['anchors'])/e['fps'];ex=training_examples(times,e['annotation']['segments'],e['fps']);starts=np.array([s['start_frame']/e['fps'] for s in e['annotation']['segments']]);mask=np.abs(times[ex['index']]-starts[ex['phase']+1])<=.5
                for arm in ('E0','E2'):
                    a=np.load(root/f'seed{seed}'/f'{arm}_{eid}.npz')['six_phase_evidence'];buffers[seed,eid,arm]=(ex['label'][mask],a[ex['index'][mask],ex['phase'][mask]],ex['weight'][mask])
        def delta(sample):
            vals=[]
            for seed in (42,43,44):
                scores=[]
                for arm in ('E0','E2'):
                    parts=[buffers[seed,eid,arm] for eid in sample];v=binary_scores(*[np.concatenate([p[k] for p in parts]) for k in range(3)])['auroc'];scores.append(v)
                vals.append(scores[1]-scores[0])
            return float(np.mean(vals))
        rng=np.random.default_rng(20261112);d=[delta([ids[i] for i in rng.integers(0,len(ids),len(ids))]) for _ in range(1000)]
        bootstrap[co]=dict(recordings=len(ids),mean_seed_paired_auc_delta=delta(ids),recording_bootstrap_95=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],replicates=1000,note='Exploratory record-level bootstrap; seeds averaged within each paired draw, not independent replicas')
        veto[co]={}
        for arm in ('E0','E1','E2'):
            ev=[dict(seed=seed,**e) for seed,s in results.items() for e in s[arm]['evaluation'][co]['commits']]
            rows={}
            for threshold in (.5,.8):
                rejected=[e for e in ev if e['evidence_probability']<threshold];early=[e for e in ev if not e['already_begun']];post=[e for e in ev if e['already_begun']]
                rows[str(threshold)]=dict(total_b0_commits=len(ev),early_total=len(early),early_rejected=sum(not e['already_begun'] for e in rejected),post_boundary_rejected=sum(e['already_begun'] for e in rejected),post_boundary_total=len(post),rejected=rejected)
            veto[co][arm]=rows
        # Event-level E0/E1/E2 comparisons at the same B0 commit.
        for seed in results:
            records={arm:{(r['episode_id'],r['ordinal']):r for r in results[seed][arm]['evaluation'][co]['commits']} for arm in ('E0','E1','E2')}
            for key,e in records['E0'].items():comparisons.append(dict(cohort=co,seed=int(seed),**{k:v for k,v in e.items() if k!='evidence_probability'},scores={arm:records[arm][key]['evidence_probability'] for arm in records}))
    ds.write_json(root/'recording_bootstrap.json',bootstrap);ds.write_json(root/'shadow_veto_diagnostic.json',veto);ds.write_json(root/'commit_comparison.json',comparisons)
    # Preserve training checkpoints; create separately named fully identified bundles.
    packaged={}
    for seed,arms in results.items():
        packaged[seed]={}
        for arm in arms:
            original=root/f'seed{seed}'/f'{arm}_final84.pt';destination=root/f'seed{seed}'/f'{arm}_identified_bundle.pt'
            if destination.exists():raise FileExistsError('Do not overwrite evidence bundle')
            payload=torch.load(original,weights_only=True,map_location='cpu');payload.update(original_checkpoint_sha256=ds.digest(original),manifest_sha256=lock['manifest_sha256'],dataset_split_sha256=manifest['split_sha256'],base_model=lock['models'][seed],feature_schema=dict(version='causal-observation-change-v1',joint_names=manifest['contract']['joint_names'],cameras=manifest['contract']['cameras'],visual_pooling='3x1024D frozen window mean features',joint_history='16x58 measured position; latest sampled position; named groups',lags_s=[.5,1.,1.5],feature_source_sha256=ds.digest('tools/tomato_transition_evidence.py'),arm=arm,input_dimension=payload['input_dimension']),scope='shadow_only_not_selected_for_deployment',true_phase_online=False)
            torch.save(payload,destination);packaged[seed][arm]=dict(path=str(destination),sha256=ds.digest(destination))
    ds.write_json(root/'bundles.json',packaged);print(bootstrap)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
