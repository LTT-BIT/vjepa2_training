"""Verify all nine trained bundles and publish a compact auditable summary."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import tomato_dataset as ds
from .train_tomato_inhand119_agents import METHODS, ARCHITECTURE


def report(run, out):
    receipt=ds.read_json(run/'receipt.json')
    results=ds.read_json(run/'results.json')
    frozen=ds.read_json(run/'all_models_frozen.json')
    expected={f'{seed}/{method}' for seed in [42,43,44] for method in METHODS}
    if receipt['status']!='complete' or set(results)!=expected or set(frozen)!=expected:
        raise ValueError('All nine completed models required')
    branches=[]
    for key,row in results.items():
        cp=Path(row['checkpoint']);saved=torch.load(cp,map_location='cpu',weights_only=False)
        if (ds.digest(cp)!=row['sha256'] or row['sha256']!=frozen[key]
                or saved['architecture']!=ARCHITECTURE
                or saved['protocol']['task_profile']!='tomato-in-hand-five-phases-v3'
                or len(saved['training_sources'])!=95):
            raise ValueError('New119 checkpoint identity mismatch')
        for target in ['current','future']:
            path=cp.with_name(cp.stem+'_'+target+'.pt')
            branch=torch.load(path,map_location='cpu',weights_only=False)
            if (branch['architecture']!='inhand119_single_rgb_memory_v1'
                    or branch['training']['updates']<=0
                    or ds.digest(path)!=saved['training'][target]['checkpoint_sha256']):
                raise ValueError('Independent trained branch missing or changed')
            branches.append(dict(model=key,target=target,sha256=ds.digest(path),
                                 training_s=branch['training']['training_s']))
    comparisons={}
    for method in METHODS:
        scores={}
        for split in ['validation','test']:
            scores[split]={}
            for target in ['current','future']:
                values=[results[f'{seed}/{method}'][split][target]['macro_f1'] for seed in [42,43,44]]
                scores[split][target]=dict(macro_f1_mean=float(np.mean(values)),
                    macro_f1_std=float(np.std(values)),seeds=values)
            replay=[r for seed in [42,43,44] for r in results[f'{seed}/{method}'][split]['five_phase_replay'].values()]
            scores[split]['five_phase_replay']=dict(recording_seed_runs=len(replay),
                independent_recordings=12,ordered_complete=sum(r['ordered_complete'] for r in replay),
                missed_boundaries=sum(r['missed'] for r in replay),
                early_gt1s=sum(r['early_gt1s'] for r in replay),
                late_gt1p5s=sum(r['late_gt1p5s'] for r in replay))
        comparisons[method]=scores
    summary=dict(status='complete',dataset='new119',paired_models=9,trained_branches=18,
        split_counts=receipt['split_counts'],methods=comparisons,branches=branches,
        head_training_s_total=sum(r['training_s'] for r in branches),
        roi_uses_future_coordinates=True,online_roi_validated=False,
        control_eligible=False,robot_calls=0,
        checkpoints={k:dict(path=r['checkpoint'],sha256=r['sha256']) for k,r in results.items()},
        evidence_sha256={name:ds.digest(run/name) for name in ['protocol.json','manifest.json',
            'train_state_statistics.npz','encoder_receipt.json','all_models_frozen.json',
            'results.json','receipt.json']},
        interpretation='Fixed models; test measured after all models frozen. Offline ROI replay only; no automatic control acceptance.')
    ds.write_json(out,summary)
    print('VERIFIED9PAIRS18BRANCHES',out)
    for method,scores in comparisons.items():
        print(method,'test',scores['test']['current']['macro_f1_mean'],
              scores['test']['future']['macro_f1_mean'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.run,a.out)
