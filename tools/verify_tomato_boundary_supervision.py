"""Verify paired initialization, recording isolation and all train-only statistics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify


def run(root):
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));m=ds.read_json(lock['manifest']);train=[e for e in m['episodes'] if e['split']=='train'];by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};receipt=ds.read_json(root/'receipt.json')
    for p,sha in receipt['inputs'].items():assert ds.digest(p)==sha
    assert ds.digest('tools/train_tomato_boundary_supervision.py')==receipt['code_sha256']
    folds=ds.read_json(root/'folds.json');assert len(set(sum(folds,[])))==84;checks=[]
    raw={e['episode_id']:np.load(by[e['episode_id']]['feature_path'])['features'] for e in train}
    for k in list(range(5))+['final84']:
        held=set(folds[k]) if k!='final84' else set();fit=[e for e in train if e['source_recording_id'] not in held];x=np.concatenate([raw[e['episode_id']] for e in fit]);mean=x.mean(0,dtype=np.float64).astype('float32');std=x.std(0,dtype=np.float64).clip(.01).astype('float32')
        for seed in (42,43,44):
            init=[]
            for method in ['C0_standard','C1_boundary_weighted']:
                name=f'{method}_fold{k}.pt' if k!='final84' else f'{method}_final84.pt';p=root/f'seed{seed}'/name;s=torch.load(p,weights_only=True,map_location='cpu');assert set(s['training_source_ids'])=={e['source_recording_id'] for e in fit};assert not held&set(s['training_source_ids']);assert np.array_equal(s['mean'].numpy(),mean) and np.array_equal(s['std'].numpy(),std);init.append(s['initialization_sha256']);checks.append(dict(seed=seed,method=method,fold=k,train_only_statistics_exact=True,checkpoint_sha256=ds.digest(p)))
            assert init[0]==init[1]
    ds.write_json(root/'verification.json',dict(checks=checks,models=len(checks),frozen_B0_verified=True,paired_initialization_exact=True,recording_overlap=0));print('VERIFIED',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
