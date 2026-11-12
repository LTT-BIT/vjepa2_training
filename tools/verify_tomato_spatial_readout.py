"""Verify training-only selection, causal cached indices and paired spatial controls."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify


def run(root):
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));m=ds.read_json(lock['manifest']);by={e['episode_id']:e for e in m['episodes']};sel=ds.read_json(root/'selection.json');ids=sel['episode_ids'];folds=ds.read_json(root/'folds.json');assert len(ids)==12 and len(set(sum(folds,[])))==12
    cache={};maxerror=0.
    for eid in ids:
        e=by[eid];assert e['split']=='train';p=root/f'{eid}_tokens.npz';z=np.load(p);receipt=ds.read_json(root/f'{eid}_receipt.json');assert ds.digest(p)==receipt['cache_sha256'];assert np.all(z['input_indices']<=z['anchors'][:,None]);assert z['tokens'].shape[1:]==(3,16,1024);cache[eid]=dict(tokens=z['tokens'].astype('float32'),positions=z['positions']);maxerror=max(maxerror,receipt['original_mean_vs_regions_max_error'])
    assert maxerror<1e-5;checks=[]
    for k,held in enumerate(folds):
        fit=[eid for eid in ids if by[eid]['source_recording_id'] not in held];v=np.concatenate([cache[e]['tokens'] for e in fit]);q=np.concatenate([cache[e]['positions'] for e in fit]);vm=v.mean(2).mean(0);vs=v.mean(2).std(0).clip(.1);qm=q.mean((0,1));qs=q.std((0,1)).clip(.01)
        for seed in (42,43,44):
            stats=[]
            for method in ('global_mean_control','spatial_4x4'):
                p=root/f'seed{seed}'/f'{method}_fold{k}.pt';s=torch.load(p,weights_only=True,map_location='cpu');assert set(s['training_source_ids'])=={by[e]['source_recording_id'] for e in fit};assert not set(s['training_source_ids'])&set(held)
                for key,expected in [('visual_mean',vm),('visual_std',vs),('state_mean',qm),('state_std',qs)]:assert np.array_equal(s[key].numpy(),expected)
                checks.append(dict(seed=seed,method=method,fold=k,normalization_exact=True,recording_overlap=0,checkpoint_sha256=ds.digest(p)))
    ds.write_json(root/'verification.json',dict(models=len(checks),checks=checks,source_recordings=12,windows=288,all_training_only=True,source_mean_vs_spatial_mean_max_error=maxerror,frozen_B0_verified=True));print('VERIFIED',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
