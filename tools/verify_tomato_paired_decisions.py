"""Confirm seven-position wrapper preserves pinned colleague first6 commits."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .compare_tomato_paired_decisions import colleague_module,BASE


def run(root):
    module,_=colleague_module();m=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');checks=[]
    for seed in (42,43,44):
        for e in m['episodes']:
            if e['split']!='val':continue
            eid=e['episode_id'];z=np.load(BASE/f'seed{seed}'/f'{eid}_input.npz');ts=z['times'];duration=min(e['length']/e['fps'],float(ts[-1]+.5));rows=[dict(observed_at=float(t),heads={name:dict(probabilities=dict(zip(module.LABELS,p))) for name,p in [('current',cp),('future',fp)]}) for cp,fp,t in zip(z['current'],z['future'],ts)]
            for name,k,ack in [('H1_future_k1_no_ack',1,False),('H3_future_k1_ack',1,True),('H2_future_k2_no_ack',2,False),('H4_future_k2_ack',2,True)]:
                original=module.replay(rows,{'K':k,'ack_enabled':ack},duration);commits=[e for e in original['events'] if e['kind']=='commit'][:6];wrapped=ds.read_json(root/f'seed{seed}'/f'{eid}_{name}.json')['changes'];assert [e['time'] for e in commits]==[e['time'] for e in wrapped]
                checks.append(dict(seed=seed,episode_id=eid,arm=name,first6_commit_times_exact=True))
    receipt=ds.read_json(root/'receipt.json')
    for p,sha in receipt['inputs'].items():assert ds.digest(p)==sha,p
    assert ds.digest('tools/compare_tomato_paired_decisions.py')==receipt['runner_sha256']
    ds.write_json(root/'verification.json',dict(checks=checks,cases=len(checks),source_and_inputs_verified=True,scope='Terminal wrapper parity, not truth-based correction'));print('VERIFIED',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
