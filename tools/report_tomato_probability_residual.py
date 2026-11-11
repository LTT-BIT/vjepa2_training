"""Paired recording-level audit and checkpoint replay for fixed residual runs."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_probability_residual import ResidualCalibration
from .train_tomato_probability_residual import history_features
from .tomato_state_recovery import StableLabel


def report(root,out):
    if out.exists():raise FileExistsError('Use a fresh report directory')
    out.mkdir(parents=True);summary=ds.read_json(root/'summary.json');rows=[];checks=[];by={}
    for name in ['summary.json','receipt.json','split.json','protocol.json']:(out/name).write_bytes((root/name).read_bytes())
    receipt=ds.read_json(root/'receipt.json');source=next(v for k,v in receipt['source_sha256'].items() if k.endswith('train_tomato_probability_residual.py'))
    assert ds.digest(root/'executed_runner.py')==source
    for seed in (42,43,44):
        folder=root/f'seed{seed}';events=ds.read_json(folder/'events.json')['development8']
        ds.write_json(out/f'seed{seed}_events.json',events)
        ck=torch.load(folder/'residual.pt',weights_only=True,map_location='cpu');s=ck['state_dict'];net=ResidualCalibration(s['mean'],s['std']);net.load_state_dict(s);net.eval()
        for eid,a in events['B0'].items():
            b=events['residual'][eid];z=np.load(folder/(eid+'.npz'));x=history_features(z['current'],z['original_future'],z['times'])
            with torch.inference_mode():p=net(torch.from_numpy(x),torch.from_numpy(np.log(np.maximum(z['original_future'],1e-6)).astype('float32')))[0].softmax(-1).numpy()
            error=float(np.max(abs(p-z['calibrated_future'])));assert error<2e-6
            assert np.array_equal(p.argmax(1),z['calibrated_future'].argmax(1))
            cutoff=b['duration_s']+z['times'][0]-.5;rule=StableLabel();ev=[]
            for prob,t in zip(p,z['times']):
                event=rule.update(prob,float(t))
                if event and t<=cutoff+1e-7:ev.append(event)
            assert ev==b['events']
            checks.append(dict(seed=seed,episode_id=eid,cpu_max_abs_error=error,argmax_and_events_identical=True))
            by.setdefault(eid,[]).append([sum(v['detected'] for v in a['boundaries']),sum(v['detected'] for v in b['boundaries']),len(a['false_events']),len(b['false_events']),len(a['boundaries']),a['duration_s']/60])
            for ordinal,(u,v) in enumerate(zip(a['boundaries'],b['boundaries']),1):
                status='retained' if u['detected'] and v['detected'] else 'lost' if u['detected'] else 'recovered' if v['detected'] else 'still_missed'
                same=[e for e in b['events'] if (e['previous'],e['label'])==(u['previous'],u['label'])]
                nearest=min(same,key=lambda e:abs(e['time']-u['time'])) if same else None
                rows.append(dict(seed=seed,episode_id=eid,ordinal=ordinal,transition=ds.LABELS[u['previous']]+'->'+ds.LABELS[u['label']],B0=u,residual=v,status=status,
                                 nearest_candidate_same_edge_offset_s=nearest['time']-u['time'] if nearest else None,nearest_scope='diagnostic only, not a match'))
    grouped=np.array([np.mean(v,axis=0) for v in by.values()]);rng=np.random.default_rng(20261111);deltas=[]
    for _ in range(4000):
        v=grouped[rng.integers(0,len(grouped),size=len(grouped))].sum(0)
        deltas.append([(v[1]-v[0])/v[4],(v[3]-v[2])/v[5]])
    quality=dict(checkpoint_replay=checks,changes={str(seed):dict(Counter(r['status'] for r in rows if r['seed']==seed)) for seed in (42,43,44)},
                 recording_paired_bootstrap_delta95=dict(recall=np.percentile(np.array(deltas)[:,0],[2.5,97.5]).tolist(),false_per_minute=np.percentile(np.array(deltas)[:,1],[2.5,97.5]).tolist(),
                                                        scope='4000 paired resamples of eight recordings; average seeds within recording; exploratory development uncertainty, not independent generalization'),
                 frozen_executed_source_hash_verified=True)
    ds.write_json(out/'event_comparisons.json',rows);ds.write_json(out/'quality.json',quality)
    print(root.name,quality['changes'],quality['recording_paired_bootstrap_delta95'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.run,a.out)
