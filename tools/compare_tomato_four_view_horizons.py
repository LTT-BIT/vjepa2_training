"""Matched G1 one-second supervision experiment; frozen visual cache is reused."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_metrics as metrics
from . import train_tomato_rgb_memory as training
from .tomato_frozen_protocol import verify
from .tomato_boundary_metrics import change_events, boundaries
from .evaluate_tomato_reviewed_boundaries import match_events
from .tomato_rgb_memory_model import RGBMemoryReader
from .tomato_runtime import initialize_runtime

COLLEAGUE = '90ac57ea9eb1de93821ebddeff5a88bb7b0edfc7'
METHOD = 'G1_local_current'


def version_checkpoint(path, cfg):
    """A shared feature cache does not imply identical prediction semantics."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    suffix = '-h10-'+ds.fingerprint(cfg)[:12]
    if not saved['model_version'].endswith(suffix):
        saved['model_version'] += suffix
        torch.save(saved, path)
    return ds.digest(path)


def relabel(rows, horizon):
    result = []
    for r in rows:
        e = r['episode']
        result.append({**r, 'target': ds.build_phase_targets(
            r['anchors'], e['annotation']['segments'], e['length'], e['fps'], horizon)})
    return result


def common_scores(rows, pred, horizon):
    current, future, cy, fy = [], [], [], []
    event_rows = []
    for r in rows:
        e = r['episode']; p = pred[e['episode_id']]
        # Both horizons use exactly the same observed timestamps and tail support.
        keep = r['anchors'] + round(1.5*e['fps']) < e['length']
        assert keep.any()
        current.append(p['current'][keep]); future.append(p['future'][keep])
        cy.append(r['target']['current_labels'][keep]); fy.append(r['target']['future_labels'][keep])
        events = change_events(p['future'][keep], r['times'][keep], .5, 2)
        truth = boundaries(dict(episode=e))
        for window in (1., 1.5):
            matched, false = match_events(events, truth, 'future', window)
            hits = [b for b in matched if b['detected']]
            event_rows.append(dict(episode_id=e['episode_id'], matching_window_s=window,
                hits=len(hits), total=len(truth), false_events=len(false),
                duration_s=float(r['times'][keep][-1]-r['times'][keep][0]+.5),
                boundaries=matched, unmatched=false))
    result = dict(horizon_s=horizon,
        current=metrics.scores(np.log(np.maximum(np.concatenate(current),1e-30)), np.concatenate(cy)),
        future=metrics.scores(np.log(np.maximum(np.concatenate(future),1e-30)), np.concatenate(fy)),
        per_episode_events=event_rows, windows={})
    for window in (1., 1.5):
        rows_w = [r for r in event_rows if r['matching_window_s'] == window]
        h = sum(r['hits'] for r in rows_w); n = sum(r['total'] for r in rows_w)
        f = sum(r['false_events'] for r in rows_w)
        result['windows'][str(window)] = dict(hits=h, total=n, recall=h/n,
            false_events=f, false_per_minute=f/(sum(r['duration_s'] for r in rows_w)/60),
            by_transition={typ:dict(hits=sum(b['detected'] for r in rows_w for b in r['boundaries'] if (b['previous'],b['label']) == pair),
                total=sum(1 for r in rows_w for b in r['boundaries'] if (b['previous'],b['label']) == pair))
                for typ,pair in [('pickup->harvest',(0,1)),('harvest->return',(1,2)),('return->adjust',(2,3)),('adjust->pickup',(3,0))]})
    return result


def verify_saved(cache, out):
    initialize_runtime(dict(device='cuda',seed=42))
    cfg=ds.read_json(out/'protocol.json');lock=verify(Path(cfg['frozen_lock']))
    manifest=ds.read_json(lock['manifest']);by={e['episode_id']:e for e in manifest['episodes']}
    index=ds.read_json(cache/'index.json');results=ds.read_json(out/'results.json');rows=[]
    for entry in index['episodes']:
        if entry['split']!='val':continue
        assert ds.digest(entry['path'])==entry['sha256']
        z=np.load(entry['path']);eid=entry['episode_id']
        rows.append(dict(episode=by[eid],anchors=z['anchors'],times=z['times'],
            visual=z['visual'],positions=z['positions'].reshape(-1,928),metadata=z['metadata']))
    rows=relabel(rows,1.);checks=[]
    for seed in cfg['seeds']:
        cp=out/f'seed{seed}'/f'{METHOD}.pt';saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert ds.digest(cp)==results[str(seed)]['training']['checkpoint_sha256']
        assert saved['protocol']['future_horizon_s']==1. and '-h10-' in saved['model_version']
        stats=np.load(index['statistics_path'])
        np.testing.assert_array_equal(saved['state_mean'].numpy(),stats['mean'])
        np.testing.assert_array_equal(saved['state_std'].numpy(),stats['std'])
        net=RGBMemoryReader(saved['state_mean'],saved['state_std'],cfg['hidden']).cuda()
        net.load_state_dict(saved['head']);pred=training.predict(net,rows,METHOD,cfg);errors=[]
        for r in rows:
            eid=r['episode']['episode_id'];expected=np.load(out/f'seed{seed}'/f'{METHOD}_{eid}.npz')
            for key in ['current','future']:
                np.testing.assert_allclose(pred[eid][key],expected[key],atol=5e-6,rtol=1e-4)
                errors.append(float(np.max(np.abs(pred[eid][key]-expected[key]))))
        checks.append(dict(seed=seed,checkpoint_sha256=ds.digest(cp),model_version=saved['model_version'],
            recomputed_windows=sum(len(r['times']) for r in rows),max_probability_error=max(errors)))
    ds.write_json(out/'verification.json',dict(checks=checks,historical_test_scored=False,default_changed=False))


def run(cache, reference, out):
    if out.exists(): raise FileExistsError('New output required; preserve checkpoints')
    original = ds.read_json(reference/'protocol.json')
    assert original['future_horizon_s'] == 1.5
    cfg = deepcopy(original); cfg.update(future_horizon_s=1., methods=[METHOD],
        version='old100-r11-G1-horizon1s-matched-v1')
    lock = verify(Path(cfg['frozen_lock'])); manifest = ds.read_json(lock['manifest'])
    rows, index, mean, std = training.load_rows(cache, original, manifest)
    train = relabel([r for r in rows if r['episode']['split']=='train'], 1.)
    dev15 = [r for r in rows if r['episode']['split']=='val']; dev10 = relabel(dev15,1.)
    assert len(train)==84 and len(dev10)==8
    out.mkdir(parents=True); ds.write_json(out/'protocol.json',cfg)
    colleague_raw = subprocess.check_output(['git','show',COLLEAGUE+':configs/tairan_deployment.json'])
    (out/'colleague_deployment.json').write_bytes(colleague_raw)
    source_files = [Path(__file__),Path(training.__file__),Path('tools/tomato_rgb_memory_model.py'),Path('tools/tomato_rgb_memory_training.py')]
    identity = dict(pretrained=lock['pretrained'], manifest_sha256=lock['manifest_sha256'],
        cache_signature=index['signature'], cache_index_sha256=ds.digest(cache/'index.json'),
        statistics_sha256=index['statistics_sha256'], joint_names=index['joint_names'],
        supervision_horizon_s=1., feature_cache_supervision_horizon_s=1.5,
        source_sha256={str(p):ds.digest(p) for p in source_files})
    for p in source_files:(out/p.name).write_bytes(p.read_bytes())
    previous = ds.read_json(reference/'results.json'); results = {}
    for seed in cfg['seeds']:
        folder = out/f'seed{seed}'; folder.mkdir()
        net, info = training.fit(train,METHOD,seed,cfg,mean,std,folder/f'{METHOD}.pt',identity)
        # Cache is deliberately shared across horizons; model identity must not be.
        checkpoint = folder/f'{METHOD}.pt'
        info['checkpoint_sha256'] = version_checkpoint(checkpoint, cfg)
        assert info['initialization_sha256']==previous[str(seed)][METHOD]['training']['initialization_sha256']
        pred = training.predict(net,dev10,METHOD,cfg)
        baseline = {}
        for r in dev10:
            eid = r['episode']['episode_id']; p = reference/f'seed{seed}'/f'{METHOD}_{eid}.npz'
            z = np.load(p); assert np.array_equal(z['anchors'],r['anchors'])
            baseline[eid] = dict(current=z['current'],future=z['future'])
            np.savez_compressed(folder/f'{METHOD}_{eid}.npz',**pred[eid],times=r['times'],anchors=r['anchors'])
        results[str(seed)] = dict(training=info,
            h10=dict(evaluation=training.evaluate(dev10,pred), common=common_scores(dev10,pred,1.)),
            h15=dict(evaluation=previous[str(seed)][METHOD]['evaluation'],common=common_scores(dev15,baseline,1.5)))
        ds.write_json(folder/'result.json',results[str(seed)])
        print('DONE',seed, {k:dict(future=v['common']['future']['macro_f1'],windows=v['common']['windows'],prompt=v['evaluation']['summary']['prompt_counts']) for k,v in results[str(seed)].items() if k!='training'},flush=True)
        del net; torch.cuda.empty_cache()
    ds.write_json(out/'results.json',results)
    ds.write_json(out/'receipt.json',dict(identity=identity,colleague_commit=COLLEAGUE,
        colleague_config_sha256=ds.digest(out/'colleague_deployment.json'),
        reference_results_sha256=ds.digest(reference/'results.json'),
        reference= str(reference.resolve()),cache=str(cache.resolve()),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        trained_models=3,encoder_unfrozen=False,historical_test_evaluated=False,
        default_changed=False,threshold_search=False,
        scope='Repeatedly exposed development8; not independent generalization or colleague model head-to-head',
        caveat='Shared readout jointly optimized: changing future supervision can also change Current outputs. No deployment change.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if a.verify_only:verify_saved(a.cache,a.out)
    else:run(a.cache,a.reference,a.out);verify_saved(a.cache,a.out)
