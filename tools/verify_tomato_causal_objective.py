"""Independent post-run provenance and fold-statistics checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_causal_objective import CausalReadouts,contexts


def run(root):
    torch.set_num_threads(1)
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))
    manifest=ds.read_json(lock['manifest']);train=[e for e in manifest['episodes'] if e['split']=='train'];active=[e for e in manifest['episodes'] if e['split']!='test']
    by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};receipt=ds.read_json(root/'receipt.json');cfg=ds.read_json(root/'protocol.json')
    for p,sha in receipt['inputs'].items():assert ds.digest(p)==sha
    for p,sha in receipt['identity']['code_sha256'].items():assert ds.digest(p)==sha
    folds=ds.read_json(root/'folds.json');assert len(set(sum(folds,[])))==84 and len(sum(folds,[]))==84
    raw={e['episode_id']:np.load(by[e['episode_id']]['feature_path'])['features'] for e in train};checks=[];probability_sources={};results=ds.read_json(root/'results.json')
    for fold in list(range(5))+['final84']:
        held=set(folds[fold]) if fold!='final84' else set();fit=[e for e in train if e['source_recording_id'] not in held];x=np.concatenate([raw[e['episode_id']] for e in fit]);mean=x.mean(0,dtype=np.float64).astype('float32');std=x.std(0,dtype=np.float64).clip(.01).astype('float32')
        for seed in cfg['seeds']:
            inits=[]
            for method in cfg['methods']:
                name=f'{method}_fold{fold}.pt' if fold!='final84' else f'{method}_final84.pt';p=root/f'seed{seed}'/name;s=torch.load(p,weights_only=True,map_location='cpu')
                assert set(s['training_source_ids'])=={e['source_recording_id'] for e in fit};assert not held&set(s['training_source_ids'])
                assert np.array_equal(s['mean'].numpy(),mean) and np.array_equal(s['std'].numpy(),std)
                info=results[str(seed)][method]['final_training'] if fold=='final84' else results[str(seed)][method]['fold_training'][fold]
                assert ds.digest(p)==info['checkpoint_sha256'];inits.append(s['training']['initialization_sha256']);checks.append(dict(seed=seed,method=method,fold=fold,checkpoint_sha256=ds.digest(p)))
                # Saved-model reload equivalence on held-out observations, using CPU inference.
                e=next(e for e in active if (e['split']=='val' if fold=='final84' else e['source_recording_id'] in held))
                z=np.load(by[e['episode_id']]['feature_path']);net=CausalReadouts(z['features'].shape[1],cfg['hidden'],cfg['dropout']).eval();net.load_state_dict(s['head'])
                c=contexts(z['times'],cfg['context'],cfg['gap_reset_s'])[:5];features=np.concatenate([np.zeros((1,z['features'].shape[1]),np.float32),(z['features']-mean)/std])
                with torch.inference_mode():pred=net(torch.from_numpy(features[c+1]),torch.from_numpy(c>=0))[:,-1].softmax(-1).numpy()
                saved=np.load(root/f'seed{seed}'/f'{method}_{e["episode_id"]}.npz')['current'][:5];np.testing.assert_allclose(pred,saved,rtol=1e-4,atol=2e-6)
            assert inits[0]==inits[1]==inits[2]
    for e in active:
        eid=e['episode_id'];z=np.load(by[eid]['feature_path']);np.testing.assert_allclose(z['times'],z['anchors']/e['fps'],atol=1e-6)
        for seed in cfg['seeds']:
            path=Path('runs/tomato/disagreement_20261112_v2')/f'seed{seed}'/f'{eid}_input.npz';prob=np.load(path)
            np.testing.assert_array_equal(z['anchors'],prob['anchors']);np.testing.assert_array_equal(z['times'],prob['times']);probability_sources[str(path)]=ds.digest(path)
            for method in cfg['methods']:
                s=np.load(root/f'seed{seed}'/f'{method}_{eid}.npz');assert s['current'].shape==(len(z['times']),4) and np.isfinite(s['current']).all();np.testing.assert_allclose(s['current'].sum(1),1,atol=1e-6);np.testing.assert_array_equal(s['times'],z['times'])
    ds.write_json(root/'verification.json',dict(checks=checks,models=len(checks),frozen_B0_verified=True,paired_initialization_exact=True,recording_overlap=0,normalization_train_fold_exact=True,reload_predictions_match=True,cached_and_prediction_timestamps_equal=True,probability_source_sha256=probability_sources,historical_test_scored=False));print('VERIFIED',len(checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
