"""Verify fold isolation, train-only normalizers and saved readout parity."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_transition_evidence import ARMS,change_features,training_examples,EvidenceReadout
from .train_tomato_transition_evidence import BASE


def run(root):
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));receipt=ds.read_json(root/'receipt.json')
    for p,sha in receipt['input_sha256'].items():assert ds.digest(p)==sha,p
    for p,sha in receipt['source_sha256'].items():assert ds.digest(p)==sha,p
    m=ds.read_json(lock['manifest']);train=[e for e in m['episodes'] if e['split']=='train'];by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};folds=ds.read_json(root/'folds.json');assert len(set(sum(folds,[])))==84
    raw={e['episode_id']:np.load(by[e['episode_id']]['feature_path'])['features'] for e in train};checks=[]
    for seed in (42,43,44):
        feature={};examples={}
        for e in train:
            eid=e['episode_id'];z=np.load(BASE/f'seed{seed}'/f'{eid}_input.npz');feature[eid]=change_features(raw[eid],z['current'],z['future'],z['times'],m['contract']['joint_names']);examples[eid]=training_examples(z['times'],e['annotation']['segments'],e['fps'])
        for arm in ARMS:
            for fold in list(range(5))+['final84']:
                name=f'{arm}_fold{fold}.pt' if fold!='final84' else f'{arm}_final84.pt';saved=torch.load(root/f'seed{seed}'/name,weights_only=True,map_location='cpu');held=set(folds[fold]) if fold!='final84' else set();fit=[e for e in train if e['source_recording_id'] not in held]
                assert set(saved['training_sources'])=={e['source_recording_id'] for e in fit} and not set(saved['training_sources'])&held
                x=np.concatenate([feature[e['episode_id']][arm][examples[e['episode_id']]['index']] for e in fit]);mean=x.mean(0,dtype=np.float64).astype('float32');std=x.std(0,dtype=np.float64).clip(.01).astype('float32')
                assert np.array_equal(mean,saved['mean'].numpy()) and np.array_equal(std,saved['std'].numpy())
                e=next(e for e in train if e['source_recording_id'] in held) if held else fit[0];eid=e['episode_id'];net=EvidenceReadout(saved['input_dimension']);net.load_state_dict(saved['state_dict']);x=torch.from_numpy((feature[eid][arm]-mean)/std)
                with torch.no_grad():actual=net.all_phases(x).numpy()
                expected=np.load(root/f'seed{seed}'/f'{arm}_{eid}.npz')['six_phase_evidence'] if held else None
                error=float(np.max(np.abs(actual-expected))) if expected is not None else None
                assert error is None or error<2e-5
                checks.append(dict(seed=seed,arm=arm,fold=fold,training_groups=len(fit),outer_overlap=0,normalization_exact=True,saved_prediction_max_error=error))
    ds.write_json(root/'verification.json',dict(checks=checks,total_checkpoints=len(checks),historical_test_read=False,frozen_B0_verified=True,source_sha256={str(Path(__file__)):ds.digest(__file__)}));print('VERIFIED',len(checks),'checkpoints')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
