"""Handoff verifier tests; synthetic fixtures never stand in for real model accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import tomato_dataset as ds, tomato_phase_model as model
from tools import train_tomato_vjepa_phase as training
from tools import verify_tairan_deployment as handoff
from tools.tomato_review_ui.handshake import DEFAULT
from tomato_helpers import make_config


@pytest.fixture
def materials(tmp_path, monkeypatch):
    cfg = {**make_config(tmp_path, length=180), 'prediction_mode': 'joint',
           'normalization_fit_split': 'train'}
    manifest = training.prepare(cfg)
    run = Path(cfg['output_dir'])
    ds.write_json(run / 'training_config.json', cfg)
    dimension = 3072 + 16 * len(cfg['joint_names'])
    model.save_checkpoint(run / 'best_head.pt', model.joint_head_model(len(cfg['joint_names'])),
                          np.zeros(dimension, np.float32), np.ones(dimension, np.float32),
                          manifest, ds.digest(run / 'manifest.json'), 1, {})
    encoder = tmp_path / 'encoder'; encoder.mkdir()
    for name in handoff.ENCODER_FILES:
        (encoder / name).write_bytes(b'synthetic pinned-encoder fixture')
    monkeypatch.setattr(model, 'verify_pretrained', lambda directory: manifest['pretrained'])
    (tmp_path / 'configs').mkdir()
    policy = tmp_path / 'configs/tairan_handshake.json'
    ds.write_json(policy, {**DEFAULT, 'late_ack_enabled': True})
    source = tmp_path / 'source.py'; source.write_text('# bound source\n')
    def entry(path):
        return dict(path=str(path), sha256=ds.digest(path), bytes=path.stat().st_size)
    inventory = dict(schema_version=1, input_contract=manifest['contract'],
                     model_contract=dict(prediction_mode='joint',future_horizon_s=1.0,dimension=dimension,
                                         joint_dimension=len(cfg['joint_names']),labels=list(ds.LABELS)),
                     artifacts=dict(run={n:entry(run/n) for n in handoff.RUN_FILES},
                                    encoder={n:entry(encoder/n) for n in handoff.ENCODER_FILES}),
                     code=dict(base_commit='fixture', source_sha256={'source.py':ds.digest(source)}),
                     handshake=dict(path='configs/tairan_handshake.json', sha256=ds.digest(policy)))
    index = tmp_path / 'inventory.json'; ds.write_json(index, inventory)
    return tmp_path, run, encoder, index, inventory


def test_valid_bound_joint_normalization_and_relocated_directories(materials):
    repo, run, encoder, index, inventory = materials
    # Original training paths need not exist on a colleague's machine.
    for entries in inventory['artifacts'].values():
        for item in entries.values():item['path']='/unavailable/original-machine'
    ds.write_json(index, inventory)
    result = handoff.verify(index, repo=repo, run_dir=run, encoder_dir=encoder)
    assert result['verified_artifacts']==9 and result['model']['prediction_mode']=='joint'
    assert result['normalization']['embedded'] and result['handshake']['prediction_ttl_s']==.75


@pytest.mark.parametrize('kind', ['artifact', 'source', 'policy_hash', 'policy_incomplete',
                                   'normalization', 'manifest_binding', 'training_contract', 'inventory_contract'])
def test_corrupted_or_mismatched_material_is_rejected(materials, kind):
    repo, run, encoder, index, inventory = materials
    if kind=='artifact':(run/'best_head.pt').write_bytes(b'corrupted')
    elif kind=='source':(repo/'source.py').write_text('# changed\n')
    elif kind in ['policy_hash','policy_incomplete']:
        policy=repo/'configs/tairan_handshake.json'; value=ds.read_json(policy)
        value.pop('prediction_ttl_s'); ds.write_json(policy,value)
        if kind=='policy_incomplete':inventory['handshake']['sha256']=ds.digest(policy)
    elif kind in ['normalization','manifest_binding']:
        path=run/'best_head.pt'; saved=torch.load(path,weights_only=True)
        if kind=='normalization':saved['std'][0]=0
        else:saved['manifest_sha256']='0'*64
        torch.save(saved,path)
        inventory['artifacts']['run']['best_head.pt'].update(sha256=ds.digest(path),bytes=path.stat().st_size)
    elif kind=='inventory_contract':inventory['input_contract']={**inventory['input_contract'],'history_frames':8}
    else:
        path=run/'training_config.json';value=ds.read_json(path);value['future_horizon_s']=2.
        ds.write_json(path,value)
        inventory['artifacts']['run']['training_config.json'].update(sha256=ds.digest(path),bytes=path.stat().st_size)
    ds.write_json(index,inventory)
    with pytest.raises(ValueError):handoff.verify(index,repo=repo)
