"""Read-only validation of the pinned Tomato Joint handoff; no robot or Encoder run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import tomato_dataset as ds, tomato_phase_model as model
from .tomato_review_ui.handshake import DEFAULT, validate

REPO = Path(__file__).resolve().parents[1]
RUN_FILES = {'best_head.pt', 'manifest.json', 'training_config.json', 'split.json',
             'validation_receipt.json', 'validation_lock.json'}
ENCODER_FILES = {'model.safetensors', 'config.json', 'weights_metadata.json'}


def verify(inventory_path, *, run_dir=None, encoder_dir=None, repo=REPO):
    """Relocation changes directories only; bytes and contracts must still match."""
    inventory = ds.read_json(inventory_path)
    if inventory.get('schema_version') != 1:
        raise ValueError('Unsupported deployment inventory schema')
    paths = {}
    for group, names, override in [('run', RUN_FILES, run_dir),
                                    ('encoder', ENCODER_FILES, encoder_dir)]:
        entries = inventory['artifacts'][group]
        if set(entries) != names:
            raise ValueError(f'Incomplete {group} artifact inventory')
        paths[group] = {}
        for name, entry in entries.items():
            path = Path(override) / name if override is not None else Path(entry['path'])
            if not path.is_file():
                raise FileNotFoundError(f'Missing handoff artifact: {path}')
            if path.stat().st_size != entry['bytes'] or ds.digest(path) != entry['sha256']:
                raise ValueError(f'Artifact SHA256/size mismatch: {name}')
            paths[group][name] = path
    for relative, expected in inventory['code']['source_sha256'].items():
        if ds.digest(Path(repo) / relative) != expected:
            raise ValueError(f'Handoff source SHA256 mismatch: {relative}')
    policy_path = Path(repo) / inventory['handshake']['path']
    if ds.digest(policy_path) != inventory['handshake']['sha256']:
        raise ValueError('Handshake config SHA256 mismatch')
    policy = ds.read_json(policy_path)
    if set(policy) != set(DEFAULT):
        raise ValueError('Handshake export must include every policy option, including TTL')
    policy = validate(policy)
    run = paths['run']
    manifest = ds.read_json(run['manifest.json'])
    training = ds.read_json(run['training_config.json'])
    splits = ds.read_json(run['split.json'])
    if splits != manifest['splits'] or ds.json_file_sha256(splits) != manifest['split_sha256']:
        raise ValueError('Manifest/split binding mismatch')
    saved, head = model.load_checkpoint(run['best_head.pt'], ds.digest(run['manifest.json']),
                                       'cpu', manifest['split_sha256'])
    if (saved['format_version'] != 2 or saved['prediction_mode'] != 'joint'
            or saved['future_horizon_s'] != 1.0 or saved['contract'] != manifest['contract']
            or saved['prediction'] != manifest['prediction']
            or saved['pretrained'] != manifest['pretrained']):
        raise ValueError('Expected the bound Joint +1.0s model/manifest contract')
    expected_description = {k: saved[k] for k in ['prediction_mode', 'future_horizon_s',
                                                 'dimension', 'joint_dimension']}
    expected_description['labels'] = saved['label_order']
    if (inventory['input_contract'] != saved['contract']
            or any(inventory['model_contract'].get(k) != value
                   for k, value in expected_description.items())):
        raise ValueError('Deployment inventory/checkpoint input contract mismatch')
    if (ds.prediction_contract(training) != saved['prediction']
            or any(training.get(k) != value for k, value in saved['contract'].items()
                   if k != 'sampler_version')
            or training.get('normalization_fit_split') != 'train'
            or manifest.get('normalization') != 'train-only'):
        raise ValueError('Training configuration/normalization provenance mismatch')
    receipt = ds.read_json(run['validation_receipt.json'])
    lock = ds.read_json(run['validation_lock.json'])
    if (receipt['manifest_sha256'] != ds.digest(run['manifest.json'])
            or receipt['split_file_sha256'] != ds.digest(run['split.json'])
            or lock['receipt_sha256'] != ds.digest(run['validation_receipt.json'])
            or lock['manifest_sha256'] != ds.digest(run['manifest.json'])
            or lock['split_file_sha256'] != ds.digest(run['split.json'])):
        raise ValueError('Training validation receipt/lock mismatch')
    provenance = model.verify_pretrained(paths['encoder']['model.safetensors'].parent)
    if provenance != saved['pretrained']:
        raise ValueError('Frozen Encoder/checkpoint provenance mismatch')
    return dict(status='PASS', scope='artifact/contract verification; no online or robot test',
                verified_artifacts=sum(len(v) for v in paths.values()),
                baseline_commit=inventory['code']['base_commit'],
                model=dict(prediction_mode=saved['prediction_mode'],
                           future_horizon_s=saved['future_horizon_s'], best_epoch=saved['best_epoch'],
                           dimension=saved['dimension'], joint_dimension=saved['joint_dimension'],
                           label_order=saved['label_order'],
                           head_parameters=sum(p.numel() for p in head.parameters())),
                normalization=dict(embedded=True, fit_split='train',
                                   mean_shape=list(saved['mean'].shape),
                                   std_shape=list(saved['std'].shape),
                                   positive_finite_std=True), handshake=policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=REPO / 'configs/tairan_deployment.json')
    parser.add_argument('--run-dir', type=Path, help='Relocated directory containing the six run files')
    parser.add_argument('--encoder-dir', type=Path, help='Relocated pinned Encoder directory')
    args = parser.parse_args()
    try:
        result = verify(args.inventory, run_dir=args.run_dir, encoder_dir=args.encoder_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(1, f'Handoff verification failed: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
