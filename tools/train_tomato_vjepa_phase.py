"""Frozen V-JEPA 2 tomato four-stage training: prepare, extract, train (local outputs only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

try:
    from . import tomato_dataset as ds
except ImportError:
    import tomato_dataset as ds

import numpy as np

try:
    from .tomato_runtime import initialize_runtime
except ImportError:
    from tomato_runtime import initialize_runtime


def grouped_split(episodes, seed=42, explicit=None):
    groups = sorted({e['source_recording_id'] for e in episodes})
    if len(groups) < 3:
        raise ValueError('At least three independent source recordings required for train/val/test')
    if explicit is None:
        shuffled = groups.copy()
        random.Random(seed).shuffle(shuffled)
        held = max(1, round(.1 * len(groups)))
        assignment = {g: 'test' if i < held else 'val' if i < 2 * held else 'train'
                      for i, g in enumerate(shuffled)}
    else:
        flat = [g for s in ('train', 'val', 'test') for g in explicit[s]]
        if len(flat) != len(set(flat)) or set(flat) != set(groups) or any(not explicit[s] for s in explicit):
            raise ValueError('Explicit split source leakage/missing/extra groups')
        assignment = {g: s for s in ('train', 'val', 'test') for g in explicit[s]}
    return {s: sorted(g for g in groups if assignment[g] == s) for s in ('train', 'val', 'test')}


def grouped_folds(episodes, folds=5, seed=42):
    groups = sorted({e['source_recording_id'] for e in episodes})
    if folds < 3 or len(groups) < folds:
        raise ValueError('Not enough independent source groups for grouped cross-validation')
    random.Random(seed).shuffle(groups)
    result = []
    for k in range(folds):
        test = groups[k::folds]
        inner = [g for g in groups if g not in test]
        if len(inner) < 2:
            raise ValueError('Fold lacks separate train and early-stopping validation sources')
        random.Random(seed + k + 1).shuffle(inner)
        held = max(1, round(.1 * len(inner)))
        result.append(dict(train=sorted(inner[held:]), val=sorted(inner[:held]), test=sorted(test)))
    return result


def _prepare_fresh(config):
    config = {**config, **ds.prediction_contract(config)}
    rows = ds.load_annotation_index(config['annotation_index'])
    run = ds.ensure_output_outside_sources(config['output_dir'], [r['dataset_root'] for r in rows],
                                            [r['coarse_annotation'] for r in rows])
    run.mkdir(parents=True, exist_ok=True)
    episodes, rejected, schema = [], [], None
    for r in rows:
        try:
            e = ds.load_episode_info(r['dataset_root'], r['episode_index'])
            names = e['info']['features']['observation.state']['names']
            if schema is not None and names != schema:
                raise ValueError('Inconsistent full state schema across datasets')
            schema = names
            ann = ds.load_reviewed_coarse(r['coarse_annotation'], e, config.get('allow_subset_metadata_mismatch', False))
            if r.get('annotation_sha256', ann['sha256']) != ann['sha256'] or r.get('revision', ann['revision']) != ann['revision']:
                raise ValueError('Locked annotation revision/SHA changed')
            before = {str(Path(p).relative_to(e['dataset_root'])): ds.digest(p) for p in [e['parquet'], *e['videos']]}
            media = ds.validate_source_and_media(e, config['joint_names'], config['joint_mode'])
            times, _ = ds.read_state(e, config['joint_names'], config['joint_mode'])
            anchors = ds.iter_phase_anchors(times)
            targets = ds.build_phase_targets(anchors, ann['segments'], e['length'], e['fps'], config['future_horizon_s'])
            counts = np.bincount(targets['current_labels'], minlength=4)
            future_counts = np.bincount(targets['future_labels'][targets['future_valid_mask']], minlength=4)
            for rel, sha in before.items():
                if ds.digest(Path(e['dataset_root']) / rel) != sha:
                    raise ValueError('Source changed during prepare')
            # Full source identity check again after decode guards changed metadata/labels as well.
            if ds.load_reviewed_coarse(r['coarse_annotation'], ds.load_episode_info(r['dataset_root'], r['episode_index']), config.get('allow_subset_metadata_mismatch', False)) != ann:
                raise ValueError('Annotation changed during prepare')
            episodes.append({**e, 'annotation': ann, 'media_validation': media, 'input_sha256': before,
                             'anchors': anchors.tolist(), 'class_counts': counts.tolist(),
                             'future_class_counts': future_counts.tolist(),
                             'future_invalid_count': int((~targets['future_valid_mask']).sum()),
                             'future_offset_frames': int(round(config['future_horizon_s'] * e['fps'])),
                             'unobserved_segments': [s for s in ann['segments'] if not np.any((anchors >= s['start_frame']) & (anchors < s['end_frame']))]})
        except (ValueError, OSError, KeyError) as exc:
            rejected.append(dict(selection=r, error=f'{type(exc).__name__}: {exc}'))
    if rejected:
        ds.write_json(run / 'prepare_rejections.json', dict(accepted=len(episodes), rejected=rejected))
        raise ValueError(f'{len(rejected)} invalid selected episode(s); see {run / "prepare_rejections.json"}; no training manifest produced')
    explicit = ds.read_json(config['split_file']) if config.get('split_file') else None
    if config.get('cv_folds'):
        if explicit is not None:
            raise ValueError('Choose explicit split or grouped CV, not both')
        all_folds = grouped_folds(episodes, config['cv_folds'], config['seed'])
        fold = config['cv_fold_index']
        if not 0 <= fold < len(all_folds):
            raise ValueError('CV fold index out of range')
        explicit = all_folds[fold]
    splits = grouped_split(episodes, config['seed'], explicit)
    summaries = {}
    for s, groups in splits.items():
        selected = [e for e in episodes if e['source_recording_id'] in groups]
        counts = np.sum([e['class_counts'] for e in selected], axis=0)
        future_counts = np.sum([e['future_class_counts'] for e in selected], axis=0)
        summaries[s] = dict(sources=len(groups), episodes=len(selected), anchors=int(counts.sum()),
                            class_counts=counts.tolist(), current_class_counts=counts.tolist(),
                            future_class_counts=future_counts.tolist(), future_valid=int(future_counts.sum()),
                            future_invalid=sum(e['future_invalid_count'] for e in selected))
        for name in ds.active_heads(config['prediction_mode']):
            active_counts = counts if name == 'current' else future_counts
            if np.any(active_counts == 0):
                raise ValueError(f'{s} {name} missing four-class anchor coverage: {active_counts.tolist()}; split is not evaluable')
        for e in selected:
            e['split'] = s
    contract_keys = ('project', 'robot', 'variant', 'model_id', 'model_revision', 'cameras', 'labels',
                     'image_size', 'history_frames', 'history_sample_hz', 'prediction_hz', 'joint_mode', 'joint_names', 'seed')
    manifest = dict(format_version=2, contract={**{k: config[k] for k in contract_keys}, 'sampler_version': ds.SAMPLER_VERSION},
                    prediction=ds.prediction_contract(config),
                    external_split_sha256=ds.digest(config['split_file']) if config.get('split_file') else None,
                    split_sha256=ds.json_file_sha256(splits),
                    allow_subset_metadata_mismatch=config.get('allow_subset_metadata_mismatch', False),
                    cross_validation=dict(folds=config.get('cv_folds'), fold_index=config.get('cv_fold_index')),
                    annotation_index_sha256=ds.digest(config['annotation_index']),
                    sampler_version=ds.SAMPLER_VERSION, episodes=episodes, splits=splits, summary=summaries,
                    warnings=['Fewer than 10 source recordings; smoke-scale result only'] if len({e['source_recording_id'] for e in episodes}) < 10 else [],
                    label_order=list(ds.LABELS), causal=True, normalization='train-only',
                    pretrained=dict(model_id=ds.MODEL_ID, revision=ds.REVISION,
                                    sha256='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b',
                                    config_sha256='3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9'),
                    model_weights_sha256='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b')
    if any(e['annotation']['allowed_metadata_mismatches'] for e in episodes):
        manifest['warnings'].append('TEST ONLY: source metadata hash mismatch allowed; see annotation.source_hashes')
    for name, value in (('manifest.json', manifest), ('split.json', splits)):
        p = run / name
        if p.exists() and ds.read_json(p) != value:
            raise ValueError('Locked manifest/split changed; use new isolated output_dir')
        if not p.exists():
            ds.write_json(p, value)
    _create_validation_receipt(run, manifest)
    ds.status(run, 'prepared', summary=summaries, warnings=manifest['warnings'])
    return manifest


def _create_validation_receipt(run, manifest):
    receipt = dict(format_version=1, validator='tomato-full-source-media-v2',
                   manifest_sha256=ds.digest(run / 'manifest.json'), split_file_sha256=ds.digest(run / 'split.json'),
                   sampler_version=manifest['sampler_version'], full_media_validation=True,
                   episode_fingerprints={e['episode_id']: ds.fingerprint(e) for e in manifest['episodes']})
    path, lock = run / 'validation_receipt.json', run / 'validation_lock.json'
    if path.exists() and ds.read_json(path) != receipt:
        raise ValueError('Locked validation receipt changed; use a new run')
    if not path.exists():
        ds.write_json(path, receipt)
    locked = dict(receipt_sha256=ds.digest(path), manifest_sha256=receipt['manifest_sha256'],
                  split_file_sha256=receipt['split_file_sha256'])
    if lock.exists() and ds.read_json(lock) != locked:
        raise ValueError('Locked receipt fingerprint changed; use a new run')
    if not lock.exists():
        ds.write_json(lock, locked)


def prepare(config, full_revalidate=False):
    if (Path(config['output_dir']) / 'manifest.json').exists():
        return load_locked_manifest(config, full_revalidate)
    return _prepare_fresh(config)


def feature_contract(manifest):
    return {**manifest['contract'], 'sampler_version': manifest['sampler_version']}


def load_locked_manifest(config, full_revalidate=False):
    run = Path(config['output_dir'])
    if not (run / 'manifest.json').is_file():
        raise ValueError('Run prepare successfully first; no locked manifest')
    manifest = ds.read_json(run / 'manifest.json')
    receipt_path, lock_path = run / 'validation_receipt.json', run / 'validation_lock.json'
    legacy = manifest.get('format_version') == 1
    if not (receipt_path.exists() and lock_path.exists()):
        if not (legacy and full_revalidate):
            raise ValueError('Locked full-validation receipt missing; legacy runs require --full-revalidate; otherwise use a new run')
    else:
        lock = ds.read_json(lock_path)
        if (lock != dict(receipt_sha256=ds.digest(receipt_path), manifest_sha256=ds.digest(run / 'manifest.json'),
                         split_file_sha256=ds.digest(run / 'split.json'))):
            raise ValueError('Locked manifest/split/receipt SHA changed; use a new run')
        receipt = ds.read_json(receipt_path)
        if (receipt.get('full_media_validation') is not True or receipt['manifest_sha256'] != lock['manifest_sha256']
                or receipt['split_file_sha256'] != lock['split_file_sha256']
                or receipt['episode_fingerprints'] != {e['episode_id']: ds.fingerprint(e) for e in manifest['episodes']}):
            raise ValueError('Locked validation receipt inconsistent')
    try:
        if manifest['annotation_index_sha256'] != ds.digest(config['annotation_index']):
            raise ValueError('annotation index SHA changed')
        expected_prediction = ds.prediction_contract(manifest.get('prediction', {}))
        if ds.prediction_contract(config) != expected_prediction:
            raise ValueError('mode/horizon/lambda/stability configuration changed')
        for key, value in manifest['contract'].items():
            if key == 'sampler_version':
                if value != manifest['sampler_version']: raise ValueError('sampler contract inconsistent')
            elif config.get(key) != value:
                raise ValueError(f'configuration {key} changed')
        if config.get('allow_subset_metadata_mismatch', False) != manifest.get('allow_subset_metadata_mismatch', False):
            raise ValueError('metadata policy configuration changed')
        if ds.read_json(run / 'split.json') != manifest['splits']:
            raise ValueError('split changed')
        if config.get('split_file'):
            if ds.read_json(config['split_file']) != manifest['splits']:
                raise ValueError('external split changed')
            if not legacy and ds.digest(config['split_file']) != manifest['external_split_sha256']:
                raise ValueError('external split SHA changed')
        selected = ds.load_annotation_index(config['annotation_index'])
        if [(r['dataset_root'], r['episode_index'], r['coarse_annotation']) for r in selected] != [
                (e['dataset_root'], e['episode_index'], e['annotation']['path']) for e in manifest['episodes']]:
            raise ValueError('selected episode membership/order changed')
        for e in manifest['episodes']:
            actual = ds.load_episode_info(e['dataset_root'], e['episode_index'])
            if any(e[k] != v for k, v in actual.items()):
                raise ValueError('episode metadata/source/path changed')
            ann = ds.load_reviewed_coarse(e['annotation']['path'], actual, config.get('allow_subset_metadata_mismatch', False))
            expected_ann = {**e['annotation']}
            if legacy: expected_ann.setdefault('allowed_metadata_mismatches', [])
            if ann != expected_ann:
                raise ValueError('reviewed annotation revision/SHA/source changed')
            for rel, expected in e['input_sha256'].items():
                if ds.digest(ds.safe_source_path(e['dataset_root'], rel)) != expected:
                    raise ValueError('input SHA changed')
            times, _ = ds.read_state(actual, config['joint_names'], config['joint_mode'])
            if ds.iter_phase_anchors(times).tolist() != e['anchors']:
                raise ValueError('anchor grid changed')
            if full_revalidate:
                media = ds.validate_source_and_media(actual, config['joint_names'], config['joint_mode'])
                if media != e['media_validation']:
                    raise ValueError('full media validation changed')
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError(f'Locked inputs/configuration changed; use new run: {exc}') from exc
    if legacy and full_revalidate and not receipt_path.exists():
        _create_validation_receipt(run, manifest)
    return manifest


def extract(config, manifest):
    initialize_runtime(config)
    try:
        from . import tomato_phase_model as model
    except ImportError:
        import tomato_phase_model as model
    run = Path(config['output_dir'])
    cache_dir = Path(config.get('feature_cache_dir', run.parent / 'feature_cache')).resolve()
    ds.ensure_output_outside_sources(cache_dir, [e['dataset_root'] for e in manifest['episodes']],
                                    [e['annotation']['path'] for e in manifest['episodes']])
    provenance = model.verify_pretrained(config['pretrained_dir'])
    if provenance != manifest['pretrained']:
        raise ValueError('Verified encoder does not match locked manifest')
    rows, encoder = [], None
    for e in manifest['episodes']:
        ds.status(run, 'extracting', episode_id=e['episode_id'], completed=len(rows), total=len(manifest['episodes']))
        path = model.cache_path(cache_dir, e, feature_contract(manifest), provenance)
        if path.exists():
            model.assert_source_unchanged(e)
            model.load_cache(path, e, feature_contract(manifest), provenance)
            performance = dict(reused=True)
        else:
            if encoder is None:
                encoder, actual = model.load_encoder(config['pretrained_dir'], config['device'], config['seed'])
                if actual != provenance: raise ValueError('Encoder provenance changed')
            path, performance = model.extract_episode(encoder, e, feature_contract(manifest), provenance, cache_dir, config['batch_windows'])
        rows.append(dict(episode_id=e['episode_id'], path=str(path), performance=performance))
    ds.write_json(run / 'feature_index.json', dict(manifest_sha256=ds.digest(run / 'manifest.json'),
                                                 pretrained=provenance, episodes=rows))
    try:
        from .tomato_runtime import hardware_report
    except ImportError:
        from tomato_runtime import hardware_report
    ds.write_json(run / 'extraction_runtime.json', dict(runtime=hardware_report(), encoder_loaded=encoder is not None,
                                                       sampler_version=manifest['sampler_version'], pretrained=provenance))
    ds.status(run, 'features_complete', episodes=len(rows))
    return rows


def load_training_arrays(config, manifest):
    try:
        from . import tomato_phase_model as model
    except ImportError:
        import tomato_phase_model as model
    run = Path(config['output_dir'])
    index = ds.read_json(run / 'feature_index.json')
    if (index['manifest_sha256'] != ds.digest(run / 'manifest.json')
            or index['pretrained'] != manifest['pretrained']):
        raise ValueError('Feature index/manifest mismatch; rerun extract after prepare')
    paths = {r['episode_id']: r['path'] for r in index['episodes']}
    if len(paths) != len(index['episodes']) or set(paths) != {e['episode_id'] for e in manifest['episodes']}:
        raise ValueError('Feature index episodes missing/duplicate')
    entries = {s: [] for s in ('train', 'val', 'test')}
    for e in manifest['episodes']:
        model.assert_source_unchanged(e)
        if ds.digest(e['annotation']['path']) != e['annotation']['sha256']:
            raise ValueError('Annotation changed since prepare; use a new manifest')
        data = model.load_cache(paths[e['episode_id']], e, feature_contract(manifest), manifest['pretrained'])
        # Always recompute supervision from locked coarse truth, never trust cached labels in NPZ.
        prediction = ds.prediction_contract(manifest.get('prediction', {}))
        targets = ds.build_phase_targets(data['anchors'], e['annotation']['segments'], e['length'], e['fps'], prediction['future_horizon_s'])
        labels = targets['current_labels']
        label_key = ds.fingerprint(dict(annotation_sha256=e['annotation']['sha256'],
                                       revision=e['annotation']['revision'], anchors=data['anchors'].tolist(), labels=list(ds.LABELS),
                                       future_horizon_s=prediction['future_horizon_s'], sampler_version=manifest['sampler_version']))
        ds.write_json(run / 'labels' / (label_key + '.json'), dict(fingerprint=label_key, label_order=list(ds.LABELS),
                                                                 label_order_current=list(ds.LABELS), **ds.serializable_targets(targets)))
        entries[e['split']].append(dict(episode=e, features=data['features'], labels=labels,
                                       **targets, prediction=prediction, times=data['times'], input_indices=data['input_indices']))
    return entries


def fit_training_statistics(features, labels):
    x, y = np.asarray(features), np.asarray(labels)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or y.shape != (len(x),):
        raise ValueError('Invalid training data for normalization')
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 3)):
        raise ValueError('Invalid training labels')
    counts = np.bincount(y, minlength=4)
    if np.any(counts == 0):
        raise ValueError('All four classes required in training')
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).clip(.01).astype(np.float32)
    weight = np.sqrt(counts.sum() / counts).astype(np.float32)
    weight /= weight.mean()
    return mean, std, weight


def fit_class_weights(labels):
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 3)):
        raise ValueError('No valid four-class supervision')
    counts = np.bincount(y, minlength=4)
    if np.any(counts == 0):
        raise ValueError('Active head missing four classes in train split')
    weight = np.sqrt(counts.sum() / counts).astype(np.float32)
    return weight / weight.mean(), counts


def validation_selection(current, future, config):
    cfg = ds.prediction_contract(config)
    values = dict(current=current, future=future)
    names = ds.active_heads(cfg['prediction_mode'])
    total = sum(cfg['lambda_' + name] for name in names)
    f1 = sum(cfg['lambda_' + name] * values[name]['macro_f1'] for name in names) / total
    nll = sum(cfg['lambda_' + name] * values[name]['nll'] for name in names) / total
    return float(f1), float(nll)


def training_batches(y_future, valid, count, batch_size, rng, mode):
    if mode == 'current_only':
        order = rng.permutation(count)
        return [order[start:start + batch_size] for start in range(0, count, batch_size)]
    valid_ids = np.flatnonzero(valid)
    if not len(valid_ids):
        raise ValueError('No valid future training samples')
    if mode == 'future_only':
        order = rng.permutation(valid_ids)
        return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    # Every current anchor occurs exactly once; every batch has real future supervision.
    # If batch_size is tiny relative to invalid-tail count, fewer, slightly larger batches
    # are necessary. This is recorded instead of fabricating a zero future loss.
    chunks = min(int(np.ceil(count / batch_size)), len(valid_ids))
    good = np.array_split(rng.permutation(valid_ids), chunks)
    tail = np.array_split(rng.permutation(np.flatnonzero(~valid)), chunks)
    return [rng.permutation(np.r_[a, b]) for a, b in zip(good, tail)]


def train(config, manifest):
    runtime = initialize_runtime(config)
    import torch
    from torch import nn
    try:
        from . import tomato_phase_model as model, tomato_metrics as metrics
        from .tomato_runtime import hardware_report
    except ImportError:
        import tomato_phase_model as model
        import tomato_metrics as metrics
        from tomato_runtime import hardware_report
    run = Path(config['output_dir'])
    prediction = ds.prediction_contract(manifest.get('prediction', {}))
    if manifest['format_version'] != 2 or manifest['sampler_version'] != ds.SAMPLER_VERSION:
        raise ValueError('Training requires a newly prepared v2 run; legacy current-only checkpoints remain readable')
    if ds.prediction_contract(config) != prediction:
        raise ValueError('Training prediction configuration differs from manifest')
    mode = prediction['prediction_mode']
    if ds.read_json(run / 'manifest.json') != manifest:
        raise ValueError('Manifest differs from locked snapshot')
    if any((run / name).exists() for name in ('best_head.pt', 'evaluation.json', 'metrics.jsonl')):
        raise ValueError('Training outputs already exist; use new output_dir; locked test is not retuned')
    entries = load_training_arrays(config, manifest)
    arrays = {}
    for split, parts in entries.items():
        arrays[split] = {key: np.concatenate([r[key] for r in parts])
                         for key in ('features', 'current_labels', 'future_labels', 'future_valid_mask')}
    x = arrays['train']['features']
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite training features')
    # Same train observations and normalization in A/B/C; class weighting is head-specific.
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).clip(.01).astype(np.float32)
    weights, counts = {}, {}
    for name in ds.active_heads(mode):
        y = arrays['train'][name + '_labels']
        if name == 'future': y = y[arrays['train']['future_valid_mask']]
        weights[name], counts[name] = fit_class_weights(y)
    rng = np.random.default_rng(config['seed'])
    device = config['device']
    head = model.mode_head_model(len(manifest['contract']['joint_names']), mode).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    class_weight = {name: torch.from_numpy(w).to(device) for name, w in weights.items()}
    training_config = {**config, **prediction, 'runtime': hardware_report(),
                       'class_counts_by_head': {h: a.tolist() for h, a in counts.items()},
                       'class_weights_by_head': {h: a.tolist() for h, a in weights.items()},
                       'head_parameters': sum(p.numel() for p in head.parameters()), 'encoder_frozen': True,
                       'optimizer_parameters': 'active head/trunk only', 'dimension': x.shape[1],
                       'normalization_fit_split': 'train', 'model_selection': 'active lambda normalized val F1 then NLL; no test tuning',
                       'joint_batch_policy': 'distribute valid/invalid anchors, each current anchor once; each batch needs valid future'}
    ds.write_json(run / 'training_config.json', training_config)
    best, stale, best_epoch, executed = (-1., -float('inf')), 0, 0, 0
    for epoch in range(1, config['epochs'] + 1):
        head.train()
        losses = dict(current=[], future=[], total=[])
        batches = training_batches(arrays['train']['future_labels'], arrays['train']['future_valid_mask'],
                                    len(x), config['batch_size'], rng, mode)
        for ids in batches:
            batch = torch.from_numpy((x[ids] - mean) / std).to(device)
            yc = torch.from_numpy(arrays['train']['current_labels'][ids]).to(device)
            yf = torch.from_numpy(arrays['train']['future_labels'][ids]).to(device)
            mask = torch.from_numpy(arrays['train']['future_valid_mask'][ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, parts = model.supervision_loss(model.forward_heads(head, batch), yc, yf, mask, class_weight, prediction)
            total.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses['total'].append(float(total.detach().cpu()))
            for name in ds.active_heads(mode): losses[name].append(float(parts[name].detach().cpu()))
        z = model.predict_heads(head, arrays['val']['features'], mean, std)
        val = dict(current=None, future=None)
        for name in ds.active_heads(mode):
            y, logits = arrays['val'][name + '_labels'], z[name + '_logits']
            if name == 'future':
                mask = arrays['val']['future_valid_mask']; y, logits = y[mask], logits[mask]
            val[name] = metrics.scores(logits, y)
        selection_f1, selection_nll = validation_selection(val['current'], val['future'], prediction)
        row = dict(epoch=epoch, train_current_ce=float(np.mean(losses['current'])) if losses['current'] else None,
                   train_future_ce=float(np.mean(losses['future'])) if losses['future'] else None,
                   train_total_loss=float(np.mean(losses['total'])), val_current=val['current'], val_future=val['future'],
                   validation_selection_score=selection_f1, validation_selection_nll=selection_nll,
                   max_actual_batch_size=max(map(len, batches)))
        if mode == 'current_only': row.update(train_loss=row['train_total_loss'], val=val['current'])
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        key = (selection_f1, -selection_nll)
        if key > best:
            best, best_epoch, stale = key, epoch, 0
            model.save_checkpoint(run / 'best_head.pt', head, mean, std, manifest,
                                  ds.digest(run / 'manifest.json'), epoch, val)
        else:
            stale += 1
        executed = epoch
        ds.status(run, 'training', epoch=epoch, best_epoch=best_epoch, validation_selection_score=selection_f1)
        if stale >= config['patience']: break
    saved, head = model.load_checkpoint(run / 'best_head.pt', ds.digest(run / 'manifest.json'), device,
                                        split_sha256=manifest['split_sha256'])
    report = dict(**prediction, best_epoch=best_epoch, selection='active lambda normalized val Macro-F1 then NLL',
                  epochs_executed=executed, source_summary=manifest['summary'], warnings=manifest['warnings'],
                  manifest_sha256=ds.digest(run / 'manifest.json'), split_sha256=manifest['split_sha256'], splits={})
    predictions = []
    for split, a in arrays.items():
        logits = model.predict_heads(head, a['features'], saved['mean'], saved['std'])
        report['splits'][split] = metrics.evaluate_heads(entries[split], logits, prediction)
        offset = 0
        for r in entries[split]:
            n = len(r['current_labels'])
            part = {name: logits[name + '_logits'][offset:offset + n] if logits[name + '_logits'] is not None else None
                    for name in ('current', 'future')}
            probs = {name: metrics.probabilities(z) if z is not None else None for name, z in part.items()}
            predictions += prediction_rows(r, probs, split, prediction, part)
            offset += n
    ds.write_json(run / 'evaluation.json', report)
    write_predictions(run, predictions)
    ds.status(run, 'training_complete', prediction_mode=mode, best_epoch=best_epoch,
              evaluation=str(run / 'evaluation.json'), warnings=manifest['warnings'])
    return report


def prediction_rows(entry, probabilities, split, config=None, logits=None):
    prediction = ds.prediction_contract(config or entry.get('prediction', {}))
    names = ds.active_heads(prediction['prediction_mode'])
    if not isinstance(probabilities, dict): probabilities = dict(current=probabilities, future=None)
    for name in names:
        p = np.asarray(probabilities[name], dtype=np.float64)
        if (p.shape != (len(entry['anchors']), 4) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1, atol=1e-6)):
            raise ValueError('Expected finite four-class normalized probabilities')
    targets = entry if 'future_labels' in entry else ds.build_phase_targets(
        entry['anchors'], entry['episode']['annotation']['segments'], entry['episode']['length'],
        entry['episode']['fps'], prediction['future_horizon_s'])
    rows = []
    for k, (i, t, history) in enumerate(zip(entry['anchors'], entry['times'], entry['input_indices'])):
        if np.max(history) > i: raise ValueError('Future frame in prediction metadata')
        valid = bool(targets['future_valid_mask'][k]); yc = int(targets['current_labels'][k])
        yf = int(targets['future_labels'][k]) if valid else None
        row = dict(episode_id=entry['episode']['episode_id'], source_recording_id=entry['episode']['source_recording_id'],
                   split=split, prediction_mode=prediction['prediction_mode'], future_horizon_s=prediction['future_horizon_s'],
                   frame_index=int(i), timestamp_s=float(t), last_input_frame=int(np.max(history)),
                   input_indices=np.asarray(history).tolist(), current_true_stage_id=yc, current_true_stage=ds.LABELS[yc],
                   future_true_stage_id=yf, future_true_stage=ds.LABELS[yf] if valid else None,
                   future_valid=valid, future_target_frame=int(targets['future_target_frames'][k]),
                   future_target_time_s=float(targets['future_target_times_s'][k]) if valid else None)
        for name in names:
            probs = np.asarray(probabilities[name][k]); pred = int(probs.argmax())
            row.update({name + '_stage_id': pred, name + '_stage': ds.LABELS[pred],
                        name + '_probabilities': dict(zip(ds.LABELS, map(float, probs))),
                        name + '_logits': np.asarray(logits[name][k]).tolist() if logits is not None else None})
        # Backward compatible alias always means CURRENT, including joint; never future.
        if 'current' in names:
            row.update(stage_id=row['current_stage_id'], stage=row['current_stage'],
                       probabilities=row['current_probabilities'], true_stage_id=yc, true_stage=ds.LABELS[yc])
        rows.append(row)
    return rows


def write_predictions(run, rows):
    import csv
    run = Path(run)
    with (run / 'predictions.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False) + '\n')
    base = ['episode_id', 'source_recording_id', 'split', 'prediction_mode', 'future_horizon_s', 'frame_index',
            'timestamp_s', 'last_input_frame', 'input_indices', 'stage_id', 'stage', 'true_stage_id', 'true_stage',
            'current_stage_id', 'current_stage', 'current_true_stage_id', 'current_true_stage', 'current_logits',
            'future_stage_id', 'future_stage', 'future_true_stage_id', 'future_true_stage', 'future_logits',
            'future_valid', 'future_target_frame', 'future_target_time_s']
    columns = base + list(ds.LABELS) + [h + '_p_' + label for h in ('current', 'future') for label in ds.LABELS]
    with (run / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in rows:
            flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items() if k in base}
            if 'probabilities' in row: flat.update(row['probabilities'])
            for name in ('current', 'future'):
                if name + '_probabilities' in row:
                    flat.update({name + '_p_' + k: v for k, v in row[name + '_probabilities'].items()})
            writer.writerow(flat)


def discover_datasets(roots, allow_subset_metadata_mismatch=False):
    """Select one explicit on-disk coarse file per episode; reject ambiguity."""
    resolved = [Path(r).expanduser().resolve(strict=True) for r in roots]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError('Provide nonempty, distinct dataset roots')
    rows = []
    for root in sorted(resolved):
        entries = ds.read_rows(root / 'meta/episodes.jsonl')
        if not entries:
            raise ValueError(f'No episodes in {root}')
        for entry in entries:
            i = entry['episode_index']
            candidates = sorted((root / 'annotations/subtasks/coarse').glob(f'episode_{i:06d}*.json'))
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one coarse annotation for {root} episode {i}; found {len(candidates)}. Use an explicit annotation index for ambiguous versions.')
            e = ds.load_episode_info(root, i)
            a = ds.load_reviewed_coarse(candidates[0], e, allow_subset_metadata_mismatch)
            rows.append(dict(dataset_root=str(root), episode_index=i, coarse_annotation=str(candidates[0].resolve()),
                             variant='normal', annotation_sha256=a['sha256'], revision=a['revision'],
                             allowed_metadata_mismatches=a['allowed_metadata_mismatches']))
    return rows


def setup_discovered_index(config, roots):
    roots = [str(Path(r).expanduser().resolve(strict=True)) for r in roots]
    out = ds.ensure_output_outside_sources(config['output_dir'], roots)
    rows = discover_datasets(roots, config.get('allow_subset_metadata_mismatch', False))
    ds.ensure_output_outside_sources(out, roots, [r['coarse_annotation'] for r in rows])
    path = out / 'annotation_index.json'
    if path.exists() and ds.read_json(path) != rows:
        raise ValueError('Discovered selection changed; use a new output directory')
    if not path.exists():
        ds.write_json(path, rows)
    config['annotation_index'] = str(path)
    return rows


def recover_run_index(config):
    generated = Path(config['output_dir']) / 'annotation_index.json'
    manifest = Path(config['output_dir']) / 'manifest.json'
    if generated.is_file() and manifest.is_file() and ds.digest(generated) == ds.read_json(manifest)['annotation_index_sha256']:
        config['annotation_index'] = str(generated)
    return config


def add_prediction_arguments(parser):
    parser.add_argument('--prediction-mode', choices=['current_only', 'future_only', 'joint'])
    parser.add_argument('--future-horizon-s', type=float)
    parser.add_argument('--lambda-current', type=float)
    parser.add_argument('--lambda-future', type=float)
    parser.add_argument('--stable-transition-k', type=int)


def apply_prediction_overrides(config, args):
    for key in ('prediction_mode', 'future_horizon_s', 'lambda_current', 'lambda_future', 'stable_transition_k'):
        value = getattr(args, key, None)
        if value is not None: config[key] = value
    config.update(ds.prediction_contract(config))
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'extract', 'train', 'run'])
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', action='append', help='LeRobot dataset root; repeat for multiple roots')
    p.add_argument('--output-dir', help='Isolated output directory for this run')
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--batch-windows', type=int)
    p.add_argument('--full-revalidate', action='store_true')
    p.add_argument('--split-file')
    p.add_argument('--feature-cache-dir')
    p.add_argument('--seed', type=int)
    add_prediction_arguments(p)
    p.add_argument('--cv-folds', type=int, help='Optional grouped outer folds; needs sufficient sources')
    p.add_argument('--cv-fold-index', type=int)
    a = p.parse_args(argv)
    config = ds.load_config(a.config)
    apply_prediction_overrides(config, a)
    for key in ('split_file', 'feature_cache_dir'):
        if getattr(a, key): config[key] = str(Path(getattr(a, key)).expanduser().resolve())
    if a.seed is not None: config['seed'] = a.seed
    if a.output_dir:
        config['output_dir'] = str(Path(a.output_dir).expanduser().resolve())
    if a.device:
        config['device'] = a.device
    if a.batch_windows is not None:
        if a.batch_windows <= 0:
            p.error('--batch-windows must be positive')
        config['batch_windows'] = a.batch_windows
    if a.cv_folds is not None or a.cv_fold_index is not None:
        if a.cv_folds is None or a.cv_fold_index is None:
            p.error('Provide both --cv-folds and --cv-fold-index')
        config['cv_folds'], config['cv_fold_index'] = a.cv_folds, a.cv_fold_index
        config['output_dir'] = str(Path(config['output_dir']) / f'cv{a.cv_folds}' / f'fold_{a.cv_fold_index}')
    roots = a.dataset if a.dataset is not None else config.get('dataset_roots', [])
    if a.command in ('run', 'prepare') and roots:
        selections = setup_discovered_index(config, roots)
    else:
        recover_run_index(config)
        selections = ds.load_annotation_index(config['annotation_index'])
    out = ds.ensure_output_outside_sources(config['output_dir'], [r['dataset_root'] for r in selections],
                                           [r['coarse_annotation'] for r in selections])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.command != 'prepare':
            initialize_runtime(config)
        if a.command == 'run':
            if any((out / name).exists() for name in ('best_head.pt', 'evaluation.json', 'metrics.jsonl')):
                raise ValueError('Training outputs already exist; use new output_dir')
            manifest = prepare(config, a.full_revalidate)
            extract(config, manifest)
            train(config, manifest)
        elif a.command == 'prepare':
            prepare(config, a.full_revalidate)
        elif a.command == 'extract':
            extract(config, load_locked_manifest(config, a.full_revalidate))
        else:
            train(config, load_locked_manifest(config, a.full_revalidate))
    except Exception as exc:
        ds.status(out, 'failed', command=a.command, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
