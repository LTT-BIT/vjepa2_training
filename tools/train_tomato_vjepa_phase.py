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


def prepare(config):
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
            ann = ds.load_reviewed_coarse(r['coarse_annotation'], e)
            if r.get('annotation_sha256', ann['sha256']) != ann['sha256'] or r.get('revision', ann['revision']) != ann['revision']:
                raise ValueError('Locked annotation revision/SHA changed')
            before = {str(Path(p).relative_to(e['dataset_root'])): ds.digest(p) for p in [e['parquet'], *e['videos']]}
            media = ds.validate_source_and_media(e, config['joint_names'], config['joint_mode'])
            times, _ = ds.read_state(e, config['joint_names'], config['joint_mode'])
            anchors = ds.iter_phase_anchors(times)
            counts = np.bincount([ds.label_at_frame(ann['segments'], int(i)) for i in anchors], minlength=4)
            for rel, sha in before.items():
                if ds.digest(Path(e['dataset_root']) / rel) != sha:
                    raise ValueError('Source changed during prepare')
            # Full source identity check again after decode guards changed metadata/labels as well.
            if ds.load_reviewed_coarse(r['coarse_annotation'], ds.load_episode_info(r['dataset_root'], r['episode_index'])) != ann:
                raise ValueError('Annotation changed during prepare')
            episodes.append({**e, 'annotation': ann, 'media_validation': media, 'input_sha256': before,
                             'anchors': anchors.tolist(), 'class_counts': counts.tolist(),
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
        summaries[s] = dict(sources=len(groups), episodes=len(selected), anchors=int(counts.sum()), class_counts=counts.tolist())
        if np.any(counts == 0):
            raise ValueError(f'{s} missing four-class anchor coverage: {counts.tolist()}; split is not evaluable')
        for e in selected:
            e['split'] = s
    contract_keys = ('project', 'robot', 'variant', 'model_id', 'model_revision', 'cameras', 'labels',
                     'image_size', 'history_frames', 'history_sample_hz', 'prediction_hz', 'joint_mode', 'joint_names', 'seed')
    manifest = dict(format_version=1, contract={k: config[k] for k in contract_keys},
                    cross_validation=dict(folds=config.get('cv_folds'), fold_index=config.get('cv_fold_index')),
                    annotation_index_sha256=ds.digest(config['annotation_index']),
                    sampler_version=ds.SAMPLER_VERSION, episodes=episodes, splits=splits, summary=summaries,
                    warnings=['Fewer than 10 source recordings; smoke-scale result only'] if len({e['source_recording_id'] for e in episodes}) < 10 else [],
                    label_order=list(ds.LABELS), causal=True, normalization='train-only',
                    pretrained=dict(model_id=ds.MODEL_ID, revision=ds.REVISION,
                                    sha256='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b',
                                    config_sha256='3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9'),
                    model_weights_sha256='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b')
    for name, value in (('manifest.json', manifest), ('split.json', splits)):
        p = run / name
        if p.exists() and ds.read_json(p) != value:
            raise ValueError('Locked manifest/split changed; use new isolated output_dir')
        if not p.exists():
            ds.write_json(p, value)
    ds.status(run, 'prepared', summary=summaries, warnings=manifest['warnings'])
    return manifest


def load_locked_manifest(config):
    run = Path(config['output_dir'])
    if not (run / 'manifest.json').is_file():
        raise ValueError('Run prepare successfully first; no locked manifest')
    previous = ds.read_json(run / 'manifest.json')
    current = prepare(config)
    if current != previous:
        raise ValueError('Locked manifest changed')
    return current


def extract(config, manifest):
    try:
        from . import tomato_phase_model as model
    except ImportError:
        import tomato_phase_model as model
    run = Path(config['output_dir'])
    cache_dir = Path(config.get('feature_cache_dir', run.parent / 'feature_cache')).resolve()
    ds.ensure_output_outside_sources(cache_dir, [e['dataset_root'] for e in manifest['episodes']],
                                    [e['annotation']['path'] for e in manifest['episodes']])
    encoder, provenance = model.load_encoder(config['pretrained_dir'], config['device'])
    if provenance != manifest['pretrained']:
        raise ValueError('Verified encoder does not match locked manifest')
    rows = []
    for e in manifest['episodes']:
        ds.status(run, 'extracting', episode_id=e['episode_id'], completed=len(rows), total=len(manifest['episodes']))
        path, performance = model.extract_episode(encoder, e, manifest['contract'], provenance, cache_dir, config['batch_windows'])
        rows.append(dict(episode_id=e['episode_id'], path=str(path), performance=performance))
    ds.write_json(run / 'feature_index.json', dict(manifest_sha256=ds.digest(run / 'manifest.json'),
                                                 pretrained=provenance, episodes=rows))
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
        data = model.load_cache(paths[e['episode_id']], e, manifest['contract'], manifest['pretrained'])
        # Always recompute supervision from locked coarse truth, never trust cached labels in NPZ.
        labels = np.array([ds.label_at_frame(e['annotation']['segments'], int(i)) for i in data['anchors']], dtype=np.int64)
        label_key = ds.fingerprint(dict(annotation_sha256=e['annotation']['sha256'],
                                       revision=e['annotation']['revision'], anchors=data['anchors'].tolist(), labels=list(ds.LABELS)))
        ds.write_json(run / 'labels' / (label_key + '.json'), dict(fingerprint=label_key, label_order=list(ds.LABELS),
                                                                 anchors=data['anchors'].tolist(), labels=labels.tolist()))
        entries[e['split']].append(dict(episode=e, features=data['features'], labels=labels,
                                       anchors=data['anchors'], times=data['times'], input_indices=data['input_indices']))
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


def train(config, manifest):
    import torch
    from torch import nn
    try:
        from . import tomato_phase_model as model
        from . import tomato_metrics as metrics
    except ImportError:
        import tomato_phase_model as model
        import tomato_metrics as metrics
    run = Path(config['output_dir'])
    if ds.read_json(run / 'manifest.json') != manifest:
        raise ValueError('Manifest differs from locked snapshot')
    if (run / 'best_head.pt').exists() or (run / 'evaluation.json').exists():
        raise ValueError('Training outputs already exist; use new output_dir; locked test is not retuned')
    entries = load_training_arrays(config, manifest)
    arrays = {s: (np.concatenate([r['features'] for r in parts]), np.concatenate([r['labels'] for r in parts]))
              for s, parts in entries.items()}
    x, y = arrays['train']
    mean, std, weight = fit_training_statistics(x, y)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    torch.use_deterministic_algorithms(True)
    device = config['device']
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select --device cpu explicitly')
        # cuBLAS needs deterministic workspace set before CUDA kernels are initialized.
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    head = model.head_model(len(manifest['contract']['joint_names'])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    class_weight = torch.from_numpy(weight).to(device)
    ds.write_json(run / 'training_config.json', {**config, 'train_class_counts': np.bincount(y, minlength=4).tolist(),
                                               'class_weights': weight.tolist(), 'head_parameters': sum(p.numel() for p in head.parameters()),
                                               'encoder_frozen': True, 'optimizer_parameters': 'head only',
                                               'dimension': x.shape[1], 'normalization_fit_split': 'train',
                                               'model_selection': 'val Macro-F1 then val NLL; test evaluated after selection'})
    best, stale, best_epoch = (-1., -float('inf')), 0, 0
    history = []
    for epoch in range(1, config['epochs'] + 1):
        head.train(); losses = []
        permutation = rng.permutation(len(y))
        for start in range(0, len(y), config['batch_size']):
            ids = permutation[start:start + config['batch_size']]
            batch = torch.from_numpy((x[ids] - mean) / std).to(device)
            target = torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(batch), target, weight=class_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite head training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # This is the only data inspected for early stopping. Never score test inside this loop.
        val = metrics.scores(model.predict_logits(head, arrays['val'][0], mean, std), arrays['val'][1])
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val=val)
        history.append(row)
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        selection = (val['macro_f1'], -val['nll'])
        if selection > best:
            best, best_epoch, stale = selection, epoch, 0
            model.save_checkpoint(run / 'best_head.pt', head, mean, std, manifest,
                                  ds.digest(run / 'manifest.json'), epoch, val)
        else:
            stale += 1
        ds.status(run, 'training', epoch=epoch, best_epoch=best_epoch, validation=val)
        if stale >= config['patience']:
            break
    saved, head = model.load_checkpoint(run / 'best_head.pt', ds.digest(run / 'manifest.json'), device)
    report = dict(best_epoch=best_epoch, selection='val Macro-F1 then NLL', epochs_executed=len(history),
                  source_summary=manifest['summary'], warnings=manifest['warnings'], splits={})
    predictions = []
    for s, (xx, yy) in arrays.items():
        logits = model.predict_logits(head, xx, saved['mean'], saved['std'])
        report['splits'][s] = metrics.evaluate_entries(entries[s], logits)
        offset = 0
        for r in entries[s]:
            n = len(r['labels'])
            predictions += prediction_rows(r, metrics.probabilities(logits[offset:offset + n]), s)
            offset += n
    ds.write_json(run / 'evaluation.json', report)
    write_predictions(run, predictions)
    ds.status(run, 'training_complete', best_epoch=best_epoch, test=report['splits']['test'],
              replay='Run eval_tomato_vjepa_phase.py for held-out replay and latency')
    return report


def prediction_rows(entry, probabilities, split):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.shape != (len(entry['anchors']), 4) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-6)):
        raise ValueError('Expected finite four-class normalized probabilities')
    rows = []
    for k, (i, t, history, y, probs) in enumerate(zip(entry['anchors'], entry['times'], entry['input_indices'], entry['labels'], p)):
        if np.max(history) > i:
            raise ValueError('Future frame in prediction metadata')
        pred = int(probs.argmax())
        rows.append(dict(episode_id=entry['episode']['episode_id'], source_recording_id=entry['episode']['source_recording_id'],
                         split=split, frame_index=int(i), timestamp_s=float(t), last_input_frame=int(np.max(history)),
                         input_indices=np.asarray(history).tolist(), stage_id=pred, stage=ds.LABELS[pred],
                         true_stage_id=int(y), true_stage=ds.LABELS[int(y)],
                         probabilities=dict(zip(ds.LABELS, map(float, probs)))))
    return rows


def write_predictions(run, rows):
    import csv
    run = Path(run)
    with (run / 'predictions.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    with (run / 'predictions.csv').open('w', newline='') as f:
        columns = ['episode_id', 'source_recording_id', 'split', 'frame_index', 'timestamp_s', 'last_input_frame',
                   'stage_id', 'stage', 'true_stage_id', 'true_stage', *ds.LABELS]
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({**{k: v for k, v in row.items() if k in columns}, **row['probabilities']})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'extract', 'train'])
    p.add_argument('--config', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--batch-windows', type=int)
    p.add_argument('--cv-folds', type=int, help='Optional grouped outer folds; needs sufficient sources')
    p.add_argument('--cv-fold-index', type=int)
    a = p.parse_args(argv)
    config = ds.load_config(a.config)
    if a.device:
        config['device'] = a.device
    if a.batch_windows:
        config['batch_windows'] = a.batch_windows
    if a.cv_folds is not None or a.cv_fold_index is not None:
        if a.cv_folds is None or a.cv_fold_index is None:
            p.error('Provide both --cv-folds and --cv-fold-index')
        config['cv_folds'], config['cv_fold_index'] = a.cv_folds, a.cv_fold_index
        config['output_dir'] = str(Path(config['output_dir']) / f'cv{a.cv_folds}' / f'fold_{a.cv_fold_index}')
    selections = ds.load_annotation_index(config['annotation_index'])
    out = ds.ensure_output_outside_sources(config['output_dir'], [r['dataset_root'] for r in selections],
                                           [r['coarse_annotation'] for r in selections])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.command == 'prepare':
            prepare(config)
        elif a.command == 'extract':
            extract(config, load_locked_manifest(config))
        else:
            train(config, load_locked_manifest(config))
    except Exception as exc:
        ds.status(out, 'failed', command=a.command, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
