#!/usr/bin/env python3
"""Annotate with Doubao, relabel frozen V-JEPA 2 features, then train a new phase head."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np

from tools import annotate_lemon_doubao as doubao
from tools import train_lemon_vjepa_phase as trainer

RUN = trainer.RUN.parent / 'lemon_vjepa2_doubao_phase_20261030'


def relabel_cache(cache_path, original, updated):
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache['fingerprint']) != trainer.feature_fingerprint(original):
            raise ValueError(f'Original feature fingerprint mismatch: {cache_path}')
        indices = cache['indices']
        expected = np.arange(0, original['frames'], trainer.STRIDE)
        features, times = cache['features'], cache['times']
        if (not np.array_equal(indices, expected) or times.shape != expected.shape
                or not np.allclose(times, expected / 10., atol=1e-5, rtol=0)
                or features.shape != (len(expected), 3 * 1024 + 16 * 12)
                or not np.isfinite(features).all()):
            raise ValueError(f'Invalid original features/timeline: {cache_path}')
        # Cached visual/state features are label independent; never reuse cached human labels.
        return dict(features=features.copy(), indices=indices.copy(), times=times.copy(),
                    labels=trainer.labels_at(times, updated['boundaries']),
                    fingerprint=np.array(trainer.feature_fingerprint(updated)))


def prepare_training(run, previous, label_output):
    original = json.loads((previous / 'manifest.json').read_text())
    expected_config = dict(model_id=trainer.MODEL_ID, revision=trainer.REVISION, clip_frames=trainer.CLIP,
                           image_size=trainer.SIZE, stride_frames=trainer.STRIDE, cameras=list(trainer.CAMERAS),
                           joints=list(trainer.JOINTS), labels=list(trainer.LABELS), boundaries=list(trainer.BOUNDARIES),
                           causal=True, seed=trainer.SEED)
    if any(original.get(k) != v for k, v in expected_config.items()):
        raise ValueError('Reference encoder, inputs or sampling configuration changed')
    config = json.loads((label_output / 'config.json').read_text())
    inputs = json.loads((label_output / 'inputs.json').read_text())
    annotations = label_output / 'doubao_annotations.json'
    expected_ids = {e['episode'] for e in original['episodes']}
    if len(expected_ids) != len(original['episodes']):
        raise ValueError('Duplicate reference episode')
    if config['dataset'] != original['dataset'] or config['annotations_sha256'] != original['annotations_sha256']:
        raise ValueError('Doubao/reference dataset provenance mismatch')
    missing = [i for i in sorted(expected_ids) if not (label_output / 'predictions' / f'{i:06d}.json').is_file()]
    if missing:
        raise ValueError(f'Annotation not finished: {len(missing)} episodes pending; no training started')
    manifest = deepcopy(original)
    manifest.update(annotations=str(annotations), annotations_sha256=trainer.digest(annotations), label_source='豆包',
                    reference_manifest=str(previous / 'manifest.json'),
                    reference_manifest_sha256=trainer.digest(previous / 'manifest.json'),
                    label_policy='Doubao only for train/val/test; human labels used only in separate final test evaluation',
                    feature_reuse='Original frozen encoder features, revalidated and relabeled; new head initialized randomly')
    episodes, excluded, features = [], [], {}
    root = Path(original['dataset'])
    assignments = {}
    for e in original['episodes']:
        assignments.setdefault(e['source'], set()).add(e['split'])
        if len(assignments[e['source']]) != 1:
            raise ValueError('Reference recording leaked across splits')
        i = e['episode']
        info = inputs[str(i)]
        prediction = json.loads((label_output / 'predictions' / f'{i:06d}.json').read_text())
        if (prediction['episode'] != i or prediction['input_fingerprint'] != info['fingerprint']
                or info['frames'] != e['frames'] or info['video_sha256'] != e['video_sha256']):
            raise ValueError(f'Prediction/reference input mismatch: {i}')
        boundaries = doubao.parse_prediction(json.dumps(dict(schema_version=1, boundaries=prediction['boundaries'])),
                                             info['last_frame_s'])
        if any(boundaries[k]['time_s'] is None for k in trainer.BOUNDARIES):
            excluded.append(dict(episode=i, reason='Doubao missing boundary; human labels not substituted'))
            continue
        updated = dict(e, boundaries=[boundaries[k]['time_s'] for k in trainer.BOUNDARIES])
        videos, parquet = trainer.paths(root, i)
        if [trainer.digest(v) for v in videos] != e['video_sha256'] or trainer.digest(parquet) != e['parquet_sha256']:
            raise ValueError(f'Source data changed since feature extraction: {i}')
        features[i] = relabel_cache(previous / 'features' / f'{i:06d}.npz', e, updated)
        episodes.append(updated)
        trainer.progress(run, 'relabeling', checked=len(episodes), total=len(original['episodes']), episode=i)
    splits = Counter(e['split'] for e in episodes)
    if any(splits[s] == 0 for s in ('train', 'val', 'test')):
        raise ValueError('No usable Doubao labels in a required split')
    train_counts = np.bincount(np.concatenate([features[e['episode']]['labels'] for e in episodes if e['split']=='train']), minlength=6)
    if np.any(train_counts == 0):
        raise ValueError('A training phase has no Doubao-labeled samples')
    manifest['episodes'] = episodes
    manifest['excluded_episodes'] = excluded
    if not any(e['episode'] == manifest['demo_episode'] and e['split'] == 'test' for e in episodes):
        manifest['demo_episode'] = next(e['episode'] for e in episodes if e['split'] == 'test')
    manifest_path = run / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Training inputs changed; use a new run directory')
    trainer.write_json(manifest_path, manifest)
    for i, data in features.items():
        out = run / 'features' / f'{i:06d}.npz'
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **data)
        tmp.replace(out)
    pretrained = run / 'pretrained'
    if pretrained.is_symlink():
        if pretrained.resolve() != (previous / 'pretrained').resolve():
            raise ValueError('Pretrained model link points to another model')
    elif pretrained.exists():
        raise ValueError('Unexpected pretrained directory; use a fresh run directory')
    else:
        pretrained.symlink_to((previous / 'pretrained').resolve(), target_is_directory=True)
    training_config = json.loads((previous / 'training_config.json').read_text())
    training_config.update(label_source='Doubao', annotations=str(annotations), splits=dict(splits),
                           train_class_counts=train_counts.tolist(), excluded_episodes=excluded,
                           human_reference='Final held-out test evaluation only; never checkpoint selection')
    trainer.write_json(run / 'training_config.json', training_config)
    for source in (Path(trainer.__file__), Path(doubao.__file__), Path(__file__)):
        shutil.copy2(source, run / source.name)
    trainer.progress(run, 'prepared', episodes=len(episodes), splits=dict(splits), excluded=len(excluded),
                     demo_episode=manifest['demo_episode'])
    return manifest, original


def evaluate_human_test(run, manifest, original):
    saved, head = trainer.load_head(run)
    old = {e['episode']: e for e in original['episodes']}
    predictions, truths, per_episode = [], [], []
    for e in manifest['episodes']:
        if e['split'] != 'test':
            continue
        with np.load(run / 'features' / f'{e["episode"]:06d}.npz', allow_pickle=False) as data:
            logits = trainer.predict(head, data['features'], saved['mean'], saved['std'])
            truth = trainer.labels_at(data['times'], old[e['episode']]['boundaries'])
        predictions.append(logits)
        truths.append(truth)
        per_episode.append(dict(episode=e['episode'], **trainer.scores(logits, truth)))
    report = dict(reference='Human annotations, held-out original recording groups only',
                  checkpoint_selected_by='Doubao validation macro F1, tie-break NLL; human test not used for selection',
                  episodes=len(per_episode), per_episode=per_episode,
                  aggregate=trainer.scores(np.concatenate(predictions), np.concatenate(truths)))
    trainer.write_json(run / 'human_test_evaluation.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'prepare'], nargs='?', default='run')
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--reference-run', type=Path, default=trainer.RUN)
    parser.add_argument('--label-output', type=Path, default=doubao.OUTPUT)
    a = parser.parse_args(argv)
    a.run, a.reference_run, a.label_output = (p.expanduser().resolve() for p in (a.run, a.reference_run, a.label_output))
    a.run.mkdir(parents=True, exist_ok=True)
    with (a.run / '.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another pipeline process is using this run directory')
        try:
            if a.mode == 'run':
                original = json.loads((a.reference_run / 'manifest.json').read_text())
                trainer.progress(a.run, 'annotating', label_status=str(a.label_output / 'status.json'))
                result = doubao.main(['annotate', '--dataset', original['dataset'], '--annotations', original['annotations'],
                                      '--output', str(a.label_output)])
                if result:
                    trainer.progress(a.run, 'interrupted' if result == 130 else 'failed', annotation_exit_code=result)
                    return result
            manifest, original = prepare_training(a.run, a.reference_run, a.label_output)
            if a.mode == 'prepare':
                return 0
            if trainer.main(['run', '--run', str(a.run), '--dataset', manifest['dataset'],
                             '--annotations', manifest['annotations']]):
                raise RuntimeError('V-JEPA 2 training failed')
            report = evaluate_human_test(a.run, manifest, original)
            trainer.progress(a.run, 'complete', human_test=report['aggregate'],
                             checkpoint=str(a.run / 'best_head.pt'),
                             video=str(a.run / f'test_episode_{manifest["demo_episode"]:06d}_phases.mp4'))
            return 0
        except BaseException as exc:
            trainer.progress(a.run, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                             error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
