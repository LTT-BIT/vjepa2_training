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
    splits = grouped_split(episodes, config['seed'], ds.read_json(config['split_file']) if config.get('split_file') else None)
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
                    annotation_index_sha256=ds.digest(config['annotation_index']),
                    sampler_version=ds.SAMPLER_VERSION, episodes=episodes, splits=splits, summary=summaries,
                    warnings=['Fewer than 10 source recordings; smoke-scale result only'] if len({e['source_recording_id'] for e in episodes}) < 10 else [],
                    label_order=list(ds.LABELS), causal=True, normalization='train-only',
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'extract', 'train'])
    p.add_argument('--config', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--batch-windows', type=int)
    a = p.parse_args(argv)
    config = ds.load_config(a.config)
    if a.device:
        config['device'] = a.device
    if a.batch_windows:
        config['batch_windows'] = a.batch_windows
    try:
        if a.command == 'prepare':
            prepare(config)
        else:
            raise RuntimeError('Extraction/training is implemented in the next implementation stage')
    except Exception as exc:
        # Only write after output isolation has been established by prepare.
        out = Path(config['output_dir'])
        if (out / 'manifest.json').exists() or (out / 'prepare_rejections.json').exists():
            ds.status(out, 'failed', command=a.command, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
