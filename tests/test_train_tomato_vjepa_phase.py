import copy
from pathlib import Path

import pytest
import numpy as np
import torch

from tools import tomato_dataset as ds
from tools import train_tomato_vjepa_phase as tr
from tomato_helpers import make_config


def synthetic_feature_index(cfg, manifest):
    from tools import tomato_phase_model as model
    rows = []
    for e in manifest['episodes']:
        t, q = ds.read_state(e, cfg['joint_names'])
        anchors = ds.iter_phase_anchors(t)
        histories = np.stack([ds.sample_causal_indices(t, int(i)) for i in anchors])
        # Explicit synthetic test fixture; no Encoder or model accuracy claim.
        x = np.random.default_rng(42).normal(size=(len(anchors), 3072 + 16 * len(cfg['joint_names']))).astype(np.float32)
        key = model.feature_fingerprint(e, manifest['contract'], manifest['pretrained'])
        path = Path(cfg['output_dir']).parent / 'synthetic_cache' / (key + '.npz')
        model.write_cache(path, x, anchors, t[anchors], histories, key)
        rows.append(dict(episode_id=e['episode_id'], path=str(path), synthetic_only=True))
    ds.write_json(Path(cfg['output_dir']) / 'feature_index.json',
                  dict(manifest_sha256=ds.digest(Path(cfg['output_dir']) / 'manifest.json'),
                       pretrained=manifest['pretrained'], episodes=rows))
    return rows


def test_group_split_reproducible_and_source_locked():
    episodes = [dict(source_recording_id=f'group-{i // 2}') for i in range(30)]
    split = tr.grouped_split(episodes)
    assert split == tr.grouped_split(list(reversed(episodes)))
    assert len({g for rows in split.values() for g in rows}) == 15
    with pytest.raises(ValueError):
        tr.grouped_split([dict(source_recording_id='same')] * 5)
    bad = copy.deepcopy(split); bad['test'].append(bad['train'][0])
    with pytest.raises(ValueError, match='leakage'):
        tr.grouped_split(episodes, explicit=bad)


def test_prepare_locks_full_manifest_no_gpu_and_rejects_changes(tmp_path):
    cfg = make_config(tmp_path)
    m = tr.prepare(cfg)
    assert m == tr.prepare(cfg)
    assert set(m['summary']) == {'train', 'val', 'test'}
    assert all(s['class_counts'] == [2, 2, 2, 2] for s in m['summary'].values())
    assert len({g for rows in m['splits'].values() for g in rows}) == 3
    old = ds.digest(Path(cfg['output_dir']) / 'manifest.json')
    ann = Path(m['episodes'][0]['annotation']['path'])
    doc = ds.read_json(ann); doc['revision'] += 1; ds.write_json(ann, doc)
    with pytest.raises(ValueError, match='Locked'):
        tr.prepare(cfg)
    assert ds.digest(Path(cfg['output_dir']) / 'manifest.json') == old


def test_prepare_rejects_source_hash_and_full_schema_change(tmp_path):
    cfg = make_config(tmp_path)
    p = Path(ds.read_json(cfg['annotation_index'])[0]['dataset_root']) / 'meta/info.json'
    doc = ds.read_json(p); doc['unrelated_export_note'] = True; ds.write_json(p, doc)
    with pytest.raises(ValueError, match='invalid selected'):
        tr.prepare(cfg)
    rejected = ds.read_json(Path(cfg['output_dir']) / 'prepare_rejections.json')
    assert 'SHA256 mismatch' in rejected['rejected'][0]['error']
    assert not (Path(cfg['output_dir']) / 'manifest.json').exists()


def test_grouped_cv_outer_test_never_used_for_inner_validation():
    episodes = [dict(source_recording_id=f'group-{i // 2}') for i in range(52)]
    folds = tr.grouped_folds(episodes)
    assert folds == tr.grouped_folds(list(reversed(episodes)))
    test_groups = []
    for fold in folds:
        sets = [set(fold[s]) for s in ('train', 'val', 'test')]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(set.union(*sets)) == 26
        test_groups += fold['test']
    assert len(test_groups) == len(set(test_groups)) == 26
    with pytest.raises(ValueError, match='enough'):
        tr.grouped_folds(episodes[:6])


def test_normalization_class_weights_train_only():
    x = np.arange(48, dtype=np.float32).reshape(8, 6)
    y = np.array([0, 1, 1, 1, 2, 2, 3, 3])
    mu, sigma, weight = tr.fit_training_statistics(x, y)
    np.testing.assert_allclose(mu, x.mean(axis=0))
    expected = np.sqrt(8 / np.bincount(y)); expected /= expected.mean()
    np.testing.assert_allclose(weight, expected, rtol=1e-6)
    val_test_outliers = np.full((10, 6), 1e30)
    np.testing.assert_array_equal(tr.fit_training_statistics(x, y)[0], mu)
    assert not np.allclose(np.concatenate([x, val_test_outliers]).mean(axis=0), mu)
    with pytest.raises(ValueError, match='four classes'):
        tr.fit_training_statistics(x, np.zeros(8, dtype=np.int64))


def test_synthetic_head_training_exports_complete_metrics_and_checkpoint(tmp_path):
    from tools import tomato_phase_model as model
    torch.set_num_threads(2)
    cfg = make_config(tmp_path)
    manifest = tr.prepare(cfg)
    synthetic_feature_index(cfg, manifest)
    result = tr.train(cfg, manifest)
    assert set(result['splits']) == {'train', 'val', 'test'}
    for s in result['splits'].values():
        assert np.array(s['confusion_true_rows']).shape == (4, 4)
        assert set(s['per_class']) == set(ds.LABELS)
        assert 'delay_p95_s' in s['temporal']
        assert len(s['per_source']) == 1
    run = Path(cfg['output_dir'])
    saved, head = model.load_checkpoint(run / 'best_head.pt', ds.digest(run / 'manifest.json'))
    entries = tr.load_training_arrays(cfg, manifest)
    train_x = np.concatenate([e['features'] for e in entries['train']])
    np.testing.assert_allclose(saved['mean'], train_x.mean(axis=0, dtype=np.float64), atol=1e-6)
    assert saved['label_order'] == list(ds.LABELS)
    rows = [__import__('json').loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 24
    assert all(r['last_input_frame'] <= r['frame_index'] for r in rows)
    assert all(sum(r['probabilities'].values()) == pytest.approx(1) for r in rows)
    assert (run / 'training_config.json').is_file() and (run / 'predictions.csv').is_file()
    with pytest.raises(ValueError, match='already exist'):
        tr.train(cfg, manifest)


def test_new_annotation_rebuilds_y_and_reuses_only_features(tmp_path):
    from tools import tomato_phase_model as model
    cfg = make_config(tmp_path); old = tr.prepare(cfg)
    old_paths = synthetic_feature_index(cfg, old)
    for e in old['episodes']:
        p = Path(e['annotation']['path']); doc = ds.read_json(p)
        doc['revision'] += 1
        doc['segments'][0]['end_frame'] = 15
        doc['segments'][1]['start_frame'] = 15
        ds.write_json(p, doc)
    cfg['output_dir'] = str(tmp_path / 'new-run')
    new = tr.prepare(cfg)
    for a, b in zip(old['episodes'], new['episodes']):
        assert model.feature_fingerprint(a, old['contract'], old['pretrained']) == model.feature_fingerprint(b, new['contract'], new['pretrained'])
    ds.write_json(Path(cfg['output_dir']) / 'feature_index.json',
                  dict(manifest_sha256=ds.digest(Path(cfg['output_dir']) / 'manifest.json'),
                       pretrained=new['pretrained'], episodes=old_paths))
    entries = tr.load_training_arrays(cfg, new)
    assert all(r['labels'][1] == 1 for parts in entries.values() for r in parts)
