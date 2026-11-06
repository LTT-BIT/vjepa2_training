import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import tomato_dataset as ds, tomato_phase_model as model, tomato_metrics as metrics
from tools import train_tomato_vjepa_phase as tr, eval_tomato_vjepa_phase as ev
from tomato_helpers import make_config
from test_train_tomato_vjepa_phase import synthetic_feature_index


def test_three_modes_train_eval_share_split_features_normalization_and_mask(tmp_path):
    torch.set_num_threads(2)
    base = make_config(tmp_path, length=240)
    expected_split = None; paths = None; means = []
    for mode in ('current_only', 'future_only', 'joint'):
        cfg = {**base, **ds.prediction_contract(dict(prediction_mode=mode)), 'output_dir': str(tmp_path / mode)}
        m = tr.prepare(cfg)
        assert m['split_sha256'] == ds.digest(Path(cfg['output_dir']) / 'split.json')
        if expected_split is None: expected_split = m['splits']
        assert m['splits'] == expected_split
        if paths is None: paths = synthetic_feature_index(cfg, m)
        else:
            ds.write_json(Path(cfg['output_dir']) / 'feature_index.json', dict(
                manifest_sha256=ds.digest(Path(cfg['output_dir']) / 'manifest.json'), pretrained=m['pretrained'], episodes=paths))
        for e, p in zip(m['episodes'], paths):
            assert Path(p['path']).stem == model.feature_fingerprint(e, m['contract'], m['pretrained'])
        result = tr.train(cfg, m)
        report, errors = ev.evaluate(cfg, video=False, measure=False)
        assert not errors
        for name in ('current', 'future'):
            assert (report[name] is not None) == (name in ds.active_heads(mode))
            if report[name] is not None:
                assert report[name]['samples'] == (16 if name == 'current' else 14)
                assert report[name]['masked_count'] == (0 if name == 'current' else 2)
                assert report[name]['confusion_true_rows'] == result['splits']['test'][name]['confusion_true_rows']
        saved, head = model.load_checkpoint(Path(cfg['output_dir']) / 'best_head.pt')
        means.append(saved['mean'])
        config = ds.read_json(Path(cfg['output_dir']) / 'training_config.json')
        for name in ds.active_heads(mode):
            counts = m['summary']['train'][name + '_class_counts']
            expected = np.sqrt(sum(counts) / np.array(counts)); expected /= expected.mean()
            np.testing.assert_allclose(config['class_weights_by_head'][name], expected, rtol=1e-6)
        logs = [json.loads(line) for line in (Path(cfg['output_dir']) / 'metrics.jsonl').read_text().splitlines()]
        for row in logs:
            f1, nll = tr.validation_selection(row['val_current'], row['val_future'], cfg)
            assert row['validation_selection_score'] == pytest.approx(f1)
            assert row['validation_selection_nll'] == pytest.approx(nll)
            c, f = row['train_current_ce'], row['train_future_ce']
            expected = c if mode == 'current_only' else f if mode == 'future_only' else (c + .5 * f) / 1.5
            assert row['train_total_loss'] == pytest.approx(expected)
        rows = [json.loads(line) for line in (Path(cfg['output_dir']) / 'predictions.jsonl').read_text().splitlines()]
        if mode == 'future_only': assert all('stage' not in r and 'current_stage' not in r for r in rows)
        else: assert all(r['stage'] == r['current_stage'] for r in rows)
        if mode != 'current_only':
            assert sum(not r['future_valid'] for r in rows) == 6
            assert all(r['future_target_time_s'] is None and r['future_target_frame'] == -1 for r in rows if not r['future_valid'])
        assert all(max(r['input_indices']) <= r['frame_index'] for r in rows)
    for mean in means[1:]: torch.testing.assert_close(means[0], mean, rtol=0, atol=0)


def test_quick_locked_validation_never_decodes_and_rejects_changed_files(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, length=240)
    m = tr.prepare(cfg)
    original = ds.validate_source_and_media
    calls = []
    def observe(*args, **kwargs):
        calls.append(args[0]['episode_id']); return original(*args, **kwargs)
    monkeypatch.setattr(ds, 'validate_source_and_media', observe)
    for _ in range(3): assert tr.load_locked_manifest(cfg) == m
    assert calls == []
    hashes = {n: ds.digest(Path(cfg['output_dir']) / n) for n in ('manifest.json', 'split.json', 'validation_receipt.json', 'validation_lock.json')}
    assert tr.load_locked_manifest(cfg, full_revalidate=True) == m and len(calls) == 3
    assert all(ds.digest(Path(cfg['output_dir']) / n) == h for n, h in hashes.items())
    for key, value in [('future_horizon_s', 2.), ('prediction_mode', 'joint'), ('lambda_current', 2.), ('stable_transition_k', 3)]:
        with pytest.raises(ValueError, match='Locked'):
            tr.load_locked_manifest({**cfg, key: value})
    # Same-size source edits must fail; this is not an mtime/size shortcut.
    video = Path(m['episodes'][0]['videos'][0]); data = bytearray(video.read_bytes()); data[-1] ^= 1; video.write_bytes(data)
    with pytest.raises(ValueError, match='Locked'):
        tr.load_locked_manifest(cfg)


@pytest.mark.parametrize('filename', ['manifest.json', 'split.json', 'validation_receipt.json'])
def test_tampered_locked_documents_are_never_silently_rewritten(tmp_path, filename):
    cfg = make_config(tmp_path)
    tr.prepare(cfg)
    path = Path(cfg['output_dir']) / filename
    path.write_text(path.read_text() + ' ')
    changed = ds.digest(path)
    with pytest.raises(ValueError, match='Locked'):
        tr.load_locked_manifest(cfg, full_revalidate=True)
    assert ds.digest(path) == changed


def test_future_only_missing_class_fails_instead_of_padding_and_joint_batches_are_valid(tmp_path):
    cfg = make_config(tmp_path, length=120)
    cfg['prediction_mode'] = 'future_only'
    with pytest.raises(ValueError, match='future missing four-class'):
        tr.prepare(cfg)
    valid = np.array([True, False, True, False])
    batches = tr.training_batches(np.array([1, -100, 2, -100]), valid, 4, 1, np.random.default_rng(42), 'joint')
    assert sorted(np.concatenate(batches)) == [0, 1, 2, 3]
    assert all(valid[b].any() for b in batches)


def test_run_initializes_before_encoder_and_train_in_one_process(tmp_path, monkeypatch):
    import os
    import yaml
    cfg = make_config(tmp_path, length=240)
    cfg['prediction_mode'] = 'joint'
    roots = [r['dataset_root'] for r in ds.read_json(cfg['annotation_index'])]
    p = tmp_path / 'config.yaml'; p.write_text(yaml.safe_dump(cfg))
    calls = []
    def extract(config, manifest):
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8')
        assert torch.are_deterministic_algorithms_enabled()
        calls.append('encoder'); synthetic_feature_index(config, manifest)
    real_train = tr.train
    def train(config, manifest):
        assert calls == ['encoder']; calls.append('train')
        return real_train(config, manifest)
    monkeypatch.setattr(tr, 'extract', extract); monkeypatch.setattr(tr, 'train', train)
    argv = ['run', '--config', str(p)]
    for root in roots: argv += ['--dataset', root]
    tr.main(argv)
    assert calls == ['encoder', 'train']


def test_extract_reuses_verified_features_without_loading_or_decoding_encoder(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, length=240); m = tr.prepare(cfg)
    paths = synthetic_feature_index(cfg, m)
    cfg['feature_cache_dir'] = str(Path(paths[0]['path']).parent)
    monkeypatch.setattr(model, 'verify_pretrained', lambda *a: m['pretrained'])
    def forbidden(*a, **kw): raise AssertionError('Cached extract reloaded encoder or decoded video')
    monkeypatch.setattr(model, 'load_encoder', forbidden)
    monkeypatch.setattr(model, 'decode_selected', forbidden)
    rows = tr.extract(cfg, m)
    assert all(r['performance']['reused'] for r in rows)


def test_legacy_manifest_cache_checkpoint_evaluate_without_rewriting_locked_files(tmp_path):
    cfg = make_config(tmp_path)
    m = tr.prepare(cfg)
    m['format_version'] = 1; m['sampler_version'] = ds.LEGACY_SAMPLER_VERSION
    m['contract'].pop('sampler_version')
    for key in ('prediction', 'split_sha256', 'external_split_sha256'): m.pop(key)
    run = Path(cfg['output_dir']); ds.write_json(run / 'manifest.json', m)
    (run / 'validation_receipt.json').unlink(); (run / 'validation_lock.json').unlink()
    original_sha = ds.digest(run / 'manifest.json')
    rows = []
    for e in m['episodes']:
        times, q = ds.read_state(e, cfg['joint_names']); anchors = ds.iter_phase_anchors(times)
        history = np.stack([ds.sample_causal_indices(times, int(i), sampler_version=ds.LEGACY_SAMPLER_VERSION) for i in anchors])
        c = tr.feature_contract(m); path = model.cache_path(tmp_path / 'old-cache', e, c, m['pretrained'])
        key = model.feature_fingerprint(e, c, m['pretrained'])
        model.write_cache(path, np.zeros((len(anchors), 3328), np.float32), anchors, times[anchors], history, key)
        rows.append(dict(episode_id=e['episode_id'], path=str(path)))
    ds.write_json(run / 'feature_index.json', dict(manifest_sha256=original_sha, pretrained=m['pretrained'], episodes=rows))
    model.save_checkpoint(run / 'best_head.pt', model.head_model(16), np.zeros(3328, np.float32), np.ones(3328, np.float32), m, original_sha, 1, {})
    saved = torch.load(run / 'best_head.pt', weights_only=True)
    for key in ('prediction', 'prediction_mode', 'future_horizon_s', 'lambda_current', 'lambda_future', 'stable_transition_k', 'split_sha256'): saved.pop(key)
    saved.update(format_version=1, sampler_version=ds.LEGACY_SAMPLER_VERSION)
    torch.save(saved, run / 'best_head.pt')
    checkpoint_sha = ds.digest(run / 'best_head.pt')
    with pytest.raises(ValueError, match='full-revalidate'):
        ev.evaluate(cfg, video=False, measure=False)
    report, errors = ev.evaluate(cfg, video=False, measure=False, full_revalidate=True)
    assert not errors and report['prediction_mode'] == 'current_only' and report['samples'] == 8
    assert ds.digest(run / 'manifest.json') == original_sha and ds.digest(run / 'best_head.pt') == checkpoint_sha
