import copy
import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from tools import tomato_dataset as ds
from tools import train_tomato_vjepa_phase as tr
from tools import eval_tomato_vjepa_phase as ev
from tomato_helpers import make_config
from test_train_tomato_vjepa_phase import synthetic_feature_index


def test_heldout_guard_and_replay_causal_staleness():
    e = dict(episode_id='locked', split='test', source_recording_id='test-source')
    manifest = dict(episodes=[e, dict(episode_id='train', split='train', source_recording_id='train-source')])
    assert ev.held_out_episode(manifest)['episode_id'] == 'locked'
    leaked = copy.deepcopy(manifest); leaked['episodes'][1]['source_recording_id'] = 'test-source'
    with pytest.raises(ValueError, match='leaked'):
        ev.held_out_episode(leaked)
    with pytest.raises(ValueError):
        ev.held_out_episode(manifest, 'train')
    schedule = ev.replay_schedule(31, [0, 15, 30])
    np.testing.assert_array_equal(schedule[:15], np.zeros(15))
    assert schedule[15] == 1 and schedule[30] == 2
    with pytest.raises(ValueError):
        ev.replay_schedule(30, [15])


def test_synthetic_complete_head_replay_and_predictions_without_encoder_accuracy_claim(tmp_path):
    torch.set_num_threads(2)
    cfg = make_config(tmp_path); m = tr.prepare(cfg)
    synthetic_feature_index(cfg, m)
    tr.train(cfg, m)
    report, errors = ev.evaluate(cfg, video=True, measure=False)
    assert not errors and report['samples'] == 8
    out = Path(cfg['output_dir']) / 'offline_eval'
    replay = ds.read_json(out / 'replay.json')
    assert replay['frames'] == 120
    with av.open(replay['path']) as c:
        frames = list(c.decode(video=0))
    assert len(frames) == 120 and frames[0].width == 768 and frames[0].height == 476
    traces = [json.loads(line) for line in Path(replay['path']).with_suffix('.frames.jsonl').read_text().splitlines()]
    assert not traces[0]['prediction_held'] and traces[1]['prediction_held']
    assert not traces[15]['prediction_held']
    assert all(r['last_input_frame'] <= r['prediction_frame'] <= r['frame_index'] for r in traces)
    assert not ds.read_json(out / 'latency.json')['measured']


def test_replay_or_benchmark_failure_does_not_hide_metrics(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = make_config(tmp_path); m = tr.prepare(cfg)
    synthetic_feature_index(cfg, m); tr.train(cfg, m)
    def broken(*args, **kwargs):
        raise RuntimeError('deliberately unavailable dependency')
    monkeypatch.setattr(ev, 'render_replay', broken)
    monkeypatch.setattr(ev, 'benchmark', broken)
    report, errors = ev.evaluate(cfg)
    out = Path(cfg['output_dir']) / 'offline_eval'
    assert report['samples'] == 8 and (out / 'evaluation.json').exists()
    assert {e['component'] for e in errors} == {'replay', 'latency'}
    assert ds.read_json(Path(cfg['output_dir']) / 'status.json')['stage'] == 'offline_evaluation_with_errors'


def test_prediction_wrapper_rejects_future_or_invalid_probabilities(tmp_path):
    cfg = make_config(tmp_path); m = tr.prepare(cfg); synthetic_feature_index(cfg, m)
    entries = tr.load_training_arrays(cfg, m); entry = entries['test'][0]
    p = np.full((8, 4), .25)
    assert list(tr.prediction_rows(entry, p, 'test')[0]['probabilities']) == list(ds.LABELS)
    with pytest.raises(ValueError):
        tr.prediction_rows(entry, p * 2, 'test')
    entry['input_indices'][0, -1] = 1
    with pytest.raises(ValueError, match='Future'):
        tr.prediction_rows(entry, p, 'test')


def test_cv_summary_requires_all_folds_and_no_outer_leakage(tmp_path):
    with pytest.raises(ValueError, match='No completed'):
        ev.summarize_cv(tmp_path)
    groups = [dict(source_recording_id=f'source-{i}') for i in range(6)]
    folds = tr.grouped_folds(groups, 3)
    for k, split in enumerate(folds):
        folder = tmp_path / f'fold_{k}'
        ds.write_json(folder / 'evaluation.json', {'splits': {'test': dict(accuracy=.5, macro_f1=.4, nll=1.)}})
        ds.write_json(folder / 'manifest.json', dict(cross_validation=dict(folds=3, fold_index=k),
                      contract={}, episodes=[dict(episode_id=str(i), source_recording_id=f'source-{i}', annotation={'sha256': str(i)}) for i in range(6)], splits=split))
    result = ev.summarize_cv(tmp_path)
    assert result['metrics']['macro_f1']['mean'] == pytest.approx(.4)
    assert result['metrics']['macro_f1']['std'] == pytest.approx(0)
    p = tmp_path / 'fold_1/manifest.json'; m = ds.read_json(p)
    m['splits']['test'] = folds[0]['test']; ds.write_json(p, m)
    with pytest.raises(ValueError, match='coverage'):
        ev.summarize_cv(tmp_path)


def test_eval_uses_auto_discovered_run_index(tmp_path):
    torch.set_num_threads(2)
    cfg = make_config(tmp_path)
    roots = [r['dataset_root'] for r in ds.read_json(cfg['annotation_index'])]
    generated_cfg = dict(cfg)
    tr.setup_discovered_index(generated_cfg, roots)
    manifest = tr.prepare(generated_cfg)
    synthetic_feature_index(generated_cfg, manifest)
    tr.train(generated_cfg, manifest)
    report, errors = ev.evaluate(cfg, video=False, measure=False)
    assert report['samples'] == 8 and not errors
