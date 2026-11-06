"""Offline model data contract checks; run with .venv-phase/bin/python -m pytest."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import lemon_phase_model as phase


def test_boundaries_and_causal_alignment():
    assert phase.labels_at([0, .9, 1, 2, 3, 4, 5], [1, 2, 3, 4]).tolist() == [0, 0, 1, 2, 3, 4, 4]
    with pytest.raises(ValueError):
        phase.labels_at([0], [1, 3, 2, 4])
    with pytest.raises(ValueError):
        phase.labels_at([0], [1, 2, 3, np.nan])
    index, valid = phase.causal_indices(np.array([9, 10, 14, 20, 26]), np.array([10, 20]), 5)
    assert index.tolist() == [-1, 0, 0, 1, 1]
    assert valid.tolist() == [False, True, True, True, False]
    assert phase.labels_at([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]).tolist() == [0, 1, 2, 3, 4, 5, 5]
    assert phase.labels_at([float(np.float32(10.7))], [1, 2, 3, 10.7, 12]).tolist() == [4]


def test_v3_manifest_group_split_and_exclusions(tmp_path, monkeypatch):
    records = tmp_path / 'conversion/records'
    records.mkdir(parents=True)
    rows = []
    for i in range(12):
        video = phase.episode_paths(tmp_path, i)[0][0]
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'video')
        rows.append(dict(episode_index=i, video_path=str(video), video_sha256=hashlib.sha256(b'video').hexdigest(),
                         frame_count=7, status='complete', frame_indices=dict(zip(phase.V3_BOUNDARIES, range(1, 6))),
                         **dict(zip(phase.V3_BOUNDARIES, range(1, 6)))))
        (records / f'{i}.json').write_text(json.dumps({'source': f'bag-{i // 2}', 'episodes': [{'episode_index': i}]}))
    rows[10]['excluded_from_training'] = True
    rows[11]['status'] = 'needs_review'
    annotations = tmp_path / 'annotations.json'
    data = dict(schema_version=3, boundary_fields=list(phase.V3_BOUNDARIES), episodes=rows)
    annotations.write_text(json.dumps(data))
    monkeypatch.setattr(phase, 'read_episode_state', lambda *_: (np.arange(7), np.zeros((7, 12))))
    result = phase.build_manifest(tmp_path, annotations, None)
    assert len(result['episodes']) == 10 and len(result['rejected']) == 2
    assert len(result['labels']) == 6
    assert {r['split'] for r in result['episodes']} == {'train', 'val', 'test'}
    for group in {r['source_recording_id'] for r in result['episodes']}:
        assert len({r['split'] for r in result['episodes'] if r['source_recording_id'] == group}) == 1
    rows[0]['frame_indices']['pour_ready_s'] = 3
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='frame/time mismatch'):
        phase.build_manifest(tmp_path, annotations, None)


def test_manifest_rejects_bad_boundary_and_source_leakage(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    records = root / 'conversion/records'
    records.mkdir(parents=True)
    annotations = tmp_path / 'annotations.tsv'
    with annotations.open('w') as handle:
        handle.write('episode_index\t' + '\t'.join(phase.BOUNDARIES) + '\n')
        handle.write('0\t1\t2\t3\t4\n1\t1\t2\t3\t9\n')
    split_file = tmp_path / 'splits.json'
    split = {'splits': {'train': [0], 'val': [1], 'test': []},
             'sources': {'train': ['bag-a'], 'val': ['bag-b'], 'test': []}}
    split_file.write_text(json.dumps(split))
    for i, source in enumerate(['bag-a', 'bag-b']):
        (records / f'{i}.json').write_text(json.dumps({'source': source, 'episodes': [{'episode_index': i}]}))
    monkeypatch.setattr(phase, 'read_episode_state', lambda *_: (np.arange(6), np.zeros((6, 12))))
    result = phase.build_manifest(root, annotations, split_file)
    assert [r['episode_index'] for r in result['episodes']] == [0]
    assert result['rejected'][0]['episode_index'] == 1
    assert 'legacy' in result['label_semantics']
    split['sources']['val'] = ['bag-a']
    split_file.write_text(json.dumps(split))
    (records / '1.json').write_text(json.dumps({'source': 'bag-a', 'episodes': [{'episode_index': 1}]}))
    with pytest.raises(ValueError, match='leakage'):
        phase.build_manifest(root, annotations, split_file)


@pytest.mark.parametrize('classes', [5, 6])
def test_complete_rosbag_keeps_head_frames_and_marks_gaps(tmp_path, monkeypatch, classes):
    pytest.importorskip('rosbags')
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    if not (phase.CONVERTER / 'src').exists():
        pytest.skip('Local converter reader is required for integration')
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    def header(t):
        return types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](t // 10**9, t % 10**9), 'test')
    bag = tmp_path / 'bag'
    with Writer(bag, version=9) as writer:
        for camera, stamps in [('cam0', [1000, 1100, 1200, 1200, 1500]),
                               ('cam1', [1050, 1150]), ('cam2', [1050, 1150])]:
            kind = 'sensor_msgs/msg/Image'
            conn = writer.add_connection(f'/{camera}/color/image_raw', kind, typestore=store)
            for time in stamps:
                msg = types[kind](header(time * 10**6), 4, 4, 'rgb8', 0, 12, np.full(48, int(camera[-1]), dtype=np.uint8))
                writer.write(conn, time * 10**6, store.serialize_cdr(msg, kind))
        for side, names in [('left', phase.JOINTS[:6]), ('right', phase.JOINTS[6:])]:
            kind = 'sensor_msgs/msg/JointState'
            conn = writer.add_connection(f'/{side}/joint_states', kind, typestore=store)
            for time in [1000, 1100, 1200, 1400]:
                msg = types[kind](header(time * 10**6), list(reversed(names)), np.arange(6, dtype=np.float64)[::-1].copy(),
                                  np.zeros(6), np.zeros(6))
                writer.write(conn, time * 10**6, store.serialize_cdr(msg, kind))
    model = tmp_path / 'model.pt'
    model.write_bytes(b'test-model')
    labels = phase.LABELS if classes == 5 else phase.V3_LABELS
    monkeypatch.setattr(phase, 'load_model', lambda *_: ({'label_semantics': 'legacy' if classes == 5 else phase.V3_SEMANTICS, 'labels': list(labels)}, None, None))
    monkeypatch.setattr(phase, 'encode', lambda encoder, images, device: np.stack([np.full(576, x[0, 0, 0]) for x in images]))
    def check_inputs(saved, head, features, state, device):
        assert np.all(features[:, :576] == 0)
        assert np.all(features[:, 576:1152] == 1)
        assert np.all(features[:, 1152:] == 2)
        np.testing.assert_array_equal(state, np.tile(np.r_[np.arange(6), np.arange(6)], (len(state), 1)))
        probabilities = np.arange(1, classes + 1, dtype=float)
        return np.tile(probabilities / probabilities.sum(), (len(state), 1))
    monkeypatch.setattr(phase, 'probabilities', check_inputs)
    output = tmp_path / 'probs.csv'
    phase.infer_bag(argparse.Namespace(converter_root=phase.CONVERTER, model=model, device='cpu', bag=bag,
                                     output=output, max_age_ms=150, image_batch=6))
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 5
    assert [r['valid'] for r in rows] == ['0', '1', '1', '1', '0']
    assert rows[0]['p0'] == ''
    assert rows[1]['phase'] == str(classes - 1)
    assert ('p5' in rows[0]) == (classes == 6)
    assert rows[2]['header_ns'] == rows[3]['header_ns']
    assert json.loads(output.with_suffix('.meta.json').read_text())['valid_frames'] == 3
