import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools import tomato_dataset as ds
from tools import build_tomato_annotation_index as indexer
from tomato_helpers import make_episode


def test_labels_repeat_half_open_and_full_coverage():
    assert ds.LABELS == ('pickup', 'harvest', 'return', 'adjust')
    segments = [dict(start_frame=i * 3, end_frame=(i + 1) * 3, label=l, reviewed=True)
                for i, l in enumerate((*ds.LABELS, 'pickup', 'harvest'))]
    ds.validate_segments(segments, 18)
    assert ds.label_at_frame(segments, 2) == 0
    assert ds.label_at_frame(segments, 3) == 1
    assert ds.label_at_frame(segments, 12) == 0
    assert ds.label_at_frame(segments, 17) == 1
    for change in (dict(start_frame=1), dict(end_frame=0), dict(end_frame=4), dict(label='unknown'),
                   dict(reviewed=False), dict(reviewed=1), dict(end_frame=19)):
        bad = copy.deepcopy(segments); bad[0].update(change)
        with pytest.raises(ValueError):
            ds.validate_segments(bad, 18)
    with pytest.raises(ValueError):
        ds.validate_segments(segments[:-1], 18)


def test_30hz_causal_sampling_and_fixed_anchors():
    times = np.arange(600) / 30
    np.testing.assert_array_equal(ds.sample_causal_indices(times, 450), np.arange(405, 451, 3))
    np.testing.assert_array_equal(ds.sample_causal_indices(times, 0), np.zeros(16))
    for i in range(len(times)):
        ids = ds.sample_causal_indices(times, i)
        assert len(ids) == 16 and np.all(ids <= i) and ids[-1] == i
    np.testing.assert_array_equal(ds.iter_phase_anchors(times), np.arange(0, 600, 15))
    irregular = np.array([0., .03, .061, .101, .137, .16, .2])
    ids = ds.sample_causal_indices(irregular, 6, history_frames=3)
    assert irregular[ids[1]] <= .1 and ids[-1] == 6
    with pytest.raises(ValueError):
        ds.sample_causal_indices(times, 600)


def test_schema_positions_only_explicit_hands_or_arms_only(tmp_path):
    e, _, names = make_episode(tmp_path / 'data')
    assert ds.resolve_joint_schema(e['info'], names) == list(range(16))
    assert ds.resolve_joint_schema(e['info'], names[:14], 'arms-only') == list(range(14))
    for bad in ([], names + [names[0]], names[:14], ['action'], names[:-1] + ['right_thumb_IP.velocity']):
        with pytest.raises(ValueError):
            ds.resolve_joint_schema(e['info'], bad)


def test_real_decode_timeline_state_ignores_actions_and_task_columns(tmp_path):
    e, ann, names = make_episode(tmp_path / 'data')
    assert ds.validate_source_and_media(e, names)['measured_finite']
    a = ds.load_reviewed_coarse(ann, e)
    assert a['revision'] == 1
    t, q = ds.read_state(e, names)
    assert q.shape == (120, 16) and np.isfinite(q).all()
    anchors = ds.iter_phase_anchors(t)
    assert [ds.label_at_frame(a['segments'], int(i)) for i in anchors] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize('change', [dict(stage='fine'), dict(revision=0), dict(saved_at=None),
                                  dict(inherited=True), dict(draft=True), dict(published=False),
                                  dict(status='draft'), dict(outcome='failure')])
def test_rejects_unsaved_unreviewed_and_draft(tmp_path, change):
    e, ann, _ = make_episode(tmp_path / 'data', length=12)
    doc = ds.read_json(ann); doc.update(change); ds.write_json(ann, doc)
    with pytest.raises(ValueError):
        ds.load_reviewed_coarse(ann, e)


def test_hash_source_mismatch_and_lock_selection(tmp_path):
    e, ann, _ = make_episode(tmp_path / 'data', length=12)
    selection = tmp_path / 'selection.json'
    row = dict(dataset_root=e['dataset_root'], episode_index=0, coarse_annotation=str(ann), variant='normal')
    ds.write_json(selection, [row])
    result = indexer.lock_selection(selection, tmp_path / 'out/index.json')
    assert result[0]['annotation_sha256'] == ds.digest(ann)
    with pytest.raises(ValueError, match='exists'):
        indexer.lock_selection(selection, tmp_path / 'out/index.json')
    ds.write_json(ann.with_name('episode_000000_r2.json'), ds.read_json(ann))
    assert all(r['candidate_count'] == 2 for r in indexer.scan([e['dataset_root']]))
    ds.write_json(selection, [row, row])
    with pytest.raises(ValueError, match='Duplicate'):
        ds.load_annotation_index(selection)
    (tmp_path / 'data/meta/tasks.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        ds.load_reviewed_coarse(ann, e)


def test_multi_dataset_same_episode_is_distinct_and_sharpa_source_required(tmp_path):
    a, _, _ = make_episode(tmp_path / 'a', length=12)
    b, _, _ = make_episode(tmp_path / 'b', length=12)
    assert a['episode_id'] != b['episode_id']
    info = ds.read_json(tmp_path / 'a/meta/info.json'); info['robot_type'] = 'dual_fr3_wuji'
    ds.write_json(tmp_path / 'a/meta/info.json', info)
    with pytest.raises(ValueError, match='Wuji'):
        ds.load_episode_info(tmp_path / 'a', 0)
    p = tmp_path / 'b/meta/episodes.jsonl'; row = json.loads(p.read_text()); del row['source_recording_id']
    p.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='source_recording_id'):
        ds.load_episode_info(tmp_path / 'b', 0)


@pytest.mark.parametrize('failure', ['times', 'joints', 'length', 'missing', 'pts', 'schema', 'collision'])
def test_corruption_rejected(tmp_path, failure):
    e, _, names = make_episode(tmp_path / 'data', length=12)
    if failure in ('times', 'joints'):
        table = pq.read_table(e['parquet']).to_pydict()
        if failure == 'times': table['timestamp'][4] = .9
        else: table['observation.state'][4][0] = float('nan')
        pq.write_table(pa.table(table), e['parquet'])
    elif failure == 'length': e['length'] += 1
    elif failure == 'schema': e['info']['features']['observation.state']['names'].reverse()
    elif failure == 'missing':
        from pathlib import Path
        Path(e['videos'][1]).unlink()
    elif failure == 'pts': e['fps'] = 25
    elif failure == 'collision':
        info = e['info']; info['video_path'] = 'videos/chunk-000/observation.images.cam0/episode_{episode_index:06d}.mp4'
        ds.write_json(tmp_path / 'data/meta/info.json', info)
    if failure == 'schema':
        # Across-dataset schema equality is checked in prepare; reordered whitelist resolves by name.
        assert ds.resolve_joint_schema(e['info'], names) == list(range(15, -1, -1))
        return
    with pytest.raises((ValueError, OSError)):
        if failure == 'collision': ds.load_episode_info(tmp_path / 'data', 0)
        else: ds.validate_source_and_media(e, names)


def test_protected_outputs_and_no_speedup(tmp_path):
    e, _, _ = make_episode(tmp_path / 'data', length=12)
    with pytest.raises(ValueError, match='protected'):
        ds.ensure_output_outside_sources(tmp_path / 'data/run', [e['dataset_root']])
    p = tmp_path / 'index.json'
    ds.write_json(p, [dict(dataset_root=e['dataset_root'], episode_index=0, coarse_annotation='x', variant='speedup')])
    with pytest.raises(ValueError, match='speedup'):
        ds.load_annotation_index(p)


def test_missing_observed_hands_requires_explicit_arms_only(tmp_path):
    e, _, names = make_episode(tmp_path / 'data', length=12)
    info = copy.deepcopy(e['info'])
    info['features']['observation.state'].update(names=names[:14], shape=[14])
    with pytest.raises(ValueError, match='arms-only'):
        ds.resolve_joint_schema(info, names[:14])
    assert len(ds.resolve_joint_schema(info, names[:14], 'arms-only')) == 14


def test_source_timestamp_gap_and_speedup_mapping_are_rejected(tmp_path):
    e, _, names = make_episode(tmp_path / 'data', length=12)
    table = pq.read_table(e['parquet']).to_pydict()
    stamps = (np.arange(12) / 30 * 1e9).astype(np.int64) + 100
    stamps[5] += 10_000_000
    table['source_timestamp_ns'] = stamps
    pq.write_table(pa.table(table), e['parquet'])
    with pytest.raises(ValueError, match='source observation'):
        ds.read_state(e, names)
    ds.write_json(tmp_path / 'data/processing/episode_000000.json', {'action_source_index': [1, 2]})
    with pytest.raises(ValueError, match='speedup'):
        ds.load_episode_info(tmp_path / 'data', 0)


def test_actual_video_pts_offset_is_rejected(tmp_path):
    import av
    from fractions import Fraction
    e, _, names = make_episode(tmp_path / 'data', length=12)
    with av.open(e['videos'][0], mode='w') as c:
        stream = c.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')
            frame.pts = i + 1; frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame): c.mux(packet)
        for packet in stream.encode(): c.mux(packet)
    with pytest.raises(ValueError, match='PTS/Parquet'):
        ds.validate_source_and_media(e, names)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_p1_integer_sampler_survives_timestamp_roundtrip(dtype):
    t = (np.arange(6000) / 30).astype(dtype).astype(np.float64)
    for anchor in (0, 2, 15, 450, 5990):
        expected = np.maximum(0, anchor - np.arange(15, -1, -1) * 3)
        np.testing.assert_array_equal(ds.sample_causal_indices(t, anchor), expected)
    if dtype == np.float32:
        assert not np.array_equal(ds.sample_causal_indices(t, 450, sampler_version=ds.LEGACY_SAMPLER_VERSION), np.arange(405, 451, 3))


def test_non30_sampler_has_explicit_integer_or_causal_fallback():
    for fps in (10, 25, 60):
        t = np.arange(100) / fps
        ids = ds.sample_causal_indices(t, 70)
        assert len(ids) == 16 and ids[-1] == 70 and np.all(t[ids] <= t[70])
        if fps % 10 == 0:
            np.testing.assert_array_equal(ids, np.maximum(0, 70 - np.arange(15, -1, -1) * (fps // 10)))
    t = (np.arange(600) / 30).astype(np.float32).astype(np.float64)
    past = ds.sample_causal_indices(t, 450)
    t[451:] += np.arange(1, 150) * .001
    np.testing.assert_array_equal(ds.sample_causal_indices(t, 450), past)


@pytest.mark.parametrize('horizon,offset', [(.5, 15), (1., 30), (2., 60)])
def test_future_targets_exact_frames_mask_and_half_open(horizon, offset):
    segments = [dict(start_frame=a, end_frame=b, label=l, reviewed=True)
                for a, b, l in [(0, 180, 'pickup'), (180, 300, 'harvest'), (300, 360, 'return'),
                                (360, 420, 'adjust'), (420, 600, 'pickup')]]
    anchors = np.array([0, 179 - offset, 180 - offset, 420 - offset, 580, 599])
    targets = ds.build_phase_targets(anchors, segments, 600, 30, horizon)
    assert targets['future_labels'][1:4].tolist() == [0, 1, 0]
    assert targets['future_target_frames'][-1] == -1 and targets['future_labels'][-1] == -100
    assert not targets['future_valid_mask'][-1] and np.isnan(targets['future_target_times_s'][-1])
    assert ds.serializable_targets(targets)['future_target_times_s'][-1] is None
    assert targets['current_labels'].tolist() == [ds.label_at_frame(segments, int(i)) for i in anchors]
    with pytest.raises(ValueError, match='exact positive integer'):
        ds.build_phase_targets(anchors, segments, 600, 30, .101)


def test_metadata_policy_function_is_unchanged_from_authorized_baseline():
    import ast
    import subprocess
    before = subprocess.check_output(['git', 'show', '11d2b57358b382aa638264303ebfac582d743029:tools/tomato_dataset.py'], cwd=ds.REPO, text=True)
    after = (ds.REPO / 'tools/tomato_dataset.py').read_text()
    def policy(text):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'load_reviewed_coarse'))
    assert policy(before) == policy(after)


def test_subset_metadata_exception_records_actual_hash_and_keeps_data_strict(tmp_path):
    from tools import tomato_phase_model as model
    e, ann, _ = make_episode(tmp_path / 'data', length=12)
    info_path = tmp_path / 'data/meta/info.json'
    info = ds.read_json(info_path); info['subset_note'] = 'exported subset'; ds.write_json(info_path, info)
    e = ds.load_episode_info(tmp_path / 'data', 0)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        ds.load_reviewed_coarse(ann, e)
    reviewed = ds.load_reviewed_coarse(ann, e, True)
    assert reviewed['allowed_metadata_mismatches'] == ['meta/info.json']
    e['annotation'] = reviewed
    e['input_sha256'] = {str(Path(p).relative_to(e['dataset_root'])): ds.digest(p) for p in [e['parquet'], *e['videos']]}
    model.assert_source_unchanged(e)
    info['subset_note'] = 'changed again'; ds.write_json(info_path, info)
    with pytest.raises(ValueError, match='Source changed'):
        model.assert_source_unchanged(e)
    Path(e['parquet']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='parquet'):
        ds.load_reviewed_coarse(ann, e, True)
