"""Read-only Sharpa normal LeRobot v2.1 + explicitly saved, reviewed coarse labels.

No actions, task indices, annotation-derived sampling, or implicit source repair.
This module and prepare deliberately do not import torch or transformers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import yaml

LABELS = ('pickup', 'harvest', 'return', 'adjust')
CAMERAS = ('cam0', 'cam1', 'cam2')
MODEL_ID = 'facebook/vjepa2-vitl-fpc64-256'
REVISION = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'
SAMPLER_VERSION = 'timestamp-floor-v1'
REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = read_json(path)
    return value if isinstance(value, list) else [value]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def status(run, stage, **kwargs):
    value = dict(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **kwargs)
    write_json(Path(run) / 'status.json', value)
    return value


def ensure_output_outside_sources(output, roots, annotations=()):
    out = Path(output).expanduser().resolve()
    protected = [Path(p).expanduser().resolve() for p in roots]
    protected += [REPO / 'experiment_results', REPO / 'annotations']
    for p in protected:
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError(f'Output overlaps protected input: {out} / {p}')
    for p in annotations:
        p = Path(p).resolve()
        if p == out or p.is_relative_to(out):
            raise ValueError(f'Output overlaps annotation: {p}')
    return out


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    fixed = dict(project='tomato_sharpa_four_stage', robot='sharpa', variant='normal',
                 cameras=list(CAMERAS), labels=list(LABELS), model_id=MODEL_ID,
                 model_revision=REVISION, history_frames=16, history_sample_hz=10,
                 prediction_hz=2, image_size=256)
    for key, value in fixed.items():
        if cfg.get(key) != value:
            raise ValueError(f'Fixed tomato contract {key} must be {value!r}')
    if cfg.get('joint_mode') not in ('arms-and-hands', 'arms-only'):
        raise ValueError('Explicit joint_mode arms-and-hands or arms-only required')
    if not cfg.get('joint_names'):
        raise ValueError('Explicit nonempty joint_names whitelist required')
    for key in ('pretrained_dir', 'annotation_index', 'output_dir'):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg.setdefault('seed', 42)
    cfg.setdefault('device', 'cuda')
    cfg.setdefault('batch_windows', 1)
    cfg.setdefault('epochs', 100)
    cfg.setdefault('patience', 15)
    cfg.setdefault('batch_size', 128)
    cfg.setdefault('learning_rate', 3e-4)
    cfg.setdefault('weight_decay', .01)
    for k in ('batch_windows', 'batch_size', 'epochs', 'patience'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f'{k} must be positive integer')
    return cfg


def resolve_joint_schema(meta_info, configured_joint_names, joint_mode='arms-and-hands'):
    state = meta_info['features']['observation.state']
    names = state.get('names')
    if not isinstance(names, list) or len(names) != len(set(names)) or state['shape'] != [len(names)]:
        raise ValueError('Invalid or duplicate state schema names/shape')
    selected = list(configured_joint_names)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Nonempty, duplicate-free joint_names whitelist required')
    if any(n not in names or not n.endswith('.position') for n in selected):
        raise ValueError('Whitelist must contain existing measured .position names only')
    arms = [f'{side}_fr3_joint{i}.position' for side in ('left', 'right') for i in range(1, 8)]
    hands = [n for n in names if n.endswith('.position') and n not in arms]
    if not set(arms).issubset(selected):
        raise ValueError('Sharpa task requires all 14 observed Franka arm positions')
    if joint_mode == 'arms-only':
        if set(selected) != set(arms):
            raise ValueError('arms-only whitelist must contain exactly 14 arm positions')
    elif joint_mode == 'arms-and-hands':
        if not hands or not set(arms + hands) == set(selected):
            raise ValueError('Missing measured hand positions; explicitly select arms-only if needed')
    else:
        raise ValueError('Unknown joint_mode')
    return [names.index(n) for n in selected]


def safe_source_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Expected safe relative source path: {relative}')
    p = (Path(root) / relative).resolve(strict=True)
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Source path escapes dataset: {p}')
    return p


def load_episode_info(dataset_root, episode_index):
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError('Invalid episode_index')
    info = read_json(root / 'meta/info.json')
    if info.get('codebase_version') != 'v2.1' or info.get('robot_type') != 'dual_fr3_sharpa':
        raise ValueError('Only dual_fr3_sharpa LeRobot v2.1 accepted; Wuji excluded')
    fps = float(info['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    entries = read_rows(root / 'meta/episodes.jsonl')
    if len({e['episode_index'] for e in entries}) != len(entries):
        raise ValueError('Duplicate episode metadata')
    found = [e for e in entries if e['episode_index'] == episode_index]
    if len(found) != 1:
        raise ValueError('Episode missing or ambiguous')
    entry = found[0]
    source = entry.get('source_recording_id')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Missing reliable source_recording_id; no episode fallback')
    if type(entry.get('length')) is not int or entry['length'] < 2:
        raise ValueError('Invalid episode length')
    for key in ('source_start_ns', 'source_end_ns'):
        if key in entry and type(entry[key]) is not int:
            raise ValueError(f'Invalid {key}')
    if ('source_start_ns' in entry and 'source_end_ns' in entry
            and entry['source_start_ns'] >= entry['source_end_ns']):
        raise ValueError('Invalid source timestamp range')
    chunks = info['chunks_size']
    if type(chunks) is not int or chunks < 1:
        raise ValueError('Invalid chunks_size')
    params = dict(episode_index=episode_index, episode_chunk=episode_index // chunks)
    parquet = safe_source_path(root, info['data_path'].format(**params))
    keys = [f'observation.images.{c}' for c in CAMERAS]
    if any(info['features'].get(k, {}).get('dtype') != 'video' for k in keys):
        raise ValueError('Missing required three video camera keys')
    videos = [safe_source_path(root, info['video_path'].format(video_key=k, **params)) for k in keys]
    if len(set([parquet, *videos])) != 4:
        raise ValueError('Camera/data path collision')
    # normal is explicitly asserted by index; reject contrary evidence, never infer from task_index.
    if entry.get('variant', 'normal') != 'normal' or info.get('variant', 'normal') != 'normal':
        raise ValueError('Only normal accepted')
    if (root / 'processing/speedup.json').exists() or 'speedup' in root.name.lower():
        raise ValueError('speedup is excluded from first experiment')
    return dict(dataset_root=str(root), episode_index=episode_index, episode_id=fingerprint([str(root), episode_index])[:24],
                length=entry['length'], fps=fps, source_recording_id=source, entry=entry,
                info=info, parquet=str(parquet), videos=list(map(str, videos)), camera_keys=keys)


def read_state(episode, joint_names, joint_mode='arms-and-hands'):
    import pyarrow.parquet as pq
    cols = resolve_joint_schema(episode['info'], joint_names, joint_mode)
    table = pq.read_table(episode['parquet'], columns=['timestamp', 'frame_index', 'episode_index', 'observation.state'])
    times = table['timestamp'].to_numpy().astype(np.float64)
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    n, fps = episode['length'], episode['fps']
    if (times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.allclose(times, np.arange(n) / fps, atol=1e-4, rtol=0)):
        raise ValueError('Parquet timestamp/FPS/length discontinuity')
    if (not np.array_equal(table['frame_index'].to_numpy(), np.arange(n))
            or not np.all(table['episode_index'].to_numpy() == episode['episode_index'])):
        raise ValueError('Parquet frame/episode index mismatch')
    if state.shape != (n, len(episode['info']['features']['observation.state']['names'])):
        raise ValueError('State array/schema dimension mismatch')
    q = state[:, cols]
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite measured joint positions; no imputation')
    return times, q


def validate_segments(segments, length):
    if not isinstance(segments, list) or not segments:
        raise ValueError('Nonempty reviewed coarse segments required')
    end = 0
    for s in segments:
        a, b = s.get('start_frame'), s.get('end_frame')
        if type(a) is not int or type(b) is not int or a != end or not a < b <= length:
            raise ValueError('Coarse [start,end) must be contiguous, nonempty, in bounds')
        if s.get('label') not in LABELS or s.get('reviewed') is not True:
            raise ValueError('Only four reviewed labels accepted; unknown/unreviewed rejected')
        end = b
    if end != length:
        raise ValueError('Coarse segments must cover full episode length')
    return segments


def source_hash_report(annotation, episode):
    declared = annotation.get('source', {}).get('sha256', {})
    required = [episode['parquet'], *episode['videos']]
    root = Path(episode['dataset_root'])
    if not isinstance(declared, dict) or not declared:
        raise ValueError('Source SHA256 declarations missing')
    if any(str(Path(p).relative_to(root)) not in declared for p in required):
        raise ValueError('Annotation must lock Parquet and all three videos by SHA256')
    rows = []
    for rel, expected in declared.items():
        path = safe_source_path(root, rel)
        actual = digest(path)
        rows.append(dict(path=rel, expected=expected, actual=actual, matched=expected == actual))
    return rows


def load_reviewed_coarse(annotation_path, episode_metadata):
    path = Path(annotation_path).expanduser().resolve(strict=True)
    doc = read_json(path)
    if (doc.get('stage') != 'coarse' or doc.get('schema_version') != 2
            or type(doc.get('revision')) is not int or doc['revision'] < 1
            or not doc.get('saved_at') or doc.get('inherited') is not False
            or doc.get('draft') is True or doc.get('status') in ('draft', 'unpublished', 'needs_review')
            or doc.get('published') is False):
        raise ValueError('Explicit saved/published coarse revision required; no draft/inherited truth')
    if doc.get('excluded_from_training') or doc.get('outcome') in ('failure', 'incomplete'):
        raise ValueError('Explicitly excluded/abnormal episode requires human disposition')
    source = doc.get('source', {})
    for key in ('episode_index', 'length', 'fps', 'source_recording_id'):
        if source.get(key) != episode_metadata[key]:
            raise ValueError(f'Coarse source {key} mismatch')
    for key in ('source_start_ns', 'source_end_ns'):
        if key in source and source[key] != episode_metadata['entry'].get(key):
            raise ValueError(f'Coarse source {key} mismatch')
    validate_segments(doc.get('segments'), episode_metadata['length'])
    hashes = source_hash_report(doc, episode_metadata)
    mismatches = [r['path'] for r in hashes if not r['matched']]
    if mismatches:
        raise ValueError('Annotation source SHA256 mismatch: ' + ', '.join(mismatches))
    return dict(path=str(path), sha256=digest(path), revision=doc['revision'], saved_at=doc['saved_at'],
                source=source, segments=doc['segments'], source_hashes=hashes,
                behavior_review='coarse is phase truth only; action success not inferred')


def validate_source_and_media(episode, joint_names, joint_mode='arms-and-hands'):
    import av
    times, q = read_state(episode, joint_names, joint_mode)
    reports = []
    for key, video in zip(episode['camera_keys'], episode['videos']):
        count, max_error = 0, 0.
        shape = episode['info']['features'][key]['shape']
        with av.open(video) as container:
            stream = container.streams.video[0]
            if stream.average_rate is None or abs(float(stream.average_rate) - episode['fps']) > .01:
                raise ValueError(f'Video FPS mismatch: {video}')
            for count, frame in enumerate(container.decode(video=0), 1):
                i = count - 1
                if i >= len(times) or [frame.height, frame.width, 3] != shape:
                    raise ValueError(f'Video frame count/dimensions mismatch: {video}')
                if frame.pts is None:
                    raise ValueError(f'Missing video PTS: {video}')
                # time origin and every PTS must agree, no relabel/shift or frame substitution.
                error = abs(float(frame.pts * frame.time_base) - times[i])
                max_error = max(max_error, error)
                if error > 1e-4:
                    raise ValueError(f'Video PTS/Parquet mismatch: {video} frame={i} error={error}')
        if count != len(times):
            raise ValueError(f'Full decode video/Parquet count mismatch: {video}: {count}')
        reports.append(dict(camera=key, decoded_frames=count, max_pts_error_s=max_error))
    return dict(media=reports, joint_dimension=q.shape[1], joint_min=q.min(axis=0).tolist(),
                joint_max=q.max(axis=0).tolist(), measured_finite=True)


def sample_causal_indices(times, anchor_index, history_frames=16, sample_hz=10):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or type(anchor_index) not in (int, np.int64, np.int32) or not 0 <= anchor_index < len(times)
            or history_frames < 1 or sample_hz <= 0):
        raise ValueError('Invalid causal timeline/history')
    query = np.maximum(times[0], times[anchor_index] - np.arange(history_frames - 1, -1, -1) / sample_hz)
    # 1e-12 only absorbs float arithmetic of exact 30Hz test timestamps, never a future frame.
    indices = np.searchsorted(times[:anchor_index + 1], query + 1e-12, side='right') - 1
    return np.maximum(indices, 0).astype(np.int64)


def iter_phase_anchors(times, prediction_hz=2):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or prediction_hz <= 0):
        raise ValueError('Invalid anchor grid/timeline')
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * prediction_hz + 1e-6)) + 1) / prediction_hz
    return np.unique(np.maximum(0, np.searchsorted(times, grid + 1e-6, side='right') - 1)).astype(np.int64)


def label_at_frame(segments, frame_index):
    if not 0 <= frame_index < segments[-1]['end_frame']:
        raise ValueError('Label frame outside episode')
    i = int(np.searchsorted([s['end_frame'] for s in segments], frame_index, side='right'))
    return LABELS.index(segments[i]['label'])


def load_annotation_index(path):
    rows = read_rows(path)
    if not rows:
        raise ValueError('Empty annotation index')
    seen, result = set(), []
    for r in rows:
        if r.get('variant') != 'normal':
            raise ValueError('Index must explicitly select normal; speedup excluded')
        root, ann = Path(r['dataset_root']).expanduser(), Path(r['coarse_annotation']).expanduser()
        if not root.is_absolute() or not ann.is_absolute():
            raise ValueError('Index paths must be absolute')
        key = (str(root.resolve()), r['episode_index'])
        if key in seen:
            raise ValueError('Duplicate annotation selection for dataset/episode')
        seen.add(key)
        result.append({**r, 'dataset_root': key[0], 'coarse_annotation': str(ann.resolve())})
    return sorted(result, key=lambda r: (r['dataset_root'], r['episode_index']))


def audit_dataset(root, config):
    """Diagnostic evidence, including invalid inputs. Never grants training eligibility."""
    rows = []
    for entry in read_rows(Path(root) / 'meta/episodes.jsonl'):
        e = load_episode_info(root, entry['episode_index'])
        row = dict(episode_index=e['episode_index'], episode_id=e['episode_id'], length=e['length'],
                   source_recording_id=e['source_recording_id'], fps=e['fps'])
        row['validation'] = validate_source_and_media(e, config['joint_names'], config['joint_mode'])
        times, _ = read_state(e, config['joint_names'], config['joint_mode'])
        anchors = iter_phase_anchors(times)
        row['anchors'] = len(anchors)
        row['annotation_candidates'] = []
        for p in sorted((Path(root) / 'annotations/subtasks/coarse').glob(f'episode_{e["episode_index"]:06d}*.json')):
            doc = read_json(p)
            a = dict(path=str(p), revision=doc.get('revision'), sha256=digest(p),
                     reviewed=[s.get('reviewed') for s in doc.get('segments', [])],
                     source_hashes=source_hash_report(doc, e))
            try:
                load_reviewed_coarse(p, e)
                a['eligible'] = True
            except ValueError as exc:
                a.update(eligible=False, error=str(exc))
            validate_segments(doc['segments'], e['length'])
            a['candidate_class_counts'] = np.bincount([label_at_frame(doc['segments'], int(i)) for i in anchors], minlength=4).tolist()
            row['annotation_candidates'].append(a)
        rows.append(row)
    return dict(diagnostic_only=True, dataset_root=str(Path(root).resolve()), labels=LABELS,
                joint_names=config['joint_names'], joint_dimension=len(config['joint_names']), episodes=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args(argv)
    output = Path(a.output).resolve()
    ensure_output_outside_sources(output.parent, [a.dataset])
    report = audit_dataset(a.dataset, load_config(a.config))
    write_json(output, report)
    print(json.dumps(dict(report=str(output), episodes=len(report['episodes']), diagnostic_only=True)))


if __name__ == '__main__':
    main()
