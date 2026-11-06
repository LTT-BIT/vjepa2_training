"""Raw held-out tomato predictions and three-camera offline replay; no robot/Agent code."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

try:
    from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as model
    from . import train_tomato_vjepa_phase as trainer
except ImportError:
    import tomato_dataset as ds
    import tomato_metrics as metrics
    import tomato_phase_model as model
    import train_tomato_vjepa_phase as trainer


def held_out_episode(manifest, episode_id=None):
    tests = sorted((e for e in manifest['episodes'] if e['split'] == 'test'), key=lambda e: e['episode_id'])
    found = tests[:1] if episode_id is None else [e for e in tests if e['episode_id'] == episode_id]
    if len(found) != 1:
        raise ValueError('Choose one locked held-out test episode_id')
    e = found[0]
    if any(r['source_recording_id'] == e['source_recording_id'] and r['split'] != 'test'
           for r in manifest['episodes']):
        raise ValueError('Held-out source leaked into train/validation')
    return e


def replay_schedule(length, anchors):
    anchors = np.asarray(anchors)
    if (anchors.ndim != 1 or not len(anchors) or anchors[0] != 0 or np.any(np.diff(anchors) <= 0)
            or anchors[-1] >= length or not np.issubdtype(anchors.dtype, np.integer)):
        raise ValueError('Replay requires ordered causal anchor grid starting at frame zero')
    return np.searchsorted(anchors, np.arange(length), side='right') - 1


def render_replay(episode, rows, output):
    import av
    import cv2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    anchors = np.array([r['frame_index'] for r in rows], dtype=np.int64)
    schedule = replay_schedule(episode['length'], anchors)
    if list(anchors) != episode['anchors'] or any(r['last_input_frame'] > r['frame_index'] for r in rows):
        raise ValueError('Replay prediction grid/future-input mismatch')
    tmp = output.with_name(output.stem + '.tmp.mp4')
    w, h = 768, 476
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), episode['fps'], (w, h))
    if not writer.isOpened():
        raise RuntimeError('Replay VideoWriter unavailable; model evaluation remains separately saved')
    traces = output.with_suffix('.frames.jsonl')
    try:
        with ExitStack() as stack, traces.open('w') as trace:
            containers = [stack.enter_context(av.open(p)) for p in episode['videos']]
            decoders = [iter(c.decode(video=0)) for c in containers]
            for i, pos in enumerate(schedule):
                row = rows[int(pos)]
                image = np.full((h, w, 3), 24, dtype=np.uint8)
                for camera, decoder in enumerate(decoders):
                    try:
                        frame = next(decoder)
                    except StopIteration as exc:
                        raise ValueError('Replay video ended early') from exc
                    tile = cv2.resize(frame.to_ndarray(format='bgr24'), (256, 256), interpolation=cv2.INTER_AREA)
                    image[:256, camera * 256:(camera + 1) * 256] = tile
                    cv2.putText(image, ds.CAMERAS[camera], (camera * 256 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                true_id = ds.label_at_frame(episode['annotation']['segments'], i)
                held = i != row['frame_index']
                lines = [f'Held-out {episode["episode_index"]} | frame {i} | {i / episode["fps"]:.3f}s',
                         f'True: {ds.LABELS[true_id]} | predicted: {row["stage"]}',
                         f'{"HELD" if held else "NEW"} prediction from frame {row["frame_index"]}; last input {row["last_input_frame"]}']
                for k, line in enumerate(lines):
                    cv2.putText(image, line, (10, 282 + k * 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
                for k, label in enumerate(ds.LABELS):
                    y = 362 + k * 26
                    p = row['probabilities'][label]
                    cv2.putText(image, f'{label:8s} {p:.3f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
                    cv2.rectangle(image, (205, y - 12), (205 + int(p * 540), y), (200, 140, 60), -1)
                writer.write(image)
                if i in (0, episode['length'] // 2, episode['length'] - 1):
                    if not cv2.imwrite(str(output.with_name(output.stem + f'.frame_{i}.png')), image):
                        raise RuntimeError('Replay preview save failed')
                trace.write(json.dumps(dict(frame_index=i, timestamp_s=i / episode['fps'],
                                            true_stage=ds.LABELS[true_id], stage=row['stage'],
                                            prediction_held=held, prediction_frame=row['frame_index'],
                                            last_input_frame=row['last_input_frame'])) + '\n')
            for decoder in decoders:
                if next(decoder, None) is not None:
                    raise ValueError('Replay video has unexpected extra frames')
        writer.release()
        with av.open(str(tmp)) as container:
            count = sum(1 for _ in container.decode(video=0))
        if count != episode['length']:
            raise ValueError('Replay output frame count mismatch')
        tmp.replace(output)
    except Exception:
        writer.release(); tmp.unlink(missing_ok=True)
        raise
    return dict(path=str(output), frames=count, fps=episode['fps'], probabilities='raw four-class softmax',
                stale_predictions='explicit HELD/NEW flag', cameras=list(ds.CAMERAS))


def benchmark(config, episode, saved, head, windows=20):
    if windows < 2:
        raise ValueError('benchmark windows must be >= 2')
    encoder, provenance = model.load_encoder(config['pretrained_dir'], config['device'])
    if provenance != saved['pretrained']:
        raise ValueError('Benchmark encoder/checkpoint provenance mismatch')
    times, q = ds.read_state(episode, saved['joint_names'], saved['contract']['joint_mode'])
    anchors = ds.iter_phase_anchors(times)[:windows + 2]
    if len(anchors) < 4:
        raise ValueError('At least four anchor windows required for latency measurements')
    histories = [ds.sample_causal_indices(times, int(i)) for i in anchors]
    views, prep_ms = model.decode_selected(episode, np.unique(histories))
    device = next(encoder.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for k, ids in enumerate(histories):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        rgb = np.stack([[np.stack([v[int(j)] for j in ids]) for v in views]])
        x = model.fuse_features(encoder, rgb, q[ids][None])
        model.predict_logits(head, x, saved['mean'], saved['std'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if k >= 2:
            durations.append((time.perf_counter() - start) * 1000)
    # Same selected source frames for all three cameras; sum actual RGB/resize work by frame.
    three_view_prep = np.array(prep_ms).reshape(3, -1).sum(axis=0)
    return dict(measured=True, encoder_and_head_mean_ms=float(np.mean(durations)),
                encoder_and_head_p95_ms=float(np.percentile(durations, 95)),
                three_view_rgb_resize_mean_ms=float(three_view_prep.mean()),
                three_view_rgb_resize_p95_ms=float(np.percentile(three_view_prep, 95)),
                warmup_windows=2, measured_windows=len(durations), device=str(device),
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                scope='Offline synchronized three-view 16-frame encoder + fusion + head. RGB conversion/resize measured separately per three-view source frame; excludes decode/I/O, ROS, other workloads. Not robot latency.')


def evaluate(config, episode_id=None, video=True, measure=True, windows=20):
    manifest = trainer.load_locked_manifest(config)
    e = held_out_episode(manifest, episode_id)
    run = Path(config['output_dir'])
    out = run / 'offline_eval'
    out.mkdir(parents=True, exist_ok=True)
    saved, head = model.load_checkpoint(run / 'best_head.pt', ds.digest(run / 'manifest.json'), config['device'])
    entries = trainer.load_training_arrays(config, manifest)
    test = entries['test']
    x = np.concatenate([r['features'] for r in test])
    logits = model.predict_logits(head, x, saved['mean'], saved['std'])
    report = metrics.evaluate_entries(test, logits)
    rows, offset = [], 0
    for r in test:
        n = len(r['labels'])
        rows += trainer.prediction_rows(r, metrics.probabilities(logits[offset:offset + n]), 'test')
        offset += n
    trainer.write_predictions(out, rows)
    ds.write_json(out / 'evaluation.json', report)
    selected_rows = [r for r in rows if r['episode_id'] == e['episode_id']]
    errors = []
    if video:
        try:
            ds.write_json(out / 'replay.json', render_replay(e, selected_rows, out / f'{e["episode_id"]}.mp4'))
        except Exception as exc:
            errors.append(dict(component='replay', error=f'{type(exc).__name__}: {exc}'))
    latency = dict(measured=False, reason='Explicitly skipped')
    if measure:
        try:
            latency = benchmark(config, e, saved, head, windows)
        except Exception as exc:
            latency = dict(measured=False, reason=f'{type(exc).__name__}: {exc}')
            errors.append(dict(component='latency', error=latency['reason']))
    ds.write_json(out / 'latency.json', latency)
    ds.write_json(out / 'errors.json', errors)
    ds.status(run, 'offline_evaluation_complete' if not errors else 'offline_evaluation_with_errors',
              evaluation=str(out / 'evaluation.json'), errors=errors)
    return report, errors


def summarize_cv(root):
    paths = sorted(Path(root).glob('fold_*/evaluation.json'))
    if not paths:
        raise ValueError('No completed grouped folds; no CV accuracy summary available')
    values, tested, expected = [], [], None
    reference = None
    protected_roots, annotation_paths = [], []
    for path in paths:
        m = ds.read_json(path.parent / 'manifest.json')
        cv = m['cross_validation']
        if not cv['folds'] or cv['folds'] != len(paths):
            raise ValueError('Incomplete CV fold collection')
        ids = {e['source_recording_id'] for e in m['episodes']}
        identity = (m['contract'], sorted((e['episode_id'], e['annotation']['sha256']) for e in m['episodes']))
        if reference is not None and identity != reference:
            raise ValueError('CV folds have different inputs/labels/configuration')
        reference = identity
        if expected is not None and ids != expected:
            raise ValueError('CV source groups differ across folds')
        expected = ids
        protected_roots += [e['dataset_root'] for e in m['episodes'] if 'dataset_root' in e]
        annotation_paths += [e['annotation']['path'] for e in m['episodes'] if 'path' in e['annotation']]
        tested += m['splits']['test']
        values.append(dict(fold_index=cv['fold_index'], path=str(path), test=ds.read_json(path)['splits']['test']))
    if len(tested) != len(set(tested)) or set(tested) != expected or len({v['fold_index'] for v in values}) != len(paths):
        raise ValueError('CV test source coverage/leakage/fold duplication')
    summary = dict(folds=values, aggregation='unweighted across grouped folds; std ddof=0',
                   metrics={k: dict(mean=float(np.mean([v['test'][k] for v in values])),
                                    std=float(np.std([v['test'][k] for v in values])))
                            for k in ('accuracy', 'macro_f1', 'nll')})
    ds.ensure_output_outside_sources(root, protected_roots, annotation_paths)
    ds.write_json(Path(root) / 'cv_summary.json', summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--episode-id')
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--skip-benchmark', action='store_true')
    p.add_argument('--benchmark-windows', type=int, default=20)
    p.add_argument('--cv-folds', type=int)
    p.add_argument('--cv-fold-index', type=int)
    p.add_argument('--summarize-cv', metavar='CV_DIRECTORY')
    a = p.parse_args(argv)
    if a.summarize_cv:
        print(json.dumps(summarize_cv(a.summarize_cv), indent=2))
        return 0
    if not a.config:
        p.error('--config required')
    cfg = ds.load_config(a.config)
    if a.device:
        cfg['device'] = a.device
    if a.cv_folds is not None or a.cv_fold_index is not None:
        if a.cv_folds is None or a.cv_fold_index is None:
            p.error('Provide both CV flags')
        cfg['cv_folds'], cfg['cv_fold_index'] = a.cv_folds, a.cv_fold_index
        cfg['output_dir'] = str(Path(cfg['output_dir']) / f'cv{a.cv_folds}' / f'fold_{a.cv_fold_index}')
    _, errors = evaluate(cfg, a.episode_id, not a.no_video, not a.skip_benchmark, a.benchmark_windows)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
