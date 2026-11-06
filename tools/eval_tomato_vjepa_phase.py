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


def frame_timeline(episode, rows, prediction=None):
    prediction = ds.prediction_contract(prediction or dict(prediction_mode=rows[0].get('prediction_mode', 'current_only')))
    names = ds.active_heads(prediction['prediction_mode'])
    anchors = np.array([r['frame_index'] for r in rows], dtype=np.int64)
    schedule = replay_schedule(episode['length'], anchors)
    if anchors.tolist() != episode['anchors'] or any(r['last_input_frame'] > r['frame_index'] for r in rows):
        raise ValueError('Replay prediction grid/future-input mismatch')
    times = np.array([r['timestamp_s'] for r in rows])
    current = metrics.temporal_report(episode, anchors, times, np.array([r['current_stage_id'] for r in rows]), prediction['stable_transition_k']) if 'current' in names else None
    targets = ds.build_phase_targets(anchors, episode['annotation']['segments'], episode['length'], episode['fps'], prediction['future_horizon_s'])
    future = metrics.future_anticipation(dict(episode=episode, times=times, **targets),
                                        np.array([r['future_stage_id'] for r in rows]), prediction['future_horizon_s']) if 'future' in names else None
    events = dict(current=current, future=future)
    def generate():
        for i, pos in enumerate(schedule):
            row = rows[int(pos)]
            true_id = ds.label_at_frame(episode['annotation']['segments'], i)
            trace = dict(frame_index=i, timestamp_s=i / episode['fps'], current_true_stage=ds.LABELS[true_id],
                         true_stage=ds.LABELS[true_id], prediction_mode=prediction['prediction_mode'],
                         prediction_held=i != row['frame_index'], prediction_frame=row['frame_index'],
                         prediction_timestamp_s=row['timestamp_s'], last_input_frame=row['last_input_frame'],
                         current_first_correct_events=[], current_stable_confirmation_events=[], future_warning_events=[])
            if current:
                for event in current['transitions']:
                    if event.get('first_correct_frame') == i: trace['current_first_correct_events'].append(event['frame_index'])
                    if event['stable_confirmation_frame'] == i: trace['current_stable_confirmation_events'].append(event['frame_index'])
            if future:
                for event in future['events']:
                    if event['first_warning_timestamp_s'] is not None and i == row['frame_index'] and abs(event['first_warning_timestamp_s'] - row['timestamp_s']) < 1e-7:
                        trace['future_warning_events'].append(dict(boundary_frame=event['boundary_frame'], lead_time_s=event['lead_time_s']))
            for name in names:
                trace[name + '_stage'] = row[name + '_stage']
                trace[name + '_probabilities'] = row[name + '_probabilities']
            if 'current' in names:
                trace['stage'] = row['current_stage']
                trace['current_wrong'] = row['current_stage'] != trace['current_true_stage']
            if 'future' in names:
                # A HELD forecast keeps its ORIGINAL anchor+delta target, not frame_i+delta.
                trace.update(future_target_frame=row['future_target_frame'], future_target_time_s=row['future_target_time_s'],
                             requested_future_time_s=row['timestamp_s'] + prediction['future_horizon_s'],
                             future_valid=row['future_valid'], future_true_stage=row['future_true_stage'],
                             future_target_status='AVAILABLE' if row['future_valid'] else 'TARGET_UNAVAILABLE')
            yield trace
    return generate(), events


def render_replay(episode, rows, output, prediction=None):
    import av
    import cv2
    prediction = ds.prediction_contract(prediction or dict(prediction_mode=rows[0].get('prediction_mode', 'current_only')))
    mode = prediction['prediction_mode']; names = ds.active_heads(mode)
    timeline, events = frame_timeline(episode, rows, prediction)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + '.tmp.mp4')
    w, h = 768, (476 if mode == 'current_only' else 520 if mode == 'future_only' else 680)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), episode['fps'], (w, h))
    if not writer.isOpened(): raise RuntimeError('Replay VideoWriter unavailable; numeric evaluation retained')
    traces = output.with_suffix('.frames.jsonl')
    ds.write_json(output.with_suffix('.events.json'), events)
    def text(image, line, y, color=(230, 230, 230), scale=.48):
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    try:
        with ExitStack() as stack, traces.open('w') as trace:
            containers = [stack.enter_context(av.open(p)) for p in episode['videos']]
            decoders = [iter(c.decode(video=0)) for c in containers]
            for item in timeline:
                i = item['frame_index']
                image = np.full((h, w, 3), 24, dtype=np.uint8)
                for camera, decoder in enumerate(decoders):
                    try: frame = next(decoder)
                    except StopIteration as exc: raise ValueError('Replay video ended early') from exc
                    tile = cv2.resize(frame.to_ndarray(format='bgr24'), (256, 256), interpolation=cv2.INTER_AREA)
                    image[:256, camera * 256:(camera + 1) * 256] = tile
                    cv2.putText(image, ds.CAMERAS[camera], (camera * 256 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                text(image, f'Held-out {episode["episode_index"]} | {mode} | frame {i} | t={item["timestamp_s"]:.3f}s', 280)
                text(image, f'True current: {item["current_true_stage"]}', 304)
                text(image, f'{"HELD" if item["prediction_held"] else "NEW"} from anchor {item["prediction_frame"]} ({item["prediction_timestamp_s"]:.3f}s); last input {item["last_input_frame"]}', 328)
                for name in names:
                    if name == 'current':
                        top, bars = 350, 374 if mode == 'joint' else 362
                        text(image, f'CURRENT at anchor: {item["current_stage"]}', top, (130, 240, 160))
                    else:
                        top = 482 if mode == 'joint' else 350
                        bars = top + 50
                        text(image, f'FUTURE: predicted stage at anchor + {prediction["future_horizon_s"]:.3f}s = {item["future_stage"]}', top, (240, 200, 100))
                        target = f't={item["future_target_time_s"]:.3f}s true={item["future_true_stage"]}' if item['future_valid'] else 'TARGET_UNAVAILABLE (not scored)'
                        text(image, 'Forecast target: ' + target, top + 24, (240, 200, 100))
                    for k, label in enumerate(ds.LABELS):
                        y = bars + k * 26; prob = item[name + '_probabilities'][label]
                        text(image, f'{label:8s} {prob:.3f}', y)
                        cv2.rectangle(image, (205, y - 12), (205 + int(prob * 540), y), (200, 140, 60), -1)
                if mode == 'joint':
                    text(image, f'Stable K={prediction["stable_transition_k"]}: evaluation only; raw argmax/probabilities shown', 654, scale=.45)
                writer.write(image)
                if i in (0, episode['length'] // 2, episode['length'] - 1):
                    if not cv2.imwrite(str(output.with_name(output.stem + f'.frame_{i}.png')), image): raise RuntimeError('Preview save failed')
                trace.write(json.dumps(item, allow_nan=False) + '\n')
            for decoder in decoders:
                if next(decoder, None) is not None: raise ValueError('Replay video has unexpected extra frames')
        writer.release()
        with av.open(str(tmp)) as container: count = sum(1 for _ in container.decode(video=0))
        if count != episode['length']: raise ValueError('Replay output frame count mismatch')
        tmp.replace(output)
    except Exception:
        writer.release(); tmp.unlink(missing_ok=True); raise
    return dict(path=str(output), frames=count, fps=episode['fps'], prediction_mode=mode,
                future_horizon_s=prediction['future_horizon_s'], probabilities='raw active-head softmax',
                stale_predictions='HELD keeps original anchor and future target', cameras=list(ds.CAMERAS),
                timeline_path=str(traces), events_path=str(output.with_suffix('.events.json')))


def benchmark(config, episode, saved, head, windows=20):
    if windows < 2:
        raise ValueError('benchmark windows must be >= 2')
    encoder, provenance = model.load_encoder(config['pretrained_dir'], config['device'], config['seed'])
    if provenance != saved['pretrained']:
        raise ValueError('Benchmark encoder/checkpoint provenance mismatch')
    times, q = ds.read_state(episode, saved['joint_names'], saved['contract']['joint_mode'])
    anchors = ds.iter_phase_anchors(times)[:windows + 2]
    if len(anchors) < 4:
        raise ValueError('At least four anchor windows required for latency measurements')
    histories = [ds.sample_causal_indices(times, int(i), sampler_version=saved['sampler_version']) for i in anchors]
    views, prep_ms = model.decode_selected(episode, np.unique(histories))
    device = next(encoder.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Benchmark encoder is not frozen/eval')
    for k, ids in enumerate(histories):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        rgb = np.stack([[np.stack([v[int(j)] for j in ids]) for v in views]])
        x = model.fuse_features(encoder, rgb, q[ids][None])
        output = model.predict_heads(head, x, saved['mean'], saved['std'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if k >= 2:
            durations.append((time.perf_counter() - start) * 1000)
    # Same selected source frames for all three cameras; sum actual RGB/resize work by frame.
    three_view_prep = np.array(prep_ms).reshape(3, -1).sum(axis=0)
    return dict(measured=True, encoder_and_head_mean_ms=float(np.mean(durations)),
                encoder_and_head_p50_ms=float(np.percentile(durations, 50)),
                encoder_and_head_p95_ms=float(np.percentile(durations, 95)),
                encoder_and_head_p99_ms=float(np.percentile(durations, 99)),
                prediction_mode=saved['prediction_mode'], future_horizon_s=saved['future_horizon_s'],
                fused_shape=list(x.shape), visual_dimension=3072, measured_joint_history_dimension=16 * q.shape[1],
                output_shapes={name: list(z.shape) if z is not None else None for name, z in output.items()},
                encoder_frozen=True, encoder_dtype=str(next(encoder.parameters()).dtype),
                first_input_indices=histories[0].tolist(), pretrained=provenance,
                three_view_rgb_resize_mean_ms=float(three_view_prep.mean()),
                three_view_rgb_resize_p95_ms=float(np.percentile(three_view_prep, 95)),
                warmup_windows=2, measured_windows=len(durations), device=str(device),
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                scope='Offline synchronized three-view 16-frame encoder + fusion + head. RGB conversion/resize measured separately per three-view source frame; excludes decode/I/O, ROS, other workloads. Not robot latency.')


def evaluate(config, episode_id=None, video=True, measure=True, windows=20, full_revalidate=False):
    config = dict(config)
    trainer.recover_run_index(config)
    manifest = trainer.load_locked_manifest(config, full_revalidate)
    trainer.initialize_runtime(config)
    prediction = ds.prediction_contract(manifest.get('prediction', {}))
    e = held_out_episode(manifest, episode_id)
    run = Path(config['output_dir'])
    out = run / 'offline_eval'
    out.mkdir(parents=True, exist_ok=True)
    saved, head = model.load_checkpoint(run / 'best_head.pt', ds.digest(run / 'manifest.json'), config['device'],
                                        split_sha256=manifest.get('split_sha256'))
    if saved['prediction'] != prediction or saved['sampler_version'] != manifest['sampler_version']:
        raise ValueError('Checkpoint/manifest prediction or sampler contract mismatch')
    entries = trainer.load_training_arrays(config, manifest)
    test = entries['test']
    x = np.concatenate([r['features'] for r in test])
    logits = model.predict_heads(head, x, saved['mean'], saved['std'])
    report = metrics.evaluate_heads(test, logits, prediction)
    report.update(manifest_sha256=ds.digest(run / 'manifest.json'), split_sha256=manifest.get('split_sha256'),
                  warnings=manifest['warnings'])
    rows, offset = [], 0
    for r in test:
        n = len(r['labels'])
        part = {name: logits[name + '_logits'][offset:offset + n] if logits[name + '_logits'] is not None else None for name in ('current', 'future')}
        probs = {name: metrics.probabilities(z) if z is not None else None for name, z in part.items()}
        rows += trainer.prediction_rows(r, probs, 'test', prediction, part)
        offset += n
    trainer.write_predictions(out, rows)
    ds.write_json(out / 'evaluation.json', report)
    selected_rows = [r for r in rows if r['episode_id'] == e['episode_id']]
    errors = []
    if video:
        try:
            ds.write_json(out / 'replay.json', render_replay(e, selected_rows, out / f'{e["episode_id"]}.mp4', prediction))
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
        identity = (m['contract'], m.get('prediction'), sorted((e['episode_id'], e['annotation']['sha256']) for e in m['episodes']))
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
    mode = values[0]['test'].get('prediction_mode', 'current_only')
    heads = {}
    for name in ds.active_heads(mode):
        reports = [v['test'].get(name, v['test']) for v in values]
        heads[name] = {k: dict(mean=float(np.mean([r[k] for r in reports])),
                              std=float(np.std([r[k] for r in reports]))) for k in ('accuracy', 'macro_f1', 'nll')}
    summary = dict(folds=values, prediction_mode=mode, heads=heads, aggregation='unweighted across grouped folds; std ddof=0')
    if mode == 'current_only': summary['metrics'] = heads['current']
    ds.ensure_output_outside_sources(root, protected_roots, annotation_paths)
    ds.write_json(Path(root) / 'cv_summary.json', summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--episode-id')
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--output-dir')
    p.add_argument('--full-revalidate', action='store_true')
    trainer.add_prediction_arguments(p)
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
    trainer.apply_prediction_overrides(cfg, a)
    if a.output_dir: cfg['output_dir'] = str(Path(a.output_dir).expanduser().resolve())
    if a.device:
        cfg['device'] = a.device
    if a.cv_folds is not None or a.cv_fold_index is not None:
        if a.cv_folds is None or a.cv_fold_index is None:
            p.error('Provide both CV flags')
        cfg['cv_folds'], cfg['cv_fold_index'] = a.cv_folds, a.cv_fold_index
        cfg['output_dir'] = str(Path(cfg['output_dir']) / f'cv{a.cv_folds}' / f'fold_{a.cv_fold_index}')
    _, errors = evaluate(cfg, a.episode_id, not a.no_video, not a.skip_benchmark, a.benchmark_windows, a.full_revalidate)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
