#!/usr/bin/env python3
"""Offline three-camera + arm-state phase classifier; never sends robot commands."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


DATASET = Path('/home/descfly/Convert_data/outputs/ur_o30/lemon_move_pi05_auto')
SPLITS = Path('/home/descfly/Downloads/lemon_pi05_2250_deploy_bundle/splits.json')
CONVERTER = Path('/home/descfly/Convert_data')
CAMERAS = ('cam0', 'cam1', 'cam2')
BOUNDARIES = ('left_arrived_s', 'scoop_start_s', 'lift_s', 'pour_complete_s')
LABELS = ('before_left_arrived', 'left_arrived_to_scoop_start',
          'scoop_start_to_lift', 'lift_to_pour_complete', 'after_pour_complete')
FEATURE_VERSION = 'mobilenet_v3_small_imagenet1k_v1_rgb_square224_v1'
JOINTS = tuple(f'{side}_{name}_joint' for side in ('left', 'right') for name in
               ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def labels_at(times, boundaries):
    b = np.asarray(boundaries, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or b[0] < 0 or np.any(np.diff(b) <= 0):
        raise ValueError('Four finite, strictly increasing, nonnegative boundaries required')
    return np.searchsorted(b, times, side='right').astype(np.int64)


def backbone(pretrained):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(net.features, net.avgpool, nn.Flatten()).eval().requires_grad_(False)


def prepare_image(rgb):
    # Full field of view: do not center-crop away the cup or wrists.
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def encode(encoder, images, device):
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float().div_(255)
    mean = batch.new_tensor([.485, .456, .406])[None, :, None, None]
    std = batch.new_tensor([.229, .224, .225])[None, :, None, None]
    return encoder((batch - mean) / std).cpu().numpy()


def classifier():
    return nn.Sequential(nn.Linear(576 * 3 + 12, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 5))


def episode_paths(root, episode):
    chunk = episode // 1000
    stem = f'episode_{episode:06d}'
    return [root / f'videos/chunk-{chunk:03d}/observation.images.{cam}/{stem}.mp4' for cam in CAMERAS], root / f'data/chunk-{chunk:03d}/{stem}.parquet'


def read_episode_state(root, episode):
    import pyarrow.parquet as pq
    _, path = episode_paths(root, episode)
    table = pq.read_table(path, columns=['timestamp', 'observation.state'])
    info = json.loads((root / 'meta/info.json').read_text())
    names = info['features']['observation.state']['names']
    indices = [names.index(name + '.position') for name in JOINTS]
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)[:, indices]
    times = table['timestamp'].to_numpy().astype(np.float64)
    if len(times) == 0 or not np.isfinite(state).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'Invalid state/timeline for episode {episode}')
    return times, state


def episode_features(root, episode, encoder, device, cache_dir, batch_size):
    videos, parquet = episode_paths(root, episode)
    signature = json.dumps([FEATURE_VERSION, [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                             for p in [*videos, parquet, root / 'meta/info.json']]])
    fingerprint = hashlib.sha256(signature.encode()).hexdigest()
    cache = cache_dir / f'episode_{episode:06d}.npz'
    times, state = read_episode_state(root, episode)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data['fingerprint']) == fingerprint:
                features = data['features']
                if features.shape == (len(times), 1728) and np.isfinite(features).all():
                    return features, state, times
    camera_features = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        batches, images, count = [], [], 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                images.append(prepare_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                count += 1
                if len(images) == batch_size:
                    batches.append(encode(encoder, images, device))
                    images = []
            if images:
                batches.append(encode(encoder, images, device))
        finally:
            cap.release()
        if count != len(times):
            raise ValueError(f'{video}: {count} video frames != {len(times)} state frames')
        camera_features.append(np.concatenate(batches))
    features = np.concatenate(camera_features, axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite visual features')
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, features=features, fingerprint=fingerprint)
    tmp.replace(cache)
    return features, state, times


def build_manifest(root, annotations, split_path):
    rows = list(csv.DictReader(annotations.open(), delimiter='\t'))
    rows = sorted((r for r in rows if all(r.get(k, '').strip() for k in BOUNDARIES)), key=lambda r: int(r['episode_index']))
    split_doc = json.loads(split_path.read_text())
    # This bundle uses compressed IDs; verify against original rosbag membership.
    split_by_id = {}
    for split, ids in split_doc['splits'].items():
        for ident in ids:
            if ident in split_by_id:
                raise ValueError('Split overlap')
            split_by_id[ident] = split
    if set(split_by_id) != set(range(len(rows))):
        raise ValueError('Annotations do not match this bundle split; provide a matching split file')
    sources = {}
    for path in sorted((root / 'conversion/records').glob('*.json')):
        record = json.loads(path.read_text())
        for episode in record['episodes']:
            sources[episode['episode_index']] = record['source']
    episodes, rejected, groups = [], [], {}
    for compressed_id, row in enumerate(rows):
        ident = int(row['episode_index'])
        split = split_by_id[compressed_id]
        source = sources[ident]
        if source not in split_doc['sources'][split]:
            raise ValueError(f'Source/split mismatch for episode {ident}')
        if source in groups and groups[source] != split:
            raise ValueError('Source rosbag leakage across splits')
        groups[source] = split
        times, _ = read_episode_state(root, ident)
        boundaries = [float(row[k]) for k in BOUNDARIES]
        item = dict(episode_index=ident, split=split, source_recording_id=source,
                    boundaries=boundaries, frames=len(times), last_frame_s=float(times[-1]))
        try:
            labels_at(times, boundaries)
            if boundaries[-1] > times[-1]:
                raise ValueError('Fourth boundary beyond last available frame')
        except ValueError as exc:
            rejected.append(dict(item, reason=str(exc)))
            continue
        episodes.append(item)
    return dict(schema_version=1, label_semantics='legacy_pour_complete_NOT_pour_ready',
                labels=LABELS, boundaries=BOUNDARIES, feature_version=FEATURE_VERSION,
                training_input='converted_10hz_three_rgb_and_12_measured_arm_positions',
                dataset=str(root.resolve()), annotations=str(annotations.resolve()),
                annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                splits_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                episodes=episodes, rejected=rejected)


def metrics(logits, targets, temperature=1.):
    probabilities = torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()
    predictions = probabilities.argmax(1)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(1)
    f1 = 2 * confusion.diagonal() / np.maximum(1, support + confusion.sum(0))
    return dict(frames=len(targets), accuracy=float((predictions == targets).mean()),
                macro_f1=float(f1.mean()), per_class_f1=f1.tolist(), support=support.tolist(),
                confusion_true_rows_predicted_columns=confusion.tolist(),
                nll=float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()))


def predict_logits(head, x, device):
    head.eval()
    with torch.inference_mode():
        return torch.cat([head(part.to(device)).cpu() for part in x.split(1024)])


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.pt').exists():
        raise ValueError('Output already contains model.pt; choose a new output directory')
    manifest = build_manifest(args.dataset, args.annotations, args.splits)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'accepted_episodes': len(manifest['episodes']), 'rejected': manifest['rejected']}, ensure_ascii=False), flush=True)
    encoder = backbone(True).to(args.device)
    records = {split: [] for split in ('train', 'val', 'test')}
    for count, item in enumerate(manifest['episodes'], 1):
        ident = item['episode_index']
        feat, state, times = episode_features(args.dataset, ident, encoder, args.device, output / 'features', args.image_batch)
        x = np.concatenate([feat, state], axis=1)
        labels = labels_at(times, item['boundaries'])
        records[item['split']].append((ident, x, labels, times))
        print(f'features {count}/{len(manifest["episodes"])} episode={ident} frames={len(times)}', flush=True)
    splits = {}
    for split, values in records.items():
        if not values:
            raise ValueError(f'Empty {split} split')
        splits[split] = (torch.from_numpy(np.concatenate([v[1] for v in values])),
                         torch.from_numpy(np.concatenate([v[2] for v in values])))
    xtrain, ytrain = splits['train']
    mean = xtrain.mean(0)
    std = xtrain.std(0).clamp_min(.02)
    for split, (x, y) in splits.items():
        splits[split] = ((x - mean) / std, y)
    xtrain, ytrain = splits['train']
    counts = torch.bincount(ytrain, minlength=5)
    if (counts == 0).any():
        raise ValueError(f'Training split lacks classes: {counts.tolist()}')
    weights = counts.float().rsqrt()
    weights /= weights.mean()
    head = classifier().to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    best_loss, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for index in torch.randperm(len(xtrain)).split(512):
            optimizer.zero_grad(set_to_none=True)
            logits = head(xtrain[index].to(args.device))
            loss = nn.functional.cross_entropy(logits, ytrain[index].to(args.device), weight=weights.to(args.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_logits = predict_logits(head, splits['val'][0], args.device)
        val_loss = nn.functional.cross_entropy(val_logits, splits['val'][1]).item()
        row = dict(epoch=epoch, train_weighted_ce=float(np.mean(losses)), val_ce=val_loss)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    head.load_state_dict(best_state)
    # A single validation-fitted temperature; the test set never tunes the model.
    val_logits = predict_logits(head, splits['val'][0], args.device).clone()
    log_temp = torch.tensor(0., requires_grad=True)
    calibration = torch.optim.LBFGS([log_temp], lr=.1, max_iter=50)
    def closure():
        calibration.zero_grad()
        loss = nn.functional.cross_entropy(val_logits / log_temp.exp().clamp(.1, 10), splits['val'][1])
        loss.backward()
        return loss
    calibration.step(closure)
    temperature = float(log_temp.detach().exp().clamp(.1, 10))
    checkpoint = dict(format_version=1, feature_version=FEATURE_VERSION, labels=list(LABELS),
                      label_semantics=manifest['label_semantics'], joints=list(JOINTS), cameras=list(CAMERAS),
                      backbone={k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                      head=best_state, mean=mean, std=std, temperature=temperature,
                      manifest_sha256=hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest())
    tmp = output / 'model.pt.tmp'
    torch.save(checkpoint, tmp)
    tmp.replace(output / 'model.pt')
    report = dict(seed=args.seed, best_epoch=best_epoch, temperature=temperature, history=history,
                  trainable_parameters=sum(p.numel() for p in head.parameters()),
                  total_parameters=sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in encoder.parameters()),
                  training_counts=counts.tolist(), label_semantics=manifest['label_semantics'],
                  splits={}, per_episode={})
    for split, (x, y) in splits.items():
        logits = predict_logits(head, x, args.device)
        report['splits'][split] = metrics(logits, y.numpy(), temperature)
        offset = 0
        for ident, _, targets, times in records[split]:
            part = logits[offset:offset + len(targets)]
            report['per_episode'][str(ident)] = dict(split=split, **metrics(part, targets, temperature))
            if split == 'test':
                probs = torch.softmax(part / temperature, dim=1).numpy()
                write_predictions(output / f'test_episode_{ident:06d}.csv', times, probs, targets)
            offset += len(targets)
    write_json(output / 'metrics.json', report)
    print(json.dumps({'model': str(output / 'model.pt'), 'metrics': report['splits']}, ensure_ascii=False), flush=True)


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format_version'] != 1 or saved['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported model format/preprocessing')
    encoder = backbone(False)
    encoder.load_state_dict(saved['backbone'])
    head = classifier()
    head.load_state_dict(saved['head'])
    return saved, encoder.to(device).eval(), head.to(device).eval()


def probabilities(saved, head, features, state, device):
    x = torch.from_numpy(np.concatenate([features, state], axis=1)).float()
    x = (x - saved['mean']) / saved['std']
    return torch.softmax(predict_logits(head, x, device) / saved['temperature'], dim=1).numpy()


def write_predictions(path, times, probs, targets=None):
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'phase', 'p0', 'p1', 'p2', 'p3', 'p4'] + (['label'] if targets is not None else []))
        for index, (time, prob) in enumerate(zip(times, probs)):
            writer.writerow([f'{time:.9f}', int(prob.argmax()), *map(float, prob)] + ([int(targets[index])] if targets is not None else []))


def infer_episode(args):
    saved, encoder, head = load_model(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feat, state, times = episode_features(args.dataset, args.episode, encoder, args.device,
                                           args.model.parent / 'features', args.image_batch)
    probs = probabilities(saved, head, feat, state, args.device)
    write_predictions(args.output, times, probs)
    print(f'{len(times)} frames -> {args.output}', flush=True)


def causal_indices(timeline, source_times, max_age_ns):
    index = np.searchsorted(source_times, timeline, side='right') - 1
    age = timeline - source_times[np.maximum(index, 0)]
    valid = (index >= 0) & (age >= 0) & (age <= max_age_ns)
    return index, valid


def infer_bag(args):
    # Reuse the existing read-only, lazy CDR/SQLite image reader (no ROS runtime).
    sys.path.insert(0, str(args.converter_root / 'src'))
    from convest.sources.gello_rosbag2 import Bag, Series, joint_values
    import yaml
    contract = yaml.safe_load((args.converter_root / 'schemas/ur_o30_contract.yaml').read_text())
    saved, encoder, head = load_model(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, valid_count = 0, 0
    with Bag(args.bag, args.converter_root / 'schemas') as bag:
        # Retain every head-camera message, including duplicate header timestamps.
        head_frames = sorted(bag.records(contract['topics']['cam0']['topic'], image=True), key=lambda v: v[0])
        if not head_frames:
            raise ValueError('No head-camera frames')
        times = np.array([r[0] for r in head_frames], dtype=np.int64)
        streams = {}
        for camera in CAMERAS[1:]:
            streams[camera] = Series.build(list(bag.records(contract['topics'][camera]['topic'], image=True)))
        for side in ('left', 'right'):
            names = list(JOINTS[:6] if side == 'left' else JOINTS[6:])
            streams[side] = Series.build([(t, joint_values(msg, names, ['position'])) for t, msg in
                bag.records(contract['topics'][f'{side}_arm_state']['topic'])])
        alignment = {name: causal_indices(times, stream.times, int(args.max_age_ms * 1e6)) for name, stream in streams.items()}
        output_tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        with output_tmp.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['frame_index', 'header_ns', 'time_s', 'valid', 'reason', 'phase', 'p0', 'p1', 'p2', 'p3', 'p4'])
            # Each inference sample has all three cameras and both arms; never fill gaps with zeroes.
            for begin in range(0, len(times), max(1, args.image_batch // 3)):
                indices = list(range(begin, min(len(times), begin + max(1, args.image_batch // 3))))
                accepted = [i for i in indices if all(v[1][i] for v in alignment.values())]
                result = {}
                if accepted:
                    images, states = [], []
                    for i in accepted:
                        refs = [head_frames[i][1]] + [streams[c].values[alignment[c][0][i]] for c in CAMERAS[1:]]
                        images.extend(prepare_image(bag.image(ref)) for ref in refs)
                        states.append(np.concatenate([streams[s].values[alignment[s][0][i]] for s in ('left', 'right')]))
                    features = encode(encoder, images, args.device).reshape(len(accepted), -1)
                    probs = probabilities(saved, head, features, np.stack(states), args.device)
                    result = dict(zip(accepted, probs))
                for i in indices:
                    prefix = [i, int(times[i]), f'{(times[i] - times[0]) / 1e9:.9f}']
                    if i in result:
                        p = result[i]
                        writer.writerow([*prefix, 1, '', int(p.argmax()), *map(float, p)])
                        valid_count += 1
                    else:
                        bad = ','.join(name for name, value in alignment.items() if not value[1][i])
                        writer.writerow([*prefix, 0, 'missing_or_stale:' + bad, '', '', '', '', '', ''])
                    count += 1
                if begin % 300 == 0:
                    print(f'bag frames {count}/{len(times)} valid={valid_count}', flush=True)
        output_tmp.replace(args.output)
    write_json(args.output.with_suffix('.meta.json'), dict(bag=str(args.bag.resolve()), model=str(args.model.resolve()),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), frames=count, valid_frames=valid_count,
        max_age_ms=args.max_age_ms, label_semantics=saved['label_semantics'], labels=saved['labels'],
        alignment='head_header_time; latest source header <= head header; no future samples',
        timestamp_zero='first head image header; NOT converted MP4 first frame'))
    print(f'{count} head frames, {valid_count} valid -> {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'infer-episode', 'infer-bag'):
        p = sub.add_parser(name)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        p.add_argument('--image-batch', type=int, default=24)
        p.add_argument('--output', type=Path, required=True)
        if name != 'infer-bag':
            p.add_argument('--dataset', type=Path, default=DATASET)
        if name != 'train':
            p.add_argument('--model', type=Path, required=True)
        if name == 'train':
            p.add_argument('--annotations', type=Path, default=DATASET / 'conversion/training_stage_annotations.tsv')
            p.add_argument('--splits', type=Path, default=SPLITS)
            p.add_argument('--epochs', type=int, default=40)
            p.add_argument('--seed', type=int, default=42)
        elif name == 'infer-episode':
            p.add_argument('--episode', type=int, required=True)
        else:
            p.add_argument('--bag', type=Path, required=True)
            p.add_argument('--converter-root', type=Path, default=CONVERTER)
            p.add_argument('--max-age-ms', type=float, default=150.)
    args = parser.parse_args()
    if args.image_batch < 1 or (args.command == 'train' and args.epochs < 1):
        parser.error('Batch size and epochs must be positive')
    if args.command == 'infer-bag' and (not math.isfinite(args.max_age_ms) or args.max_age_ms <= 0):
        parser.error('--max-age-ms must be finite and positive')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.device == 'cuda':
        # Bound this small offline job's allocator; do not evict the policy server.
        torch.cuda.set_per_process_memory_fraction(.06)
    {'train': train, 'infer-episode': infer_episode, 'infer-bag': infer_bag}[args.command](args)


if __name__ == '__main__':
    main()
