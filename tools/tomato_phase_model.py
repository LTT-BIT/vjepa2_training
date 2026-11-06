"""Tomato-only frozen V-JEPA 2 preprocessing, fusion, MLP and safe checkpoints.

Mirrors train_lemon_vjepa_phase.encode_clips: RGB full resize, ImageNet
normalization, shared encoder, last_hidden_state token mean. No legacy service.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch import nn

try:
    from . import tomato_dataset as ds
except ImportError:
    import tomato_dataset as ds

WEIGHTS_SHA256 = '25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b'
CONFIG_SHA256 = '3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9'
FEATURE_VERSION = 'three-view-mean-rgb-imagenet-v1'


def verify_pretrained(directory):
    directory = Path(directory)
    for filename in ('config.json', 'model.safetensors', 'weights_metadata.json'):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f'Missing pinned pretrained artifact: {directory / filename}')
    meta = ds.read_json(directory / 'weights_metadata.json')
    if meta.get('revision') != ds.REVISION or meta.get('sha256') != WEIGHTS_SHA256:
        raise ValueError('Pretrained revision/official SHA256 mismatch; no fallback')
    actual = ds.digest(directory / 'model.safetensors')
    if actual != WEIGHTS_SHA256:
        raise ValueError('Pretrained model bytes do not match official SHA256')
    config = ds.read_json(directory / 'config.json')
    if (ds.digest(directory / 'config.json') != CONFIG_SHA256
            or config.get('model_type') != 'vjepa2' or config.get('hidden_size') != 1024):
        raise ValueError('Expected pinned V-JEPA 2 ViT-L 1024D configuration')
    return dict(model_id=ds.MODEL_ID, revision=ds.REVISION, sha256=actual,
                config_sha256=ds.digest(directory / 'config.json'))


def load_encoder(directory, device='cuda'):
    provenance = verify_pretrained(directory)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select --device cpu explicitly for FP32')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable on selected GPU; no silent dtype fallback')
        dtype = torch.bfloat16
    elif device == 'cpu':
        dtype = torch.float32
    else:
        raise ValueError('device must be cpu or cuda')
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(str(directory), local_files_only=True,
                                       torch_dtype=dtype, attn_implementation='sdpa')
    encoder = model.encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return encoder, provenance


def preprocess(clips, device, dtype):
    if clips.dtype != np.uint8 or clips.ndim != 5 or clips.shape[1:] != (16, 256, 256, 3):
        raise ValueError('Expected uint8 [B*3,16,256,256,3] RGB')
    pixels = torch.from_numpy(np.ascontiguousarray(clips)).to(device=device, dtype=dtype).permute(0, 1, 4, 2, 3) / 255
    mean = pixels.new_tensor([.485, .456, .406])[None, None, :, None, None]
    std = pixels.new_tensor([.229, .224, .225])[None, None, :, None, None]
    return (pixels - mean) / std


@torch.inference_mode()
def encode_clips(encoder, clips):
    parameter = next(encoder.parameters())
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Encoder must be eval() and fully frozen')
    features = encoder(pixel_values_videos=preprocess(clips, parameter.device, parameter.dtype)).last_hidden_state.mean(dim=1).float().cpu().numpy()
    if features.shape != (len(clips), 1024) or not np.isfinite(features).all():
        raise ValueError('Invalid 1024D encoder output')
    return features


def head_model(joint_dimension):
    if type(joint_dimension) is not int or joint_dimension < 1:
        raise ValueError('Verified positive joint dimension required')
    return nn.Sequential(nn.Linear(3072 + 16 * joint_dimension, 256), nn.GELU(),
                         nn.Dropout(.3), nn.Linear(256, 4))


def feature_fingerprint(episode, contract, provenance):
    return ds.fingerprint(dict(input_sha256=episode['input_sha256'], cameras=contract['cameras'],
                               camera_keys=episode['camera_keys'], joint_names=contract['joint_names'],
                               state_schema=episode['info']['features']['observation.state'], fps=episode['fps'],
                               joint_mode=contract['joint_mode'], history_frames=16, history_sample_hz=10,
                               prediction_hz=2, sampler=ds.SAMPLER_VERSION, image_size=256,
                               pooling='last_hidden_state.mean(dim=1)', preprocessing=FEATURE_VERSION,
                               model=provenance))


def assert_source_unchanged(episode):
    expected_hashes = {**episode.get('annotation', {}).get('source', {}).get('sha256', {}),
                       **episode['input_sha256']}
    for rel, expected in expected_hashes.items():
        if ds.digest(ds.safe_source_path(episode['dataset_root'], rel)) != expected:
            raise ValueError(f'Source changed since manifest: {rel}')


def decode_selected(episode, selected_indices):
    """Decode all PTS; retain only frames needed by fixed causal windows, bounded RAM.

    Full-frame RGB conversion/resize matches the lemon path. Decoding unselected
    source frames here never puts them into a model window or a feature vector.
    """
    import av
    import cv2
    selected = set(map(int, selected_indices))
    views, prep_ms = [], []
    for key, path in zip(episode['camera_keys'], episode['videos']):
        frames, count = {}, 0
        with av.open(path) as container:
            for count, frame in enumerate(container.decode(video=0), 1):
                i = count - 1
                if i in selected:
                    started = time.perf_counter()
                    frames[i] = cv2.resize(frame.to_ndarray(format='rgb24'), (256, 256), interpolation=cv2.INTER_AREA)
                    prep_ms.append((time.perf_counter() - started) * 1000)
        if count != episode['length'] or set(frames) != selected:
            raise ValueError(f'Decode/frame mismatch: {key}')
        views.append(frames)
    return views, prep_ms


def fuse_features(encoder, rgb, joint_history):
    """Only image and measured position tensors enter the network feature path."""
    q = np.asarray(joint_history, dtype=np.float32)
    if (rgb.ndim != 6 or rgb.shape[1:] != (3, 16, 256, 256, 3)
            or q.ndim != 3 or q.shape[:2] != (len(rgb), 16) or not np.isfinite(q).all()):
        raise ValueError('Expected aligned [B,3,16,256,256,3] and [B,16,D]')
    visual = encode_clips(encoder, rgb.reshape(-1, 16, 256, 256, 3)).reshape(len(rgb), 3072)
    return np.concatenate([visual, q.reshape(len(q), -1)], axis=1).astype(np.float32)


def cache_path(directory, episode, contract, provenance):
    return Path(directory) / (feature_fingerprint(episode, contract, provenance) + '.npz')


def write_cache(path, features, anchors, times, histories, key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.tmp.npz')
    np.savez_compressed(tmp, features=features, anchors=anchors, times=times,
                        input_indices=histories, fingerprint=np.array(key))
    tmp.replace(path)
    ds.write_json(path.with_suffix('.json'), dict(fingerprint=key, content_sha256=ds.digest(path)))


def load_cache(path, episode, contract, provenance):
    key = feature_fingerprint(episode, contract, provenance)
    receipt = ds.read_json(Path(path).with_suffix('.json'))
    if receipt != dict(fingerprint=key, content_sha256=ds.digest(path)):
        raise ValueError('Feature cache content/fingerprint mismatch')
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    t, _ = ds.read_state(episode, contract['joint_names'], contract['joint_mode'])
    anchors = ds.iter_phase_anchors(t)
    histories = np.stack([ds.sample_causal_indices(t, int(i)) for i in anchors])
    if (str(data['fingerprint']) != key or not np.array_equal(data['anchors'], anchors)
            or not np.array_equal(data['times'], t[anchors])
            or not np.array_equal(data['input_indices'], histories)
            or data['features'].shape != (len(anchors), 3072 + 16 * len(contract['joint_names']))
            or data['features'].dtype != np.float32 or not np.isfinite(data['features']).all()):
        raise ValueError('Invalid cache dimensions, timeline, or feature values')
    return data


def extract_episode(encoder, episode, contract, provenance, directory, batch_windows=1):
    assert_source_unchanged(episode)
    path = cache_path(directory, episode, contract, provenance)
    if path.exists():
        load_cache(path, episode, contract, provenance)
        return path, dict(reused=True)
    if type(batch_windows) is not int or batch_windows < 1:
        raise ValueError('batch_windows must be positive')
    times, q = ds.read_state(episode, contract['joint_names'], contract['joint_mode'])
    anchors = ds.iter_phase_anchors(times)
    histories = np.stack([ds.sample_causal_indices(times, int(i)) for i in anchors])
    views, prep_ms = decode_selected(episode, np.unique(histories))
    features, latency = [], []
    device = next(encoder.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(anchors), batch_windows):
        ids = histories[start:start + batch_windows]
        rgb = np.stack([[np.stack([v[int(j)] for j in history]) for v in views] for history in ids])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        features.append(fuse_features(encoder, rgb, q[ids]))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        latency.append((time.perf_counter() - began) * 1000 / len(ids))
    assert_source_unchanged(episode)
    key = feature_fingerprint(episode, contract, provenance)
    write_cache(path, np.concatenate(features), anchors, times[anchors], histories, key)
    report = dict(reused=False, encoder_ms_per_anchor_mean=float(np.mean(latency)),
                  encoder_ms_per_anchor_p95=float(np.percentile(latency, 95)),
                  view_frame_rgb_resize_mean_ms=float(np.mean(prep_ms)),
                  view_frame_rgb_resize_p95_ms=float(np.percentile(prep_ms, 95)),
                  batch_windows=batch_windows, device=str(device),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  scope='Offline feature extraction; synchronized encoder, transfer, tensor normalization and fusion; RGB resize measured separately; excludes head')
    ds.write_json(path.with_suffix('.performance.json'), report)
    return path, report


def save_checkpoint(path, head, mean, std, manifest, manifest_sha256, epoch, validation):
    c = manifest['contract']
    payload = dict(format_version=1, label_order=list(ds.LABELS), joint_names=c['joint_names'],
                   joint_dimension=len(c['joint_names']), dimension=3072 + 16 * len(c['joint_names']),
                   contract=c, sampler_version=ds.SAMPLER_VERSION,
                   pretrained=manifest['pretrained'], feature_version=FEATURE_VERSION,
                   mean=torch.as_tensor(mean).float().cpu(), std=torch.as_tensor(std).float().cpu(),
                   head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                   manifest_sha256=manifest_sha256, best_epoch=epoch, validation=validation)
    path = Path(path)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, manifest_sha256=None, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format_version') != 1 or saved.get('label_order') != list(ds.LABELS)
            or saved.get('sampler_version') != ds.SAMPLER_VERSION or saved.get('feature_version') != FEATURE_VERSION
            or saved.get('contract', {}).get('model_id') != ds.MODEL_ID
            or saved.get('contract', {}).get('model_revision') != ds.REVISION
            or saved.get('pretrained', {}).get('sha256') != WEIGHTS_SHA256
            or saved.get('pretrained', {}).get('config_sha256') != CONFIG_SHA256):
        raise ValueError('Not a compatible tomato four-class V-JEPA 2 checkpoint')
    for k, value in dict(cameras=list(ds.CAMERAS), labels=list(ds.LABELS), history_frames=16,
                         history_sample_hz=10, prediction_hz=2, image_size=256,
                         robot='sharpa', variant='normal').items():
        if saved['contract'].get(k) != value:
            raise ValueError(f'Checkpoint task/sampling contract mismatch: {k}')
    if manifest_sha256 is not None and saved['manifest_sha256'] != manifest_sha256:
        raise ValueError('Checkpoint/manifest hash mismatch')
    d = len(saved['joint_names'])
    if (saved['joint_dimension'] != d or saved['dimension'] != 3072 + 16 * d
            or saved['joint_names'] != saved['contract']['joint_names']
            or saved['mean'].shape != (saved['dimension'],) or saved['std'].shape != saved['mean'].shape
            or not torch.isfinite(saved['mean']).all() or not torch.isfinite(saved['std']).all()
            or torch.any(saved['std'] <= 0)):
        raise ValueError('Invalid checkpoint dimensions/normalization')
    head = head_model(d).to(device).eval()
    head.load_state_dict(saved['head'])
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Nonfinite checkpoint parameters')
    return saved, head


@torch.inference_mode()
def predict_logits(head, features, mean, std, batch_size=512):
    device = next(head.parameters()).device
    x = np.asarray(features, dtype=np.float32)
    mu, sigma = torch.as_tensor(mean).to(device).float(), torch.as_tensor(std).to(device).float()
    if x.ndim != 2 or x.shape[1] != mu.numel() or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid fused prediction features')
    head.eval()
    return torch.cat([head((torch.from_numpy(a).to(device) - mu) / sigma)
                      for a in np.array_split(x, max(1, int(np.ceil(len(x) / batch_size))))]).cpu().numpy()
