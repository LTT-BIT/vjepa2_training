import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools import tomato_dataset as ds
from tools import tomato_phase_model as model
from tools import train_tomato_vjepa_phase as tr
from tomato_helpers import make_config


class SyntheticEncoder(nn.Module):
    """Shape/freeze fixture, never a V-JEPA accuracy substitute."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.eval().requires_grad_(False)

    def forward(self, pixel_values_videos):
        x = pixel_values_videos.mean(dim=(1, 2, 3, 4))[:, None, None]
        return SimpleNamespace(last_hidden_state=x.expand(-1, 2, 1024) * self.scale)


def test_frozen_shared_three_view_fusion_and_only_head_gradients():
    torch.set_num_threads(2)
    encoder = SyntheticEncoder()
    rgb = np.zeros((1, 3, 16, 256, 256, 3), dtype=np.uint8)
    rgb[:, 1] = 100; rgb[:, 2] = 200
    q = np.arange(16 * 58, dtype=np.float32).reshape(1, 16, 58)
    x = model.fuse_features(encoder, rgb, q)
    assert x.shape == (1, 4000)
    assert x[0, 0] < x[0, 1024] < x[0, 2048]
    np.testing.assert_array_equal(x[:, 3072:], q.reshape(1, -1))
    head = model.head_model(58)
    optimizer = torch.optim.AdamW(head.parameters())
    loss = nn.functional.cross_entropy(head(torch.from_numpy(x)), torch.tensor([1]))
    loss.backward()
    assert encoder.scale.grad is None and all(not p.requires_grad for p in encoder.parameters())
    assert all(p.grad is not None for p in head.parameters())
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in head.parameters()}
    encoder.train()
    with pytest.raises(ValueError, match='frozen'):
        model.fuse_features(encoder, rgb, q)


def test_preprocessing_rgb_imagenet_matches_lemon_semantics():
    x = np.full((1, 16, 256, 256, 3), 255, dtype=np.uint8)
    pixels = model.preprocess(x, 'cpu', torch.float32)
    assert pixels.shape == (1, 16, 3, 256, 256)
    np.testing.assert_allclose(pixels[0, 0, :, 0, 0], (1 - np.array([.485, .456, .406])) / np.array([.229, .224, .225]), rtol=1e-6)
    with pytest.raises(ValueError):
        model.preprocess(x.astype(np.float32), 'cpu', torch.float32)


def test_checkpoint_safe_roundtrip_identical_logits_and_contract(tmp_path):
    cfg = make_config(tmp_path)
    manifest = tr.prepare(cfg)
    head = model.head_model(16).eval()
    dim = 3072 + 16 * 16
    x = np.random.default_rng(42).normal(size=(4, dim)).astype(np.float32)
    mean, std = np.zeros(dim, np.float32), np.ones(dim, np.float32)
    expected = model.predict_logits(head, x, mean, std)
    path = tmp_path / 'head.pt'
    model.save_checkpoint(path, head, mean, std, manifest, 'manifest-hash', 1, {'macro_f1': 0.})
    saved, loaded = model.load_checkpoint(path, 'manifest-hash')
    actual = model.predict_logits(loaded, x, saved['mean'], saved['std'])
    np.testing.assert_array_equal(expected, actual)
    p = torch.softmax(torch.from_numpy(actual), dim=1).numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-6)
    with pytest.raises(ValueError, match='manifest'):
        model.load_checkpoint(path, 'different')
    saved['label_order'] = ['lemon'] * 6; torch.save(saved, path)
    with pytest.raises(ValueError, match='compatible'):
        model.load_checkpoint(path)


def test_weights_missing_and_wrong_revision_fail_without_model_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match='config.json'):
        model.verify_pretrained(tmp_path)
    ds.write_json(tmp_path / 'config.json', {'model_type': 'vjepa2', 'hidden_size': 1024})
    (tmp_path / 'model.safetensors').write_bytes(b'not real weights')
    ds.write_json(tmp_path / 'weights_metadata.json', {'revision': 'wrong', 'sha256': model.WEIGHTS_SHA256})
    with pytest.raises(ValueError, match='revision'):
        model.verify_pretrained(tmp_path)
    ds.write_json(tmp_path / 'weights_metadata.json', {'revision': ds.REVISION, 'sha256': model.WEIGHTS_SHA256})
    with pytest.raises(ValueError, match='bytes'):
        model.verify_pretrained(tmp_path)


def test_cache_invalidates_media_not_labels_and_rejects_corruption(tmp_path):
    cfg = make_config(tmp_path)
    m = tr.prepare(cfg); e = m['episodes'][0]; c = m['contract']; provenance = m['pretrained']
    encoder = SyntheticEncoder()
    path, report = model.extract_episode(encoder, e, c, provenance, tmp_path / 'cache', 2)
    data = model.load_cache(path, e, c, provenance)
    assert 'labels' not in data
    assert data['features'].shape == (8, 3328)
    assert (data['input_indices'] <= data['anchors'][:, None]).all()
    assert model.extract_episode(encoder, e, c, provenance, tmp_path / 'cache')[1]['reused']
    revised = copy.deepcopy(e); revised['annotation']['revision'] = 2
    revised['annotation']['segments'][0]['label'] = 'adjust'
    assert model.feature_fingerprint(e, c, provenance) == model.feature_fingerprint(revised, c, provenance)
    changed = copy.deepcopy(e); changed['input_sha256'][next(iter(changed['input_sha256']))] = 'different'
    assert model.feature_fingerprint(e, c, provenance) != model.feature_fingerprint(changed, c, provenance)
    changed_c = copy.deepcopy(c); changed_c['joint_names'].reverse()
    assert model.feature_fingerprint(e, c, provenance) != model.feature_fingerprint(e, changed_c, provenance)
    changed_p = {**provenance, 'sha256': 'changed'}
    assert model.feature_fingerprint(e, c, provenance) != model.feature_fingerprint(e, c, changed_p)
    changed_schema = copy.deepcopy(e)
    changed_schema['info']['features']['observation.state']['names'].reverse()
    assert model.feature_fingerprint(e, c, provenance) != model.feature_fingerprint(changed_schema, c, provenance)
    with path.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='content'):
        model.load_cache(path, e, c, provenance)
    Path(e['videos'][0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        model.extract_episode(encoder, e, c, provenance, tmp_path / 'cache')
