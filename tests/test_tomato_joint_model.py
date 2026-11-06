import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import tomato_dataset as ds, tomato_phase_model as model
from tools import train_tomato_vjepa_phase as tr
from tomato_helpers import make_config


@pytest.mark.parametrize('mode,params', [('current_only', 1025284), ('future_only', 1025284), ('joint', 1026312)])
def test_modes_parameter_shapes_roundtrip_and_inactive_heads(tmp_path, mode, params):
    cfg = make_config(tmp_path); manifest = tr.prepare(cfg)
    manifest['prediction'] = ds.prediction_contract(dict(prediction_mode=mode))
    head = model.mode_head_model(58, mode)
    assert sum(p.numel() for p in head.parameters()) == params
    head.eval()
    outputs = model.forward_heads(head, torch.zeros(2, 4000))
    for name in ('current', 'future'):
        assert (outputs[name + '_logits'] is not None) == (name in ds.active_heads(mode))
    # Fixture uses verified 16D joint schema; separately exercise safe mode roundtrip.
    head = model.mode_head_model(16, mode).eval()
    x = np.random.default_rng(1).normal(size=(3, 3328)).astype(np.float32)
    mu, sigma = np.zeros(3328, np.float32), np.ones(3328, np.float32)
    path = tmp_path / 'new.pt'
    model.save_checkpoint(path, head, mu, sigma, manifest, 'manifest', 1, {})
    saved, loaded = model.load_checkpoint(path, 'manifest', split_sha256=manifest['split_sha256'])
    assert saved['format_version'] == 2 and saved['prediction_mode'] == mode
    before, after = model.predict_heads(head, x, mu, sigma), model.predict_heads(loaded, x, mu, sigma)
    for key in before:
        if before[key] is None: assert after[key] is None
        else: np.testing.assert_array_equal(before[key], after[key])
    if mode != 'current_only':
        with pytest.raises(ValueError, match='time semantics'):
            model.predict_logits(head, x, mu, sigma)
    with pytest.raises(ValueError, match='split'):
        model.load_checkpoint(path, split_sha256='wrong')
    saved['lambda_future'] = 100; torch.save(saved, path)
    with pytest.raises(ValueError, match='contract'):
        model.load_checkpoint(path)


def test_legacy_v1_still_current_only_and_identical(tmp_path):
    cfg = make_config(tmp_path); manifest = tr.prepare(cfg)
    torch.manual_seed(42); before = model.head_model(16).eval()
    path = tmp_path / 'v1.pt'
    model.save_checkpoint(path, before, np.zeros(3328, np.float32), np.ones(3328, np.float32), manifest, 'old', 1, {})
    saved = torch.load(path, weights_only=True)
    for key in ('prediction', 'prediction_mode', 'future_horizon_s', 'lambda_current', 'lambda_future', 'stable_transition_k', 'split_sha256'):
        saved.pop(key)
    saved.update(format_version=1, sampler_version=ds.LEGACY_SAMPLER_VERSION)
    torch.save(saved, path)
    restored, after = model.load_checkpoint(path, 'old')
    assert restored['prediction_mode'] == 'current_only'
    x = np.ones((2, 3328), np.float32)
    np.testing.assert_array_equal(model.predict_logits(before, x, restored['mean'], restored['std']),
                                  model.predict_heads(after, x, restored['mean'], restored['std'])['current_logits'])
    saved['prediction_mode'] = 'future_only'; torch.save(saved, path)
    with pytest.raises(ValueError, match='v1'):
        model.load_checkpoint(path)


def test_loss_independent_lambdas_mask_exact_combination_and_gradients():
    head = model.joint_head_model(58)
    x = torch.ones(3, 4000)
    out = model.forward_heads(head, x)
    yc, yf, valid = torch.tensor([0, 1, 2]), torch.tensor([1, 2, -100]), torch.tensor([True, True, False])
    weights = dict(current=torch.tensor([1., 2, 3, 4]), future=torch.tensor([4., 3, 2, 1]))
    cfg = dict(prediction_mode='joint', lambda_current=1., lambda_future=.5)
    total, losses = model.supervision_loss(out, yc, yf, valid, weights, cfg)
    c = torch.nn.functional.cross_entropy(out['current_logits'], yc, weight=weights['current'])
    f = torch.nn.functional.cross_entropy(out['future_logits'][:2], yf[:2], weight=weights['future'])
    assert total.item() == pytest.approx(((c + .5 * f) / 1.5).item())
    changed, _ = model.supervision_loss(out, yc, yf, valid, weights, {**cfg, 'lambda_current': .5, 'lambda_future': 1.})
    assert changed.item() == pytest.approx(((.5 * c + f) / 1.5).item())
    c.backward(retain_graph=True)
    assert head.current.weight.grad is not None and head.future.weight.grad is None
    head.zero_grad(); total.backward()
    assert head.current.weight.grad is not None and head.future.weight.grad is not None
    with pytest.raises(ValueError, match='No valid future'):
        model.supervision_loss(out, yc, yf, torch.zeros(3, dtype=torch.bool), weights, cfg)
    for key, value in [('lambda_current', 0), ('lambda_future', -1), ('lambda_future', float('nan'))]:
        with pytest.raises(ValueError): ds.prediction_contract({**cfg, key: value})


def test_current_only_initialization_unchanged_and_cache_ignores_supervision(tmp_path):
    torch.manual_seed(42); old = model.head_model(58)
    torch.manual_seed(42); new = model.mode_head_model(58, 'current_only')
    for a, b in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    cfg = make_config(tmp_path); m = tr.prepare(cfg); e = m['episodes'][0]
    key = model.feature_fingerprint(e, m['contract'], m['pretrained'])
    for mode in ('current_only', 'future_only', 'joint'):
        c = {**m['contract'], **ds.prediction_contract(dict(prediction_mode=mode, future_horizon_s=2., lambda_future=3.))}
        assert model.feature_fingerprint(e, c, m['pretrained']) == key
    legacy = {**m['contract'], 'sampler_version': ds.LEGACY_SAMPLER_VERSION}
    assert model.feature_fingerprint(e, legacy, m['pretrained']) != key
