from dataclasses import replace
import numpy as np
import pytest
import torch

from tools.tomato_rgb_memory import RGBStateMemory, RGBStateWindow
from tools.tomato_interaction_rgb import crop_rgb, square_context, encode_four_views
from tools.tomato_rgb_memory_model import RGBMemoryReader, reader_batch


NAMES = tuple([f'{side}_fr3_joint{i}' for side in ('left', 'right') for i in range(7)]
              + [f'hand_{i}' for i in range(44)])


def memory(**kwargs):
    return RGBStateMemory('session', 'episode', NAMES, 'encoder', 'roi', **kwargs)


def window(i=0, **kwargs):
    times = np.arange(16)*.1 + i*.5
    return replace(RGBStateWindow('session', 'episode', str(i), 'encoder', 'roi',
        float(times[-1]+.01), times, times.copy(), np.zeros((3, 1024)), np.ones(1024),
        np.arange(928).reshape(16, 58)/100, np.tile([300, 180, 500, 380], (16, 1)),
        np.ones(16, dtype=bool), times.copy(), 'cam0', (640, 400), 'causal_prediction'), **kwargs)


def test_full_state_window_preserved_without_stage_or_outcome_inputs():
    m = memory(); w = window(); m.observe(w)
    s = m.read(now=1.51)
    np.testing.assert_array_equal(s['rows'][0]['positions'], w.positions)
    assert not s['stage_inputs'] and not s['outcome_inputs']
    assert not s['suggests_prompt_update'] and s['robot_action'] is None
    w.positions[:] = -1
    assert m.read(now=1.51)['rows'][0]['positions'][0, 0] == 0
    s['rows'][0]['positions'][:] = -2
    assert m.read(now=1.51)['rows'][0]['positions'][0, 0] == 0


@pytest.mark.parametrize('change', [
    dict(session_id='other'), dict(encoder_signature='other'), dict(roi_signature='other'),
    dict(positions=np.zeros(58)), dict(positions=np.full((16, 58), np.nan)),
    dict(state_times=np.arange(16)*.1+.01), dict(times=np.arange(16)*.2),
    dict(roi_mode='manual_offline'), dict(roi_available_at=np.arange(16)*.1+1),
    dict(arrival_timestamp=10), dict(boxes=np.zeros((16, 4))),
    dict(source_camera='unknown'), dict(roi_valid=np.ones(16)),
])
def test_rejected_input_never_mutates_history(change):
    m = memory(); m.observe(window())
    with pytest.raises(ValueError): m.observe(window(1, **change))
    assert len(m.rows) == 1 and m.last == 1.5


def test_offline_provenance_cannot_be_silently_used_online():
    w = window(roi_mode='manual_offline', roi_available_at=np.full(16, 10.))
    m = memory(mode='offline_diagnostic'); m.observe(w)
    assert m.read(now=1.51)['mode'] == 'offline_diagnostic'
    assert m.read(now=1.51)['rows'][0]['roi_mode'] == 'manual_offline'
    with pytest.raises(ValueError): memory().observe(w)


def test_bounded_history_read_limits_and_gap_reset():
    m = memory()
    for i in range(20): m.observe(window(i))
    s = m.read(now=11.01)
    assert len(s['rows']) == 16 and s['endpoint_span_s'] == 7.5
    assert s['raw_coverage_s'] == 9
    assert len(m.read(now=11.01, limit=1)['rows']) == 1
    assert len(m.read(now=11.01, limit=8)['rows']) == 8
    with pytest.raises(ValueError): m.read(now=11.01, limit=17)
    assert not m.read(now=13)['valid']
    assert m.observe(window(23))['gap_reset']
    assert len(m.rows) == 1
    m.reset('new', 'episode'); assert not m.read(now=0)['valid']
    with pytest.raises(ValueError): m.observe(window())


def test_history_changes_readout_when_latest_state_is_identical():
    torch.manual_seed(42)
    a, b = memory(), memory()
    a.observe(window(0)); b.observe(window(0, local_visual=np.full(1024, -5.)))
    a.observe(window(1)); b.observe(window(1))
    model = RGBMemoryReader(np.zeros(58), np.ones(58)).eval()
    ba, bb = (reader_batch(m.read(now=2.01)) for m in (a, b))
    with torch.no_grad(): oa, ob = model(**ba), model(**bb)
    assert oa['current_logits'].shape == (1, 4)
    assert not torch.allclose(oa['current_logits'], ob['current_logits'])
    # Same newest observation alone is identical; older input is the only change.
    with torch.no_grad():
        la = model(**reader_batch(a.read(now=2.01, limit=1)))
        lb = model(**reader_batch(b.read(now=2.01, limit=1)))
    torch.testing.assert_close(la['current_logits'], lb['current_logits'])
    model.train(); model(**ba)['current_logits'].sum().backward()
    assert model.attention.in_proj_weight.grad.abs().sum() > 0


def test_roi_geometry_enters_reader_and_invalid_is_explicit():
    a, b = memory(), memory()
    a.observe(window()); b.observe(window(roi_valid=np.zeros(16, dtype=bool)))
    ba, bb = (reader_batch(m.read(now=1.51)) for m in (a, b))
    assert not torch.equal(ba['metadata'], bb['metadata'])
    assert ba['positions'].shape == (1, 1, 928)


def test_crop_at_native_resolution_and_edge_padding_and_fresh_fallback():
    image = np.arange(400*640*3, dtype=np.uint8).reshape(400, 640, 3)
    r = crop_rgb(image, [400, 230, 490, 310], size=135)
    left, top, right, bottom = r['crop_box']
    np.testing.assert_array_equal(r['rgb'], image[top:bottom, left:right])
    edge = square_context([0, 0, 640, 400], 640, 400)
    assert edge[2]-edge[0] == edge[3]-edge[1] == 640
    old = crop_rgb(np.zeros_like(image), valid=False)
    fresh = crop_rgb(np.full_like(image, 255), valid=False)
    assert not fresh['valid'] and fresh['fallback']
    assert old['rgb'].max() == 0 and fresh['rgb'].max() == 255


def test_fourth_view_reaches_frozen_encoder_and_does_not_replace_original_views():
    # Synthetic encoder verifies actual routing, NOT real visual quality.
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        def forward(self, pixel_values_videos):
            from types import SimpleNamespace
            means = pixel_values_videos.mean((1, 2, 3, 4))
            return SimpleNamespace(last_hidden_state=means[:, None, None].expand(-1, 2, 1024))
    encoder = Encoder().eval()
    global_rgb = np.zeros((1, 3, 16, 256, 256, 3), dtype=np.uint8)
    dark = np.zeros((1, 16, 256, 256, 3), dtype=np.uint8)
    a = encode_four_views(encoder, global_rgb, dark)
    b = encode_four_views(encoder, global_rgb, np.full_like(dark, 255))
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    assert not np.array_equal(a[:, 3], b[:, 3])
