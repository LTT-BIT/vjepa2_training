import numpy as np
import pytest
from tools import tomato_boundary_metrics as bm
from tools.tomato_decision_attribution import decisions
from tools.tomato_state_recovery import StateRecovery, OrderedBeam, CLASSES


def p(k, confidence=.97):
    row = np.full(4, (1-confidence)/3)
    row[k] = confidence
    return row


def replay(mode, c, f=None, times=None):
    f = c if f is None else f
    times = np.arange(len(c))*.5 if times is None else times
    s = StateRecovery(mode)
    return [s.update(a, b, t) for a, b, t in zip(c, f, times)]


def test_b0_exact_frozen_warning_and_prompt_algorithms():
    rng = np.random.default_rng(20261110)
    c, f = rng.dirichlet([.2]*4, 200), rng.dirichlet([.2]*4, 200)
    ts = np.arange(len(c))*.5
    expected_warnings = bm.change_events(f, ts, .5, 2)
    _, expected_prompts = decisions(c, f, ts, .8, 2, True)
    a, b = replay('baseline', c, f), replay('detect', c, f)
    assert [r['warning'] for r in a if r['warning']] == expected_warnings
    assert [r['phase'] for r in a] == [r['phase'] for r in expected_prompts]
    for x, y in zip(a, b):
        assert (x['warning'], x['phase'], x['changed']) == (y['warning'], y['phase'], y['changed'])


@pytest.mark.parametrize('mode', ['baseline', 'detect', 'hypotheses', 'resync'])
def test_causal_prefix_and_future_alone_never_advances_prompt(mode):
    c = np.asarray([p(0)]*10 + [p(1)]*10 + [p(2)]*10)
    full = replay(mode, c)
    assert replay(mode, c[:17]) == full[:17]
    r = replay(mode, [p(0)]*50, [p(3)]*50)
    assert all(v['phase'] == 0 for v in r)


@pytest.mark.parametrize('mode', ['hypotheses', 'resync'])
def test_repeated_class_argmax_cannot_teleport_or_skip_unobserved_stage(mode):
    # High second-pickup-like observations provide no evidence of adjust.
    c = [p(0)]*8 + [p(1)]*8 + [p(2)]*8 + [p(0)]*20 + [p(1)]*20 + [p(2)]*20
    r = replay(mode, c)
    assert max(v['phase'] for v in r) <= 2
    assert max(h['phase'] for v in r for h in v['hypotheses']) <= 2
    for v in r:
        assert len(v['hypotheses']) <= 3
        assert abs(sum(h['probability'] for h in v['hypotheses'])-1) < 1e-10
    # Full observed sequence must distinguish both passes and both return stages.
    c = [p(k) for k in CLASSES for _ in range(8)]
    r = replay(mode, c)
    assert r[-1]['phase'] == 6
    assert all(v['phase'] >= v['previous_phase'] for v in r)
    for v in r:
        if v['changed']:
            assert [w['phase'] for w in v['witness']] == list(range(v['previous_phase']+1, v['phase']+1))
            assert all(w['end'] <= v['time'] and w['count'] >= 2 for w in v['witness'])


def test_invalid_identity_timestamp_and_gap_do_not_complete_evidence():
    s = StateRecovery('resync')
    s.update(p(0), p(0), 0.)
    for kw in ({'timestamp': 0.}, {'timestamp': 1., 'now': 3.},
               {'timestamp': 1., 'session_id': 'other'}, {'timestamp': 1., 'model_id': 'other'}):
        assert not s.update(p(1), p(1), **kw)['valid']
    assert not s.update([np.nan]*4, p(1), 1.)['valid']
    s.update(p(1), p(1), 2.)
    s.update(p(1), p(1), 4.)
    assert s.beam.runs[1] == 1
    assert s.phase == 0
    s.reset('new')
    assert s.phase == 0 and s.last is None


def test_single_ambiguous_frame_and_time_head_are_not_evidence():
    r = replay('resync', [p(0)]*4 + [p(3)] + [p(0)]*4)
    assert all(x['phase'] == 0 for x in r)
    with pytest.raises(TypeError):
        StateRecovery('resync').update(p(0), p(1), 0., truth_phase=3)
    with pytest.raises(ValueError):
        OrderedBeam(width=7)


def test_resync_requires_witnessed_intermediate_phase_and_does_not_repeat():
    # A short weak harvest is actually observed, but never reaches the commit
    # posterior. Strong return can recover only along that witnessed path.
    c = [p(0)]*4 + [p(1, .51)]*2 + [p(2)]*6
    a, b = replay('hypotheses', c), replay('resync', c)
    assert a[-1]['phase'] == 0
    recovered = [v for v in b if v['resynchronized']]
    assert len(recovered) == 1
    assert [v['phase'] for v in recovered[0]['witness']] == [1, 2]
    assert recovered[0]['phase'] == 2 and b[-1]['prompt_version'] == 1
    absent = replay('resync', [p(0)]*4 + [p(2)]*20)
    assert absent[-1]['phase'] == 0


def test_mismatch_metrics_include_latency_but_persistent_is_separate():
    from tools.evaluate_tomato_state_recovery import state_score, prompt_score
    ts = np.arange(8)*.5
    r = state_score([0,0,0,0,0,0,1,1], np.ones(8), ts)
    assert r['mismatch_s'] == 3 and r['persistent_runs'] == 1
    assert r['runs'][0]['recovered']
    r = state_score([0,1,1,1,1,1,1,1], np.ones(8), ts)
    assert r['mismatch_s'] == .5 and r['persistent_runs'] == 0
    # A 0->2 recovery recommendation is one update; phase1 gets no invented credit.
    trace = [dict(time=1., previous_phase=0, phase=2, prompt_version=1,
                  changed=True, resynchronized=True)]
    truth = [dict(time=.5,end=2.,previous=0,label=1), dict(time=1.,end=3.,previous=1,label=2)]
    score = prompt_score(trace,truth,ts)
    assert score['counts'] == {'missing':1,'on_time':1}
    assert score['skipped_prompt_positions'] == 1


def test_attachable_detector_reports_contradiction_never_resets_repeated_phase():
    from tools.tomato_state_recovery import StateInconsistencyDetector
    d = StateInconsistencyDetector()
    # State stuck at first return; pickup contradicts allowed return/adjust.
    assert not d.update(p(0), 0., 2)['inconsistent']
    r = d.update(p(0), .5, 2)
    assert r['inconsistent'] and not r['recommends_reset']
    # Neither first nor second return can be distinguished by a return emission.
    for phase in [2, 6]:
        d = StateInconsistencyDetector()
        assert not d.update(p(2), 0., phase)['inconsistent']
        assert not d.update(p(2), .5, phase)['inconsistent']
    d = StateInconsistencyDetector()
    assert not d.update(p(3), 0., 2)['inconsistent']  # allowed next class
    assert not d.update(p(3), .5, 2)['inconsistent']


def test_attachable_detector_invalid_and_gapped_input_cannot_confirm():
    from tools.tomato_state_recovery import StateInconsistencyDetector
    d = StateInconsistencyDetector()
    assert not d.update('invalid', 0., 2)['valid']
    assert not d.update(p(0), 'invalid', 2)['valid']
    assert not d.update(p(0), 0., 9)['valid']
    assert not d.update(p(0), 0., 2)['inconsistent']
    assert not d.update(p(0), 2., 2)['inconsistent']
    assert not d.update(p(0), 2., 2)['valid']
    assert not d.update(p(0), 2.5, 2)['inconsistent']
