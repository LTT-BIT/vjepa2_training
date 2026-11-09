import pytest
from tools.tomato_review_ui.handshake import *

def evidence(t,p=.9,current=0,future=1,lat=0):
    def probs(k,p):v=[(1-p)/3]*4;v[k]=p;return tuple(v)
    return Evidence(t,t+lat,probs(current,.95),probs(future,p))

def test_k2_prepare_then_commit_without_third_anchor():
    h=Handshake(dict(K=2));h.prediction(evidence(53,.728));h.prediction(evidence(53.5,.876))
    assert h.state=='SWITCH_REQUESTED' and h.pending['request_at']==53.5
    h.planning(53.5);assert h.active=='harvest' and h.version==1

def test_k1_same_anchor_and_catchup_no_current_gate():
    h=Handshake({});h.prediction(evidence(10,current=1));h.planning(10)
    assert h.version==1 and next(e for e in h.events if e['kind']=='propose')['pending']['catch_up']

@pytest.mark.parametrize('value',[.60])
def test_withdraw(value):
    h=Handshake(dict(K=2));h.prediction(evidence(0,.728));h.prediction(evidence(.5,value));assert h.pending is None

def test_ack_requires_post_commit_observation():
    h=Handshake({});h.prediction(evidence(0));h.planning(.5)
    h.prediction(evidence(.5,current=1));assert h.state=='WAIT_CURRENT_ACK'
    h.prediction(evidence(1,current=1));assert any(e['kind']=='ack' for e in h.events)

@pytest.mark.parametrize('timeout',[1,1.5,2])
def test_timeout_retains_and_blocks(timeout):
    h=Handshake(dict(ack_timeout_s=timeout));h.prediction(evidence(0));h.planning(.5);h.timer(.5+timeout)
    assert h.state=='RECOVERY_REQUIRED' and h.active=='harvest'
    h.prediction(evidence(3,current=1,future=2));assert h.version==1
    h.recover(4,'operator_review');assert h.state=='READY'

@pytest.mark.parametrize('config',[dict(theta_prepare=.9,theta_commit=.8),dict(theta_commit=float('nan')),dict(K=3),dict(theta_prepare='x'),dict(ack_timeout_s=.5)])
def test_bad_config(config):
    with pytest.raises(ValueError):validate(config)

def test_distinct_anchors_and_span():
    h=Handshake(dict(K=2));h.prediction(evidence(0))
    with pytest.raises(ValueError):h.prediction(evidence(0))
    h.prediction(evidence(.5));assert h.pending['last_confirm_available_at']==.5

def test_gap_cancels_and_illegal():
    h=Handshake(dict(K=2));h.prediction(evidence(0));h.timer(1);assert h.pending is None
    h.prediction(evidence(1,future=2));assert h.events[-1]['kind']=='illegal_candidate'

def test_replay_commit_boundaries_and_gt_not_input():
    def row(t):return dict(observed_at=t,heads={k:dict(probabilities=dict(zip(LABELS,evidence(t).current if k=='current' else evidence(t).future))) for k in ['current','future']},future_valid=False,current_true_stage='return')
    r=replay([row(10),row(10.5)],dict(K=2),12)
    commit=next(e for e in r['events'] if e['kind']=='commit');assert commit['time']==11
    assert any(v['start_s']==11 for v in r['active_intervals'])
    assert commit['evidence']['first_available_at']==10.09
    r2=replay([row(10),row(10.5)],dict(K=2,plan_offset_s=.25),12)
    assert next(e for e in r2['events'] if e['kind']=='commit')['time']==10.75

def test_prompt_snapshot_is_immutable_and_not_retroactive():
    from dataclasses import FrozenInstanceError
    h=Handshake({});snapshot=h.snapshot();h.prediction(evidence(0));h.planning(0)
    assert snapshot.active_prompt=='pickup' and snapshot.version==0
    with pytest.raises(FrozenInstanceError):snapshot.version=4

def test_latest_margin_is_required_only_at_request():
    h=Handshake(dict(K=2,margin=.9));h.prediction(evidence(0,.728));h.prediction(evidence(.5,.876));assert h.state=='CANDIDATE_PENDING'
    h.prediction(evidence(1,.99));assert h.state=='SWITCH_REQUESTED'

def test_preplanning_single_pulse_retract_and_future_equal_active():
    h=Handshake({});h.prediction(evidence(0,.99));h.prediction(evidence(.5,future=0,current=1));h.planning(1);assert h.version==0

def test_invalid_quality_never_confirms_and_clears_candidate():
    h=Handshake(dict(K=2));h.prediction(evidence(0));m=evidence(.5)
    h.prediction(Evidence(m.observed_at,m.available_at,m.current,m.future,valid=False));assert h.pending is None
