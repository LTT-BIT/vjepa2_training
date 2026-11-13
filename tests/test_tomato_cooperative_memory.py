"""Synthetic contract tests only; these are not reviewed operation labels."""
from dataclasses import replace
import numpy as np
import pytest
from tools.tomato_cooperative_memory import CooperativeMemory, Frame
from tools.tomato_operation_memory import EvidenceRecord

NAMES = tuple(f'{side}_fr3_joint{i}.position' for side in ('left','right') for i in range(1,8)) + tuple(f'{side}_hand{i}.position' for side in ('left','right') for i in range(22))


def new():return CooperativeMemory('s','e','frozen-b0',NAMES,'frozen3cam-mean-v1')


def observe(m,i,label=0,q=0.):
    p=np.full(4,.01/3);p[label]=.99;t=i*.5
    row=Frame('s','e',str(i),t,t+.01,t,t,np.ones((3,1024)),np.full(58,q),p,p)
    return m.observe(row,encoder_signature='frozen3cam-mean-v1')


def evidence(key,t,event,verdict='positive'):
    return EvidenceRecord(key,'s','e','test-only',t,event,'verified_operation_result',
        'independent_operation_review',None,verdict=verdict,reviewers=('synthetic-a','synthetic-b'),
        independent_label=True,reason='SYNTHETIC UNIT FIXTURE, NOT REAL HUMAN REVIEW')


def test_same_hand_pose_different_goal_context_without_claiming_success():
    m=new();observe(m,0);first=m.read(now=.01)
    for i,k in enumerate([1,1,2,2],1):observe(m,i,k)
    returned=m.read(now=2.01)
    assert np.array_equal(first['recent'][-1]['positions'],returned['recent'][-1]['positions'])
    assert first['goal']=='acquire_truss' and returned['goal']=='place_and_release_truss'
    assert first['result_status']==returned['result_status']=='unknown'
    assert returned['robot_action'] is None and not returned['suggests_prompt_update']


def test_feature_and_motion_memory_do_not_confirm_outcome():
    m=new();observe(m,0);observe(m,1,q=1.)
    r=m.read(now=.51)
    assert np.allclose(r['recent'][-1]['velocity'],2.)
    assert r['result_status']=='unknown' and r['guidance']=='need_result_evidence'
    assert len(r['recent'])==2 and r['goal_anchors'][0]['kind']=='stage_decision'


def test_independent_result_is_advisory_not_prompt_change():
    m=new();observe(m,0);observe(m,1)
    assert m.add_result(evidence('lift',.5,'truss_stably_lifted'),available_at=.5,now=.51)
    r=m.read(now=.51)
    assert r['result_status']=='verified' and r['guidance']=='request_transition_review'
    assert m.b0.phase==0 and r['robot_action'] is None
    assert not m.add_result(evidence('lift',.5,'truss_stably_lifted'),available_at=.5,now=.51)


def test_same_result_type_not_reused_for_second_harvest():
    m=new()
    for i,k in enumerate([0,1,1]):observe(m,i,k)
    m.add_result(evidence('first_detach',1.,'tomato_detached'),available_at=1.,now=1.01)
    assert m.read(now=1.01)['result_status']=='verified'
    for i,k in enumerate([2,2,3,3,0,0,1,1],3):observe(m,i,k)
    assert m.b0.phase==5 and m.read(now=5.01)['result_status']=='unknown'


def test_failure_unknown_and_conflict_remain_distinct():
    m=new();observe(m,0);observe(m,1)
    m.add_result(evidence('bad',.5,'truss_stably_lifted','negative'),available_at=.5,now=.51)
    assert m.read(now=.51)['result_status']=='failure_evidence'
    assert not m.add_result(evidence('good',.5,'truss_stably_lifted'),available_at=.5,now=.51)
    assert m.read(now=.51)['result_status']=='unknown'
    assert m.ledger.snapshot(now=.51)['conflicts']


def test_stage_proxy_and_unregistered_verifier_rejected():
    m=new();observe(m,0)
    e=evidence('x',0.,'truss_stably_lifted')
    with pytest.raises(ValueError):m.add_result(replace(e,kind='stage_prediction'),available_at=0.,now=.01)
    with pytest.raises(ValueError):m.add_result(replace(e,source='outcome_verifier:unregistered',reviewers=(),independent_label=False,qualified_verifier=True,probability=.99,support_ids=('a','b')),available_at=0.,now=.01)


def test_gap_stale_input_and_schema_are_conservative():
    m=new();observe(m,0);observe(m,4,q=1.)
    assert len(m.recent)==1 and m.recent[-1]['velocity'] is None
    assert not m.read(now=3.1)['valid']
    p=np.array([1.,0.,0.,0.]);bad=Frame('s','e','bad',2.5,2.51,2.,2.5,np.ones((3,1024)),np.zeros(58),p,p)
    with pytest.raises(ValueError):m.observe(bad,encoder_signature='frozen3cam-mean-v1')
    assert m.last==2.


def test_bounds_reset_and_snapshot_copy():
    m=new()
    for i in range(30):observe(m,i,q=i*.01)
    assert len(m.recent)==16 and len(m.anchors)<=12
    r=m.read(now=14.51);r['recent'][-1]['positions'][:]=99
    assert not np.any(m.recent[-1]['positions']==99)
    m.reset('s2','e2');assert not m.read(now=15.)['valid']


def test_result_expiry_and_future_evidence_rejected():
    m=new();observe(m,0);observe(m,1)
    with pytest.raises(ValueError):m.add_result(evidence('future',1.,'truss_stably_lifted'),available_at=1.,now=1.01)
    m.add_result(replace(evidence('past',.5,'truss_stably_lifted'),valid_until=.7),available_at=.5,now=.51)
    assert m.read(now=.8)['result_status']=='unknown'


def test_second_pickup_retrieves_return_and_adjustment_context():
    m=new()
    for i,k in enumerate([0,1,1,2,2,3,3,0,0]):observe(m,i,k)
    r=m.read(now=4.01)
    assert r['phase']==4 and r['context_phases']==(2,3,4)
    assert {a['goal_phase'] for a in r['goal_anchors']}=={2,3,4}
    assert all(a['historical_only'] for a in r['goal_anchors'] if a['goal_phase']!=4)


def test_result_cannot_be_backdated_from_later_available_evidence():
    m=new();observe(m,0);observe(m,1)
    with pytest.raises(ValueError):
        m.add_result(evidence('backdated',0.,'truss_stably_lifted'),available_at=.5,now=.51)
    assert m.read(now=.51)['result_status']=='unknown'
