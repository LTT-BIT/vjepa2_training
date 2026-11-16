import dataclasses
import json
import math
import subprocess
import sys
import pytest
from tools.tomato_prompt_policy import PolicyInput, PromptPolicy
from tools.tomato_prompt_replay import replay, whitelist


def msg(t,stage=1,**kwargs):
    p=[.01]*4;p[stage]=.97
    return PolicyInput('episode','source',round(t*30),t,t,'future_only',tuple(p),**kwargs)


def cfg(name='P0',**kwargs):
    return dict(id=name,theta=.8,K=2,margin=0.,d=.25,prediction_ttl_s=.75,**kwargs)


def test_no_torch_import():
    subprocess.run([sys.executable,'-c',"import sys; import tools.tomato_prompt_replay; import tools.evaluate_tomato_prompt_switch; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"],check=True)


def test_gt_and_mask_physically_removed():
    r=dict(episode_id='e',source_recording_id='s',frame_index=0,timestamp_s=0,
        future_probabilities=dict(pickup=.1,harvest=.7,**{'return':.1},adjust=.1),
        current_true_stage='return',future_true_stage='harvest',future_valid=False,future_target_time_s=None)
    a=whitelist(r,'future_only');r.update(future_valid=True,future_target_time_s=999,current_true_stage='pickup')
    assert a==whitelist(r,'future_only')
    assert not any('true' in x.name or x.name=='future_valid' for x in dataclasses.fields(a))
    with pytest.raises(TypeError):PolicyInput(**{**dataclasses.asdict(a),'future_valid':True})


@pytest.mark.parametrize('p',[(float('nan'),0,0,1),(-.1,.1,0,1),(0,0,0,0),(1,0)])
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):PolicyInput('e','s',0,0,0,'m',p)


@pytest.mark.parametrize('latency,expected',[(0.,0.),(.09,.5)])
def test_availability_before_same_time_planning(latency,expected):
    r=replay([msg(0)],cfg(),1.,latency=latency)
    assert r['commits'][0]['time']==expected
    assert r['commits'][0]['time']>=latency


def test_k_needs_distinct_anchors_not_planning_hold():
    r=replay([msg(0)],cfg('P2'),.7,period=.05)
    assert not r['commits']
    r=replay([msg(0),msg(.5)],cfg('P2'),1.,period=.05)
    assert r['commits'][0]['time']==.5
    with pytest.raises(ValueError):replay([msg(0),msg(0)],cfg('P2'),1.)


def test_retract_same_time_before_pending_commit():
    r=replay([msg(0),msg(.5),msg(1.,0)],cfg('P3'),2.,period=.5)
    assert not r['commits']
    assert any(e['kind']=='cancel' and e['reason']=='candidate_equals_active' for e in r['events'])


def test_delayed_release_and_original_target_do_not_move():
    r=replay([msg(0),msg(.5)],cfg('P3'),1.2,period=.25)
    c=r['commits'][0]
    assert c['time']==.75 and c['active']['source_target_time']==1.5
    assert c['request']['release_at']==.75


def test_candidate_change_cannot_inherit_count():
    r=replay([msg(0),msg(.5,2)],cfg('P2'),1.,period=.1)
    assert not r['commits']


def test_drop_gap_expires_and_end_does_not_finish():
    c=cfg('P3');c['d']=1.
    r=replay([msg(0),msg(.5)],c,2.)
    assert not r['commits'] and any(e['kind']=='expire' for e in r['events'])
    assert r['events'][-1]['reason']=='data_stopped_not_FINISH'
    assert all(a['stage']!='FINISH' for a in r['intervals'])


def test_reverse_unknown_and_noncausal_fail():
    with pytest.raises(ValueError):replay([msg(.5),msg(0)],cfg(),2.)
    with pytest.raises(ValueError):PolicyInput('e','s',30,1.,.9,'m',(.25,)*4)
    with pytest.raises(ValueError):PromptPolicy(dict(id='undefined'))
    with pytest.raises(ValueError):replay([msg(0)],cfg(),2.,latency=.6)


def test_p4_legal_edges_and_no_gt_reset():
    r=replay([msg(0,2),msg(.5,2),msg(1.,1),msg(1.5,1)],cfg('P4'),2.5,period=.25)
    assert sum(e['kind']=='illegal_edge' for e in r['events'])==2
    assert [(c['from_stage'],c['to_stage']) for c in r['commits']]==[('pickup','harvest')]


def test_gating_requires_actual_current_head_and_is_separate():
    c=cfg();c['current_gate']=True
    with pytest.raises(ValueError):replay([msg(0)],c,1.)
    r=replay([msg(0,joint_current_probabilities=(.01,.97,.01,.01))],c,1.)
    assert not r['commits']


def test_tail_messages_not_filtered_and_no_late_result_at_end():
    r=replay([msg(0),msg(.5,2)],cfg(),.75,latency=.09,period=.25)
    assert r['consumed_anchors']==2
    assert r['commits'][0]['time']==.25
    late=replay([msg(0),msg(.5,2)],cfg(),.55,latency=.09)
    assert late['consumed_anchors']==1


def test_deterministic_normalized_event_logs():
    a=replay([msg(0),msg(.5)],cfg('P3'),2.,latency=.09,period=.25)
    b=replay([msg(0),msg(.5)],cfg('P3'),2.,latency=.09,period=.25)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)


def test_increasing_latency_never_uses_future_result():
    for latency in [0,.09,.2,.3]:
        r=replay([msg(0),msg(.5)],cfg('P2'),1.5,latency=latency,period=.25)
        assert all(c['time']>=c['request']['source_available_at'] for c in r['commits'])
        assert r['commits'][0]['time']>=.5+latency
