import pytest
from tools.evaluate_tomato_prompt_switch import score_episode,match,aggregate,bootstrap_pair
from tools.tomato_prompt_replay import replay
from test_tomato_prompt_replay import msg,cfg


def ep():
    return dict(episode_id='episode',source_recording_id='source',fps=30,length=60,
        annotation=dict(segments=[dict(label='pickup',start_frame=0,end_frame=30),dict(label='harvest',start_frame=30,end_frame=60)]))


def test_exact_continuous_duration_and_no_initial_boundary():
    r=replay([msg(0),msg(.5)],cfg(),2.)
    s=score_episode(ep(),r)
    assert s['wrong_prompt_duration_s']==1.
    assert s['early_wrong_prompt_duration_s']==1.
    assert s['matched_transitions']==s['eligible_transitions']==1
    assert s['deltas']==[-1.]
    assert s['full_episode_order_exact']
    assert s['post_first_error_recovery_s']==1.
    assert s['strict_window_matched']==0
    assert aggregate([s])['premature_ge_05']==1


def test_matching_is_ordered_one_to_one_and_pair_specific():
    b=[dict(time=1.,from_stage='pickup',to_stage='harvest'),dict(time=1.5,from_stage='pickup',to_stage='harvest')]
    c=[dict(time=1.2,from_stage='pickup',to_stage='harvest')]
    assert match(b,c)==((0,0),)
    c[0]['from_stage']='return'
    assert match(b,c)==()


def test_sequence_correct_does_not_imply_timing_correct():
    s=score_episode(ep(),replay([msg(0),msg(.5)],cfg(),2.))
    assert s['full_episode_order_exact'] and s['premature_ge_1']==1


def test_no_commit_metrics_null_not_fabricated():
    s=score_episode(ep(),replay([msg(0,0)],cfg(),2.))
    a=aggregate([s]);assert a['commit_delta_s']['p50'] is None
    assert a['commit_delta_s']['reason']=='no_matched_commit'
    assert s['post_first_error_recovery_s'] is None


def test_bootstrap_paired_identity_and_mismatch_failure():
    s=score_episode(ep(),replay([msg(0)],cfg(),2.))
    ci=bootstrap_pair([s],[s],n=20)
    assert all(v==[0.,0.] for v in ci['intervals'].values())
    with pytest.raises(ValueError):bootstrap_pair([s],[{**s,'source_recording_id':'bad'}],n=2)


def test_confirmation_release_planning_timing_components_sum():
    r=replay([msg(0),msg(.5)],cfg('P3'),2.,latency=.09,period=.5)
    s=score_episode(ep(),r);t=s['timing'][0]
    assert t['confirmation_wait_s']==pytest.approx(.5)
    assert sum(t[k] for k in ['observation_to_available_s','confirmation_wait_s','deferred_release_wait_s','planning_wait_s'])==pytest.approx(t['observation_to_commit_s'])
