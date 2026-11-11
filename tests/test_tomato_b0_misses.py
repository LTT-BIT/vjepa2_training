import numpy as np
from tools.analyze_tomato_b0_misses import classify_miss


def diag(labels,events):
    t=np.array([8.5,9.,9.5,10.,10.5,11.])
    p=np.full((6,4),.01/3)
    for i,k in enumerate(labels):p[i,k]=.99
    return classify_miss(t,p,dict(time=10.,end=15.,previous=0,label=1),1,0.,events)


def test_early_event_is_not_a_new_event_inside_correct_probability_window():
    r=diag([1]*6,[dict(time=8.,previous=0,label=1)])
    assert r['kind']=='too_early_accepted_transition' and r['correct_argmax_samples']==3


def test_existing_K2_confirmation_can_straddle_boundary():
    assert diag([0,0,1,1,1,1],[dict(time=10.,previous=0,label=1)])['kind']=='K2_confirmation_crosses_boundary'


def test_transient_candidate_and_no_candidate_are_different():
    assert diag([0,0,1,0,1,1],[dict(time=11.,previous=0,label=1)])['kind']=='transient_future_candidate_not_stable'
    assert diag([0,0,0,1,1,1],[dict(time=10.5,previous=0,label=1)])['kind']=='no_qualifying_future_in_window'


def test_wrong_source_class_is_not_credited_as_correct_transition():
    assert diag([2,1,1,1,1,1],[dict(time=9.,previous=2,label=1)])['kind']=='wrong_from_class'
