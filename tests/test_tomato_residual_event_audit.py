import numpy as np
import pytest
from tools.tomato_residual_event_audit import boundary_evidence,false_changes,nested_folds,accepted_events


def test_cross_boundary_confirmation_does_not_backdate_event():
    t=np.arange(8.,11.,.5);p=np.eye(4)[[0,0,0,1,1,1]]
    events=accepted_events(p,t)
    r=boundary_evidence(t,p,dict(time=10.,end=15.,previous=0,label=1),0,events)
    assert r['first_in_strict_window_s']==9.5 and r['first_target_K2_candidate_s']==10.
    assert r['cross_boundary_K2'] and events[0]['time']==10.


def test_early_accepted_class_is_distinct_from_window_candidate():
    t=np.arange(6.,11.,.5);p=np.eye(4)[[0,0]+[1]*8]
    events=accepted_events(p,t);r=boundary_evidence(t,p,dict(time=10.,end=15.,previous=0,label=1),0,events)
    assert len(r['early_accepted_events'])==1 and r['first_in_strict_window_s']==8.5
    assert not r['cross_boundary_K2'] and r['potential_target_pair_spans_boundary']


def test_false_shift_pairing_does_not_match_one_event_twice():
    event=lambda t:dict(time=t,previous=3,label=0)
    r=false_changes([event(5),event(6)],[event(5.5)])
    assert len(r['removed_exact'])==2 and len(r['added_exact'])==1
    assert len(r['shifted_false_to_false_diagnostic'])==1


def test_nested_fit_selection_outer_are_disjoint_and_cover_pool_once():
    ids=[str(i) for i in range(10)];folds=nested_folds(ids)
    assert sorted(v for f in folds for v in f['outer'])==sorted(ids)
    for f in folds:
        assert all(type(v) is str for values in f.values() for v in values)
        assert len(f['fit'])==6 and len(f['selection'])==2 and len(f['outer'])==2 and len(f['refit'])==8
        assert not set(f['refit'])&set(f['outer'])
    assert nested_folds(list(reversed(ids)))==folds
    with pytest.raises(ValueError):nested_folds(ids[:-1]+[ids[0]])
