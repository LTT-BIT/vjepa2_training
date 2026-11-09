import numpy as np
from tools.audit_tomato_transitions import match_events
from tools import tomato_boundary_metrics as b, tomato_dataset as ds


def truth(t=2.,end=5.):
    return dict(time=t,end=end,previous=0,label=1)


def test_strict_matching_is_one_to_one_and_half_open():
    events=[dict(time=1.5,previous=0,label=1)]
    rows,false=match_events(events,[truth(),truth(2.2)],'future',1.)
    assert [r['detected'] for r in rows]==[True,False] and not false
    assert not match_events([dict(time=2.,previous=0,label=1)],[truth()],'future',1.)[0][0]['detected']
    assert match_events([dict(time=1.,previous=0,label=1)],[truth()],'future',1.)[0][0]['detected']
    assert not match_events([dict(time=1.5,previous=2,label=1)],[truth()],'future',1.)[0][0]['detected']


def test_current_only_matches_after_boundary_and_before_segment_end():
    for t,expected in [(1.9,False),(2.,True),(3.9,True),(4.,False)]:
        rows,_=match_events([dict(time=t,previous=0,label=1)],[truth()],'current',1.)
        assert rows[0]['detected']==expected
    assert not match_events([dict(time=3.,previous=0,label=1)],[truth(end=3.)],'current',1.)[0][0]['detected']


def test_confirmation_count_and_duration_have_different_rate_effects():
    t2=np.arange(8)/2;t4=np.arange(16)/4
    p2=np.eye(4)[(t2>=1).astype(int)];p4=np.eye(4)[(t4>=1).astype(int)]
    assert b.change_events(p2,t2,.5,2)[0]['time']==1.5
    assert b.change_events(p4,t4,.5,2)[0]['time']==1.25
    assert b.change_events(p4,t4,.5,3)[0]['time']==1.5


def test_real_four_hz_grid_is_fresh_and_causal():
    ts=np.arange(120)/30
    a=ds.iter_phase_anchors(ts,4)
    assert len(set(a))==len(a) and set(np.diff(a))=={7,8}
    for i in a:
        h=ds.sample_causal_indices(ts,int(i))
        assert h[-1]==i and np.all(h<=i)
