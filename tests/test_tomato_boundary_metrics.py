import numpy as np
import pytest
from tools import tomato_boundary_metrics as b


def test_events_are_causal_confirmed_and_initialization_is_not_a_change():
    p=np.eye(4)[[0,0,1,1,2,2]];t=np.arange(6)*.5
    events=b.change_events(p,t,.8,2)
    assert events==[dict(time=1.5,previous=0,label=1),dict(time=2.5,previous=1,label=2)]
    for n in range(1,7):
        assert b.change_events(p[:n],t[:n],.8,2)==[v for v in events if v['time']<=t[n-1]]
    assert b.change_events(np.full((3,4),.25),np.arange(3)*.5,.8,2)==[]


def entry():
    return dict(episode=dict(episode_id='s',fps=2.,annotation=dict(segments=[
        dict(start_frame=0,end_frame=4,label='pickup'),dict(start_frame=4,end_frame=10,label='harvest')])),
        times=np.arange(10)*.5,future_valid_mask=np.arange(10)<8)


def test_strict_early_warning_and_false_events():
    r=entry();z=np.eye(4)[[0,0,0,1,1,1,1,1,1,1]]*10
    early=b.event_report([r],z,'future',1.,.8,1)
    assert early['recall']==1 and early['lead_p50_s']==.5 and early['unmatched_events']==0
    late=b.event_report([r],z,'current',1.,.8,1)
    assert late['recall']==0 and late['unmatched_events']==1
    at_boundary=np.eye(4)[[0,0,0,0,1,1,1,1,1,1]]*10
    assert b.event_report([r],at_boundary,'future',1.,.8,1)['recall']==0
    assert b.event_report([r],at_boundary,'current',1.,.8,1)['recall']==1
