import numpy as np
import pytest
from tools.tomato_causal_probability_filter import CausalProbabilityFilter
from tools.tomato_state_recovery import B0State


def test_half_life_and_actual_timestamp_not_frame_counter():
    f=CausalProbabilityFilter()
    f.update([1,0,0,0],0)
    p,reset=f.update([0,1,0,0],.25)
    assert not reset
    assert np.allclose(p,[2**-.5,1-2**-.5,0,0])
    assert p.sum()==pytest.approx(1)


def test_prefix_causality_and_no_array_alias():
    prefix=[[1,0,0,0],[0,1,0,0],[0,1,0,0]]
    def run(tail):
        f=CausalProbabilityFilter()
        return [f.update(p,i*.5)[0] for i,p in enumerate(prefix+tail)]
    a,b=run([[0,0,1,0]]),run([[0,0,0,1]])
    assert np.array_equal(a[:3],b[:3])
    f=CausalProbabilityFilter();p=np.array([1.,0,0,0]);r,_=f.update(p,0)
    p[:]=0;r[:]=0
    assert f.value[0]==1


def test_gap_and_episode_restart_discard_history():
    f=CausalProbabilityFilter();f.update([1,0,0,0],0)
    p,reset=f.update([0,0,0,1],1.)
    assert reset and np.array_equal(p,[0,0,0,1])
    assert CausalProbabilityFilter().update([0,1,0,0],0)[1]


def test_stale_invalid_observations_cannot_update_filter():
    f=CausalProbabilityFilter();f.update([1,0,0,0],1)
    for p,t in [([0,1,0,0],1),([0,1,0,0],.5),([0,float('nan'),0,0],2),([1,1,0,0],2)]:
        with pytest.raises(ValueError):f.update(p,t)
        assert f.last==1 and f.value[0]==1


def test_filtering_future_does_not_modify_current_prompt():
    base,other=B0State(),B0State();f=CausalProbabilityFilter()
    for i in range(20):
        current=np.eye(4)[(i//4)%4]
        future=np.eye(4)[(i//2)%4]
        smooth,_=f.update(future,i*.5)
        a=base.update(current,future,i*.5);b=other.update(current,smooth,i*.5)
        assert (a['phase'],a['changed'])==(b['phase'],b['changed'])
