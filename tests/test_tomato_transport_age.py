from tools import tomato_dataset as ds
from tools.tomato_task_memory import TaskMemory,Observation


def test_shadow_age_limit_does_not_equal_pinned_bridge_limit():
    lock=ds.read_json('configs/tomato_frozen_independent_20261112/b0.lock.v3.json')
    assert lock['rules']['max_prediction_age_s']==2.
    m=TaskMemory('age','b0',mode='M1');p=(.001,.997,.001,.001)
    for i in range(2):
        t=i*.5;o=Observation('age',i,f'age:{i}',t,t+1.5,p,p,'b0')
        assert not m.observe(o,b0_phase=0)['valid']
    assert m.phase==0 and not m.confirmed
