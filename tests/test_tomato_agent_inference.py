import time
import numpy as np
import pytest
from tools import tomato_agent_inference as agent


def test_named_joint_alignment_and_active_head_contract(monkeypatch):
    names = [f'joint{i}.position' for i in range(58)]
    instance = object.__new__(agent.TomatoAgent)
    instance.joint_names = names
    instance.encoder = object()
    instance.head = object()
    instance.model_version = 'checkpoint-sha'
    instance.saved = dict(mean=np.zeros(4000), std=np.ones(4000), future_horizon_s=1., prediction_mode='joint')
    seen = {}
    def fuse(encoder, rgb, q):
        seen['q'] = q
        return np.zeros((1,4000),np.float32)
    monkeypatch.setattr(agent.model, 'fuse_features', fuse)
    monkeypatch.setattr(agent.model, 'predict_heads', lambda *a:dict(current_logits=np.array([[1,2,3,4]]),future_logits=np.array([[4,3,2,1]])))
    q = np.tile(np.arange(58)[::-1],(16,1))
    out = instance.predict(np.zeros((3,16,256,256,3),np.uint8),q,names[::-1],observation_id='1',session_id='s',timestamp=time.time())
    np.testing.assert_array_equal(seen['q'][0,0],np.arange(58))
    assert out['labels']==['pickup','harvest','return','adjust']
    assert len(out['current_probabilities'])==4
    assert np.isclose(sum(out['future_probabilities']),1)
    with pytest.raises(ValueError,match='58 named'):
        instance.predict(None,q,names[:-1],observation_id='1',session_id='s',timestamp=time.time())
    with pytest.raises(ValueError,match='timestamp'):
        instance.predict(None,q,names,observation_id='1',session_id='s',timestamp=time.time()-10)
