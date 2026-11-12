import numpy as np
import torch
from tools.tomato_causal_current_session import CausalCurrentSession
from tools.tomato_causal_objective import CausalReadouts,contexts


def session():
    torch.manual_seed(2)
    return CausalCurrentSession(CausalReadouts(3,8,0),np.zeros(3),np.ones(3))


def test_stream_matches_causal_batch_including_startup():
    s=session();rng=np.random.default_rng(1);raw=rng.normal(size=(7,3)).astype('float32');times=np.arange(7)*.5;ids=contexts(times);padded=np.r_[np.zeros((1,3),np.float32),raw]
    with torch.inference_mode():expected=s.net(torch.from_numpy(padded[ids+1]),torch.from_numpy(ids>=0))[:,-1].softmax(-1).numpy()
    actual=np.stack([s.update(x,t)['current_probabilities'] for x,t in zip(raw,times)])
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_gap_resets_history_without_inventing_observations():
    s=session();s.update(np.ones(3),0);s.update(np.ones(3),.5);r=s.update(np.ones(3),2)
    assert r['context_observation_timestamps']==[2.]
    assert r['valid']


def test_reordered_or_invalid_input_produces_no_probability():
    s=session();s.update(np.ones(3),1)
    for x,t in [(np.ones(3),1),(np.ones(3),.5),(np.ones(2),2),(np.full(3,np.nan),2)]:
        r=s.update(x,t);assert not r['valid'] and r['current_probabilities'] is None and not s.history
    assert s.update(np.ones(3),2)['context_observation_timestamps']==[2.]
