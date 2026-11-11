import numpy as np
import pytest
import torch
from tools.tomato_probability_residual import ProbabilityHistory, ResidualCalibration


def test_prefix_causality_and_lag_timestamps():
    h=ProbabilityHistory();a=h.update([1,0,0,0],[0,1,0,0],0)
    assert a.shape==(36,) and np.array_equal(a[-4:],[1,0,0,0])
    b=h.update([0,0,1,0],[0,0,0,1],.5)
    assert np.array_equal(b[8:16],a[:8]) and np.array_equal(b[-4:],[1,1,0,0])
    h.update([0,1,0,0],[1,0,0,0],1)
    assert np.array_equal(a[-4:],[1,0,0,0])


def test_gap_and_new_episode_have_no_past_truth_or_history():
    h=ProbabilityHistory();h.update([1,0,0,0],[1,0,0,0],0)
    a=h.update([0,1,0,0],[0,1,0,0],2)
    assert np.array_equal(a[-4:],[1,0,0,0])
    assert np.array_equal(a[8:32],np.zeros(24))
    assert len(ProbabilityHistory().rows)==0


def test_no_future_slot_fill_or_duplicate_observation():
    h=ProbabilityHistory();h.update([1,0,0,0],[1,0,0,0],0)
    a=h.update([0,1,0,0],[0,1,0,0],.4)
    assert a[-3]==0 # t-.5 precedes start, cannot use t=0
    with pytest.raises(ValueError):h.update([1,0,0,0],[1,0,0,0],.4)
    with pytest.raises(ValueError):h.update([1,1,0,0],[1,0,0,0],.9)


def test_identity_initialization_and_training_leaves_base_logits_unchanged():
    net=ResidualCalibration(np.zeros(36),np.ones(36))
    assert sum(p.numel() for p in net.parameters())==148
    base=torch.log(torch.tensor([[.1,.2,.3,.4]]));saved=base.clone()
    x=torch.randn(1,36);z,delta=net(x,base)
    assert torch.equal(z,base) and delta.abs().sum()==0
    torch.nn.functional.cross_entropy(z,torch.tensor([1])).backward()
    assert net.correction.weight.grad.abs().sum()>0 and torch.equal(base,saved)
