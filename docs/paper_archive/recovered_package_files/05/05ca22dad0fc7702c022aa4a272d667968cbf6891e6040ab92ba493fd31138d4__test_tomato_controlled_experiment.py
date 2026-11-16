import numpy as np
import torch
from tools.test_tomato_controlled_experiment import common_batches, gradient_statistics


def test_paired_batches_cover_tail_and_repeat_exactly():
    valid = np.ones(21706, dtype=bool); valid[-150:] = False
    batches = common_batches(valid, 42, 1, 128)
    assert len(batches) == 170
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(valid)))
    assert all(valid[b].any() for b in batches)
    assert all(np.array_equal(a,b) for a,b in zip(batches,common_batches(valid,42,1,128)))
    assert not np.array_equal(batches[0],common_batches(valid,43,1,128)[0])


def test_gradient_measurement_preserves_backward_and_rng():
    p = torch.tensor([1., 2.], requires_grad=True)
    c, f = p.square().sum(), -2*p.square().sum()
    rng = torch.get_rng_state().clone()
    gc = torch.autograd.grad(c, [p], retain_graph=True)
    gf = torch.autograd.grad(f, [p], retain_graph=True)
    s = gradient_statistics(gc,gf)
    assert abs(s['cosine'] + 1.) < 1e-6
    assert abs(s['weighted_future_current_ratio'] - 1.) < 1e-6
    assert p.grad is None and torch.equal(rng,torch.get_rng_state())
    ((c+.5*f)/1.5).backward()
    torch.testing.assert_close(p.grad,torch.zeros_like(p))


def test_real_joint_gradient_logging_does_not_change_dropout_or_optimizer_update():
    import copy
    from tools import tomato_phase_model as model
    torch.manual_seed(81)
    first = model.joint_head_model(1).train()
    second = copy.deepcopy(first)
    x = torch.randn(8,3088); yc = torch.tensor([0,1,2,3,0,1,2,3])
    yf = torch.tensor([1,2,3,0,1,2,-100,-100]); mask = yf>=0
    for logging,head in [(False,first),(True,second)]:
        torch.manual_seed(100)
        opt = torch.optim.AdamW(head.parameters(),lr=.0003)
        total,parts = model.supervision_loss(head(x),yc,yf,mask,
            {'current':torch.ones(4),'future':torch.ones(4)}, {'prediction_mode':'joint'})
        if logging:
            trunk=list(head.trunk.parameters())
            a=torch.autograd.grad(parts['current'],trunk,retain_graph=True)
            b=torch.autograd.grad(parts['future'],trunk,retain_graph=True)
            gradient_statistics(a,b)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        opt.step()
    for a,b in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
