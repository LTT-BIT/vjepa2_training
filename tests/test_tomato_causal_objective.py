import numpy as np
import pytest
import torch
from tools.tomato_causal_objective import contexts,CausalReadouts,supervised_loss,overlap_loss
from tools.train_tomato_causal_objective import pack


def test_contexts_causal_startup_and_gap():
    got=contexts([0,.5,1,2,2.5])
    np.testing.assert_array_equal(got,[[-1,-1,-1,0],[-1,-1,0,1],[-1,0,1,2],[-1,-1,-1,3],[-1,-1,3,4]])
    with pytest.raises(ValueError):contexts([1,0])


def test_pack_never_crosses_episode():
    rows=[dict(times=np.arange(3)*.5,features=np.ones((3,2)),labels=np.arange(3)) for _ in range(2)]
    _,_,ix,previous=pack(rows,dict(context=4,gap_reset_s=.6))
    np.testing.assert_array_equal(ix[3],[-1,-1,-1,3])
    np.testing.assert_array_equal(previous[3],[-1]*4)
    assert np.all(ix[3:][ix[3:]>=0]>=3)


def test_model_future_cannot_change_earlier_readout():
    torch.manual_seed(4);net=CausalReadouts(3,8,0).eval()
    x=torch.randn(2,4,3);v=torch.ones(2,4,dtype=torch.bool)
    a=net(x,v);x[:,3]+=100
    torch.testing.assert_close(a[:,:3],net(x,v)[:,:3])


def test_padding_does_not_create_history():
    net=CausalReadouts(3,8,0).eval();x=torch.randn(1,4,3);v=torch.tensor([[False,False,True,True]])
    a=net(x,v);x[:,:2]+=1000
    torch.testing.assert_close(a[:,2:],net(x,v)[:,2:])
    torch.testing.assert_close(a[:,2:],net(x[:,2:],v[:,2:]))


def test_shared_timestamp_alignment_not_adjacent_smoothing():
    # A:1,2,3,4; B:0,1,2,3. Labels can change sharply at timestamp3.
    a=torch.tensor([[[8.,0,0,0],[8.,0,0,0],[0,8.,0,0],[0,0,8.,0]]])
    b=torch.cat([torch.zeros(1,1,4),a[:,:3]],1)
    ids_a=torch.tensor([[1,2,3,4]]);ids_b=torch.tensor([[0,1,2,3]])
    assert overlap_loss(a,b,ids_a,ids_b).item()==0
    b[:,3]=a[:,0]
    assert overlap_loss(a,b,ids_a,ids_b)>0
    assert overlap_loss(a,b,ids_a,ids_b+100).item()==0


def test_dense_only_supervises_valid_readouts():
    z=torch.randn(1,4,4,requires_grad=True);y=torch.tensor([[0,0,1,2]]);v=torch.tensor([[False,False,True,True]])
    loss=supervised_loss(z,y,v,torch.ones(4),True);loss.backward()
    assert z.grad[:,:2].abs().sum()==0
    assert z.grad[:,2:].abs().sum()>0


def test_last_objective_matches_last_cross_entropy():
    z=torch.randn(3,4,4);y=torch.randint(0,4,(3,4));v=torch.ones(3,4,dtype=torch.bool);cw=torch.tensor([1.,2.,3.,4.])
    torch.testing.assert_close(supervised_loss(z,y,v,cw),torch.nn.functional.cross_entropy(z[:,-1],y[:,-1],weight=cw))
