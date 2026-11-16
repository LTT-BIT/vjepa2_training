import numpy as np
import torch
from tools.tomato_rgb_memory_model import RGBMemoryReader
from tools.tomato_rgb_memory_training import history_length, memory_scope
from tools.tomato_rgb_memory_inference import RGBMemorySession


def test_local_memory_cannot_read_past_global_or_state():
    torch.manual_seed(42)
    net=RGBMemoryReader(np.zeros(58),np.ones(58),16,memory_scope='local_only').eval()
    v=torch.randn(1,4,4,1024,requires_grad=True)
    q=torch.randn(1,4,928,requires_grad=True)
    m=torch.randn(1,4,116,requires_grad=True)
    a=net(v,q,m)
    a['future_logits'].square().sum().backward()
    assert v.grad[:,:-1,:3].abs().sum()==0
    assert q.grad[:,:-1].abs().sum()==0
    assert m.grad[:,:-1,96:112].abs().sum()==0
    assert v.grad[:,:-1,3].abs().sum()>0
    assert v.grad[:,-1,:3].abs().sum()>0
    assert q.grad[:,-1].abs().sum()>0
    changed_v=v.detach().clone();changed_v[:,:-1,:3]+=100
    changed_q=q.detach().clone();changed_q[:,:-1]+=100
    changed_m=m.detach().clone();changed_m[:,:-1,96:112]+=100
    b=net(changed_v,changed_q,changed_m)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_single_window_scope_is_exactly_same_model_and_capacity():
    torch.manual_seed(42);a=RGBMemoryReader(np.zeros(58),np.ones(58),16)
    torch.manual_seed(42);b=RGBMemoryReader(np.zeros(58),np.ones(58),16,memory_scope='local_only')
    v=torch.randn(3,1,4,1024);q=torch.randn(3,1,928);m=torch.randn(3,1,116)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for key,value in a(v,q,m).items():torch.testing.assert_close(value,b(v,q,m)[key],atol=0,rtol=0)
    assert history_length('G3_local_only_history8')==8
    assert memory_scope('G3_local_only_history8')=='local_only'


def test_local_memory_stream_drops_old_global_and_state_and_matches_batch(tmp_path):
    mean=np.linspace(-1,1,58,dtype=np.float32)
    net=RGBMemoryReader(mean,np.ones(58),16,memory_scope='local_only').eval()
    path=tmp_path/'local.pt'
    torch.save(dict(architecture='old100_rgb_memory_reader_v2',format_version=1,
        method='G3_local_only_history8',state_mean=torch.from_numpy(mean),state_std=torch.ones(58),
        head=net.state_dict(),identity=dict(cache_signature='test'),
        protocol=dict(hidden=16,future_horizon_s=1.5),model_version='test-local-only',
        input_schema=dict(labels=['pickup','harvest','return','adjust'])),path)
    session=RGBMemorySession(path,session_id='s',episode_id='e')
    rng=np.random.default_rng(42);vs=[];qs=[];ms=[]
    for i in range(3):
        t=1.5+i*.5;v=rng.normal(size=(4,1024)).astype('float32')
        q=rng.normal(size=(16,58)).astype('float32');m=np.zeros(116,np.float32)
        m[80:96]=np.linspace(-1.5,0,16);vs.append(v);qs.append(q.flatten());ms.append(m)
        result=session.observe(v,q,m,timestamp=t,arrival_timestamp=t+.01,
            session_id='s',episode_id='e',cache_signature='test')
        assert result['valid']
    assert result['history_windows']==3
    for _,v,q,m in list(session.rows)[:-1]:
        assert not v[:3].any()
        np.testing.assert_array_equal(q,np.tile(mean,16))
    metadata=np.stack(ms)[None]
    for i in range(3):metadata[0,i,80:96]+=(i-2)*.5;metadata[0,i,112]=(i-2)*.5
    with torch.inference_mode():
        expected=net(torch.tensor(np.stack(vs)[None]),torch.tensor(np.stack(qs)[None]),torch.tensor(metadata))
    for key in ['current','future']:
        np.testing.assert_allclose(result[key+'_probabilities'],expected[key+'_logits'][0].softmax(-1).numpy(),atol=1e-6)
