import numpy as np
import torch
from tools.tomato_independent_rgb_memory import SingleRGBReader,IndependentRGBReader,restore_reader,ARCHITECTURE


def inputs():
    return dict(visual=torch.randn(2,3,4,1024),positions=torch.randn(2,3,928),metadata=torch.zeros(2,3,116))


def test_independent_loss_cannot_update_other_branch():
    a=SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','current')
    b=SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','future')
    net=IndependentRGBReader(a,b);before={k:v.clone() for k,v in b.state_dict().items()}
    optimizer=torch.optim.AdamW(a.parameters(),lr=.001)
    logits=net(**inputs());loss=torch.nn.functional.cross_entropy(logits['current_logits'],torch.tensor([0,1]))
    loss.backward();optimizer.step()
    assert all(p.grad is None for p in b.parameters())
    for k,v in b.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    assert isinstance(a.future,torch.nn.Identity) and isinstance(b.current,torch.nn.Identity)


def test_independent_checkpoint_roundtrip_and_local_history_mask():
    torch.manual_seed(42)
    net=IndependentRGBReader(SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','current'),
        SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','future')).eval()
    saved=dict(state_mean=torch.zeros(58),state_std=torch.ones(58),method='G3_local_only_history8',
        protocol=dict(hidden=16),head=net.state_dict())
    restored=restore_reader(saved).eval();x=inputs();changed={k:v.clone() for k,v in x.items()}
    changed['visual'][:,:-1,:3]+=10;changed['positions'][:,:-1]+=10
    for key,value in net(**x).items():
        torch.testing.assert_close(value,restored(**x)[key],atol=0,rtol=0)
        torch.testing.assert_close(value,restored(**changed)[key],atol=0,rtol=0)


def test_independent_reader_session_loads_both_outputs(tmp_path):
    from tools.tomato_rgb_memory_inference import RGBMemorySession
    net=IndependentRGBReader(SingleRGBReader(np.zeros(58),np.ones(58),16,target='current'),
        SingleRGBReader(np.zeros(58),np.ones(58),16,target='future'))
    path=tmp_path/'independent.pt'
    torch.save(dict(architecture=ARCHITECTURE,format_version=1,state_mean=torch.zeros(58),state_std=torch.ones(58),
        method='G1_local_current',protocol=dict(hidden=16,future_horizon_s=1.5),head=net.state_dict(),
        identity=dict(cache_signature='test'),model_version='synthetic-independent',
        input_schema=dict(labels=['pickup','harvest','return','adjust'])),path)
    session=RGBMemorySession(path,session_id='s',episode_id='e')
    m=np.zeros(116,np.float32);m[80:96]=np.linspace(-1.5,0,16)
    result=session.observe(np.ones((4,1024)),np.zeros((16,58)),m,timestamp=2.,arrival_timestamp=2.01,
        session_id='s',episode_id='e',cache_signature='test')
    assert result['valid'] and not result['suggests_prompt_update']
    for key in ['current','future']:
        assert np.isclose(sum(result[key+'_probabilities']),1)


def test_four_view_bridge_encodes_once_for_independent_heads(tmp_path,monkeypatch):
    import time
    from tools.tomato_rgb_memory_agent import TomatoRGBMemoryAgent
    net=IndependentRGBReader(SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','current'),
        SingleRGBReader(np.zeros(58),np.ones(58),16,'local_only','future'))
    names=[f'q{i}' for i in range(58)];path=tmp_path/'independent.pt'
    torch.save(dict(architecture=ARCHITECTURE,format_version=1,state_mean=torch.zeros(58),state_std=torch.ones(58),
        method='G3_local_only_history8',protocol=dict(hidden=16,future_horizon_s=1.5,roi_uses_future_coordinates=True),
        head=net.state_dict(),identity=dict(cache_signature='test',joint_names=names,pretrained={'test':'encoder'}),
        model_version='synthetic',input_schema=dict(labels=['pickup','harvest','return','adjust'])),path)
    calls=[]
    monkeypatch.setattr('tools.tomato_rgb_memory_agent.model.load_encoder',lambda *a:(object(),{'test':'encoder'}))
    def encode(encoder,rgb):
        calls.append(rgb.shape)
        return np.ones((4,1024),np.float32)
    monkeypatch.setattr('tools.tomato_rgb_memory_agent.model.encode_clips',encode)
    agent=TomatoRGBMemoryAgent(path,'unused',device='cpu',offline_replay=True)
    m=np.zeros(116,np.float32);m[80:96]=np.linspace(-1.5,0,16)
    result=agent.predict(np.zeros((4,16,256,256,3),np.uint8),np.zeros((16,58)),names,
        roi_metadata=m,roi_contract=agent.contract,observation_id='one',session_id='s',timestamp=time.time())
    assert calls==[(4,16,256,256,3)]
    assert not result['control_eligible']
    assert len(result['current_probabilities'])==len(result['future_probabilities'])==4
