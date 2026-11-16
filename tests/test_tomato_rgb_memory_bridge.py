"""Real readout + synthetic encoder: bridge contracts, not accuracy claims."""
import time
import numpy as np
import pytest
import torch
from tools.tomato_rgb_memory_model import RGBMemoryReader
from tools.tomato_rgb_memory_agent import TomatoRGBMemoryAgent
from tools import tomato_dataset as ds


def checkpoint(tmp_path):
    p=tmp_path/'head.pt';names=[f'q{i}' for i in range(58)];net=RGBMemoryReader(np.zeros(58),np.ones(58),16)
    torch.save(dict(architecture='old100_rgb_memory_reader_v2',format_version=1,method='G2_local_history8',head=net.state_dict(),state_mean=torch.zeros(58),state_std=torch.ones(58),identity=dict(cache_signature='offline',joint_names=names,pretrained={'test':'encoder'}),protocol=dict(hidden=16,future_horizon_s=1.5,roi_uses_future_coordinates=True,online_roi_validated=False),model_version='test-only',input_schema=dict(labels=['pickup','harvest','return','adjust'])),p)
    return p,names


def test_offline_bridge_preserves_wire_protocol_and_cannot_control(tmp_path,monkeypatch):
    p,names=checkpoint(tmp_path)
    monkeypatch.setattr('tools.tomato_rgb_memory_agent.model.load_encoder',lambda *a:(object(),{'test':'encoder'}))
    monkeypatch.setattr('tools.tomato_rgb_memory_agent.model.encode_clips',lambda e,r:np.tile(r.mean((1,2,3,4))[:,None],(1,1024)).astype('float32'))
    with pytest.raises(ValueError,match='offline-replay-only'):TomatoRGBMemoryAgent(p,'unused',device='cpu')
    agent=TomatoRGBMemoryAgent(p,'unused',device='cpu',offline_replay=True);rgb=np.zeros((4,16,256,256,3),np.uint8);q=np.zeros((16,58),np.float32);m=np.zeros(116,np.float32);m[80:96]=np.linspace(-1.5,0,16);t=time.time()
    r=agent.predict(rgb,q,names,roi_metadata=m,roi_contract='offline_bidirectional_local_rgb_v6',observation_id='one',session_id='s',timestamp=t)
    assert r['model_version']==ds.digest(p) and r['control_eligible'] is False and r['history_windows']==1
    for h in ['current','future']:
        assert len(r[h+'_probabilities'])==4 and abs(sum(r[h+'_probabilities'])-1)<1e-6
        z=np.array(r[h+'_logits']);actual=np.exp(z-z.max());actual/=actual.sum();np.testing.assert_allclose(actual,r[h+'_probabilities'])
    with pytest.raises(ValueError,match='ROI contract mismatch'):agent.predict(rgb,q,names,roi_metadata=m,roi_contract='live-causal-mixed-hand-v2',observation_id='two',session_id='s',timestamp=time.time())
    with pytest.raises(ValueError,match='four real'):agent.predict(rgb[:3],q,names,roi_metadata=m,roi_contract=agent.contract,observation_id='bad',session_id='s',timestamp=time.time())
