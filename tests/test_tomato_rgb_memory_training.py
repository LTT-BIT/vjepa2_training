import numpy as np
from tools.tomato_causal_local_roi import CausalLocalROI
from tools.tomato_interaction_rgb import crop_rgb

NAMES=[f'{s}_fr3_joint{i}.position' for s in ['left','right'] for i in range(7)]+[f'{s}_finger{i}.position' for s in ['left','right'] for i in range(22)]


def test_roi_prefix_uses_no_future_or_phase_inputs():
    d=[dict(label=1,score=.99,box=[100,100,140,140]),dict(label=3,score=.9,box=[138,115,160,135])]
    a=CausalLocalROI(NAMES,np.ones(58));b=CausalLocalROI(NAMES,np.ones(58))
    for i in range(6):
        q=np.zeros(58);q[7:14]=i*.01
        x=a.update(d,q,640,400);y=b.update(d,q,640,400)
        np.testing.assert_array_equal(x['interaction_box'],y['interaction_box'])
        assert x['valid'] and x['roles']==y['roles']
    # Altering later observations cannot retroactively alter returned boxes.
    box=x['interaction_box'].copy();a.update([],np.ones(58),640,400)
    np.testing.assert_array_equal(x['interaction_box'],box)


def test_missing_hand_is_fresh_fallback_not_stale_crop():
    s=CausalLocalROI(NAMES,np.ones(58));r=s.update([],np.zeros(58),640,400)
    assert not r['valid'] and r['interaction_box'] is None
    red=np.full((400,640,3),80,np.uint8);blue=np.full((400,640,3),120,np.uint8)
    x=crop_rgb(red,valid=False);y=crop_rgb(blue,valid=False)
    assert x['rgb'].shape==(256,256,3) and not np.array_equal(x['rgb'],y['rgb'])


def test_masked_padding_does_not_change_latest_prediction():
    import torch
    from tools.tomato_rgb_memory_model import RGBMemoryReader
    torch.manual_seed(3);net=RGBMemoryReader(np.zeros(58),np.ones(58),hidden=16).eval()
    v=torch.randn(1,4,4,1024);q=torch.randn(1,4,928);m=torch.zeros(1,4,116);mask=torch.tensor([[False,False,True,True]])
    with torch.inference_mode():
        a=net(v,q,m,mask);b=net(v[:,2:],q[:,2:],m[:,2:])
        for key in a:torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-5)


def test_history_alignment_keeps_query_causal_and_resets_gaps():
    import torch
    from tools.tomato_rgb_memory_training import history_indices,gather_inputs
    ix=history_indices([1.5,2,2.5,4],3)
    np.testing.assert_array_equal(ix,[[-1,-1,0],[-1,0,1],[0,1,2],[-1,-1,3]])
    v=torch.zeros(5,4,1024);q=torch.zeros(5,928);m=torch.zeros(5,116);m[:,80:96]=torch.linspace(-1.5,0,16);t=torch.tensor([0,1.5,2,2.5,4.])
    got=gather_inputs(v,q,m,t,torch.tensor(ix),'G2_local_history8')
    assert got['metadata'][2,0,112]==-1 and got['metadata'][2,-1,112]==0
    assert not got['history_valid'][3,:2].any()


def test_readout_runtime_identity_stale_and_gap(tmp_path):
    import torch
    from tools.tomato_rgb_memory_model import RGBMemoryReader
    from tools.tomato_rgb_memory_inference import RGBMemorySession
    model=RGBMemoryReader(np.zeros(58),np.ones(58),hidden=16);p=tmp_path/'head.pt'
    torch.save(dict(architecture='old100_causal_rgb_memory_reader_v1',format_version=1,method='G2_local_history8',head=model.state_dict(),state_mean=torch.zeros(58),state_std=torch.ones(58),identity=dict(cache_signature='test'),protocol=dict(hidden=16,future_horizon_s=1.5),model_version='synthetic-test-only',input_schema=dict(labels=['pickup','harvest','return','adjust'])),p)
    s=RGBMemorySession(p,session_id='s',episode_id='e');v=np.ones((4,1024),np.float32);q=np.zeros((16,58),np.float32);m=np.zeros(116,np.float32);m[80:96]=np.linspace(-1.5,0,16)
    def call(t,**kw):return s.observe(v,q,m,timestamp=t,arrival_timestamp=kw.get('arrival',t+.04),session_id=kw.get('session','s'),episode_id='e',cache_signature='test')
    assert call(1.5)['history_windows']==1
    assert call(2.)['history_windows']==2
    assert not call(2.5,arrival=4.)['valid'] and len(s.rows)==2
    assert not call(2.5,session='other')['valid'] and len(s.rows)==2
    r=call(4.);assert r['history_windows']==1 and not r['suggests_prompt_update'] and r['physical_success']=='unknown'
    m[95]=.1;assert not call(4.5)['valid']


def test_right_anchor_never_follows_fast_left_transport():
    s=CausalLocalROI(NAMES,np.ones(58))
    d=[dict(label=1,score=.99,box=[400,100,440,140]),dict(label=2,score=.99,box=[40,100,80,140])]
    a=s.update(d,np.zeros(58),640,400)
    q=np.zeros(58);q[s.left]=100
    b=s.update(d,q,640,400)
    assert a['roles']==b['roles']==[1]
    np.testing.assert_array_equal(a['interaction_box'],b['interaction_box'])
    assert b['interaction_box'][0]>350
    assert not s.update(d[1:],q,640,400)['valid']


def test_left_contact_and_current_fruit_join_right_anchor():
    s=CausalLocalROI(NAMES,np.ones(58));rgb=np.zeros((400,640,3),np.uint8);rgb[110:130,145:160]=[255,0,0]
    d=[dict(label=1,score=.99,box=[100,100,140,140]),dict(label=2,score=.99,box=[165,100,195,140])]
    r=s.update(d,np.zeros(58),640,400,rgb=rgb)
    assert r['roles']==[1,2] and r['interaction_box'][2]>=195


def test_recorded_r11_checkpoint_cannot_silently_run_live(tmp_path):
    import pytest,torch
    from tools.tomato_rgb_memory_model import RGBMemoryReader
    from tools.tomato_rgb_memory_inference import RGBMemorySession
    net=RGBMemoryReader(np.zeros(58),np.ones(58),hidden=16);p=tmp_path/'offline.pt'
    torch.save(dict(architecture='old100_rgb_memory_reader_v2',format_version=1,method='G1_local_current',head=net.state_dict(),state_mean=torch.zeros(58),state_std=torch.ones(58),identity=dict(cache_signature='r11'),protocol=dict(hidden=16,future_horizon_s=1.5,roi_uses_future_coordinates=True),model_version='offline-only-test',input_schema=dict(labels=['pickup','harvest','return','adjust'])),p)
    with pytest.raises(ValueError,match='offline-replay-only'):RGBMemorySession(p,session_id='s',episode_id='e')
    assert RGBMemorySession(p,session_id='s',episode_id='e',offline_replay=True).offline_only


def test_recorded_r11_binding_rejects_different_source():
    import pytest
    from tools.prepare_tomato_recorded_roi_cache import validate_binding
    rows=[dict(dataset_root='old',episode_index=i) for i in range(100)]
    mapping=[dict(source_dataset='old',source_episode_index=i) for i in range(100)]
    info=dict(features={k:dict(dtype='video') for k in ['observation.images.cam0','observation.images.cam1','observation.images.cam2','observation.images.interaction']},total_frames=425556,fps=30)
    validate_binding(dict(episodes=rows),dict(mapping=mapping),info)
    mapping[0]['source_dataset']='new119'
    with pytest.raises(ValueError,match='identity'):validate_binding(dict(episodes=rows),dict(mapping=mapping),info)
