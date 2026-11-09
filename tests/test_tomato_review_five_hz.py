from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tools.tomato_review_ui.five_hz import episode_grid,validate_hz,load_features
from tools.tomato_review_ui.catalog import write,sha
from tools.tomato_review_ui.timeline import validate_rows,held


def test_source_grid_not_history_or_forecast_rate():
    e=dict(fps=30,length=31,anchors=[0,15,30])
    assert episode_grid(e,5)['anchors']==[0,6,12,18,24,30]
    assert episode_grid(e,2) is e
    assert e['anchors']==[0,15,30]


@pytest.mark.parametrize('hz',[True,'5',0,3,5.5,float('nan')])
def test_reject_unregistered_rates(hz):
    with pytest.raises(ValueError):validate_hz(hz)


def test_held_at_5hz_uses_real_anchor_and_fixed_target():
    e=dict(episode_id='e',source_recording_id='s',split='train',fps=30,length=60,anchors=[0,15,30,45])
    rows=[dict(episode_id='e',source_recording_id='s',split='train',prediction_mode='future_only',frame_index=i,timestamp_s=i/30,
        last_input_frame=i,input_indices=[i]*16,future_probabilities=dict(pickup=.7,harvest=.1,**{'return':.1},adjust=.1)) for i in episode_grid(e,5)['anchors']]
    with pytest.raises(ValueError):validate_rows(rows,e,'future_only')
    rr=validate_rows(rows,episode_grid(e,5),'future_only')
    assert held(rr,8)['frame_index']==6
    assert held(rr,8)['heads']['future']['target_at']==1.2
    assert not held(rr,59)['heads']['future']['target_scorable']


@pytest.fixture

def cached_setup(tmp_path,monkeypatch):
    from tools import tomato_dataset as ds,tomato_phase_model as model
    times=np.arange(60,dtype=np.float64)/30;q=np.arange(60,dtype=np.float32)[:,None]
    e=dict(episode_id='e',fps=30,length=60,anchors=[0,15,30,45])
    contract=dict(joint_names=['j'],joint_mode='arms-and-hands',sampler_version=ds.SAMPLER_VERSION)
    pretrained=dict(model='fixture');m=dict(contract=contract,pretrained=pretrained)
    catalog=SimpleNamespace(root=tmp_path,manifests={'joint':m,'future_only':m},get=lambda eid:e)
    for mode in ['joint','future_only']:
        run=tmp_path/mode;run.mkdir();write(run/'manifest.json',m);write(run/'training_config.json',dict(seed=42,pretrained_dir=str(tmp_path)))
    monkeypatch.setattr(ds,'read_state',lambda *a,**k:(times,q))
    monkeypatch.setattr(model,'assert_source_unchanged',lambda e:None)
    monkeypatch.setattr(model,'feature_fingerprint',lambda *a:'original-two-hz')
    anchors=np.array(e['anchors']);hist=np.stack([ds.sample_causal_indices(times,int(i)) for i in anchors])
    original=tmp_path/'original.npz';features=np.concatenate([np.repeat(anchors[:,None],3072,axis=1),q[hist].reshape(len(anchors),-1)],axis=1).astype(np.float32)
    model.write_cache(original,features,anchors,times[anchors],hist,'original-two-hz')
    for mode in ['joint','future_only']:
        write(tmp_path/mode/'feature_index.json',dict(manifest_sha256=sha(tmp_path/mode/'manifest.json'),pretrained=pretrained,episodes=[dict(episode_id='e',path=str(original))]))
    calls=[]
    monkeypatch.setattr(model,'load_encoder',lambda *a:(object(),pretrained))
    monkeypatch.setattr(model,'decode_selected',lambda e,indices:([{int(i):np.zeros((256,256,3),np.uint8) for i in indices} for _ in range(3)],[]))
    def fuse(encoder,rgb,history):
        calls.append(float(history[0,-1,0]))
        return np.concatenate([np.full((1,3072),history[0,-1,0],np.float32),history.reshape(1,-1)],axis=1)
    monkeypatch.setattr(model,'fuse_features',fuse)
    return catalog,tmp_path/'new-ui',original,calls


def test_new_anchors_encoded_shared_between_heads_and_original_untouched(cached_setup):
    c,out,original,calls=cached_setup;before=sha(original)
    d,p=load_features(c,'e','joint',out,'cpu')
    assert p['reused_2hz_anchors']==2 and p['encoded_anchors']==8 and len(calls)==8
    np.testing.assert_array_equal(d['anchors'],np.arange(0,60,6))
    np.testing.assert_array_equal(d['features'][:,0],d['anchors'])
    second,receipt=load_features(c,'e','future_only',out,'cpu')
    assert receipt['feature_cache_hit'] and not receipt['encoder_executed'] and len(calls)==8
    assert sha(original)==before
    np.testing.assert_array_equal(d['features'],second['features'])
    write(out/'features_5hz'/(p['fingerprint']+'.json'),dict(fingerprint='wrong',content_sha256=sha(original)))
    with pytest.raises(ValueError):load_features(c,'e','joint',out,'cpu')


def test_failed_encoder_does_not_publish_cache(cached_setup,monkeypatch):
    from tools import tomato_phase_model as model
    c,out,_,_=cached_setup
    def fail(*a):raise RuntimeError('injected encoder failure')
    monkeypatch.setattr(model,'load_encoder',fail)
    with pytest.raises(RuntimeError):load_features(c,'e','joint',out,'cpu')
    assert not list(out.rglob('*.npz'))


def test_float32_timestamps_cannot_shift_five_hz_anchors_one_frame():
    from tools.tomato_review_ui.five_hz import source_grid
    e=dict(fps=30,length=1065,anchors=[])
    times=(np.arange(1065)/30).astype(np.float32).astype(np.float64)
    assert source_grid(e,times).tolist()==list(range(0,1065,6))
    assert source_grid(e,times)[-1]==1062
    times[15]+=.003
    with pytest.raises(ValueError):source_grid(e,times)
