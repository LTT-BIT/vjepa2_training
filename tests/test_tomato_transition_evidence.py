import numpy as np
import pytest
import torch
from tools.tomato_transition_evidence import change_features,past_indices,training_examples,EvidenceReadout


def names():
    return [f'{s}_fr3_joint{i}.position' for s in ('left','right') for i in range(1,8)]+[f'{s}_finger{i}.position' for s in ('left','right') for i in range(22)]


def inputs(n=12):
    rng=np.random.default_rng(8);x=rng.normal(size=(n,4000)).astype('float32');c=np.tile([.7,.1,.1,.1],(n,1));f=c.copy();t=np.arange(n)*.5
    return x,c,f,t


def test_prefix_causality_and_dimensions():
    x,c,f,t=inputs();full=change_features(x,c,f,t,names())
    assert [full[k].shape[1] for k in ('E0','E1','E2')]==[35,381,416]
    for n in range(1,len(t)+1):
        part=change_features(x[:n],c[:n],f[:n],t[:n],names())
        for k in part:np.testing.assert_allclose(part[k],full[k][:n],atol=1e-6)


def test_true_timestamps_and_episode_reset():
    assert past_indices(np.array([0,.5,2.]),1.).tolist()==[0,0,1]
    x,c,f,t=inputs(1);out=change_features(x,c,f,t,names())
    assert np.allclose(out['E1'],0,atol=1e-5)


def test_joint_reordering_by_name_preserves_group_speed():
    x,c,f,t=inputs();orig=change_features(x,c,f,t,names())['E1']
    order=np.arange(58)[::-1];y=x.copy();y[:,3072:]=x[:,3072:].reshape(-1,16,58)[:,:,order].reshape(-1,928)
    reordered=change_features(y,c,f,t,[names()[i] for i in order])['E1']
    for k in range(3):np.testing.assert_allclose(orig[:,k*126+122:k*126+126],reordered[:,k*126+122:k*126+126],rtol=1e-5)


def test_invalid_schema_and_time_rejected():
    x,c,f,t=inputs()
    with pytest.raises(ValueError):change_features(x,c,f,t,[names()[0]]*58)
    with pytest.raises(ValueError):change_features(x,c,f,np.zeros(len(t)),names())


def test_labels_before_after_and_balanced_groups():
    labels=['pickup','harvest','return','adjust','pickup','harvest','return'];seg=[dict(start_frame=i*60,end_frame=(i+1)*60,label=k) for i,k in enumerate(labels)]
    t=np.arange(0,14,.5);r=training_examples(t,seg,30)
    for k in range(6):
        for y in (0,1):
            mask=(r['phase']==k)&(r['label']==y);assert np.isclose(r['weight'][mask].sum(),1)
            assert np.all((t[r['index'][mask]]>=(k+1)*2)==bool(y))


def test_small_readout_no_hidden_network():
    m=EvidenceReadout(35);x=torch.zeros(4,35)
    assert m.all_phases(x).shape==(4,6)
    assert sum(p.numel() for p in m.parameters())==216
    assert m(x,torch.tensor([0,1,2,5])).shape==(4,)


def test_rank_metrics_handle_ties_and_weights():
    from tools.train_tomato_transition_evidence import binary_scores
    assert binary_scores([1,1,0,0],[.9,.8,.2,.1])['auroc']==1
    tied=binary_scores([1,0],[.5,.5],[2,2])
    assert tied['auroc']==.5 and tied['ap']==.5
    assert binary_scores([0,0],[.2,.3])['auroc'] is None


def test_no_boundary_or_oracle_arguments_in_features():
    x,c,f,t=inputs()
    with pytest.raises(TypeError):change_features(x,c,f,t,names(),boundary_time=3)


def make_shadow(tmp_path):
    from tools import tomato_dataset as ds
    from tools.tomato_transition_evidence_runtime import EvidenceShadow
    from pathlib import Path
    m=EvidenceReadout(416);path=tmp_path/'shadow.pt'
    torch.save(dict(scope='shadow_only_not_selected_for_deployment',architecture='six_conditional_linear_transition_evidence',arm='E2',input_dimension=416,state_dict=m.state_dict(),mean=torch.zeros(416),std=torch.ones(416),feature_schema=dict(joint_names=names(),feature_source_sha256=ds.digest(Path('tools/tomato_transition_evidence.py')))),path)
    return EvidenceShadow(path,'s')


def test_bounded_online_history_matches_batch_and_never_writes_prompt(tmp_path):
    m=make_shadow(tmp_path);x,c,f,t=inputs(20);batch=change_features(x,c,f,t,names())['E2']
    for i in range(len(t)):
        r=m.observe(x[i],c[i],f[i],t[i],3,session_id='s')
        np.testing.assert_allclose(m.last_vector,batch[i],atol=1e-5)
        assert r['candidate_phase']=='pickup_2' and r['suggests_prompt_update'] is False
        assert len(m.history)<=4
    assert m.observe(x[-1],c[-1],f[-1],t[-1]+.5,6,session_id='s')['candidate_phase'] is None


def test_runtime_rejects_stale_or_different_session(tmp_path):
    m=make_shadow(tmp_path);x,c,f,t=inputs(1)
    assert not m.observe(x[0],c[0],f[0],0,0,session_id='wrong')['valid']
    assert not m.observe(x[0],c[0],f[0],0,0,session_id='s',now=2)['valid']
    assert not m.history


def test_runtime_gap_marks_insufficient_evidence(tmp_path):
    m=make_shadow(tmp_path);x,c,f,t=inputs(4)
    for i in range(4):r=m.observe(x[i],c[i],f[i],t[i],0,session_id='s')
    assert r['history_complete'] and r['evidence_usable']
    r=m.observe(x[-1],c[-1],f[-1],5,0,session_id='s')
    assert r['valid'] and not r['history_complete'] and not r['evidence_usable']
    assert r['suggests_prompt_update'] is False
