import numpy as np
import pytest
import torch
from tools.tomato_transition_time import survival_targets,hazard_nll,event_distribution,PredictiveTransitionDecision,transition_fields


def test_event_bins_and_partial_right_censoring():
    s=[dict(label='adjust',start_frame=0,end_frame=30),dict(label='pickup',start_frame=30,end_frame=90)]
    t=survival_targets(s,[0,15,23,30,75,89],30,90)
    assert t['labels'][0].tolist()==[0,0,1,0]
    assert t['labels'][1].tolist()==[0,1,0,0]
    assert t['labels'][2].tolist()==[1,0,0,0]
    assert t['mask'][-1].sum()==0
    assert t['mask'][-2].tolist()==[1,0,0,0]
    assert not t['event_observed'][3]


def test_probabilities_and_censored_likelihood():
    z=torch.zeros(2,4,requires_grad=True);p=event_distribution(z)
    assert torch.allclose(p.sum(1),torch.ones(2))
    assert p[0].tolist()==[.5,.25,.125,.0625,.0625]
    mask=torch.tensor([[1.,0,0,0],[0.,0,0,0]])
    loss=hazard_nll(z,torch.zeros_like(z),mask);loss.backward()
    assert abs(loss.item()-np.log(2))<1e-6
    assert z.grad[0,1:].abs().sum()==0 and z.grad[1].abs().sum()==0


def test_same_class_segments_not_events_and_exact_boundary_uses_next_event():
    s=[dict(label='pickup',start_frame=0,end_frame=10),dict(label='pickup',start_frame=10,end_frame=30),dict(label='harvest',start_frame=30,end_frame=100)]
    t=survival_targets(s,[10,30],30,100)
    assert abs(t['time_to_event'][0]-2/3)<1e-8
    assert not t['event_observed'][1]


def test_warning_never_alone_switches_and_repeated_pickup_cannot_skip():
    d=PredictiveTransitionDecision();p=[.9,.02,.02,.01,.05]
    for t in [0,.5,1.,1.5]:
        r=d.update([1,0,0,0],[0,1,0,0],p,t,t+.01)
        assert not r['suggest_switch'] and d.phase==0
    assert d.update([0,1,0,0],[0,1,0,0],p,2,2.01)['phase_index']==0
    assert d.update([0,1,0,0],[0,1,0,0],p,2.5,2.51)['phase_index']==1
    for t in [3.,3.5,4.]:
        assert not d.update([1,0,0,0],[1,0,0,0],p,t,t+.01)['suggest_switch']


@pytest.mark.parametrize('case',['stale','timestamp','nan','badmass'])
def test_conservative_invalid_inputs(case):
    d=PredictiveTransitionDecision();kwargs=dict(current=[1,0,0,0],future=[0,1,0,0],masses=[.2]*5,timestamp=1.,now=1.1,transition_timestamp=1.)
    if case=='stale':kwargs['now']=3.
    if case=='timestamp':kwargs['transition_timestamp']=.5
    if case=='nan':kwargs['current']=[float('nan')]*4
    if case=='badmass':kwargs['masses']=[1.]*5
    assert not d.update(**kwargs)['valid'] and d.phase==0


def test_schema_additive_and_probability_semantics():
    response=dict(current_probabilities=[1,0,0,0],future_probabilities=[0,1,0,0])
    response.update(transition_fields([0]*4,2.,'version'))
    assert response['current_probabilities']==[1,0,0,0]
    assert sum(response['transition_time']['probabilities'])==1


def test_response_adapter_identity_and_no_duplicate_trigger():
    from tools import tomato_dataset as ds
    d=PredictiveTransitionDecision()
    r=dict(valid=True,labels=list(ds.LABELS),model_version='cf',session_id='a',input_timestamp=0.,current_probabilities=[1,0,0,0],future_probabilities=[0,1,0,0],**transition_fields([0]*4,0.,'h'))
    assert d.update_response(r,.01)['valid']
    assert not d.update_response(r,.01)['valid']
    r['input_timestamp']=.5;r['transition_time']['input_timestamp']=.5;r['transition_time']['model_version']='changed'
    assert not d.update_response(r,.51)['valid']


def test_optional_inference_preserves_base_response_and_rejects_pair_mismatch(tmp_path):
    from tools.tomato_transition_time import TransitionTimeHead
    from tools.tomato_time_inference import TransitionTimePredictor
    path=tmp_path/'time.pt'
    torch.save(dict(identity=dict(version='tomato-transition-time-v1',original_classifiers=dict(sha256='base')),mean=torch.zeros(4000),std=torch.ones(4000),head=TransitionTimeHead().state_dict()),path)
    p=TransitionTimePredictor(path,'base')
    response=dict(valid=True,input_timestamp=1.,current_probabilities=[1,0,0,0],future_probabilities=[0,1,0,0],model_version='base')
    out=p.augment(response,np.zeros((1,4000)))
    assert 'transition_time' not in response and out['model_version']==response['model_version']
    assert out['current_probabilities']==response['current_probabilities']
    with pytest.raises(ValueError):TransitionTimePredictor(path,'wrong')


def test_missing_transition_timestamp_and_bad_types_are_conservative():
    from tools import tomato_dataset as ds
    d=PredictiveTransitionDecision()
    r=dict(valid=True,labels=list(ds.LABELS),model_version='cf',session_id='a',input_timestamp=0.,current_probabilities=[1,0,0,0],future_probabilities=[0,1,0,0],transition_time=dict(schema_version='tomato-transition-time-v1',model_version='h',probabilities=[.2]*5))
    assert not d.update_response(r,.01)['valid']
    assert not d.update([None]*4,[0]*4,[0]*5,None,None)['valid']


def test_full_seven_phase_causal_replay_and_chunk_version_mock():
    from tools.tomato_transition_time import CLASSES
    d=PredictiveTransitionDecision();t=0.;changes=[];chunk_version=0
    for phase in range(6):
        current=np.eye(4)[CLASSES[phase]];nxt=np.eye(4)[CLASSES[phase+1]];p=[.9,.02,.02,.01,.05]
        for _ in range(2):
            r=d.update(current,nxt,p,t,t+.01);t+=.5
            assert not r['suggest_switch']
        for _ in range(2):
            r=d.update(nxt,nxt,p,t,t+.01);t+=.5
            if r['suggest_switch']:
                assert chunk_version!=r['prompt_version']
                chunk_version=r['prompt_version'];changes.append(r['phase_index'])
        assert d.phase==phase+1 and chunk_version==d.phase
    assert changes==[1,2,3,4,5,6]
    for _ in range(3):
        r=d.update(np.eye(4)[2],np.eye(4)[0],p,t,t+.01);t+=.5
        assert not r['suggest_switch'] and d.phase==6
