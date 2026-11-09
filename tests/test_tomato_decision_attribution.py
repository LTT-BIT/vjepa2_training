"""Causal research-policy and historical early-Prompt regression checks."""
import numpy as np
import pytest
from tools import tomato_dataset as ds
from tools.tomato_decision_attribution import decisions,historical_bridge,ROOT


def test_joint_requires_current_and_never_advances_from_future_alone():
    t=np.arange(6)*.5;c=np.eye(4)[[0]*6];f=np.eye(4)[[1]*6]
    assert decisions(c,f,t,.8,2,True,True)[0]==[]
    c=np.eye(4)[[0,0,1,1,1,1]];f=np.eye(4)[[0,0,0,0,1,1]]
    assert decisions(c,f,t,.8,2,True,True)[0]==[dict(time=2.5,previous=0,label=1)]


def test_no_future_stream_samples_used_and_no_cross_episode_state():
    t=np.arange(8)*.5;c=np.eye(4)[[0,0,1,1,2,2,3,3]];f=c.copy()
    complete=decisions(c,f,t,.8,2,True,True)[0]
    for n in range(1,9):
        assert decisions(c[:n],f[:n],t[:n],.8,2,True,True)[0]==[r for r in complete if r['time']<=t[n-1]]
    assert decisions(np.eye(4)[[1,1]],np.eye(4)[[1,1]],t[:2],.8,2,True)[0][0]['previous']==0


def test_order_blocks_skip_and_pending_resets_when_evidence_breaks():
    t=np.arange(7)*.5;c=np.eye(4)[[0,2,2,1,0,1,1]]
    assert decisions(c,c,t,.8,2,True)[0]==[dict(time=3.,previous=0,label=1)]
    with pytest.raises(ValueError):decisions(c,c,np.zeros(7),.8,2)


@pytest.fixture
def bridge():
    if not (ROOT/'historical_openpi/src/openpi/subtasks/agent_bridge.py').exists():
        pytest.skip('Read-only historical OpenPI snapshot unavailable')
    return historical_bridge()


def response(bridge,oid,t,current,future):
    c=np.full(4,.001);f=c.copy();c[current]=.997;f[future]=.997
    return dict(protocol=bridge.PROTOCOL,labels=list(bridge.LABELS),valid=True,error=None,observation_id=oid,session_id='audit',model_version='locked',prediction_mode='joint',future_horizon_s=1.5,input_timestamp=t,result_timestamp=t+.01,current_probabilities=c.tolist(),future_probabilities=f.tolist(),current_logits=np.log(c).tolist(),future_logits=np.log(f).tolist())


def test_original_provider_two_high_current_predictions_can_advance_before_human_boundary(bridge):
    """Expected contract behavior, not a bug-fix test: no annotation reaches provider."""
    p=bridge.PredictionPromptProvider();p.reset('audit')
    # Advance the three genuine earlier phases before the focal adjust->pickup.
    for i,label in enumerate([1,1,2,2,3,3]):
        v=response(bridge,str(i),10+i*.5,label,label);p.accept_prediction(v,observation_id=str(i),now=v['input_timestamp']+.02)
    assert p.phase==3 and p.version==3
    # Real case's confidence pattern at71.5 and72.0 precedes annotation73.3.
    a=response(bridge,'71.5',71.5,0,0);c1=p.accept_prediction(a,observation_id='71.5',now=71.52)
    assert not c1.changed and p.count==1
    old=bridge.bind_chunk(c1,'old')
    b=response(bridge,'72.0',72.,0,0);c2=p.accept_prediction(b,observation_id='72.0',now=72.02)
    assert c2.changed and c2.phase=='pickup_2' and c2.version==4
    assert not bridge.chunk_matches_prompt(old,c2)
    assert bridge.chunk_matches_prompt(bridge.bind_chunk(c2,'new'),c2)


def test_future_warning_alone_cannot_change_prompt_version(bridge):
    p=bridge.PredictionPromptProvider();p.reset('audit')
    for i in range(4):
        v=response(bridge,str(i),10+i*.5,0,1);c=p.accept_prediction(v,observation_id=str(i),now=v['input_timestamp']+.02)
        assert not c.changed and c.version==0 and c.phase=='pickup_1' and c.future_warning=='harvest'
