from types import SimpleNamespace
import numpy as np
import pytest
from tools.tomato_offline_contract import OfflineChunkGate,check_request_response


class Bridge:
    @staticmethod
    def bind_chunk(c,key):return dict(chunk_id=key,prompt_version=c.version,phase=c.phase,prompt=c.prompt,session_id=c.session_id,model_version=c.model_version,input_timestamp=c.input_timestamp,observation_id='o')
    @staticmethod
    def chunk_matches_prompt(b,c):return all(b[k]==getattr(c,attr) for k,attr in [('prompt_version','version'),('phase','phase'),('prompt','prompt'),('session_id','session_id'),('model_version','model_version')])


def context(phase='pickup_1',version=0,t=10.):return SimpleNamespace(phase=phase,version=version,prompt='pickup text',session_id='s',model_version='b0',input_timestamp=t)


def test_prompt_version_distinguishes_repeated_pickup_even_with_same_text():
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'first');nextc=context('pickup_2',4)
    assert not g.result(b,np.zeros((50,58)),nextc,10.1,safe_boundary=True)['accepted']


def test_same_prompt_does_not_make_expired_chunk_fresh():
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'one')
    assert Bridge.chunk_matches_prompt(b,c)
    assert g.result(b,np.zeros((50,58)),c,12.1,safe_boundary=True)['reason']=='expired_or_future_result'


def test_safe_boundary_duplicate_and_action_values_are_preserved():
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'one');x=np.arange(50*58).reshape(50,58)/100.;before=x.copy()
    assert not g.result(b,x,c,10.1,safe_boundary=False)['accepted']
    assert g.result(b,x,c,10.1,safe_boundary=True)['accepted'] and np.array_equal(x,before)
    assert not g.result(b,x,c,10.2,safe_boundary=True)['accepted']
    newer=g.request(context(t=10.5),'two');result=g.result(newer,x+10,context(t=10.5),10.6,safe_boundary=True)
    assert result['first_action_max_delta']>0 and not result['continuity_is_policy_guarantee']


@pytest.mark.parametrize('actions',[np.zeros((15,58)),np.full((50,58),np.nan),'not-an-action'])
def test_invalid_action_chunks_rejected(actions):
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'one')
    assert g.result(b,actions,c,10.1,safe_boundary=True)['reason']=='invalid_action_chunk'


def test_request_observation_time_and_locked_model_are_bound():
    packet=dict(observation_id='o',input_timestamp=10.,model_version='b0',future_horizon_s=1.5)
    check_request_response(packet,observation_id='o',input_timestamp=10.,model_version='b0')
    for key,value in [('input_timestamp',10.1),('model_version','other'),('observation_id','other'),('future_horizon_s',1.)]:
        with pytest.raises(ValueError):check_request_response(dict(packet,**{key:value}),observation_id='o',input_timestamp=10.,model_version='b0')


def test_tampered_chunk_binding_is_rejected():
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'one')
    assert g.result(dict(b,observation_id='another'),np.zeros((50,58)),c,10.1,safe_boundary=True)['reason']=='request_binding_mismatch'


def test_invalid_result_time_cannot_apply_actions():
    g=OfflineChunkGate(Bridge);c=context();b=g.request(c,'one')
    assert not g.result(b,np.zeros((50,58)),c,'bad',safe_boundary=True)['accepted']
    assert g.last_action is None
