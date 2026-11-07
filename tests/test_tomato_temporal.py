import numpy as np
import pytest
import torch
from tools import tomato_temporal_heads as tm,tomato_time_sampling as sampler


def test_sampling_causal_latest_and_noninteger_horizon():
 times=np.arange(300)/30
 for design in [(8,10,False),(16,10,False),(16,5,False),(16,10,True)]:
  for anchor in [0,1,45,299]:
   ids=sampler.sample(times,anchor,*design)
   assert len(ids)==design[0] and ids[-1]==anchor and np.all(ids<=anchor) and np.all(ids>=0)
 assert len({str(sampler.contract(*d)) for d in [(8,10,False),(16,10,False),(16,5,False),(16,10,True)]})==4
 assert sampler.horizon_frames(.25,30)['offset_frames']==8
 assert sampler.horizon_frames(.75,30)['offset_frames']==23
 assert sampler.horizon_frames(1.,30)['effective_s']==1.


@pytest.mark.parametrize('arch',['gru','tcn'])
def test_causal_head_has_no_future_or_cross_episode(arch):
 spec=dict(architecture=arch,dimension=4000,context=4,boundary_weight=.2)
 torch.manual_seed(1);head=tm.TemporalHead(spec).eval();x=np.random.default_rng(1).normal(size=(8,4000)).astype(np.float32)
 indices=tm.history_indices(8,4)
 assert np.array_equal(indices[0],[0,0,0,0]) and np.all(indices<=np.arange(8)[:,None])
 first=tm.predict(head,x,indices);x[5:]=1000.;second=tm.predict(head,x,indices)
 for key in first:np.testing.assert_array_equal(first[key][:5],second[key][:5])
 # A separate episode starts at its own offset, never previous episode tail.
 assert np.array_equal(tm.history_indices(3,4)[0]+8,[8,8,8,8])


def test_boundary_targets_repeated_classes_tail_and_missing_stage():
 episode=dict(length=120,fps=30,annotation=dict(segments=[dict(start_frame=0,end_frame=30,label='pickup'),dict(start_frame=30,end_frame=60,label='pickup'),dict(start_frame=60,end_frame=120,label='return')]))
 for segment in episode['annotation']['segments']:segment['reviewed']=True
 y,valid=tm.boundary_targets(episode,np.array([0,30,45,60,90,105]),1.)
 assert y.tolist()==[0,1,1,0,0,0]
 assert valid.tolist()==[True,True,True,True,False,False]


def test_velocity_uses_position_names_and_real_causal_delta(monkeypatch):
 names=[f'j{i}.position' for i in range(58)];seen={}
 def state(e,selected,mode):
  seen['names']=selected
  return np.arange(30)/30,np.zeros((30,58),np.float32)
 monkeypatch.setattr(tm.ds,'read_state',state)
 q=np.zeros((2,16,58),np.float32);q[1,-1]=1.
 features=np.concatenate([np.zeros((2,3072),np.float32),q.reshape(2,-1)],1)
 ids=np.zeros((2,16),np.int64);ids[1,-2:]=[3,6]
 entry=dict(episode=dict(info={'features':{'observation.state':{'names':names+['j0.velocity']}}}),features=features,input_indices=ids)
 actual=tm.append_velocity(entry)
 assert seen['names']==names and actual.shape==(2,4058)
 np.testing.assert_array_equal(actual[0,-58:],0)
 np.testing.assert_allclose(actual[1,-58:],10.)


def synthetic_checkpoint(path,arch='gru'):
 from tools import tomato_dataset as ds,tomato_phase_model as model
 spec=dict(architecture=arch,dimension=4000,context=4 if arch in ('gru','tcn') else 1,boundary_weight=0.,horizon_s=1.5)
 names=[f'{s}_fr3_joint{i}.position' for s in ['left','right'] for i in range(1,8)]+[f'hand{i}.position' for i in range(44)]
 payload=dict(format_version=4,architecture_version='tomato-causal-head-v1',spec=spec,
  mean=torch.zeros(4000),std=torch.ones(4000),head=tm.TemporalHead(spec).state_dict(),training_seed=42,
  input_schema=dict(label_order=list(ds.LABELS),joint_names=names,state_history_shape=[16,58],feature_hz=2,causal=True,
  feature_sampler_version=ds.SAMPLER_VERSION,arms_indices=list(range(14)),hands_indices=list(range(14,58))),
  pretrained=dict(sha256=model.WEIGHTS_SHA256,config_sha256=model.CONFIG_SHA256),feature_version=model.FEATURE_VERSION,
  normalization_split='train only',provenance=dict(spec=spec,seed=42))
 torch.save(payload,path);return payload


@pytest.mark.parametrize('arch',['gru','tcn'])
def test_online_context_matches_offline_and_resets_episode(tmp_path,arch):
 path=tmp_path/'head.pt';payload=synthetic_checkpoint(path,arch);session=tm.CausalFeatureSession(path)
 x=np.random.default_rng(4).normal(size=(8,4000)).astype(np.float32);expected=tm.predict(session.head,x,tm.history_indices(8,4))
 actual=[session.predict(v,float(i)*.5) for i,v in enumerate(x)]
 for key in expected:np.testing.assert_allclose(np.concatenate([v[key] for v in actual]),expected[key],atol=1e-5,rtol=1e-5)
 with pytest.raises(ValueError,match='Fresh'):session.predict(x[0],3.5)
 session.reset();assert session.last_time is None
 first=session.predict(x[0],0.)
 for key in expected:np.testing.assert_allclose(first[key],expected[key][:1],atol=1e-5,rtol=1e-5)


def test_runtime_one_encoder_and_rejects_wrong_cadence_or_schema(tmp_path,monkeypatch):
 from tools import tomato_causal_agent as agent
 path=tmp_path/'head.pt';payload=synthetic_checkpoint(path);saved,_=agent.load_checked(path)
 instance=object.__new__(agent.CausalVideoAgent);instance.saved=saved;instance.session=tm.CausalFeatureSession(path)
 instance.encoder=object();instance.version='sha';instance.names=saved['input_schema']['joint_names'];instance.session_id='s'
 calls=[]
 monkeypatch.setattr(agent.time,'time',lambda:101.)
 def fuse(*args):calls.append(1);return np.zeros((1,4000),np.float32)
 monkeypatch.setattr(agent.model,'fuse_features',fuse)
 q=np.zeros((16,58),np.float32);times=100.-np.arange(15,-1,-1)/10
 out=instance.predict(None,q,instance.names,times,timestamp=100.,observation_id='a',session_id='s')
 assert len(calls)==1 and len(out['future_probabilities'])==4
 with pytest.raises(ValueError,match='higher-rate'):instance.predict(None,q,instance.names,times+.2,timestamp=100.2,observation_id='b',session_id='s')
 with pytest.raises(ValueError,match='sampling times'):instance.predict(None,q,instance.names,100.5-np.arange(15,-1,-1)/5,timestamp=100.5,observation_id='c',session_id='s')
 assert len(calls)==1
 payload['input_schema']['joint_names'][0]='wrong.position';torch.save(payload,path)
 with pytest.raises(ValueError,match='name/order'):agent.load_checked(path)


def test_probability_mixture_is_exact_and_preserves_current():
 from tools.tomato_agent_mixture import MixtureHead
 from tools import tomato_phase_model as model
 c=model.head_model(58).eval();f=model.head_model(58).eval();new=tm.TemporalHead(dict(architecture='mlp',dimension=4000,context=1,boundary_weight=0.)).eval()
 head=MixtureHead(c,f,new,.5).eval();x=torch.randn(3,4000)
 with torch.inference_mode():
  z=head(x);expected=(f(x).softmax(-1)+new(x[:,None])['future_logits'].softmax(-1))*.5
  torch.testing.assert_close(z['current_logits'],c(x));torch.testing.assert_close(z['future_logits'].softmax(-1),expected)


def test_binary_average_precision_handles_tied_scores():
 from tools.train_tomato_temporal import binary_scores
 y=np.array([1,0,1,0])
 tied=binary_scores(np.full(4,.5),y)
 assert tied['average_precision']==pytest.approx(.5) and tied['roc_auc']==pytest.approx(.5)
 perfect=binary_scores(np.array([.9,.1,.8,.2]),y)
 assert perfect['average_precision']==1 and perfect['roc_auc']==1


def test_available_warning_excludes_results_delivered_after_boundary():
 from tools.select_tomato_temporal import availability
 def event(time):
  return dict(per_episode={'e':dict(events=[dict(time=time,previous=0,label=1)],
   boundaries=[dict(time=2.,end=4.,previous=0,label=1)],duration_s=5.)})
 assert availability(event(1.98),'future',1.,.05)['recall']==0
 # Processing delay must not rescue a forecast outside its input-time horizon.
 assert availability(event(.99),'future',1.,.05)['recall']==0
 valid=availability(event(1.1),'future',1.,.05)
 assert valid['recall']==1 and valid['lead_p50_s']==pytest.approx(.85)
 assert availability(event(2.),'current',1.,.05)['delay_p95_s']==pytest.approx(.05)
