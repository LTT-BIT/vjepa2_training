import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.tomato_joint_memory.cache_reader import CausalBuffer,history_indices,causal_motion,past_beliefs,ema
from tools.tomato_joint_memory.models import make_model,load_memory_checkpoint
from tools.tomato_joint_memory.prediction_memory import folds
from tools.tomato_joint_memory.train import TensorData
from tools.tomato_joint_memory.contracts import config,write
from tools.tomato_phase_model import supervision_loss
from tools.tomato_joint_memory.evaluate import anticipation,warning_events,match_events

C=dict(hidden=128,dropout=.3,device='cpu',gap_reset_s=.75,memory_lengths=dict(T0=1,T1=4,T2=8,K1=4,P1=4),oof=dict(folds=5,seed=42,inner_val_fraction=.1))

def entry(n=10,eid='a'):
    rng=np.random.default_rng(1)
    return dict(episode=dict(episode_id=eid,source_recording_id=eid,split='train',fps=30,length=n*15,annotation=dict(segments=[dict(start_frame=0,end_frame=60,label='pickup'),dict(start_frame=60,end_frame=n*15,label='harvest')])),features=rng.normal(size=(n,4000)).astype('float32'),times=np.arange(n)*.5,anchors=np.arange(n)*15,input_indices=np.maximum(0,np.arange(n)[:,None]*15-np.arange(15,-1,-1)[None,:]*3),current_labels=np.arange(n)%4,future_labels=np.arange(n)%4,future_valid_mask=np.arange(n)<n-2)

@pytest.mark.parametrize('length',[1,4,8])
def test_online_offline_left_padding_and_gap_equal(length):
    x=entry()['features'];t=np.r_[np.arange(5)*.5,4+np.arange(5)*.5];ii,mask,dt=history_indices(t,length)
    b=CausalBuffer(length);b.reset('a','s','m')
    for i,tt in enumerate(t):
        xx,mm,dd=b.push(x[i],tt,'a','s','m');expected=x[ii[i].clip(0)].copy();expected[~mask[i]]=0
        np.testing.assert_array_equal(xx,expected);np.testing.assert_array_equal(mm,mask[i]);np.testing.assert_array_equal(dd,dt[i])
        assert np.all(ii[i][mask[i]]<=i)
    assert mask[5].sum()==1

@pytest.mark.parametrize('failure',['duplicate','reversed','session','episode','model','invalid','expired'])
def test_rejected_observation_does_not_mutate(failure):
    b=CausalBuffer(4);b.reset('a','s','m');x=np.zeros(4000,'float32');b.push(x,1,'a','s','m');before=list(b.values)
    kwargs=dict(t=1.5,episode='a',session='s',model='m')
    if failure=='duplicate':kwargs['t']=1
    if failure=='reversed':kwargs['t']=.5
    if failure in ['session','episode','model']:kwargs[failure]='wrong'
    if failure=='invalid':kwargs['valid']=False
    if failure=='expired':kwargs['available_at']=3
    with pytest.raises(ValueError):b.push(x,**kwargs)
    assert len(b.values)==len(before) and b.values[-1][0]==before[-1][0]
    np.testing.assert_array_equal(b.values[-1][1],before[-1][1])
    b.reset('b','new','m');assert not b.values

@pytest.mark.parametrize('group',['T0','T1','T2','K1','P1'])
def test_joint_shapes_future_tampering_and_reload(group,tmp_path):
    torch.manual_seed(1);torch.set_num_threads(1);e=entry();p=np.tile([.25]*8,(10,1)).astype('float32');beliefs={'a':p}
    train=TensorData([e],C,group,beliefs=beliefs);model=make_model(group,C).eval();z=train.predict(model)
    assert z['current'].shape==z['future'].shape==(10,4)
    modified=copy.deepcopy(e);modified['features'][6:]=12345;modified['current_labels'][6:]=3;modified['future_labels'][6:]=2
    pp=p.copy();pp[6:]=np.tile([1,0,0,0,0,1,0,0],(4,1))
    other=TensorData([modified],C,group,train.stats,{'a':pp});zz=other.predict(model)
    np.testing.assert_array_equal(z['current'][:6],zz['current'][:6]);np.testing.assert_array_equal(z['future'][:6],zz['future'][:6])
    binding={'statistics':'locked'};path=tmp_path/'model.pt';torch.save(dict(format='tomato-joint-memory-v1',group=group,state=model.state_dict(),bindings=binding),path)
    _,loaded=load_memory_checkpoint(path,C,binding);np.testing.assert_array_equal(z['current'],train.predict(loaded)['current'])
    with pytest.raises(ValueError):load_memory_checkpoint(path,C,{'statistics':'wrong'})

@pytest.mark.parametrize('n',[1,4,8])
def test_actual_streaming_predictions_match_training_constructor(n):
    e=entry();group={1:'T0',4:'T1',8:'T2'}[n];d=TensorData([e],C,group);m=make_model(group,C).eval();z=d.predict(m)
    buffer=CausalBuffer(n);buffer.reset('a','s','m')
    for i,t in enumerate(e['times']):
        raw,mask,dt=buffer.push(e['features'][i],t,'a','s','m');xx=(raw-d.mean)/d.std;xx[~mask]=0
        with torch.inference_mode():out=m(torch.tensor(xx[None]),torch.tensor(mask[None]),torch.tensor(dt[None]))
        np.testing.assert_allclose(out['current_logits'].numpy()[0],z['current'][i],atol=1e-6,rtol=1e-5)


def test_motion_and_beliefs_are_strictly_past_measurements():
    e=entry();x=e['features'];v=causal_motion(x,e['times']);np.testing.assert_array_equal(v[0],np.zeros(58));np.testing.assert_allclose(v[2],(x[2,-58:]-x[1,-58:])/.5)
    p=np.tile([.25]*8,(10,1)).astype('float32');p2=p.copy();p2[5:]=[1,0,0,0,0,1,0,0]
    np.testing.assert_array_equal(past_beliefs(p,e['times'])[:6],past_beliefs(p2,e['times'])[:6]);assert not past_beliefs(p,e['times'])[0].any()


def test_oof_membership_excludes_held_and_original_val_test():
    ee=[entry(eid=str(i)) for i in range(75)];ff=folds(ee,C);held=[]
    for f in ff:
        assert len(f['train'])==54 and len(f['val'])==6 and len(f['held'])==15
        assert not (set(f['held'])&(set(f['train'])|set(f['val'])));held+=f['held']
    assert len(set(held))==75


def test_joint_mask_loss_and_both_heads_gradients():
    a=torch.randn(3,4,requires_grad=True);b=torch.randn(3,4,requires_grad=True);yc=torch.tensor([0,1,2]);yf=torch.tensor([1,2,-1]);mask=torch.tensor([True,True,False]);w=dict(current=torch.ones(4),future=torch.ones(4));cfg=dict(prediction_mode='joint',lambda_current=1.,lambda_future=.5)
    loss,_=supervision_loss(dict(current_logits=a,future_logits=b),yc,yf,mask,w,cfg)
    expected=(torch.nn.functional.cross_entropy(a,yc)+.5*torch.nn.functional.cross_entropy(b[:2],yf[:2]))/1.5
    torch.testing.assert_close(loss,expected);loss.backward();assert a.grad.abs().sum()>0 and b.grad[:2].abs().sum()>0;assert not b.grad[2].any()


def test_strict_warning_rejects_post_boundary_and_nonadjacent_pair():
    e=entry();pc=np.tile([1,0,0,0],(10,1));pf=pc.copy();pf[4:]=[0,1,0,0]
    result=anticipation(e,pc,pf);assert result['tp']==0 and result['fp']==1 and result['fn']==1
    pf[3]=[0,1,0,0];assert anticipation(e,pc,pf)['tp']==1
    pc[3]=[0,0,0,1];assert anticipation(e,pc,pf)['tp']==0


def test_stable_warning_not_backdated_to_first_support():
    t=np.arange(5)*.5;pc=np.tile([1,0,0,0],(5,1));pf=np.tile([0,1,0,0],(5,1));ev=warning_events(t,pc,pf,.8,2)
    assert len(ev)==1 and ev[0]['time']==.5 and ev[0]['start_time']==0


def test_ema_is_causal_and_resets():
    t=np.array([0,.5,2]);p=np.tile([.8,.1,.05,.05],(3,1));p[2]=[.1,.8,.05,.05]
    z=ema(p,t);np.testing.assert_array_equal(z[:2],p[:2]);np.testing.assert_array_equal(z[2],p[2])


def test_checkpoint_bytes_hash_enforced(tmp_path):
    from tools.tomato_joint_memory.contracts import sha
    m=make_model('T0',C);p=tmp_path/'head.pt';torch.save(dict(format='tomato-joint-memory-v1',group='T0',state=m.state_dict(),bindings={}),p)
    load_memory_checkpoint(p,C,{},sha(p))
    with pytest.raises(ValueError,match='Checkpoint SHA'):load_memory_checkpoint(p,C,{},'bad-sha')


def test_training_integration_joint_sealed_reload_and_no_encoder(tmp_path,monkeypatch):
    from tools.tomato_joint_memory.train import fit,predict_entries
    from tools import tomato_phase_model as baseline
    monkeypatch.setattr(baseline,'load_encoder',lambda *a,**k:pytest.fail('Encoder must not load'))
    c={**C,'epochs':2,'patience':15,'batch_size':4,'learning_rate':3e-4,'weight_decay':.01,'grad_clip_norm':1.,'lambda_current':1.,'lambda_future':.5}
    e=entry(12);model,saved,meta,pred=fit([e],[e],c,'T1',42,tmp_path/'run',{})
    assert meta['updates']==6 and meta['encoder_new_calls']==0
    loaded,head=load_memory_checkpoint(tmp_path/'run/best_head.pt',c,saved['bindings'],meta['checkpoint_sha256'])
    np.testing.assert_allclose(pred['a']['future'],predict_entries(head,loaded,[e],c,'T1')['a']['future'],atol=1e-6)


def test_reversed_arrival_clock_does_not_mutate_stream():
    b=CausalBuffer(4);b.reset('a','s','m');x=np.zeros(4000,'float32');b.push(x,1,'a','s','m',available_at=1.7)
    with pytest.raises(ValueError,match='arrival'):b.push(x,1.5,'a','s','m',available_at=1.6)
    assert len(b.values)==1 and b.values[-1][0]==1 and b.last_arrival==1.7


def test_future_raw_rgb_measured_state_and_labels_cannot_change_earlier_outputs():
    """Synthetic shape/causality fixture, never evidence of real V-JEPA accuracy."""
    from types import SimpleNamespace
    from tools import tomato_dataset as ds,tomato_phase_model as baseline
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.ones(1));self.eval().requires_grad_(False)
        def forward(self,pixel_values_videos):
            z=pixel_values_videos.mean((1,2,3,4))[:,None,None]
            return SimpleNamespace(last_hidden_state=z.expand(-1,1,1024))
    torch.set_num_threads(1);e=entry(4);times=np.arange(60)/30
    rgb=np.zeros((60,3,256,256,3),np.uint8);q=np.arange(60*58,dtype=np.float32).reshape(60,58)/1000
    def encode(video,state):
        ids=np.stack([ds.sample_causal_indices(times,int(a)) for a in e['anchors']]);clips=video[ids].transpose(0,2,1,3,4,5)
        return baseline.fuse_features(Encoder(),clips,state[ids])
    first=encode(rgb,q);rgb[30:]=255;q[30:]=-123;second=encode(rgb,q)
    np.testing.assert_array_equal(first[:2],second[:2]);assert not np.array_equal(first[2:],second[2:])
    e['features']=first;data=TensorData([e],C,'T1');model=make_model('T1',C).eval();before=data.predict(model)
    changed=copy.deepcopy(e);changed['features']=second;changed['current_labels'][2:]=3;changed['future_labels'][2:]=0
    after=TensorData([changed],C,'T1',data.stats).predict(model)
    for h in ['current','future']:np.testing.assert_array_equal(before[h][:2],after[h][:2])
