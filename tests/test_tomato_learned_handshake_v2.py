import copy,dataclasses
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.tomato_review_ui.handshake import Handshake,LABELS
from tools.tomato_review_ui.learned_handshake.contracts import H0_CONFIG,TASK_POLICY,Observation,DecisionProposal
from tools.tomato_review_ui.learned_handshake.feature_builder import FeatureBuilder,observation
from tools.tomato_review_ui.learned_handshake.shell_adapter import ExecutionShell
from tools.tomato_review_ui.learned_handshake.rule_policies import RulePolicy
from tools.tomato_review_ui.learned_handshake.replay import run,h0
from tools.tomato_review_ui.learned_handshake.models import PolicyNet,LearnedPolicy,load_checkpoint
from tools.tomato_review_ui.learned_handshake.targets import label,weights
from tools.tomato_review_ui.learned_handshake.contracts import sha,json_sha


def row(t,current='pickup',future='harvest'):
 p=lambda stage:{s:.97 if s==stage else .01 for s in LABELS}
 return dict(frame_index=int(t*30),observed_at=t,heads={h:dict(probabilities=p(s)) for h,s in [('current',current),('future',future)]})

def obs(t=0,**kw):return observation(row(t,**kw),'s','B0','P')
def proposal(o,c='harvest',request=True,score=.9):return DecisionProposal(o.observation_id,o.observed_at,o.available_at,c,request,score,'test',o.model_version,o.policy_version)


def test_reference_ack_shell_same_active_as_h0():
 rows=[row(0)]+[row(t,'harvest','harvest') for t in [.5,1,1.5,2]]
 a=run(rows,3,RulePolicy('H0'));b=h0(rows,3)
 assert a['active_intervals']==b['active_intervals']
 assert [(e['kind'],e['time']) for e in a['events'] if e['kind'] in ['ack','late_ack','timeout','hard_timeout']]==[(e['kind'],e['time']) for e in b['events'] if e['kind'] in ['ack','late_ack','timeout','hard_timeout']]


def test_no_available_before_planning_and_atomic_commit():
 s=ExecutionShell();o=obs();s.ingest(o);s.apply(proposal(o));assert s.core.active=='pickup' and s.core.version==0
 s.core.planning(.5);assert s.core.active=='harvest' and s.core.version==1
 # .5 observation is available .59: cannot have acknowledged the .5 commit at .5.
 rr=run([row(.5)],1.6,RulePolicy('H0'));assert next(e['time'] for e in rr['events'] if e['kind']=='commit')==1.


def test_ack_late_hard_window_and_no_bypass():
 s=ExecutionShell();o=obs();s.ingest(o);s.apply(proposal(o));s.core.planning(.5)
 s.core.timer(2);assert s.core.state=='WAIT_LATE_ACK'
 for t in [2.,2.5]:
  o=obs(t,current='harvest',future='return');s.ingest(o);s.apply(proposal(o,'return',True));
  if t==2:assert s.core.active=='harvest' and s.core.state=='WAIT_LATE_ACK'
 s.core.planning(3);assert s.core.active=='return' and s.core.version==2
 s.core.timer(8);assert s.core.state=='RECOVERY_REQUIRED'
 o=obs(8.5,current='return',future='adjust');s.ingest(o);s.apply(proposal(o,'adjust'));s.core.planning(9);assert s.core.active=='return'


@pytest.mark.parametrize('failure',['duplicate','reversed','session','model','policy','held','expired','nan','arrival'])
def test_rejected_envelopes_leave_feature_history_unchanged(failure):
 b=FeatureBuilder('s','B0','P');o=obs(1);b.observe(o);q=obs(1.5)
 if failure in ['duplicate','held']:q=o
 if failure=='reversed':q=obs(.5)
 if failure in ['session','model','policy']:q=dataclasses.replace(q,**{{'session':'session_id','model':'model_version','policy':'policy_version'}[failure]:'wrong'})
 if failure=='expired':q=dataclasses.replace(q,available_at=3)
 if failure=='nan':q=dataclasses.replace(q,current=(float('nan'),0,0,0))
 if failure=='arrival':q=dataclasses.replace(q,available_at=1.0)
 with pytest.raises(ValueError):b.observe(q)
 assert len(b.history)==1 and b.history[-1]==o


def test_task_policy_external_and_identity_mask():
 s=ExecutionShell({'pickup':['return']});o=obs(future='return');s.ingest(o)
 assert s.apply(dataclasses.replace(proposal(o,'return'),policy_version='wrong'))=='proposal_identity_mismatch'
 assert s.core.active=='pickup';s.apply(proposal(o,'return'));s.core.planning(.5);assert s.core.active=='return'
 s=ExecutionShell({});o=obs();s.ingest(o);assert s.apply(proposal(o))=='task_mask';s.core.planning(.5);assert s.core.active=='pickup'


def test_no_gt_end_length_or_future_input_in_features_and_trace():
 rows=[row(t,'pickup' if t<2 else 'harvest','harvest') for t in [0,.5,1,1.5,2,2.5]];changed=copy.deepcopy(rows)
 for r in changed:r.update(annotation={'bad':'GT'},future_valid=False,current_true_stage=3,future_true_stage=2,episode_length=99,source_recording_id='test')
 a=run(rows,3,RulePolicy('R1'));b=run(changed,3,RulePolicy('R1'))
 assert a['events']==b['events'] and a['active_intervals']==b['active_intervals']
 for x,y in zip(a['records'],b['records']):np.testing.assert_array_equal(x['features']['flat'],y['features']['flat'])
 altered=copy.deepcopy(rows);altered[3:]=[row(t,'return','adjust') for t in [1.5,2,2.5]];c=run(altered,3,RulePolicy('R1'))
 assert [e for e in a['events'] if e['time']<1.5]==[e for e in c['events'] if e['time']<1.5]


def test_gap_resets_mask_and_duplicate_cannot_satisfy_k2():
 r=row(0);r['heads']['future']['probabilities']={k:.9 if k=='harvest' else .1/3 for k in LABELS}
 rr=run([r,r],1,RulePolicy('R1'));assert rr['summary']['commits']==0
 b=FeatureBuilder('s','B0','P');b.observe(obs());b.observe(obs(2));assert len(b.history)==1
 f=b.build(Handshake(H0_CONFIG),TASK_POLICY);assert f['mask'].tolist()==[False,False,False,True] and f['flat'].shape==(92,) and f['state'].shape==(48,)


@pytest.mark.parametrize('name',['L0','L1a','L1b','L2','L3a'])
def test_network_batch_stream_prefix_and_checkpoint(name,tmp_path):
 torch.set_num_threads(1);torch.manual_seed(42);b=FeatureBuilder('s','B0','P');s=Handshake(H0_CONFIG);features=[]
 for t in [0,.5,1,1.5]:b.observe(obs(t));features.append(b.build(s,TASK_POLICY))
 stats={k+suffix:np.zeros_like(features[0][k]) if suffix=='_mean' else np.ones_like(features[0][k]) for k in ['flat','sequence','state'] for suffix in ['_mean','_std']}
 model=PolicyNet(name).eval();p=LearnedPolicy(name,model,stats,{'hazard_switch_cost':[4,3,2,0,.3,4],'hazard_hold_cost':[0,.25,1,3,4,0]})
 xx={k:torch.tensor(np.array([f[k] for f in features])) for k in ['flat','sequence','state','mask']}
 with torch.inference_mode():bat=model(xx['flat'],xx['sequence'],xx['state'],xx['mask'])
 for i,f in enumerate(features):np.testing.assert_allclose(p.predict(f)['action'],bat['action'][i].numpy(),atol=1e-6,rtol=1e-5)
 saved=dict(name=name,seed=42,best_epoch=1,bindings={'sha':'x'},schema=dict(flat=92,state=48,sequence=[4,11]),state=model.state_dict(),stats=stats,stats_sha256=json_sha({k:v.tolist() for k,v in stats.items()}));path=tmp_path/'best_policy.pt';torch.save(saved,path);(tmp_path/'checkpoint.sha256').write_text(sha(path));_,loaded=load_checkpoint(path,{'sha':'x'},{'dropout':.1})
 np.testing.assert_array_equal(p.predict(features[-1])['action'],LearnedPolicy(name,loaded,stats,p.c).predict(features[-1])['action'])
 with pytest.raises(ValueError):load_checkpoint(path,{'sha':'wrong'},{'dropout':.1})


def test_targets_strict_bins_tail_and_late_catchup():
 e=dict(fps=30,annotation=dict(segments=[dict(start_frame=0,end_frame=90,label='pickup'),dict(start_frame=90,end_frame=180,label='harvest'),dict(start_frame=180,end_frame=270,label='return')]))
 b=FeatureBuilder('s','B0','P');b.observe(obs());f=b.build(Handshake(H0_CONFIG),TASK_POLICY)
 for tick,bin_id in [(1.,0),(2.,1),(2.5,2),(3.,3),(3.5,4),(4.5,-100)]:
  ff={**f,'commit_tick':tick};l=label(e,ff);assert l['time_bin']==bin_id
  if tick==4.5:assert l['gate']==1 and l['late_catchup']
 ff={**f,'active':'return','commit_tick':8};l=label(e,ff);assert l['next_stage']==-1 and l['action']==0 and l['time_bin']==-100 and l['right_censored']
 assert not label(e,{**f,'active':'return','commit_tick':9.5})['supervision_valid']
 # Repeated coarse classes locate supervision with local GT segment only.
 e['annotation']['segments'] +=[dict(start_frame=270,end_frame=300,label='adjust'),dict(start_frame=300,end_frame=330,label='pickup'),dict(start_frame=330,end_frame=400,label='harvest')]
 assert label(e,{**f,'commit_tick':10.5})['event']==4


def test_equal_source_weighting_not_negative_anchor_count():
 labs=[dict(event=0,time_bin=0)]*100+[dict(event=0,time_bin=3)]+[dict(event=0,time_bin=0)]*10+[dict(event=0,time_bin=3)];src=['a']*101+['b']*11;w=weights(src,labs)
 assert np.isclose(w[:101].sum(),w[101:].sum()) and np.isclose(w[:100].sum(),w[100])


@pytest.mark.parametrize('name',['L0','L1a','L1b','L2','L3a'])
def test_all_learned_policies_ignore_gt_and_future_suffix(name):
 torch.manual_seed(42);torch.set_num_threads(1);model=PolicyNet(name).eval();stats={}
 for key,shape in [('flat',(92,)),('sequence',(11,)),('state',(48,))]:stats[key+'_mean']=np.zeros(shape,np.float32);stats[key+'_std']=np.ones(shape,np.float32)
 c={'hazard_switch_cost':[4,3,2,0,.3,4],'hazard_hold_cost':[0,.25,1,3,4,0]};policy=LearnedPolicy(name,model,stats,c)
 rows=[row(t,'pickup' if t<1.5 else 'harvest') for t in [0,.5,1,1.5,2,2.5]];changed=copy.deepcopy(rows)
 for r in changed:r.update(annotation={'segments':['random']},future_valid=False,current_true_stage=2,future_true_stage=3,total_duration=99999)
 a=run(rows,3,policy);b=run(changed,3,policy);assert a['events']==b['events'] and a['active_intervals']==b['active_intervals']
 altered=copy.deepcopy(changed)
 for r in altered[3:]:r['heads']=row(r['observed_at'],'return','adjust')['heads']
 d=run(altered,3,policy);assert [e for e in a['events'] if e['time']<1.5]==[e for e in d['events'] if e['time']<1.5]


def test_censored_and_off_trajectory_supervision_are_distinct():
 e=dict(fps=30,length=180,annotation=dict(segments=[dict(start_frame=0,end_frame=90,label='pickup'),dict(start_frame=90,end_frame=180,label='harvest')]))
 b=FeatureBuilder('s','B0','P');b.observe(obs());f=b.build(Handshake(H0_CONFIG),TASK_POLICY)
 tail=label(e,{**f,'active':'harvest','commit_tick':5.5});assert tail['right_censored'] and tail['time_bin']==-100 and tail['next_stage']==-1 and tail['action']==0
 unknown=label(e,{**f,'active':'return','commit_tick':5.5});assert not unknown['right_censored'] and unknown['time_bin']==5
 outside=label(e,{**f,'active':'harvest','commit_tick':6.});assert not outside['supervision_valid'] and outside['action']==outside['time_bin']==-100


@pytest.mark.parametrize('bucket,expected',[(0,False),(1,False),(2,False),(3,True),(4,True),(5,False)])
def test_hazard_cost_matrix_is_fixed_and_differs_from_mass_ablation(bucket,expected):
 class Fixed(torch.nn.Module):
  def forward(self,x,seq,state,mask):
   a=torch.full((len(x),4),-30.);a[:,1]=30.;t=torch.full((len(x),6),-30.);t[:,bucket]=30.
   return dict(action=a,time=t)
 stats={}
 for key,shape in [('flat',(92,)),('sequence',(11,)),('state',(48,))]:stats[key+'_mean']=np.zeros(shape,np.float32);stats[key+'_std']=np.ones(shape,np.float32)
 c={'hazard_switch_cost':[4,3,2,0,.3,4],'hazard_hold_cost':[0,.25,1,3,4,0]};p=LearnedPolicy('L3a',Fixed(),stats,c);b=FeatureBuilder('s','B0','P');o=obs();b.observe(o);core=Handshake(H0_CONFIG);f=b.build(core,TASK_POLICY)
 assert p.propose(o,f,core,TASK_POLICY).request_commit==expected
 if bucket==2:p.mapping='mass';assert p.propose(o,f,core,TASK_POLICY).request_commit


@pytest.mark.parametrize('mode',['nan_score','bad_score','backdated','stale','nan_time'])
def test_bad_proposal_holds_and_serializes_without_bypassing_planner(mode):
 import json
 s=ExecutionShell();o=obs();s.ingest(o);s.apply(proposal(o));p=proposal(o)
 if mode=='nan_score':p=dataclasses.replace(p,proposal_score=float('nan'))
 if mode=='bad_score':p=dataclasses.replace(p,proposal_score='bad')
 if mode=='backdated':p=dataclasses.replace(p,available_at=.01)
 if mode=='stale':p=dataclasses.replace(p,available_at=1.)
 if mode=='nan_time':p=dataclasses.replace(p,available_at=float('nan'))
 assert s.apply(p) in ['invalid_score','stale_proposal'];s.core.planning(1.5)
 assert s.core.active=='pickup' and s.core.version==0
 json.dumps(s.core.events,allow_nan=False)


@pytest.mark.parametrize('current,future_probability',[('harvest',.7),('pickup',.85)])
def test_r2_accelerates_current_catchup_without_current_veto(current,future_probability):
 r=row(0,current,'harvest');r['heads']['future']['probabilities']={s:future_probability if s=='harvest' else (1-future_probability)/3 for s in LABELS};o=observation(r,'s','B0','P');b=FeatureBuilder('s','B0','P');b.observe(o);core=Handshake(H0_CONFIG);f=b.build(core,TASK_POLICY)
 assert RulePolicy('R2').propose(o,f,core,TASK_POLICY).request_commit
 if future_probability==.7:assert not RulePolicy('H0').propose(o,f,core,TASK_POLICY).request_commit


def test_logistic_has_explicit_l2_penalty_and_validation_excludes_it():
 from tools.tomato_review_ui.learned_handshake.train import loss
 model=PolicyNet('L0');model.l2_coefficient=.01;n=4
 t=dict(flat=torch.zeros(n,92),sequence=torch.zeros(n,4,11),state=torch.zeros(n,48),mask=torch.ones(n,4,dtype=torch.bool),weight=torch.ones(n),gate=torch.tensor([0.,1.,0.,1.]))
 base=loss(model,t,slice(None),{},torch.tensor(1.),regularize=False);penalized=loss(model,t,slice(None),{},torch.tensor(1.),regularize=True)
 torch.testing.assert_close(penalized-base,.005*model.action.weight.square().sum())
