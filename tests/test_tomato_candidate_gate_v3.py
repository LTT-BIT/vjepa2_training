import copy,dataclasses,json,types,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.tomato_review_ui.handshake import Handshake,Evidence,LABELS,replay as baseline_replay
from tools.tomato_review_ui.candidate_gate_v3.contract import *
from tools.tomato_review_ui.candidate_gate_v3.candidate import CandidateProvider
from tools.tomato_review_ui.candidate_gate_v3.features import GateFeatures,make_observation,D,schema
from tools.tomato_review_ui.candidate_gate_v3.port import CandidateRequestPort
from tools.tomato_review_ui.candidate_gate_v3.policy import GateNet,GatePolicy,R1Policy,load_checkpoint
from tools.tomato_review_ui.candidate_gate_v3.replay import run,pure_h0
from tools.tomato_review_ui.candidate_gate_v3.targets import label
from tools.tomato_review_ui.candidate_gate_v3.train import objective


def row(t,c='pickup',f='harvest',prob=.97):
 p=lambda s,q:{k:q if k==s else (1-q)/3 for k in LABELS}
 return dict(frame_index=round(t*30),observed_at=t,heads={h:dict(probabilities=p(s,q)) for h,s,q in [('current',c,.97),('future',f,prob)]})
def setup(t=0,c='pickup',f='harvest',prob=.97):
 o=make_observation(row(t,c,f,prob),'s','p');port=CandidateRequestPort();features=GateFeatures('s','B0-seed42','p');features.observe(o);port.ingest(o);provider=CandidateProvider();cand=provider.update(o,port.core.active,TASK_POLICY);port.bind_candidate(cand);ft=features.snapshot(o,port.core,cand,TASK_POLICY);return o,port,features,provider,cand,ft

def decision(o,c,action='REQUEST',score=.9):return GateDecision(c.stage,action,score,o.observation_id,o.observed_at,o.available_at,'p')


def test_real_evidence_public_interface_never_mutates_future():
 o,p,b,provider,c,f=setup();assert p.core.last.future is o.future and p.core.pending is None
 assert p.core.state=='READY';p.apply_decision(decision(o,c),'G');assert p.core.active=='pickup';p.planning_tick(.5);assert p.core.active=='harvest'
 assert not p.planning_tick(.5) and p.core.version==1


@pytest.mark.parametrize('stage,confidence,expected',[('harvest',.65,'harvest'),('harvest',.7,'harvest'),('harvest',.64,None),('pickup',.99,None),('return',.99,None)])
def test_single_candidate_prepare_only(stage,confidence,expected):
 *_,c,f=setup(f=stage,prob=confidence);assert c.stage==expected


def test_no_gate_can_create_another_legal_candidate_or_duplicate_support():
 o,p,_,_,c,_=setup();d=dataclasses.replace(decision(o,c),candidate_stage='return');assert p.apply_decision(d,'G')=='candidate_mismatch'
 p.apply_decision(decision(o,c),'G');count=p.core.pending['count'];assert p.apply_decision(decision(o,c),'G')=='duplicate_gate_decision';assert p.core.pending is None or p.core.pending['count']==count


def test_original_h0_and_instrumented_trace_exact_and_r1_old_trace():
 rows=[row(0)]+[row(t,'harvest','harvest') for t in [.5,1,1.5,2]];a=run(rows,3);b=pure_h0(rows,3)
 for k in ['events','active_intervals','pending_intervals','summary']:assert a[k]==b[k]
 from tools.tomato_review_ui.learned_handshake.replay import run as old_run
 from tools.tomato_review_ui.learned_handshake.rule_policies import RulePolicy
 a=run(rows,3,R1Policy());b=old_run(rows,3,RulePolicy('R1'))
 for k in ['events','active_intervals','pending_intervals','summary']:assert a[k]==b[k]


def test_custom_allowed_graph_and_future_none_force_hold():
 o,p,b,provider,c,f=setup(f='return');assert c.stage is None
 provider=CandidateProvider();c=provider.update(o,'pickup',{'pickup':['return','harvest']});assert c.stage=='return'
 p=CandidateRequestPort({'pickup':['return']});p.ingest(o);p.bind_candidate(c);p.apply_decision(decision(o,c),'G');p.planning_tick(.5);assert p.core.active=='return'


@pytest.mark.parametrize('failure',['duplicate','reversed','session','model','policy','expired','nan','held'])
def test_envelope_rejects_without_new_history(failure):
 o,p,b,provider,c,f=setup();q=make_observation(row(.5),'s','p')
 if failure=='duplicate':q=o
 if failure=='reversed':q=dataclasses.replace(q,observed_at=-.5)
 if failure=='held':q=dataclasses.replace(q,observed_at=0.)
 if failure in ['session','model','policy']:q=dataclasses.replace(q,**{{'session':'session_id','model':'model_version','policy':'policy_version'}[failure]:'other'})
 if failure=='expired':q=dataclasses.replace(q,available_at=2.)
 if failure=='nan':q=dataclasses.replace(q,future=(float('nan'),0,0,0))
 with pytest.raises(ValueError):b.observe(q)
 assert len(b.rows)==1


def test_gap_reset_and_no_future_padding_or_hidden_session():
 o,p,b,provider,c,f=setup();assert f['mask'].tolist()==[False,False,False,True]
 q=make_observation(row(2),'s','p');b.observe(q);p.ingest(q);c=provider.update(q,p.core.active,TASK_POLICY);f=b.snapshot(q,p.core,c,TASK_POLICY);assert f['mask'].tolist()==[False,False,False,True]
 assert GateFeatures('new','B0-seed42','p').rows==__import__('collections').deque(maxlen=4)


@pytest.mark.parametrize('group',['G0','G1','G2','G3'])
def test_same_input_only_one_output_and_batch_stream_reload_gt_isolation(group,tmp_path):
 torch.set_num_threads(1);torch.manual_seed(1);o,p,b,provider,c,f=setup();stats=dict(mean=np.zeros(D,np.float32),std=np.ones(D,np.float32));m=GateNet(group).eval();policy=GatePolicy(group,m,stats,'p')
 with torch.inference_mode():z=m(torch.tensor(f['sequence'][None]),torch.tensor(f['mask'][None]))
 assert z['gate'].shape==(1,) and (z['hazard'] is None or z['hazard'].shape==(1,6))
 a=policy.decide(o,f,c,p.core);changed={**f,'annotation':{'bad':1},'future_valid':False,'true_stage':3,'source_id':'other','length':9999};assert a==policy.decide(o,changed,c,p.core)
 none=dataclasses.replace(c,stage=None);assert policy.decide(o,f,none,p.core).action=='HOLD'
 saved=dict(format='candidate-gate-v3',group=group,bindings={'sha':'x'},schema=schema(),stats=stats,stats_sha256=json_sha({k:v.tolist() for k,v in stats.items()}),state=m.state_dict());path=tmp_path/'best.pt';torch.save(saved,path);(tmp_path/'checkpoint.sha256').write_text(sha(path));_,loaded=load_checkpoint(path,{'sha':'x'},{'dropout':.1});assert a==GatePolicy(group,loaded,stats,'p').decide(o,f,c,p.core)
 with pytest.raises(ValueError):load_checkpoint(path,{'sha':'wrong'},{'dropout':.1})


def meta():return dict(fps=30,length=180,annotation=dict(segments=[dict(start_frame=0,end_frame=90,label='pickup'),dict(start_frame=90,end_frame=180,label='harvest')]))


@pytest.mark.parametrize('tick,expected',[(4.,0),(3.5,1),(3.,1),(2.9,2),(2.8,2),(2.5,3),(2.,4),(1.5,5)])
def test_hazard_observed_bucket_edges(tick,expected):
 *_,f=setup();target=label(meta(),{**f,'commit_tick':tick});assert target['hazard_bucket']==expected


@pytest.mark.parametrize('tick,allowed',[(4.,[5]),(5.,[5]),(5.5,[4,5]),(5.9,[2,3,4,5])])
def test_right_censoring_is_not_false_no_transition(tick,allowed):
 *_,f=setup();target=label(meta(),{**f,'active':'harvest','candidate':'return','commit_tick':tick});assert target['censored'] and not target['observed'];assert np.flatnonzero(target['hazard_allowed']).tolist()==allowed
 assert not label(meta(),{**f,'commit_tick':6})['eligible']


def test_auxiliary_mask_has_no_nan_gradient_and_does_not_control_action():
 m=GateNet('G3');n=3;t=dict(sequence=torch.zeros(n,4,D),mask=torch.ones(n,4,dtype=torch.bool),weight=torch.ones(n),gate=torch.tensor([0.,1.,0.]),allowed=torch.tensor([[False]*6,[False]*5+[True],[False,False,True,True,True,True]]),hazard_valid=torch.tensor([False,True,True]));cfg={'hazard_lambda':.25,'explicit_l2':.01};v,bce,hz=objective(m,t,slice(None),torch.tensor(1.),cfg);v.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());torch.testing.assert_close(v,bce+.25*hz)
 o,p,_,_,c,f=setup();stats=dict(mean=np.zeros(D,np.float32),std=np.ones(D,np.float32));policy=GatePolicy('G3',m,stats,'p');a=policy.decide(o,f,c,p.core)
 with torch.no_grad():m.hazard.weight.fill_(100);m.hazard.bias.fill_(-100)
 assert a==policy.decide(o,f,c,p.core)


def test_no_prediction_until_available_and_hard_timer_without_new_results():
 class Request:
  name='G';version='p'
  def decide(self,o,f,c,core):return decision(o,c,score=1.)
 a=run([row(.5)],7,Request());assert next(e['time'] for e in a['events'] if e['kind']=='commit')==1.;assert next(e['time'] for e in a['events'] if e['kind']=='hard_timeout')==6.
 # Coincident completed result is processed before planner.
 r=row(.41);r['available_at']=.5;a=run([r],1,Request());assert next(e['time'] for e in a['events'] if e['kind']=='commit')==.5


def test_decision_clock_spoof_and_nonfinite_scores_are_rejected():
 o,p,_,_,c,_=setup();assert p.apply_decision(dataclasses.replace(decision(o,c),available_at=1.),'G')=='decision_clock_mismatch'
 assert p.apply_decision(dataclasses.replace(decision(o,c),score=float('nan')),'G')=='invalid_score';p.planning_tick(.5);assert p.core.version==0


def test_full_replay_gt_fields_removed_byte_identical_causal_features():
 rows=[row(t,'pickup' if t<1 else 'harvest','harvest' if t<1.5 else 'return') for t in [0,.5,1,1.5,2]]
 enriched=[{**r,'annotation':{'segments':[{'label':'adjust','start_frame':999}]},'true_stage':'adjust','future_valid':False,'source_recording_id':'poison','length':100000} for r in rows]
 a=run(rows,3,R1Policy(),'s');b=run(enriched,3,R1Policy(),'s')
 assert a['events']==b['events']
 for x,y in zip(a['records'],b['records']):
  assert x['decision']==y['decision'];assert x['feature']['sequence'].tobytes()==y['feature']['sequence'].tobytes();assert x['feature']['mask'].tobytes()==y['feature']['mask'].tobytes()


def test_calibration_tied_scores_uses_threshold_grouped_average_precision():
 from tools.tomato_review_ui.candidate_gate_v3.analyze import calibration
 r=calibration([1,0,1,0],[.5,.5,.5,.5]);assert r['pr_auc']==.5 and r['brier']==.25


@pytest.mark.parametrize('remaining,expected',[(.3,True),(.2,True),(.1,False),(.01,False)])
def test_censored_tail_main_gate_unknown_inside_anticipatory_window(remaining,expected):
 *_,f=setup();t=label(meta(),{**f,'active':'harvest','candidate':'return','commit_tick':6-remaining});assert t['censored'] and t['hazard_valid'];assert t['gate_valid']==expected


def test_unknown_gate_has_zero_gate_gradient_but_hazard_remains_supervised():
 m=GateNet('G3');t=dict(sequence=torch.zeros(1,4,D),mask=torch.ones(1,4,dtype=torch.bool),weight=torch.ones(1),gate=torch.zeros(1),gate_valid=torch.zeros(1,dtype=torch.bool),allowed=torch.tensor([[False,False,True,True,True,True]]),hazard_valid=torch.ones(1,dtype=torch.bool));total,bce,hz=objective(m,t,slice(None),torch.tensor(1.),{'hazard_lambda':.25});assert bce==0 and hz>0;total.backward();assert m.gate.weight.grad.abs().sum()==0 and m.hazard.weight.grad.abs().sum()>0
