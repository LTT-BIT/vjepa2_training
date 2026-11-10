from dataclasses import replace
import json
import numpy as np
import pytest
from tools.tomato_task_memory import TaskMemory, Observation, LABELS
from tools.tomato_state_recovery import StateRecovery, CLASSES


def p(k,conf=.97):
    a=np.full(4,(1-conf)/3);a[k]=conf;return tuple(a)


def obs(i,k=0,**kw):
    return replace(Observation('test',i,f'test:{i}',i*.5,i*.5+.01,p(k),p(k),'b0'),**kw)


def replay(classes,mode='M2'):
    m=TaskMemory('test','b0',mode=mode);b=StateRecovery('baseline');rows=[]
    for i,k in enumerate(classes):
        o=obs(i,k);r=b.update(o.current,o.future,o.timestamp,now=o.arrival_timestamp)
        rows.append(m.observe(o,b0_phase=r['phase']))
    return m,rows


def test_shadow_cannot_mutate_b0_and_recognition_is_not_success():
    m,rows=replay([k for k in CLASSES for _ in range(6)],'M1')
    assert m.phase==6 and len(m.confirmed)==6
    assert all(not r['suggests_prompt_update'] and not r['progress']['task_complete'] for r in rows)
    assert all(e.physical_success=='unobserved' for e in m.events)
    assert not any(e.event_type in ('operation_command','operation_result') for e in m.events)
    assert len({e['event_id'] for e in m.confirmed})==6
    assert [e['to_phase'] for e in m.confirmed]==list(range(1,7))
    assert all(e['confirmation_time'] is not None for e in m.confirmed)


def test_causal_prefix_and_no_truth_api():
    _,a=replay([0]*4+[1]*4+[2]*4)
    _,b=replay([0]*4+[1]*2)
    assert a[:6]==b
    with pytest.raises(TypeError):TaskMemory('test','b0').observe(obs(0),truth_phase=4)


def test_future_candidate_cannot_confirm_current_and_expires_without_observations():
    m=TaskMemory('test','b0',mode='M2');o=obs(0,future=p(1));m.observe(o)
    event=m.advisory;assert event.status=='candidate'
    m.poll(2.);assert event.status=='expired' and m.advisory is None and m.phase==0
    assert m.view()['progress']['physical_completion']=='unknown'


def test_wrong_candidate_rejected_and_history_never_erased():
    m,rows=replay([0]*3+[1]*3+[2]+[1]*4)
    assert m.phase==1 and len(m.confirmed)==1
    assert any(e.event_type=='transition' and e.status=='rejected' for e in m.events)
    ids=m.view()['confirmed_event_ids'];m.observe(obs(12,3));assert m.view()['confirmed_event_ids']==ids


@pytest.mark.parametrize('mode',['M1','M2'])
def test_four_classes_cannot_teleport_between_repeated_positions(mode):
    m,_=replay([0]*4+[1]*4+[2]*4+[0]*8+[1]*8+[2]*8,mode)
    assert m.phase==2
    m,rows=replay([k for k in CLASSES for _ in range(6)],mode)
    assert m.phase==6
    assert all(b['confirmed_phase']-a['confirmed_phase'] in (0,1) for a,b in zip(rows,rows[1:]))


def test_bounded_memory_preserves_compact_confirmed_history_and_reset_isolated():
    logs=[];m=TaskMemory('test','b0',mode='M2',short_size=4,long_size=16,sink=logs.append)
    for i,k in enumerate([0]*3+[1]*3+[2]*3+[3,0]*60):m.observe(obs(i,k))
    assert len(m.short)==4 and len(m.events)==16 and len(m.confirmed)==2
    assert len(logs)>16
    m.reset('next');assert m.phase==0 and not m.confirmed and not m.short and m.last_valid is None
    assert all(e.event_id.startswith('next:') for e in m.events)
    assert not m.observe(obs(0))['valid'] and m.last_valid is None


@pytest.mark.parametrize('change',[
 dict(arrival_timestamp=3.),dict(timestamp=-1.),dict(sequence=-1),
 dict(observation_id='wrong'),dict(labels=tuple(reversed(LABELS))),
 dict(current=(np.nan,0.,0.,0.)),dict(current=(1.,0.,0.)),dict(model_version='other')])
def test_invalid_inputs_never_confirm_or_rewrite_facts(change):
    m=TaskMemory('test','b0',mode='M2');m.observe(obs(0));m.observe(obs(1,1))
    assert not m.observe(obs(2,1,**change))['valid']
    assert m.phase==0 and not m.confirmed


def test_duplicates_and_out_of_order_and_gap_break_streak():
    m=TaskMemory('test','b0',mode='M2');m.observe(obs(0));m.observe(obs(1,1))
    assert not m.observe(obs(1,1))['valid']
    assert m.observe(obs(2,1))['confirmed_phase']==0
    assert m.observe(obs(3,1))['confirmed_phase']==1
    assert not m.observe(obs(4,2,timestamp=1.))['valid']
    m=TaskMemory('test','b0',mode='M2');m.observe(obs(0));m.observe(obs(1,1))
    assert m.observe(obs(5,1))['confirmed_phase']==0
    assert any(e.status=='expired' for e in m.events)


def test_named_robot_state_is_causal_and_not_physical_success():
    names=tuple([f'right_fr3_{i}' for i in range(7)]+[f'left_fr3_{i}' for i in range(7)]+[f'hand_{i}' for i in range(44)])
    m=TaskMemory('test','b0',mode='M2',joint_names=names)
    assert not m.observe(obs(0,positions=tuple(np.zeros(58)),state_timestamp=.1,joint_names=names))['valid']
    for i in range(2):assert m.observe(obs(i,positions=tuple(np.ones(58)*i),state_timestamp=i*.5,joint_names=names))['valid']
    assert m.motion['arm_velocity_rms']==2.
    assert m.view()['progress']['physical_completion']=='unknown'
    assert not m.observe(obs(2,positions=tuple(np.zeros(58)),state_timestamp=1.,joint_names=names[::-1]))['valid']


def test_snapshot_is_detached_and_json_serializable():
    m,_=replay([0]*3+[1]*3);v=m.snapshot();json.dumps(v,allow_nan=False)
    v['confirmed_transitions'].clear()
    # snapshot must not grant callers write access to memory.
    assert len(m.confirmed)==1


def test_nested_snapshot_cannot_rewrite_confirmed_ledger():
    m,_=replay([0]*3+[1]*3)
    v=m.snapshot();v['confirmed_transitions'][0]['to_phase']=6
    v['short_term'][0]['current'][0]=0.
    assert m.ledger_phase()==1 and m.short[0]['current'][0]>.9


def test_ledger_cannot_silently_reset_from_corrupted_history():
    m,_=replay([0]*3+[1]*3);m.confirmed[0]['to_phase']=4
    with pytest.raises(RuntimeError):m.ledger_phase()


def test_readonly_b0_snapshot_never_licenses_skipping_stage():
    m=TaskMemory('test','b0',mode='M1')
    v=m.observe(obs(0),b0_phase=4)
    assert v['confirmed_phase']==0 and 'invalid_b0_snapshot' in v['uncertainty']


def test_fault_schedule_never_requires_gt_or_boundary():
    from tools.evaluate_tomato_task_memory import messages
    o=obs(8,0)
    assert messages(o,obs(7),'missing_observations',8,0)==[]
    dup=messages(o,obs(7),'duplicate_message',8,0)
    assert len(dup)==2 and dup[0]==dup[1]
    late=messages(o,obs(7),'out_of_order',8,0)
    assert late[1].timestamp<late[0].timestamp
    cand=messages(o,obs(7),'wrong_candidate',8,0)[0]
    assert np.argmax(cand.current)==1


def test_nonfinite_rejected_timestamps_leave_serializable_memory():
    m=TaskMemory('test','b0',mode='M2')
    assert not m.observe(obs(0,timestamp=float('nan')))['valid']
    json.dumps(m.snapshot(),allow_nan=False)


def test_single_false_candidate_does_not_propagate_into_confirmed_history():
    m,_=replay([0]*3+[1]+[0]*5)
    assert m.phase==0 and not m.confirmed
    assert any(e.status=='rejected' and e.event_type=='transition' for e in m.events)


def test_saved_contract_required_fields_and_scopes_match_runtime_snapshot():
    from pathlib import Path
    schema=json.loads(Path('configs/tomato_task_memory_20261110/memory.schema.json').read_text())
    m,_=replay([0]*3+[1]*3)
    snap=m.snapshot()
    assert set(schema['required'])<=snap.keys()
    spec=schema['$defs']['EventRecord']
    for e in snap['long_term']:
        assert set(spec['required'])<=e.keys()
        assert e['status'] in spec['properties']['status']['enum']
        assert e['scope'] in spec['properties']['scope']['enum']
        assert e['physical_success']=='unobserved'
    assert snap['suggests_prompt_update'] is False


def test_healthy_input_clears_transport_alarm_but_keeps_rejection_audit():
    m=TaskMemory('test','b0',mode='M2');m.observe(obs(0));m.observe(obs(0))
    assert 'duplicate_or_old_sequence' in m.view()['uncertainty']
    m.observe(obs(1))
    assert 'duplicate_or_old_sequence' not in m.view()['uncertainty']
    assert any(e.event_type=='observation_rejected' for e in m.events)
