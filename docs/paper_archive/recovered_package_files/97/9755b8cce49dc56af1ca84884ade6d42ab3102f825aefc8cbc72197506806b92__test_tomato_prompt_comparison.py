import copy
import json
from pathlib import Path
import pytest
from tools.tomato_prompt_replay import preflight,write,jsonl,sha
from tools.compare_tomato_prompt_policies import selection_key,select_shared
from tools.simulate_tomato_prompt_switch import run_case


@pytest.fixture
def frozen(tmp_path):
    episodes=[];split=dict(train=[],val=[],test=[]);predictions=[]
    labels=['pickup','harvest','return','adjust','pickup','harvest','return']
    for i in range(93):
        s='train' if i<75 else 'val' if i<84 else 'test';source=f's{i}';eid=f'e{i}';split[s].append(source)
        seq=labels if i else labels[:3];length=len(seq)*30
        ann=dict(segments=[dict(label=x,start_frame=j*30,end_frame=(j+1)*30,reviewed=True) for j,x in enumerate(seq)])
        path=tmp_path/f'annotation{i}.json';write(path,ann)
        anchors=list(range(0,length,15))
        episodes.append(dict(episode_id=eid,source_recording_id=source,split=s,length=length,fps=30.,anchors=anchors,
            annotation=dict(path=str(path),sha256=sha(path),**ann)))
        for k in anchors:
            p=dict(zip(['pickup','harvest','return','adjust'],[.97,.01,.01,.01]))
            predictions.append(dict(episode_id=eid,source_recording_id=source,split=s,frame_index=k,timestamp_s=k/30,
                last_input_frame=k,input_indices=[k]*16,future_probabilities=p,current_probabilities=p,future_valid=False))
    for model in ['future_only','joint','current_only']:
        d=tmp_path/model;d.mkdir();write(d/'split.json',split)
        write(d/'manifest.json',dict(contract={},splits=split,split_sha256=sha(d/'split.json'),episodes=episodes))
        write(d/'training_config.json',dict(labels=['pickup','harvest','return','adjust'],future_horizon_s=1.,prediction_hz=2))
        jsonl(d/'predictions.jsonl',predictions);(d/'best_head.pt').write_bytes(b'synthetic-no-model-loaded')
    return tmp_path


def test_paired_preflight_all_sources_and_original_rows_do_not_enter_policy(frozen):
    m,s,r=preflight(frozen)
    assert r['success_sources']==93 and r['sequence_counts']=={'3':1,'7':92}
    assert not hasattr(s['joint']['e92'][0],'future_valid')


@pytest.mark.parametrize('change',['duplicate','missing','reversed','future_input','source_mix','nan','split'])
def test_bad_input_fails_without_padding(frozen,change):
    path=frozen/'future_only/predictions.jsonl';rr=[json.loads(l) for l in path.read_text().splitlines()]
    if change=='duplicate':rr.insert(1,rr[0])
    elif change=='missing':rr.pop(0)
    elif change=='reversed':rr[0],rr[1]=rr[1],rr[0]
    elif change=='future_input':rr[0]['last_input_frame']=1
    elif change=='source_mix':rr[0]['source_recording_id']='s1'
    elif change=='nan':rr[0]['future_probabilities']['pickup']=float('nan')
    elif change=='split':rr[0]['split']='test'
    # Allow NaN in the intentionally corrupted fixture only.
    path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    with pytest.raises((ValueError,KeyError)):preflight(frozen)


def summary(wrong=.03,recall=.9,premature=.05):
    return dict(duration_s=100,wrong_prompt_duration_s=wrong*100,eligible_transitions=100,
        premature_ge_05=premature*100,transition_recall=recall,
        commit_delta_s=dict(p50=0),mean_absolute_matched_delta_s=.2)


def test_selection_requires_both_models_and_never_test_results():
    protocol=dict(max_wrong_time_fraction=.05,max_premature_ge_05_fraction=.1,min_transition_recall=.8)
    a=dict(future_only=dict(summary=summary()),joint=dict(summary=summary()))
    b=copy.deepcopy(a);b['joint']['summary']=summary(wrong=.09,recall=1)
    selected,ranks=select_shared({'good':a,'bad':b},protocol)
    assert selected=='good' and ranks[0]['feasible']
    b['joint']['summary']=summary(wrong=.01,recall=.7)
    assert not selection_key(b['future_only']['summary'],b['joint']['summary'],protocol,'b')[1]['feasible']


def test_all_failed_constraints_are_explicit_fallback():
    protocol=dict(max_wrong_time_fraction=.05,max_premature_ge_05_fraction=.1,min_transition_recall=.8)
    key,row=selection_key(summary(wrong=.2),summary(wrong=.2),protocol,'bad')
    assert not row['feasible'] and key[0]==1 and row['normalized_constraint_excess']>0


def test_full_replay_artifacts_and_refuse_overwrite(frozen,tmp_path):
    m,s,i=preflight(frozen);out=tmp_path/'out'
    r=run_case(m,s,i,'future_only','val',dict(id='P0',name='test'),out)
    assert r['summary']['episodes']==9
    assert (out/'replay_events.jsonl').exists() and (out/'active_prompt_intervals.jsonl').exists()
    with pytest.raises(FileExistsError):run_case(m,s,i,'future_only','val',dict(id='P0'),out)


def test_orchestrator_locks_before_test_and_finishes_synthetic_cases(frozen,tmp_path,monkeypatch):
    import sys
    import subprocess
    from tools import compare_tomato_prompt_policies as compare
    from tools import visualize_tomato_prompt_timeline as visual
    from tools import report_tomato_prompt_switch as report
    config=json.loads((compare.ROOT/'configs/tomato_prompt_switch_offline.yaml').read_text())
    config['policies']=config['policies'][:1]
    config['sensitivity']=dict(latency_s=[0],plan_period_s=[.5],phase_fraction=[0],references=['P0_raw'])
    path=tmp_path/'config.json';write(path,config);out=tmp_path/'end_to_end'
    argv=['run','--run-root',str(frozen),'--policy-config',str(path),'--output-dir',str(out)]
    child='''
import sys
from types import SimpleNamespace
from tools import compare_tomato_prompt_policies as compare
from tools import visualize_tomato_prompt_timeline as visual
from tools import report_tomato_prompt_switch as report
original=compare.subprocess.run
def gate(cmd,**kwargs):
    if len(cmd)>3 and cmd[1:3]==['-m','pytest']:return SimpleNamespace(returncode=0)
    return original(cmd,**kwargs)
compare.subprocess.run=gate
visual.visualize=lambda *a,**k:None
report.report=lambda *a,**k:None
sys.argv=ARGV
compare.main()
'''.replace('ARGV',repr(argv))
    subprocess.run([sys.executable,'-c',child],check=True)
    assert (out/'completion.json').exists()
    lock=sha(out/'selected_policy.lock.json')
    assert json.loads((out/'test_main/joint/evaluation_control.json').read_text())['selected_policy_lock_sha256']==lock
    assert json.loads((out/'completion.json').read_text())['sensitivity_cases']==4
