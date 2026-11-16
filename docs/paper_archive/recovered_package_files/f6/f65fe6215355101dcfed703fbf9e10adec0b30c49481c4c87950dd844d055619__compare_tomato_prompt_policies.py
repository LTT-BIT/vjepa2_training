"""One-shot preregistered validation selection, fixed test comparison and P6 matrix."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from .tomato_prompt_replay import read,write,sha,jsonl,preflight
from .evaluate_tomato_prompt_switch import bootstrap_pair,percentile
from .simulate_tomato_prompt_switch import ROOT,run_case,csv_write


def selection_key(b,c,protocol,name):
    wrong=sum(v['wrong_prompt_duration_s']/v['duration_s'] for v in [b,c])/2
    premature=sum(v['premature_ge_05']/v['eligible_transitions'] for v in [b,c])/2
    recall=sum(v['transition_recall'] for v in [b,c])/2
    # Feasibility is conservative across both models, ranking gives them equal weight.
    excess=sum(max(0,v['wrong_prompt_duration_s']/v['duration_s']/protocol['max_wrong_time_fraction']-1)+
        max(0,v['premature_ge_05']/v['eligible_transitions']/protocol['max_premature_ge_05_fraction']-1)+
        max(0,(protocol['min_transition_recall']-v['transition_recall'])/protocol['min_transition_recall']) for v in [b,c])
    feasible=excess<1e-12
    delay=sum(abs(v['commit_delta_s']['p50']) if v['commit_delta_s']['p50'] is not None else 1e9 for v in [b,c])/2
    # p50 diagnostic only; ranking uses mean absolute matched delta below (explicit summary field).
    absolute=sum(v['mean_absolute_matched_delta_s'] if v['mean_absolute_matched_delta_s'] is not None else 1e9 for v in [b,c])/2
    key=(0,-recall,wrong,absolute,name) if feasible else (1,excess,wrong,premature,-recall,name)
    return key,dict(feasible=feasible,normalized_constraint_excess=excess,mean_wrong_fraction=wrong,
                    mean_premature_fraction=premature,mean_recall=recall,mean_absolute_matched_delta_s=absolute)


def select_shared(results,protocol):
    ranked=[]
    for name,models in results.items():
        key,summary=selection_key(models['future_only']['summary'],models['joint']['summary'],protocol,name)
        ranked.append((key,name,summary))
    ranked.sort()
    return ranked[0][1],[dict(name=n,ranking=list(k),**v) for k,n,v in ranked]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=ROOT/'runs/tomato/local93_20261107')
    p.add_argument('--policy-config',type=Path,default=ROOT/'configs/tomato_prompt_switch_offline.yaml')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=20261108)
    a=p.parse_args();out=a.output_dir
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    config=read(a.policy_config)
    if config['seed']!=a.seed:raise ValueError('Seed differs from preregistered config')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_hash={str(q.relative_to(ROOT)) if q.is_relative_to(ROOT) else str(q):sha(q) for q in list((ROOT/'tools').glob('*tomato_prompt*.py'))+list((ROOT/'tests').glob('test_tomato_prompt*.py'))+[a.policy_config]}
    write(out/'selection_protocol.json',dict(**config,git_head=head,source_sha256=source_hash,
        policy_config_sha256=sha(a.policy_config),selection_scope='val only, shared B/C, no test controller scores yet',
        risk_constraints='Both models must individually satisfy max wrong time 5%, premature>=0.5s<=10% boundaries, recall>=80%',
        match_main_window_s=[-1,2],strict_window_s=[-.2,.5],
        phases=['preflight','P1_P3_gate','val_grid','selected_lock','test_main','auxiliary','P5_auto','P6_fixed_matrix','final_report'],
        deferred=['human_video_review','physical_safety','FINISH','overloaded_queue','new_data_collection']))
    # Protect every completed local93 output, not merely the files consumed here.
    oldfiles=[q for model in ['current_only','future_only','joint'] for q in (a.run_root/model).rglob('*') if q.is_file()]
    protected={str(q):sha(q) for q in oldfiles}
    write(out/'protected_original_sha256.json',protected)
    try:m,s,integrity=preflight(a.run_root)
    except Exception as exc:
        write(out/'replay_integrity.json',dict(status='FAIL',error_type=type(exc).__name__,failure_logs=[str(exc)]));raise
    write(out/'replay_integrity.json',dict(status='PASS',**integrity))
    with (out/'unit_gate.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','-q',*[str(q) for q in sorted((ROOT/'tests').glob('test_tomato_prompt*.py'))]],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(out/'stage_status.json',dict(P0='PASS',P1='PASS_UNIT_TEST_GATE',P2='PASS_UNIT_TEST_GATE',P3='PASS_UNIT_TEST_GATE',P4='RUNNING'))
    latency=config['main_latency_s'];period=config['main_plan_period_s'];offset=config['main_plan_offset_s']
    candidates={v['name']:{**v,'initial_prompt':config['initial_prompt'],'prediction_ttl_s':config['prediction_ttl_s'],'seed':a.seed} for v in config['policies']}
    val={}
    for name,policy in candidates.items():
        val[name]={}
        for model in ['future_only','joint']:
            r=run_case(m,s,integrity,model,'val',policy,out/'val_grid'/name/model,latency,period,offset)
            val[name][model]=r
        print('val',name,[(k,round(v['summary']['active_prompt_time_accuracy'],4),v['summary']['matched_transitions']) for k,v in val[name].items()],flush=True)
    selected,ranks=select_shared(val,config['selection'])
    write(out/'val_selection_rankings.json',ranks)
    csv_write(out/'val_candidates.csv',[{**{k:v for k,v in r['summary'].items() if not isinstance(v,dict)},'candidate':n,'model':model} for n,mm in val.items() for model,r in mm.items()])
    policy=candidates[selected]
    lock=dict(policy=policy,name=selected,selection_protocol_sha256=sha(out/'selection_protocol.json'),
        policy_config_sha256=sha(a.policy_config),input_sha256=integrity['input_sha256'],
        val_rankings_sha256=sha(out/'val_selection_rankings.json'),git_head=head,source_sha256=source_hash,
        selected_using='validation only, equal weight B/C, conservative feasibility then declared ranking',
        constraints_satisfied=ranks[0]['feasible'],main_latency_s=latency,main_plan_period_s=period,main_plan_offset_s=offset,
        auxiliary_rule=config['auxiliary'],sensitivity=config['sensitivity'])
    write(out/'selected_policy.lock.json',lock);lock_sha=sha(out/'selected_policy.lock.json')
    print('LOCKED',selected,'feasible',lock['constraints_satisfied'],lock_sha,flush=True)
    test={model:run_case(m,s,integrity,model,'test',policy,out/'test_main'/model,latency,period,offset,lock_sha) for model in ['future_only','joint']}
    paired=[]
    for b,c in zip(test['future_only']['per_episode'],test['joint']['per_episode']):
        paired.append(dict(source_recording_id=b['source_recording_id'],episode_id=b['episode_id'],
            B_time_accuracy=b['active_prompt_time_accuracy'],C_time_accuracy=c['active_prompt_time_accuracy'],
            C_minus_B_time_accuracy=c['active_prompt_time_accuracy']-b['active_prompt_time_accuracy'],
            B_matches=b['matched_transitions'],C_matches=c['matched_transitions'],eligible=b['eligible_transitions'],
            B_wrong_commits=b['wrong_to_stage_commits'],C_wrong_commits=c['wrong_to_stage_commits'],
            B_exact=b['full_episode_order_exact'],C_exact=c['full_episode_order_exact']))
    csv_write(out/'test_paired_sources.csv',paired)
    ci=bootstrap_pair(test['future_only']['per_episode'],test['joint']['per_episode'],a.seed)
    ci['source_wins_losses_ties_time_accuracy']={k:sum((r['C_minus_B_time_accuracy']>1e-9 if k=='wins' else r['C_minus_B_time_accuracy']< -1e-9 if k=='losses' else abs(r['C_minus_B_time_accuracy'])<=1e-9) for r in paired) for k in ['wins','losses','ties']}
    write(out/'test_cluster_bootstrap.json',ci)
    # Repeat actual selected val streams: canonical event logs must match byte for byte.
    repro=run_case(m,s,integrity,'joint','val',policy,out/'determinism_repeat_joint_val',latency,period,offset)
    comparison={n:sha(out/'val_grid'/selected/'joint'/n)==sha(out/'determinism_repeat_joint_val'/n) for n in ['replay_events.jsonl','active_prompt_intervals.jsonl','evaluation_control.json']}
    if not all(comparison.values()):raise ValueError('Nondeterministic replay')
    write(out/'determinism.json',comparison)
    auxiliary={}
    for split in ['val','test']:
        for kind,model,pol in [('joint_current_gate','joint',{**policy,'current_gate':True}),('current_only_reactive','current_only',policy)]:
            r=run_case(m,s,integrity,model,split,pol,out/'auxiliary'/split/kind,latency,period,offset,lock_sha)
            auxiliary[split+'_'+kind]=r['summary']
    write(out/'auxiliary_comparison.json',auxiliary)
    special_ids={eid for eid,v in integrity['sequences'].items() if len(v['sequence'])==3}
    special={}
    for eid in special_ids:
        special_split=integrity['sequences'][eid]['split']
        for model in ['future_only','joint']:
            r=run_case(m,s,integrity,model,special_split,policy,out/'special_three_stage'/model,latency,period,offset,lock_sha,only_episode_ids={eid})
            special[model]=dict(summary=r['summary'],per_episode=r['per_episode'])
    write(out/'special_three_stage_summary.json',special)
    from .visualize_tomato_prompt_timeline import visualize
    visualize(out,m,s,policy,val[selected],test,config)
    sensitivity=[];refs=list(dict.fromkeys([selected]+config['sensitivity']['references']))
    for split in ['val','test']:
        for lat in config['sensitivity']['latency_s']:
            for per in config['sensitivity']['plan_period_s']:
                for phase in config['sensitivity']['phase_fraction']:
                    for name in refs:
                        for model in ['future_only','joint']:
                            key=f'{split}_{name}_{model}_L{lat:g}_P{per:g}_O{phase*per:g}'
                            r=run_case(m,s,integrity,model,split,candidates[name],out/'sensitivity'/key,lat,per,phase*per,lock_sha,compressed=True)
                            summary=r['summary'];sensitivity.append(dict(scenario=key,candidate=name,**summary))
        print('sensitivity complete',split,len(sensitivity),flush=True)
    write(out/'sensitivity_matrix.json',sensitivity)
    csv_write(out/'sensitivity_matrix.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in sensitivity])
    intact=all(sha(q)==h for q,h in protected.items())
    write(out/'original_artifact_integrity.json',dict(protected_files=len(protected),all_unchanged=intact))
    assert intact and sha(out/'selected_policy.lock.json')==lock_sha
    write(out/'stage_status.json',dict(P0='PASS',P1='PASS',P2='PASS',P3='PASS',P4_val='PASS',P4_test='PASS',P5_auto='PASS',
        P5_manual_review='DEFERRED / NOT_PERFORMED',P6='PASS',real_robot='NOT_RUN',FINISH='NOT_SUPPORTED'))
    write(out/'completion.json',dict(duration_s=time.monotonic()-began,git_head=head,selected_policy=selected,
        selection_lock_sha256=lock_sha,sensitivity_cases=len(sensitivity),P5_manual_review='NOT_PERFORMED / DEFERRED',
        torch_imported='torch' in sys.modules,transformers_imported='transformers' in sys.modules,python=sys.version))
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    from .report_tomato_prompt_switch import report
    report(out)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
