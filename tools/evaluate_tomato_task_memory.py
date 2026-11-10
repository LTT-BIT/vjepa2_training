"""No-training full-episode shadow/fault replay; labels are scorer-only inputs."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import subprocess
import time
import sys
from collections import deque
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_boundary_metrics as bm
from .evaluate_tomato_reviewed_boundaries import revise_episode
from .evaluate_tomato_state_recovery import state_score, annotate_state, prompt_score, event_score, summarize
from .tomato_state_recovery import StateRecovery, CLASSES
from .tomato_task_memory import TaskMemory, Observation

CFG=Path('configs/tomato_task_memory_20261110/protocol.json')
SOURCE=Path('runs/tomato/transition_time_20261109')
SCENARIOS=('normal','single_class_error','consecutive_class_errors','missing_observations','stale_results','duplicate_message','out_of_order','wrong_candidate')


def corrupt(row,label):
    p=np.full(4,.01/3);p[label]=.99
    return replace(row,current=tuple(p))


def messages(o,previous,scenario,start,phase):
    """Injection schedule depends only on index and causal outputs, never labels."""
    i=o.sequence
    if scenario=='missing_observations' and start<=i<start+4:return []
    if scenario=='stale_results' and start<=i<start+4:return [replace(o,timestamp=o.timestamp-2.)]
    if scenario=='single_class_error' and i==start:return [corrupt(o,(int(np.argmax(o.current))+1)%4)]
    if scenario=='consecutive_class_errors' and start<=i<start+6:return [corrupt(o,(int(np.argmax(o.current))+1)%4)]
    if scenario=='wrong_candidate' and i==start:return [corrupt(o,CLASSES[phase+1] if phase<6 else 0)]
    if scenario=='duplicate_message' and i==start:return [o,o]
    if scenario=='out_of_order' and i==start and previous is not None:return [o,replace(previous,arrival_timestamp=o.arrival_timestamp)]
    return [o]


def standard_trace(times,phases):
    rows=[];previous=version=0
    for t,phase in zip(times,phases):
        changed=phase!=previous;version+=int(changed)
        rows.append(dict(time=float(t),phase=int(phase),previous_phase=int(previous),changed=changed,prompt_version=version,resynchronized=False));previous=phase
    return rows


def recoveries(phase,clean,truth,times,start,end):
    def first(target):
        for i in range(end+1,len(times)-1):
            if all(phase[j]==target[j] for j in (i,i+1)):return float(times[i]-times[end])
        return None
    deviated=any(phase[i]!=clean[i] for i in range(start,len(times)))
    return dict(state_deviation_from_clean=deviated,return_to_clean_policy_s=first(clean) if deviated else None,return_to_truth_s=first(truth),definition='First two consecutive matching scheduled samples after last injected index; GT scorer only, no online reset')


def object_bytes(obj,seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return 0
    seen.add(id(obj));size=sys.getsizeof(obj)
    if isinstance(obj,dict):size+=sum(object_bytes(k,seen)+object_bytes(v,seen) for k,v in obj.items() if k!='sink')
    elif isinstance(obj,(list,tuple,deque,set)):size+=sum(object_bytes(x,seen) for x in obj)
    elif hasattr(obj,'__dict__'):size+=object_bytes(vars(obj),seen)
    return size


def run_episode(e,z,state_times,q,names,scenario,folder=None,model_id="b0",short_size=16,long_size=256):
    times=z['times'];start=max(8,len(times)//3);end=min(len(times)-1,start+(5 if scenario=='consecutive_class_errors' else 3 if scenario in ('stale_results','missing_observations') else 0))
    baseline=StateRecovery('baseline',session_id=e['episode_id'],model_id=model_id)
    control=StateRecovery('baseline',session_id=e['episode_id'],model_id=model_id)
    files={};mem={};logs={};latencies={k:[] for k in ('M0','M1','M2')}
    for mode in ('M1','M2'):
        if folder is not None:
            folder.mkdir(parents=True,exist_ok=True);files[mode]=(folder/(mode+'_events.jsonl')).open('w')
            sink=lambda event,handle=files[mode]:handle.write(json.dumps(event,allow_nan=False)+'\n')
        else:sink=None
        mem[mode]=TaskMemory(e['episode_id'],model_id,mode=mode,joint_names=names,sink=sink,short_size=short_size,long_size=long_size)
    phases={k:[] for k in ('M0','M1','M2')};views={k:[] for k in ('M1','M2')};events=[];ingress=[];previous=None;max_bytes={k:0 for k in mem};resident={k:0 for k in mem}
    for i,t in enumerate(times):
        j=int(np.searchsorted(state_times,t,side='right')-1);assert 0<=j<len(q) and state_times[j]<=t
        o=Observation(e['episode_id'],i,f"{e['episode_id']}:{i}",float(t),float(t)+.01,tuple(z['current_probabilities'][i]),tuple(z['future_probabilities'][i]),model_id,positions=tuple(q[j]),state_timestamp=min(float(t),float(state_times[j])),joint_names=tuple(names))
        batch=messages(o,previous,scenario,start,baseline.phase)
        ingress.append(dict(schedule_time=float(t),messages=[dict(sequence=x.sequence,observation_id=x.observation_id,timestamp=x.timestamp,arrival_timestamp=x.arrival_timestamp,current=list(x.current),future=list(x.future),state_timestamp=x.state_timestamp) for x in batch]))
        if not batch:
            for m in mem.values():m.poll(o.arrival_timestamp)
        for item in batch:
            begin=time.perf_counter_ns();b=baseline.update(item.current,item.future,item.timestamp,now=item.arrival_timestamp,session_id=item.session_id,model_id=item.model_version);latencies['M0'].append((time.perf_counter_ns()-begin)/1e6)
            expected=control.update(item.current,item.future,item.timestamp,now=item.arrival_timestamp,session_id=item.session_id,model_id=item.model_version)
            frozen_before=json.dumps(b,sort_keys=True)
            for mode,m in mem.items():
                begin=time.perf_counter_ns();m.observe(item,b0_phase=int(b['phase']));latencies[mode].append((time.perf_counter_ns()-begin)/1e6)
            assert json.dumps(b,sort_keys=True)==frozen_before and b==expected,'Shadow interface isolation failure'
            if b.get('warning'):events.append(b['warning'])
        for mode in phases:phases[mode].append(baseline.phase if mode=='M0' else mem[mode].phase)
        for mode,m in mem.items():
            v=m.view();views[mode].append(v)
            assert not v['suggests_prompt_update'] and not v['progress']['task_complete']
            assert m.ledger_phase()==m.phase
            if i%16==0 or i==len(times)-1:
                max_bytes[mode]=max(max_bytes[mode],len(json.dumps(m.snapshot(),allow_nan=False).encode()))
                resident[mode]=max(resident[mode],object_bytes(m))
        previous=o
    for f in files.values():f.close()
    for mode,m in mem.items():
        ids=[r['to_phase'] for r in m.confirmed];assert ids==list(range(1,m.phase+1))
        if folder is not None:ds.write_json(folder/(mode+'_final.json'),m.snapshot())
    return dict(input_trace=ingress,phases=phases,views=views,warnings=events,latencies=latencies,max_snapshot_bytes=max_bytes,max_resident_object_bytes=resident,fault_start_index=start,fault_end_index=end,rejected_inputs={mode:m.rejected_inputs for mode,m in mem.items()})


def score_episode(e,z,replay,mode):
    ts=z['times'];truth=bm.boundaries(dict(episode=e));gt=np.searchsorted([b['time'] for b in truth],ts,side='right');ph=replay['phases'][mode]
    valid=z['anchors']+45<e['length'];duration=float(ts[valid][-1]-ts[valid][0]+.5)
    warnings=[r for r in replay['warnings'] if r['time']<=ts[valid][-1]]
    trace=standard_trace(ts,ph)
    confusion=np.zeros((7,7),int)
    for a,b in zip(gt,ph):confusion[a,b]+=1
    per=dict(warnings=event_score(warnings,truth,duration),prompt=prompt_score(trace,truth,ts),state=annotate_state(state_score(ph,gt,ts),truth,ph,ts),position_confusion=confusion.tolist(),second_half_position_accuracy=float(np.mean(np.asarray(ph)[len(ph)//2:]==gt[len(ph)//2:])),phases=ph)
    if mode!='M0':per['memory_conflict_samples']=sum('history_current_conflict' in v['uncertainty'] for v in replay['views'][mode])
    return per,gt


def run(out):
    if out.exists():raise FileExistsError('Choose a fresh directory; never overwrite frozen runs')
    out.mkdir(parents=True);cfg=ds.read_json(CFG)
    lock=ds.read_json('runs/tomato/state_recovery_20261110_release/receipt.json')
    # Frozen inputs, not former implementation hashes; new module is intentionally separate.
    for path,sha in lock['inputs'].items():assert ds.digest(path)==sha,(path,'changed since baseline')
    manifest_path='runs/tomato/campaign94_20261107/prepared/manifest.json';manifest=ds.read_json(manifest_path);by={e['episode_id']:e for e in manifest['episodes']};names=manifest['contract']['joint_names']
    proposals=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals'];recs=[r for r in ds.read_json(SOURCE/'data_identity.json')['episodes'] if r['split']=='val'];assert len(recs)==9
    states={};frozen={str(CFG):ds.digest(CFG)};models={}
    for rec in recs:
        e=by[rec['episode_id']];states[e['episode_id']]=ds.read_state(e,names,'arms-and-hands');frozen[e['parquet']]=ds.digest(e['parquet'])
    summaries={};all_results={};timing={k:[] for k in ('M0','M1','M2')};max_storage={k:0 for k in ('M1','M2')};max_resident={k:0 for k in ('M1','M2')};classifications={};event_comparisons=[]
    for seed in cfg['seeds']:
        scenarios={};normal_replays={};normal_scores={};cls={h:[[],[]] for h in ('current','future')}
        prior=ds.read_json(SOURCE/f'seed{seed}/result.json');baseline_op=prior['external_validation_thresholds_are_ablation_not_selection'][str(prior['selected_threshold'])]
        classifier=prior['identity']['original_classifiers'];assert ds.digest(classifier['path'])==classifier['sha256'];models[str(seed)]=classifier;frozen[classifier['path']]=classifier['sha256']
        for scenario in SCENARIOS:
            if seed!=42 and scenario!='normal':continue
            per={m:{} for m in ('M0','M1','M2')};case_results={}
            for rec in recs:
                eid=rec['episode_id'];e=revise_episode(by[eid],proposals.get(eid,[]));path=SOURCE/f'seed{seed}'/(eid+'.npz');z=np.load(path);frozen[str(path)]=ds.digest(path)
                folder=out/f'seed{seed}'/scenario/eid
                replay=run_episode(e,z,*states[eid],names,scenario,folder,model_id=classifier['sha256'],short_size=cfg['short_term_samples'],long_size=cfg['long_term_recent_events'])
                if scenario=='normal':
                    normal_replays[eid]=replay
                    assert replay['phases']['M0']==replay['phases']['M1'], 'Shadow changed B0 phase or failed to mirror valid normal decision'
                    old=ds.read_json('runs/tomato/state_recovery_20261110_release/results.json')[str(seed)]['per_episode']['baseline'][eid]['full_current_timeline']
                    previous_trace=ds.read_json('runs/tomato/state_recovery_20261110_release/results.json')[str(seed)]['per_episode']['baseline'][eid]['trace']
                    assert replay['phases']['M0'][:len(previous_trace)]==[r['phase'] for r in previous_trace]
                for mode in per:
                    scored,gt=score_episode(e,z,replay,mode);per[mode][eid]=scored
                    if scenario=='normal':normal_scores.setdefault(mode,{})[eid]=scored
                    else:scored['recovery']=recoveries(replay['phases'][mode],normal_scores[mode][eid]['phases'],gt,z['times'],replay['fault_start_index'],replay['fault_end_index'])
                    for k,v in enumerate(scored['prompt']['events']):event_comparisons.append(dict(seed=seed,scenario=scenario,mode=mode,episode_id=eid,**v))
                if scenario=='normal':
                    assert per['M0'][eid]['warnings']['events']==baseline_op['per_episode']['B0'][eid]['events']
                    assert per['M0'][eid]['prompt']['counts']==old['prompt']['counts']
                    assert per['M0'][eid]['state']['mismatch_s']==old['state']['mismatch_s']
                    for head,offset in [('current',0),('future',45)]:
                        keep=np.ones(len(z['times']),bool) if head=='current' else z['anchors']+offset<e['length']
                        ys=np.array([ds.label_at_frame(e['annotation']['segments'],int(a)+offset) for a in z['anchors'][keep]])
                        cls[head][0].append(np.log(np.maximum(z[head+'_probabilities'][keep],1e-30)));cls[head][1].append(ys)
                    for mode,values in replay['latencies'].items():timing[mode]+=values
                for mode,nb in replay['max_snapshot_bytes'].items():max_storage[mode]=max(max_storage[mode],nb)
                for mode,nb in replay['max_resident_object_bytes'].items():max_resident[mode]=max(max_resident[mode],nb)
                case_results[eid]=dict(**replay,fault_rule='Index schedule and causal model outputs only; never GT',source_recording_id=e['source_recording_id'],positions_state_source=e['parquet'])
            summaries_seed={mode:summarize(p) for mode,p in per.items()}
            if scenario=='normal':
                assert summaries_seed['M0']['warnings']==summaries_seed['M1']['warnings']
                assert summaries_seed['M0']['warnings']['hits']==baseline_op['summaries']['B0']['detected']
            for mode in per:
                s=summaries_seed[mode];s['second_half_position_accuracy_mean']=float(np.mean([p['second_half_position_accuracy'] for p in per[mode].values()]));s['position_confusion']=np.sum([p['position_confusion'] for p in per[mode].values()],axis=0).tolist()
                if scenario!='normal':s['recovery_by_episode']={eid:p['recovery'] for eid,p in per[mode].items()}
            scenarios[scenario]=dict(summary=summaries_seed,per_episode=per)
            ds.write_json(out/f'seed{seed}'/scenario/'replay.json',case_results)
        classifications[str(seed)]={head:metrics.scores(np.concatenate(v[0]),np.concatenate(v[1])) for head,v in cls.items()}
        all_results[str(seed)]=scenarios;summaries[str(seed)]={scenario:v['summary'] for scenario,v in scenarios.items()}
    ds.write_json(out/'results.json',all_results);ds.write_json(out/'summary.json',summaries);ds.write_json(out/'classification.json',classifications);ds.write_json(out/'event_comparisons.json',event_comparisons)
    ds.write_json(out/'resource.json',dict(cpu_update_ms={k:dict(p50=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=max(v),n=len(v)) for k,v in timing.items()},max_serialized_snapshot_bytes=max_storage,max_resident_object_bytes=max_resident,note='CPU state update only. No encoder/head timing included; audit JSONL and saved full traces live on disk, not resident memory. No CUDA allocations by TaskMemory.'))
    ds.write_json(out/'receipt.json',dict(start_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=frozen,prior_baseline_receipt_sha=ds.digest('runs/tomato/state_recovery_20261110_release/receipt.json'),models=models,source_hashes={p:ds.digest(p) for p in ['tools/tomato_task_memory.py',__file__]},test_arrays_read=False,encoder_calls=0,training=False,remote_push=False,baseline_parity=True,shadow_isolation=True,truth_used_only_in_scorers=True))
    print(json.dumps(summaries['42']['normal'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
