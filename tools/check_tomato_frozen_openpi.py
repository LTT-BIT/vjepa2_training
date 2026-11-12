"""Pinned OpenPI Git snapshot + dev8 + TaskMemory shadow, hardware-free."""
from pathlib import Path
import argparse,hashlib,importlib,json,subprocess,sys
import numpy as np
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_offline_contract import prediction_packet,check_request_response,OfflineChunkGate
from .tomato_state_recovery import B0State
from .tomato_task_memory import TaskMemory,Observation
from .tomato_full_chain import reconstruct_shadow
from .evaluate_tomato_complete100_history import SOURCE


def load_snapshot(lock,directory):
    root=directory/'src';package=root/'openpi';binding=lock['openpi']
    for rel,sha in binding['files'].items():
        archive=Path(binding.get('archived_snapshot_dir',''))/rel
        raw=archive.read_bytes() if archive.is_file() else subprocess.check_output(['git','-C',binding['repository'],'show',binding['snapshot_commit']+':src/openpi/'+rel])
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Pinned bridge Git blob changed')
        p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    for p in [package,package/'subtasks',package/'policies']:(p/'__init__.py').touch()
    if any(k=='openpi' or k.startswith('openpi.') for k in sys.modules):raise RuntimeError('Run isolated CLI; refusing another loaded OpenPI version')
    sys.path.insert(0,str(root.resolve()));return importlib.import_module('openpi.subtasks.agent_bridge')


def run(lock_path,out):
    if out.exists():raise FileExistsError('Preserve old integration artifacts')
    lock=verify(lock_path);out.mkdir(parents=True);bridge=load_snapshot(lock,out/'openpi_snapshot');m=ds.read_json(lock['manifest']);results=[];inputs={};states={}
    for seed,model in lock['models'].items():
        folder=out/f'seed{seed}';folder.mkdir()
        for e in m['episodes']:
            if e['split']!='val':continue
            eid=e['episode_id'];path=SOURCE/f'seed{seed}'/(eid+'.npz');inputs[str(path)]=ds.digest(path);z=np.load(path);normal=bridge.PredictionPromptProvider();control=bridge.PredictionPromptProvider()
            normal.reset(eid);control.reset(eid);b0=B0State();logs=[];memory=TaskMemory(eid,model['model_version'],mode='M1',sink=logs.append)
            if eid not in states:states[eid]=ds.read_state(e,lock['contract']['joint_names'],lock['contract']['joint_mode'])
            state_times,q=states[eid];names=lock['contract']['joint_names'];policy_names=[n+'.position' for n in bridge.ORDER]
            assert set(policy_names)==set(names)
            policy_indices=[names.index(n) for n in policy_names]
            gate=OfflineChunkGate(bridge);old=None;changed=0;invalid_old=0;trace=[];faults=0;named_checks=0
            for i,t in enumerate(z['times']):
                oid=f'{eid}:{i}';absolute=1e6+float(t);c=z['current_probabilities'][i];f=z['future_probabilities'][i]
                packet=prediction_packet(bridge,c,f,eid,oid,absolute,model['model_version']);check_request_response(packet,observation_id=oid,input_timestamp=absolute,model_version=model['model_version'])
                context=normal.accept_prediction(packet,observation_id=oid,now=absolute+.05)
                counter=control.accept_prediction(prediction_packet(bridge,c,np.full(4,.25),eid,oid,absolute,model['model_version']),observation_id=oid,now=absolute+.05)
                ref=b0.update(c,f,float(t));assert (context.phase_index,context.version,context.prompt)==(counter.phase_index,counter.version,counter.prompt)
                assert context.phase_index==ref['phase'] and context.changed==ref['changed']
                anchor=int(z['anchors'][i]);assert abs(state_times[anchor]-t)<1e-6
                measured_delta=None
                if context.changed or i==0:
                    indices=ds.sample_causal_indices(state_times,anchor,sampler_version=lock['contract']['sampler_version'])
                    expected=q[indices];policy_history=expected[:,policy_indices]
                    restored=bridge.named_agent_history(policy_history.tolist(),bridge.ORDER,names)
                    assert np.array_equal(restored,expected);named_checks+=1
                    if i>0:measured_delta=float(np.max(abs(q[anchor]-q[int(z['anchors'][i-1])])))
                view=memory.observe(Observation(eid,i,oid,float(t),float(t)+.05,tuple(c),tuple(f),model['model_version']),b0_phase=int(context.phase_index))
                assert view['confirmed_phase']==context.phase_index and not view['suggests_prompt_update']
                if context.changed:
                    changed+=1
                    if old is not None:
                        assert not bridge.chunk_matches_prompt(old,context);invalid_old+=1
                        denied=gate.result(old,np.zeros((50,58)),context,absolute+.05,safe_boundary=True)
                        assert not denied['accepted']
                binding=gate.request(context,oid)
                # Alternate mock actions intentionally expose continuity risk;
                # metadata correctness cannot guarantee VLA output smoothness.
                actions=np.full((50,58),.1*(i%2));before=actions.copy()
                admitted=gate.result(binding,actions,context,absolute+.05,safe_boundary=True);assert admitted['accepted'] and np.array_equal(actions,before)
                if i==0:
                    stale=prediction_packet(bridge,c,f,eid,oid+'-stale',absolute-3,model['model_version'])
                    before_state=(normal.phase,normal.count,normal.version)
                    try:normal.accept_prediction(stale,observation_id=oid+'-stale',now=absolute+.05)
                    except ValueError:faults+=1
                    else:raise AssertionError('Stale prediction accepted')
                    assert before_state==(normal.phase,normal.count,normal.version)
                    wrong=dict(packet,model_version='wrong-locked-model')
                    try:check_request_response(wrong,observation_id=oid,input_timestamp=absolute,model_version=model['model_version'])
                    except ValueError:faults+=1
                    else:raise AssertionError('Wrong model accepted')
                trace.append(dict(observation_timestamp=float(t),mock_request_timestamp=absolute,result_timestamp=packet['result_timestamp'],arrival_timestamp=absolute+.05,
                                  context=context.__dict__,chunk_binding=binding,mock_action_admission=admitted,
                                  strict_future_warning=ref['warning'],bridge_display_advisory=context.future_warning,
                                  memory_version=view['memory_version'],counterfactual_future_prompt_identical=True,
                                  measured_position_delta_at_prompt_change=measured_delta,
                                  measured_delta_scope='Recorded observed joint difference, not VLA action discontinuity'))
                old=binding
            assert reconstruct_shadow(logs)==memory.view()['history_entered_positions']
            with (folder/(eid+'_trace.jsonl')).open('x') as f:
                for row in trace:f.write(json.dumps(row,allow_nan=False)+'\n')
            with (folder/(eid+'_memory_journal.jsonl')).open('x') as f:
                for row in logs:f.write(json.dumps(row,allow_nan=False)+'\n')
            ds.write_json(folder/(eid+'_memory.json'),memory.snapshot())
            results.append(dict(seed=seed,episode_id=eid,observations=len(trace),prompt_changes=changed,obsolete_chunks_rejected=invalid_old,
                                injected_prediction_faults_rejected=faults,memory_history_faithful=True,
                                named_state_reorder_checks=named_checks,
                                max_mock_action_jump=max([r['mock_action_admission']['first_action_max_delta'] or 0 for r in trace]),
                                VLA_run=False,robot_connected=False,RTC_modified=False))
    live=lock['openpi'];ds.write_json(out/'summary.json',dict(episodes=results,pinned_openpi=live,
        counterfactual_future_changes_prompt=0,strict_future_and_bridge_display_are_different_fields=True,
        scope='Pinned historical interface and synthetic chunks replay, not current deployment or real VLA action continuity/success',
        mock_timestamp_origin=1e6,synthetic_result_latency_s=.04,synthetic_arrival_latency_s=.05))
    verify(lock_path)
    ds.write_json(out/'receipt.json',dict(lock_sha256=ds.digest(lock_path),input_sha256=inputs,source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_offline_contract.py')]},
                 current_openpi_sha=subprocess.check_output(['git','-C',live['repository'],'rev-parse','HEAD'],text=True).strip(),
                 current_openpi_status=subprocess.check_output(['git','-C',live['repository'],'status','--porcelain'],text=True),robot_connected=False))
    print('INTERFACE',len(results),'episode-seeds',sum(r['observations'] for r in results),'observations; Future counterfactual Prompt changes0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,default=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.lock,a.out)
