"""Real-probability Shadow replay, deterministic restore and abrupt-exit test.

No training, ROI inference, robot IO or Prompt writes. Truth is used only by the
unchanged evaluator after replay, never by the durable memory or controller.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_task_memory import TaskMemory,Observation
from .tomato_durable_memory import DurableTaskMemoryShadow
from .tomato_disagreement_policy import DisagreementPolicy
from .evaluate_tomato_disagreement import score,summarize

CONFIG=Path('configs/tomato_durable_memory_20261112/protocol.json')
BASE=Path('runs/tomato/disagreement_20261112_v2')


def prepare_episode(seed,e,model,joint_names):
    eid=e['episode_id'];path=BASE/f'seed{seed}'/f'{eid}_input.npz';z=np.load(path);identities=ds.read_json('runs/tomato/complete100_20261111/data_identity.json');row=next(r for r in identities['episodes'] if r['episode_id']==eid);features=Path(row['feature_path']);assert ds.digest(features)==row['feature_sha256'];f=np.load(features);np.testing.assert_array_equal(f['times'],z['times']);np.testing.assert_array_equal(f['anchors'],z['anchors'])
    session=f'memory-shadow-s{seed}-{eid}';identity=dict(session_id=session,episode_id=eid,model_version=model['model_version'],model_sha256=model['sha256'],manifest_sha256=ds.read_json(CONFIG)['manifest_sha256'],joint_names=joint_names);policy=DisagreementPolicy('B0');commands=[];traces=[]
    for i,(t,c,p,q) in enumerate(zip(z['times'],z['current'],z['future'],f['features'][:,-58:])):
        # q is measured position at the latest canonical frame, validated against
        # raw Parquet across all 92 train/dev recordings by the companion audit.
        t=float(t);arrival=t+.04;r=policy.update(c,p,t,now=arrival);traces.append(r)
        o=Observation(session,i,f'{session}:{i}',t,arrival,tuple(c),tuple(p),model['model_version'],positions=tuple(float(v) for v in q),state_timestamp=t,joint_names=tuple(joint_names));batch=[]
        if r['changed']:
            fact=dict(session_id=session,model_version=model['model_version'],decision_id=f'{session}:prompt{r["prompt_version"]}',from_phase=r['previous_phase'],to_phase=r['phase'],prompt_version=r['prompt_version'],observation_id=o.observation_id,observation_timestamp=t,decision_timestamp=t,arrival_timestamp=arrival,source='controller_current_confirmation');batch.append((f'fact:{r["prompt_version"]}','controller_fact',fact))
        batch.append((f'observation:{i}','observation',asdict(o)));commands.append((o,r,batch))
    return identity,commands,traces,z,{str(path):ds.digest(path),str(features):ds.digest(features)}


def abrupt_worker(database,seed,e,model,joint_names,cut):
    identity,commands,_,_,_=prepare_episode(seed,e,model,joint_names);memory=DurableTaskMemoryShadow(database,identity)
    for n,(o,r,batch) in enumerate(commands):
        for cid,kind,payload in batch:assert memory.submit(cid,kind,payload)['accepted']
        if n+1==cut:
            ds.write_json(database.with_suffix('.before_exit.json'),memory.view(now=o.arrival_timestamp))
            # Actual process exit: deliberately no close/SQLite checkpoint/finalizer.
            os._exit(77)
    raise RuntimeError('Abrupt cut was not reached')


def replay_episode(database,seed,e,model,joint_names,*,abrupt=False):
    identity,commands,traces,z,inputs=prepare_episode(seed,e,model,joint_names);cut=len(commands)//2 if abrupt else 0
    if abrupt:
        child=subprocess.run([sys.executable,'-m',__spec__.name,'--abrupt-worker','--database',str(database),'--seed',str(seed),'--episode',e['episode_id'],'--cut',str(cut)],check=False)
        assert child.returncode==77
    reference=TaskMemory(identity['session_id'],identity['model_version'],mode='M1',joint_names=joint_names);memory=DurableTaskMemoryShadow(database,identity);restores=0;parity=0;latency=[];retried=False
    for n,(o,r,batch) in enumerate(commands):
        expected=reference.observe(o,b0_phase=r['phase'])
        if n<cut:
            if n+1==cut:
                assert ds.fingerprint(memory.view(now=o.arrival_timestamp))==ds.fingerprint(ds.read_json(database.with_suffix('.before_exit.json')))
                assert ds.fingerprint(memory.memory.snapshot())==ds.fingerprint(reference.snapshot())
                ordinal=memory.ordinal;cid,kind,payload=batch[-1];assert memory.submit(cid,kind,payload)['accepted'] and memory.ordinal==ordinal;retried=True
            continue
        for cid,kind,payload in batch:
            started=time.perf_counter();result=memory.submit(cid,kind,payload);latency.append((time.perf_counter()-started)*1000);assert result['accepted'] and result['suggests_prompt_update'] is False
        v=memory.view(now=o.arrival_timestamp)
        assert v['controller_reported_phase']==r['phase'] and v['controller_prompt_version']==r['prompt_version']
        assert ds.fingerprint(v['perception'])==ds.fingerprint(reference.snapshot())
        assert v['perception_is_fresh'] and not v['suggests_prompt_update'] and v['physical_success']=='unknown';parity+=1
        if not abrupt and n+1 in [len(commands)//3,2*len(commands)//3]:
            before=ds.fingerprint(v);memory.close();memory=DurableTaskMemoryShadow(database,identity);assert ds.fingerprint(memory.view(now=o.arrival_timestamp))==before;assert not memory.view()['perception_is_fresh'];restores+=1
    final=memory.view(now=commands[-1][0].arrival_timestamp);assert final['controller_reported_phase']==6 and [f['to_phase'] for f in final['controller_facts']]==list(range(1,7));assert len(memory.memory.short)<=16 and len(memory.memory.events)<=256
    integrity=memory.conn.execute('PRAGMA integrity_check').fetchone()[0];assert integrity=='ok';memory.close()
    # Freshness is checked explicitly after reopening; history alone is not live.
    memory=DurableTaskMemoryShadow(database,identity);assert not memory.view()['perception_is_fresh'];assert ds.fingerprint(memory.view(now=commands[-1][0].arrival_timestamp))==ds.fingerprint(final);memory.close();restores+=1
    result,trace=score(e,z['times'],z['anchors'],z['current'],z['future'],'B0');assert [r['phase'] for r in traces]==[r['phase'] for r in trace]
    return dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],observations=len(commands),snapshots_compared=parity,restarts_verified=restores,abrupt_exit=abrupt,retry_after_abrupt_verified=retried,full_legacy_snapshot_parity=True,controller_prefix_preserved=True,confirmed_controller_facts=len(final['controller_facts']),journal_commands=final['journal_ordinal'],journal_sha256=final['journal_sha256'],sqlite_integrity=integrity,shadow_false_prompt_updates=0,latency_ms=dict(p50=float(np.median(latency)),p95=float(np.percentile(latency,95)),max=float(max(latency))),scoring=result),inputs


def run(out):
    if out.exists():raise FileExistsError('Preserve previous Memory journals')
    cfg=ds.read_json(CONFIG);lock=verify(Path(cfg['frozen_lock']));m=ds.read_json(lock['manifest']);assert ds.digest(lock['manifest'])==cfg['manifest_sha256'];audit=ds.read_json('runs/tomato/memory_state_audit_20261112_v1/state_mapping.json');assert len(audit['episodes'])==92;out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);results={};inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256'],'runs/tomato/memory_state_audit_20261112_v1/state_mapping.json':ds.digest('runs/tomato/memory_state_audit_20261112_v1/state_mapping.json')};started=time.perf_counter()
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();rows=[]
        for e in m['episodes']:
            if e['split']!='val':continue
            r,src=replay_episode(folder/f'{e["episode_id"]}.sqlite',seed,e,lock['models'][str(seed)],m['contract']['joint_names']);inputs.update(src);rows.append(r);print('PARITY',seed,e['episode_id'],r['observations'],flush=True)
        summary=summarize({r['episode_id']:r['scoring'] for r in rows});results[str(seed)]=dict(episodes=rows,baseline_summary=summary)
    e=next(e for e in m['episodes'] if e['split']=='val');abrupt,src=replay_episode(out/'real_abrupt.sqlite',42,e,lock['models']['42'],m['contract']['joint_names'],abrupt=True);inputs.update(src)
    ds.write_json(out/'results.json',results);ds.write_json(out/'abrupt_real_replay.json',abrupt);ds.write_json(out/'receipt.json',dict(inputs=inputs,source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_durable_memory.py'),Path('tools/audit_tomato_memory_state.py')]},git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_s=time.perf_counter()-started,real_recordings=8,seeds=cfg['seeds'],snapshots_compared=sum(r['snapshots_compared'] for v in results.values() for r in v['episodes']),restarts_verified=sum(r['restarts_verified'] for v in results.values() for r in v['episodes']),ordered_controller_facts=sum(r['confirmed_controller_facts'] for v in results.values() for r in v['episodes']),memory_training=False,roi_executed=False,default_changed=False,openpi_modified=False,historical_test_scored=False,hardware_power_loss_tested=False));verify(Path(cfg['frozen_lock']));print('DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--abrupt-worker',action='store_true');p.add_argument('--database',type=Path);p.add_argument('--seed',type=int);p.add_argument('--episode');p.add_argument('--cut',type=int);a=p.parse_args()
    if a.abrupt_worker:
        cfg=ds.read_json(CONFIG);lock=ds.read_json(cfg['frozen_lock']);m=ds.read_json(lock['manifest']);e=next(e for e in m['episodes'] if e['episode_id']==a.episode);abrupt_worker(a.database,a.seed,e,lock['models'][str(a.seed)],m['contract']['joint_names'],a.cut)
    else:
        if a.out is None:p.error('--out required')
        run(a.out)
