"""Fixed-probability paired causal policy comparison with shared seven-position cap."""
import argparse
from collections import Counter
from pathlib import Path
import heapq
import importlib.util
import json
import subprocess
import sys
import time
import hashlib
import numpy as np
from . import tomato_dataset as ds
from .tomato_state_recovery import B0State,CLASSES
from .tomato_boundary_metrics import boundaries
from .evaluate_tomato_state_recovery import prompt_score,state_score
from .tomato_full_chain import cascade_diagnostics
from .tomato_frozen_protocol import verify

CONFIG=Path('configs/tomato_paired_decisions_20261112/protocol.json')
COMMIT='07f93115d5427b3ebaaf1735e9bd18a1e1240aa5'
BASE=Path('runs/tomato/disagreement_20261112_v2')


def colleague_module():
    raw=subprocess.check_output(['git','show',COMMIT+':tools/tomato_review_ui/handshake.py']);name='_tomato_paired_colleague_'+COMMIT
    if name not in sys.modules:
        import types
        module=types.ModuleType(name);sys.modules[name]=module;exec(compile(raw,COMMIT+'/handshake.py','exec'),module.__dict__)
    return sys.modules[name],raw


def colleague_replay(c,f,times,duration,k,ack):
    module,_=colleague_module()
    class Seven(module.Handshake):
        def planning(self,t):
            if self.version>=6:return
            before=self.version;super().planning(t)
            if self.version!=before:
                assert self.version==before+1 and module.LABELS.index(self.active)==CLASSES[self.version]
        def prediction(self,m):
            if self.version<6:return super().prediction(m)
            # Terminal outer contract: allow final post-commit ACK, forbid new candidates.
            if self.state in ('WAIT_CURRENT_ACK','RECOVERY_REQUIRED'):
                super().prediction(m)
            self.pending=None
            if self.state not in ('WAIT_CURRENT_ACK','RECOVERY_REQUIRED'):self.state='TERMINAL'
    model=Seven({'K':k,'ack_enabled':ack});queue=[]
    for i,(cp,fp,t) in enumerate(zip(c,f,times)):
        m=module.Evidence(float(t),round(float(t)+.09,9),tuple(cp),tuple(fp));heapq.heappush(queue,(m.available_at,0,i,'prediction',m));heapq.heappush(queue,(round(m.available_at+.75+1e-8,9),1,i,'timer',None))
    for i,t in enumerate(np.arange(0,duration,.5)):heapq.heappush(queue,(round(float(t),9),2,i,'planning',None))
    while queue:
        t,_,i,kind,m=heapq.heappop(queue)
        if t>=duration:continue
        before=model.version
        if kind=='prediction':model.prediction(m)
        elif kind=='planning':model.planning(t)
        else:model.timer(t)
        if model.version!=before and ack:heapq.heappush(queue,(round(model.deadline,9),1,100000+model.version,'timer',None))
    commits=[e for e in model.events if e['kind']=='commit'];assert len(commits)<=6
    return [dict(time=e['time'],phase=i+1) for i,e in enumerate(commits)],dict(acks=sum(e['kind']=='ack' for e in model.events),timeouts=sum(e['kind']=='timeout' for e in model.events),final_state=model.state,events=model.events)


def b0_replay(c,f,times,duration,same_clock):
    if not same_clock:
        m=B0State();changes=[]
        for cp,fp,t in zip(c,f,times):
            r=m.update(cp,fp,float(t))
            if r['changed']:changes.append(dict(time=float(t),phase=m.phase))
        return changes,dict(acks=0,timeouts=0,final_state='TERMINAL' if m.phase==6 else 'READY',events=[])
    phase=count=0;last=None;changes=[];queue=[]
    for i,(cp,fp,t) in enumerate(zip(c,f,times)):heapq.heappush(queue,(round(float(t)+.09,9),0,i,'prediction',cp))
    for i,t in enumerate(np.arange(0,duration,.5)):heapq.heappush(queue,(round(float(t),9),2,i,'planning',None))
    while queue:
        t,_,i,kind,p=heapq.heappop(queue)
        if t>=duration or phase==6:continue
        if kind=='prediction':
            target=CLASSES[phase+1];count=count+1 if int(p.argmax())==target and p[target]>=.8 else 0;last=t
        elif count>=2 and last is not None and t-last<=.75:
            phase+=1;count=0;changes.append(dict(time=t,phase=phase))
    return changes,dict(acks=0,timeouts=0,final_state='TERMINAL' if phase==6 else 'READY',events=[])


def score(e,times,changes,meta):
    truth=boundaries(dict(episode=e));trace=[];phase=0;j=0
    for t in sorted(set([float(v) for v in times]+[v['time'] for v in changes])):
        before=phase
        if j<len(changes) and abs(changes[j]['time']-t)<1e-8:phase=changes[j]['phase'];j+=1
        trace.append(dict(time=t,phase=phase,previous_phase=before,changed=phase!=before,prompt_version=phase,resynchronized=False))
    prompt=prompt_score(trace,truth,times)
    # Integrate exact state mismatch on union of observed support and true/commit breakpoints.
    duration=min(e['length']/e['fps'],float(times[-1]+.5));points=sorted(set([0.,duration]+[b['time'] for b in truth if b['time']<duration]+[v['time'] for v in changes if v['time']<duration]));mismatch=0.;wrong_runs=[];opened=None
    for a,b in zip(points,points[1:]):
        p=sum(v['time']<=a for v in changes);g=sum(v['time']<=a for v in truth)
        if p!=g:
            mismatch+=b-a
            if opened is None:opened=a
        elif opened is not None:wrong_runs.append(dict(start_s=opened,end_s=a,duration_s=a-opened));opened=None
    if opened is not None:wrong_runs.append(dict(start_s=opened,end_s=duration,duration_s=duration-opened))
    offsets=[r['offset_s'] for r in prompt['events'] if r['offset_s'] is not None];deltas=[v['time'] for v in changes];sample_ph=np.searchsorted(deltas,times,side='right');gt=np.searchsorted([b['time'] for b in truth],times,side='right');st=state_score(sample_ph,gt,times);cas=cascade_diagnostics(st['persistent'],sample_ph,gt,truth)
    return dict(prompt=prompt,within_delay={str(d):sum(x is not None and 0<=x<=d+1e-9 for x in [r['offset_s'] for r in prompt['events']]) for d in [.5,1.,1.5,2.]},
        mismatch_exact_s=mismatch,mismatch_sampled_s=st['mismatch_s'],persistent_runs=wrong_runs,potential_cascade_boundaries=sum(len(v['potential_downstream_ordinals']) for v in cas),
        ordered_complete=len(changes)==6,all_prompts_on_time=all(r['outcome']=='on_time' for r in prompt['events']),acks=meta['acks'],timeouts=meta['timeouts'],final_state=meta['final_state'])


def summarize(per):
    ps=list(per.values());ev=[r for p in ps for r in p['prompt']['events']]
    return dict(recordings=len(ps),events=len(ev),prompt_counts=dict(Counter(r['outcome'] for r in ev)),within_delay={str(d):sum(p['within_delay'][str(d)] for p in ps) for d in [.5,1.,1.5,2.]},
        mismatch_exact_s=sum(p['mismatch_exact_s'] for p in ps),mismatch_sampled_s=sum(p['mismatch_sampled_s'] for p in ps),ordered_complete=sum(p['ordered_complete'] for p in ps),all_prompts_on_time=sum(p['all_prompts_on_time'] for p in ps),timeouts=sum(p['timeouts'] for p in ps),acks=sum(p['acks'] for p in ps),potential_cascade_boundaries=sum(p['potential_cascade_boundaries'] for p in ps),
        by_transition={typ:dict(Counter(r['outcome'] for r in ev if r['transition']==typ)) for typ in sorted({r['transition'] for r in ev})})


def run(out):
    if out.exists():raise FileExistsError('New output required')
    cfg=ds.read_json(CONFIG);lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));manifest=ds.read_json(lock['manifest']);old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);out.mkdir(parents=True);module,raw=colleague_module();(out/'colleague_handshake_executed.py').write_bytes(raw);ds.write_json(out/'protocol.json',cfg);summary={};inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256']};start=time.perf_counter();differences=[]
    for seed in (42,43,44):
        folder=out/f'seed{seed}';folder.mkdir();per={a:{} for a in cfg['arms']};cohorts={}
        for e in manifest['episodes']:
            if e['split']=='test':continue
            eid=e['episode_id'];cohorts[eid]='development8' if e['split']=='val' else 'training74_seen' if e['source_recording_id'] in seen else 'training10_unseen_exposed';p=BASE/f'seed{seed}'/f'{eid}_input.npz';inputs[str(p)]=ds.digest(p);z=np.load(p);c,f,ts=z['current'],z['future'],z['times'];assert np.array_equal(z['anchors'],e['anchors']) and np.allclose(ts,z['anchors']/e['fps']);duration=min(e['length']/e['fps'],float(ts[-1]+.5))
            for arm in cfg['arms']:
                if arm.startswith('B0'):changes,meta=b0_replay(c,f,ts,duration,arm=='B0_same_clock')
                else:changes,meta=colleague_replay(c,f,ts,duration,2 if '_k2_' in arm else 1,arm.endswith('_ack') and not arm.endswith('no_ack'))
                per[arm][eid]=score(e,ts,changes,meta);ds.write_json(folder/f'{eid}_{arm}.json',dict(changes=changes,metadata=meta,score=per[arm][eid]))
            for arm in cfg['arms'][1:]:
                for b,q in zip(per['B0_native'][eid]['prompt']['events'],per[arm][eid]['prompt']['events']):differences.append(dict(seed=seed,cohort=cohorts[eid],episode_id=eid,ordinal=b['ordinal'],transition=b['transition'],arm=arm,baseline=b,candidate=q))
        summary[str(seed)]={co:{a:summarize({eid:s for eid,s in per[a].items() if cohorts[eid]==co}) for a in cfg['arms']} for co in sorted(set(cohorts.values()))}
        if seed==42:assert summary[str(seed)]['development8']['B0_native']['prompt_counts']==dict(on_time=44,early=2,late=2)
        ds.write_json(folder/'per_episode.json',per);print('SEED',seed,summary[str(seed)]['development8'],flush=True)
    ds.write_json(out/'summary.json',summary);ds.write_json(out/'event_comparison.json',differences);ds.write_json(out/'receipt.json',dict(inputs=inputs,colleague_commit=COMMIT,colleague_source_sha256=hashlib.sha256(raw).hexdigest(),runner_sha256=ds.digest(__file__),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_s=time.perf_counter()-start,real_latency_measured=False,model_comparison=False,input_horizon_s=1.5,original_colleague_horizon_metadata_s=1.,note='Original source_target_time metadata remains t+1 but is never used by policy. Shared inputs t+1.5; no model or horizon quality comparison.',default_changed=False,historical_test_scored=False,seven_cap='Outer wrapper changes terminal behavior only; first6 commits retain pinned colleague policy'))
    verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
