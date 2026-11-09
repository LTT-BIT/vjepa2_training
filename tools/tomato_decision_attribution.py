"""Fixed offline decision ablations; truth enters scoring/oracle diagnostics only."""
import collections
import importlib
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .audit_tomato_transitions import match_events,manifest,PRIOR

ROOT=Path('runs/tomato/boundary_review_20261109')
SOURCE=Path('runs/tomato/transition_audit_20261109')
SEQ=('pickup','harvest','return','adjust','pickup','harvest','return')
POLICIES={
 'current_argmax':(0.,1,False,False),
 'current_conf05':(.5,1,False,False),
 'current_conf05_k2':(.5,2,False,False),
 'current_conf08':(.8,1,False,False),
 'current_conf08_k2':(.8,2,False,False),
 'ordered_current08_k1':(.8,1,True,False),
 'ordered_current08_k2':(.8,2,True,False),
 'ordered_current_future08_k2':(.8,2,True,True),
}


def decisions(current,future,times,threshold,confirmations,ordered=False,joint=False):
    """No labels/episode boundaries as inputs; fresh monotonic probability stream."""
    c=np.asarray(current);f=np.asarray(future);t=np.asarray(times)
    if c.shape!=(len(t),4) or f.shape!=c.shape or not len(t) or np.any(np.diff(t)<=0):raise ValueError('Invalid stream')
    if not np.isfinite(c).all() or not np.isfinite(f).all() or np.any(c<0) or np.any(f<0) or not np.allclose(c.sum(1),1) or not np.allclose(f.sum(1),1):raise ValueError('Invalid probabilities')
    if confirmations<1 or not 0<=threshold<=1:raise ValueError('Invalid operating point')
    accepted=0 if ordered else None;phase=0;pending=None;count=0;events=[];trace=[]
    for i,(row,ts) in enumerate(zip(c,t)):
        label=int(row.argmax());expected=ds.LABELS.index(SEQ[phase+1]) if ordered and phase+1<len(SEQ) else None
        valid=row[label]>=threshold and (not ordered or label==expected)
        if joint:valid=valid and int(f[i].argmax())==label and f[i,label]>=threshold
        if valid:
            if pending==label:count+=1
            else:pending=label;count=1
        else:pending=None;count=0
        changed=False
        if valid and count>=confirmations and label!=accepted:
            if accepted is not None:events.append(dict(time=float(ts),previous=int(accepted),label=label))
            accepted=label;changed=True
            if ordered:phase+=1;pending=None;count=0
        trace.append(dict(time=float(ts),accepted=accepted,phase=phase if ordered else None,count=count,changed=changed))
    return events,trace


def historical_bridge():
    lock=ds.read_json(ROOT/'protocol.json');base=ROOT/'historical_openpi/src';package=base/'openpi'
    allowed={'subtasks/agent_bridge.py','subtasks/tomato.py','policies/fr3_sharpa_protocol.py'}
    if set(lock['historical_openpi']['files'])!=allowed:raise ValueError('Unexpected historical source paths')
    for name,sha in lock['historical_openpi']['files'].items():
        path=package/name
        if not path.exists():
            raw=subprocess.check_output(['git','-C','/home/user/tomato-openpi','show',lock['historical_openpi']['commit']+':src/openpi/'+name])
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Historical Git blob mismatch')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    for folder in [package,package/'subtasks',package/'policies']:
        init=folder/'__init__.py'
        if not init.exists():init.write_text('')
    for name,sha in lock['historical_openpi']['files'].items():
        if ds.digest(package/name)!=sha:raise ValueError('Historical source hash mismatch')
    existing=sys.modules.get('openpi.subtasks.agent_bridge')
    if existing and Path(existing.__file__).resolve()!= (package/'subtasks/agent_bridge.py').resolve():raise RuntimeError('Unexpected already imported OpenPI version')
    sys.path.insert(0,str(base.resolve()))
    return importlib.import_module('openpi.subtasks.agent_bridge')


def interface_trace(e,anchors,z,version):
    bridge=historical_bridge();normal=bridge.PredictionPromptProvider();counter=bridge.PredictionPromptProvider()
    for provider in [normal,counter]:provider.reset(e['episode_id'])
    probs={h:metrics.probabilities(z[h+'_logits']) for h in ['current','future']};trace=[];old=None
    for i,a in enumerate(anchors):
        t=float(a/e['fps']);base=1e6+t;before=dict(phase=normal.phase,count=normal.count,version=normal.version)
        r=dict(protocol='tomato-stage-prediction-v1',labels=list(ds.LABELS),valid=True,error=None,observation_id=str(i),session_id=e['episode_id'],model_version=version,prediction_mode='joint',future_horizon_s=1.5,input_timestamp=base,result_timestamp=base+.04,
            **{h+'_logits':z[h+'_logits'][i].astype(float).tolist() for h in probs},**{h+'_probabilities':probs[h][i].tolist() for h in probs})
        context=normal.accept_prediction(r,observation_id=str(i),now=base+.05)
        other=counter.accept_prediction(dict(r,future_logits=[0.]*4,future_probabilities=[.25]*4),observation_id=str(i),now=base+.05)
        assert (context.phase,context.version,context.prompt)==(other.phase,other.version,other.prompt)
        binding=bridge.bind_chunk(context,str(i));assert bridge.chunk_matches_prompt(binding,context)
        old_valid=bridge.chunk_matches_prompt(old,context) if old is not None else None
        if context.changed and old is not None:assert not old_valid
        expected=SEQ[before['phase']+1] if before['phase']+1<len(SEQ) else None
        qualifies=expected is not None and ds.LABELS[int(probs['current'][i].argmax())]==expected and float(probs['current'][i].max())>=.8
        trace.append(dict(time=t,anchor_frame=int(a),human_label=ds.LABELS[ds.label_at_frame(e['annotation']['segments'],int(a))],current_probabilities=probs['current'][i].tolist(),future_probabilities=probs['future'][i].tolist(),before=before,
            expected_next=expected,qualifies=qualifies,count_on_arrival=before['count']+1 if qualifies else 0,
            after=dict(phase=normal.phase,count=normal.count,version=normal.version),context=context.__dict__,chunk=binding,previous_chunk_still_matches=old_valid,
            counterfactual_future_prompt_unchanged=True))
        old=binding
    return trace


def run():
    lock=ds.read_json(ROOT/'protocol.json')
    for p,h in lock['read_only_sources'].items():assert ds.digest(p)==h
    m,episodes=manifest();saved=np.load(PRIOR/'evaluations/independent_h15_s42_val_logits.npz');offset=0;per={name:{} for name in POLICIES};future_per={};oracles={};event_rows=[];interfaces={}
    version=ds.read_json('configs/tomato_transition_20261109/recommended_inference.json')['model_version']
    for e in episodes:
        a=np.array(e['anchors']);ts=a/e['fps'];n=len(a);z={h+'_logits':saved[h+'_logits'][offset:offset+n] for h in ['current','future']};offset+=n
        c=metrics.probabilities(z['current_logits']);f=metrics.probabilities(z['future_logits']);truth=bm.boundaries(dict(episode=e));valid=a+45<e['length'];interfaces[e['episode_id']]=interface_trace(e,a,z,version)
        for name,args in POLICIES.items():
            ev,trace=decisions(c,f,ts,*args);rows,false=match_events(ev,truth,'current',1.5)
            per[name][e['episode_id']]=dict(boundaries=rows,events=ev,false_events=false,trace=trace,duration_s=float(ts[-1]-ts[0]+.5))
        # Same exact event matcher on Future, so confirmation timing can be isolated.
        for name,threshold,k in [('future_argmax',0.,1),('future_conf05',.5,1),('future_conf05_k2',.5,2),('future_conf08',.8,1),('future_conf08_k2',.8,2)]:
            ev,_=decisions(f[valid],f[valid],ts[valid],threshold,k)
            rows,false=match_events(ev,truth,'future',1.5)
            future_per.setdefault(name,{})[e['episode_id']]=dict(boundaries=rows,events=ev,false_events=false,duration_s=float(ts[valid][-1]-ts[valid][0]+.5))
        # GT used here only to quantify ideal-label sampling/confirmation ceiling.
        for head in ['current','future']:
            keep=np.ones(n,bool) if head=='current' else valid
            y=np.array([ds.label_at_frame(e['annotation']['segments'],int(v)+(45 if head=='future' else 0)) for v in a[keep]])
            p=np.eye(4)[y];ev,_=decisions(p,p,ts[keep],.8,2,head=='current')
            rows,false=match_events(ev,truth,head,1.5)
            oracles.setdefault(head,{})[e['episode_id']]=dict(boundaries=rows,events=ev,false_events=false,duration_s=float(ts[keep][-1]-ts[keep][0]+.5))
        for j,b in enumerate(truth):
            target=b['label'];win=(ts>=b['time']-1.5)&(ts<b['time'])&valid;hit=win&(f.argmax(1)==target)&(f[:,target]>=.5);stable=hit&np.r_[False,hit[:-1]]
            target_times=ts[hit];maxp=float(f[win,target].max()) if win.any() else None
            f1=future_per['future_conf05'][e['episode_id']]['boundaries'][j];f2=future_per['future_conf05_k2'][e['episode_id']]['boundaries'][j]
            # Enumerate nearby actual transitions rather than force primary cause by priority.
            raw=future_per['future_conf05'][e['episode_id']]['events'];near=[v for v in raw if v['previous']==b['previous'] and v['label']==b['label'] and abs(v['time']-b['time'])<=5]
            allr={name:rr[e['episode_id']]['boundaries'][j] for name,rr in per.items()}
            before=(ts>=b['time']-1.5)&(ts<b['time']);after=(ts>=b['time'])&(ts<min(b['end'],b['time']+2))
            event_rows.append(dict(event_id=e['episode_id']+f'_b{j+1}',episode_id=e['episode_id'],ordinal=j+1,from_stage=ds.LABELS[b['previous']],to_stage=ds.LABELS[target],boundary_s=b['time'],
                future_strict_k1=f1,future_strict_k2=f2,window_target_argmax_count=int(hit.sum()),window_stable_target_present=bool(stable.any()),window_max_target_probability=maxp,
                future_nearby_k1_same_path=near,current_new_class_before_boundary=int(np.sum(before&(c.argmax(1)==target))),current_wrong_after_boundary=int(np.sum(after&(c.argmax(1)!=target))),
                current_policy_results=allr,confirmation_lost_future_event=bool(f1['detected'] and not f2['detected']),
                stage_order_changed_outcome=allr['current_conf08_k2']['detected']!=allr['ordered_current08_k2']['detected'],
                joint_changed_outcome=allr['ordered_current08_k2']['detected']!=allr['ordered_current_future08_k2']['detected']))
        # Our fixed ordered comparator must reproduce the frozen real interface.
        expected=per['ordered_current08_k2'][e['episode_id']]['events'];actual=[dict(time=v['time'],previous=ds.LABELS.index(SEQ[v['before']['phase']]),label=ds.LABELS.index(SEQ[v['after']['phase']])) for v in interfaces[e['episode_id']] if v['context']['changed']]
        assert expected==actual
    def aggregate(methods,head):
        result={}
        for name,eps in methods.items():
            rows=[b for e in eps.values() for b in e['boundaries']];matched=[b for b in rows if b['detected']];false=sum(len(v['false_events']) for v in eps.values());duration=sum(v['duration_s'] for v in eps.values())
            offsets=[b['offset_s'] for b in matched];bytype=collections.defaultdict(lambda:dict(total=0,hits=0,false=0))
            for b in rows:
                typ=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']];bytype[typ]['total']+=1;bytype[typ]['hits']+=int(b['detected'])
            for e in eps.values():
                for ev in e['false_events']:bytype[ds.LABELS[ev['previous']]+'->'+ds.LABELS[ev['label']]]['false']+=1
            result[name]=dict(head=head,total=len(rows),hits=len(matched),recall=len(matched)/len(rows),false_events=false,false_per_minute=false/(duration/60),
                conditional_offset_p50_s=float(np.median(offsets)) if offsets else None,conditional_offset_p95_s=float(np.percentile(offsets,95)) if offsets else None,by_transition=dict(bytype))
        return result
    ds.write_json(ROOT/'decisions.json',dict(current=per,future=future_per,oracles=oracles))
    ds.write_json(ROOT/'decision_summary.json',dict(current=aggregate(per,'current'),future=aggregate(future_per,'future'),oracles={h:aggregate({h:v},h)[h] for h,v in oracles.items()},
        classifications={h:metrics.scores(saved[h+'_logits'][np.concatenate([np.ones(len(e['anchors']),bool) if h=='current' else np.array(e['anchors'])+45<e['length'] for e in episodes])],np.concatenate([np.array([ds.label_at_frame(e['annotation']['segments'],int(i)+(45 if h=='future' else 0)) for i in e['anchors'] if h=='current' or i+45<e['length']]) for e in episodes])) for h in ['current','future']},
        scope='Policies fixed before analysis, val only, no model selection/default changes. Oracle truth is diagnostic ONLY, never an online input.'))
    ds.write_json(ROOT/'event_attribution.json',dict(events=event_rows))
    ds.write_json(ROOT/'interface_traces.json',dict(episodes=interfaces,source=lock['historical_openpi'],all_ordered_replays_equal=True))
    focus=[r for r in event_rows if r['event_id']=='3e645bf4e137d0f836c9d7ff_b4'][0];b=focus['boundary_s'];trace=[r for r in interfaces[focus['episode_id']] if b-5<=r['time']<=b+3]
    ds.write_json(ROOT/'early_prompt_trace.json',dict(event=focus,annotated_boundary_s=b,rows=trace,full_episode_history_replayed=True,truth_not_used_by_provider=True,defect_status='No implementation contract violation found; early Current prediction satisfies documented .8/K2 rule. Human boundary semantics still unresolved.'))
    print('CURRENT', {k:(v['hits'],v['false_events']) for k,v in aggregate(per,'current').items()});print('FUTURE',{k:(v['hits'],v['false_events']) for k,v in aggregate(future_per,'future').items()});print('ORACLES',{h:aggregate({h:v},h)[h]['hits'] for h,v in oracles.items()})


if __name__=='__main__':run()
