"""Validated whitelist inputs and deterministic event replay. No torch dependency."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import heapq
import json
import math
from .tomato_prompt_policy import LABELS, PolicyInput, PromptPolicy


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,o):
    Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(p,rr):
    with Path(p).open('w') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def whitelist(r, model, latency=0., reactive=False):
    head='current' if reactive else 'future'
    p=r[head+'_probabilities']
    q=r.get('current_probabilities') if model=='joint' else None
    if set(p)!=set(LABELS) or (q is not None and set(q)!=set(LABELS)):
        raise ValueError('Expected exactly four named probabilities')
    if model=='joint' and q is None:raise ValueError('Missing actual Joint Current head')
    return PolicyInput(r['episode_id'],r['source_recording_id'],r['frame_index'],r['timestamp_s'],
        round(r['timestamp_s']+latency,9),model,tuple(p[k] for k in LABELS),
        tuple(q[k] for k in LABELS) if q else None,0. if reactive else 1.)


def preflight(root):
    manifests={};streams={};hashes={};reference=None
    for model in ['future_only','joint','current_only']:
        d=Path(root)/model;m=read(d/'manifest.json');split=read(d/'split.json')
        if sha(d/'split.json')!=m['split_sha256'] or split!=m['splits']:raise ValueError('Split hash mismatch')
        cfg=read(d/'training_config.json')
        if cfg['labels']!=list(LABELS) or cfg['future_horizon_s']!=1. or cfg['prediction_hz']!=2:raise ValueError('Task contract mismatch')
        signature=(m['contract'],m['splits'],[(e['episode_id'],e['source_recording_id'],e['split'],e['length'],e['fps'],e['anchors'],e['annotation']) for e in m['episodes']])
        if reference is None:reference=signature
        elif signature!=reference:raise ValueError('Model manifests not paired')
        if len(m['episodes'])!=93 or {s:len(v) for s,v in split.items()}!={'train':75,'val':9,'test':9}:raise ValueError('Unexpected source membership')
        if len(set().union(*(set(x) for x in split.values())))!=93:raise ValueError('Source split overlap')
        for n in ['manifest.json','split.json','predictions.jsonl','training_config.json','best_head.pt']:
            hashes[str(d/n)]=sha(d/n)
        es={e['episode_id']:e for e in m['episodes']};ss={k:[] for k in es};seen=set();last={}
        for line in (d/'predictions.jsonl').open():
            r=json.loads(line);eid=r['episode_id'];e=es[eid];frame=r['frame_index'];t=r['timestamp_s']
            if (r['source_recording_id']!=e['source_recording_id'] or r['split']!=e['split']
                or not math.isfinite(t) or abs(t-frame/e['fps'])>1e-6
                or r['last_input_frame']>frame or len(r['input_indices'])!=16
                or any(i>frame or i<0 for i in r['input_indices'])):raise ValueError('Noncausal or inconsistent prediction')
            key=(e['source_recording_id'],eid,frame,round(t,6))
            if key in seen or frame<=last.get(eid,-1):raise ValueError('Duplicate/reversed anchor')
            seen.add(key);last[eid]=frame
            ss[eid].append(whitelist(r,model,reactive=model=='current_only'))
        for eid,rr in ss.items():
            if [r.anchor_frame for r in rr]!=es[eid]['anchors']:raise ValueError('Missing or extra anchor')
        manifests[model]=m;streams[model]=ss
    for eid in streams['joint']:
        if any([(r.source_recording_id,r.anchor_frame,r.observed_at) for r in streams[m][eid]]!=
               [(r.source_recording_id,r.anchor_frame,r.observed_at) for r in streams['joint'][eid]] for m in streams):raise ValueError('Unpaired anchor keys')
    # Scoring snapshots remain separate from DTOs. No annotation or video is edited.
    sequences={}
    for e in manifests['joint']['episodes']:
        segments=e['annotation']['segments'];seq=[s['label'] for i,s in enumerate(segments) if i==0 or s['label']!=segments[i-1]['label']]
        sequences[e['episode_id']]=dict(source_recording_id=e['source_recording_id'],split=e['split'],sequence=seq)
        if sha(e['annotation']['path'])!=e['annotation']['sha256']:raise ValueError('Annotation changed')
    counts={}
    for e in sequences.values():counts[str(len(e['sequence']))]=counts.get(str(len(e['sequence'])),0)+1
    if counts!={'7':92,'3':1}:raise ValueError(f'Stage protocol anomaly: {counts}')
    return manifests,streams,dict(success_sources=93,failed_sources=0,anchors_per_model=sum(len(v) for v in streams['joint'].values()),
        input_sha256=hashes,alignment_errors=[],failure_logs=[],sequence_counts=counts,sequences=sequences)


def replay(messages, config, duration, latency=0., period=.5, offset=0.):
    if latency<0 or period<=0 or not 0<=offset<period or not math.isfinite(duration) or duration<=0:raise ValueError('Invalid clock')
    if latency>=.5:raise ValueError('Main experiment serial latency must be below 2Hz input period; queue overload not implemented')
    policy=PromptPolicy(config);events=[];queue=[];serial=0
    def push(t,priority,kind,msg=None):
        nonlocal serial
        t=round(t,9)
        if 0<=t<duration:
            heapq.heappush(queue,(t,priority,serial,kind,msg));serial+=1
    previous=-1.;seen=set()
    for msg in messages:
        if msg.observed_at<=previous or msg.anchor_frame in seen:raise ValueError('Duplicate/reversed stream')
        previous=msg.observed_at;seen.add(msg.anchor_frame)
        m=PolicyInput(**{**asdict(msg),'available_at':round(msg.observed_at+latency,9)})
        push(m.available_at,0,'prediction',m)
        push(m.available_at+config.get('prediction_ttl_s',.75)+1e-8,1,'timer')
    n=0
    while offset+n*period<duration:
        push(offset+n*period,2,'planning');n+=1
    intervals=[];start=0.;active=dict(policy.active)
    while queue:
        now,_,_,kind,msg=heapq.heappop(queue)
        if kind=='prediction': rr=policy.on_prediction(msg)
        elif kind=='planning':rr=policy.on_planning_boundary(now)
        else:
            rr=policy.on_timer(now)
            if kind=='release':rr.append(policy._event('release',now,'deferred_timer'))
        if kind=='prediction' and policy.pending:
            push(policy.pending['release_at'],1,'release') if policy.pending['release_at']>now else None
        for event in rr:
            if event['kind']=='commit':
                intervals.append(dict(start_s=start,end_s=now,**active))
                start=now;active=dict(event['active'])
        events.extend(rr)
        # Planning events logged even if HOLD; no confirmation is added.
        if kind=='planning':events.append(policy._event('planning',now,'clock_boundary'))
    events.extend(policy.finalize(duration))
    intervals.append(dict(start_s=start,end_s=duration,**active))
    return dict(events=events,intervals=[r for r in intervals if r['end_s']>r['start_s']],
        initial_prompt=config.get('initial_prompt','pickup'),
        input_anchors=len(messages),consumed_anchors=sum(e['kind']=='prediction' for e in events),
        commits=[e for e in events if e['kind']=='commit'])
