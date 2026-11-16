"""Independent scorers: labels are only visible here, never to memory/handshake."""
from functools import lru_cache
import numpy as np
from tools.tomato_metrics import scores,probabilities,anchor_widths
from .contracts import LABELS,write


def classification(p,y):
    if not len(y):return None
    p=np.asarray(p,dtype=float);sc=scores(np.log(p.clip(1e-30)),y)
    hot=np.eye(4)[y];sc['brier']=float(((p-hot)**2).sum(1).mean());conf=p.max(1);ok=p.argmax(1)==y;ece=0
    for lo in np.arange(0,1,.1):
        keep=(conf>=lo)&((conf<lo+.1) if lo<.9-1e-9 else (conf<=1))
        if keep.any():ece+=keep.mean()*abs(ok[keep].mean()-conf[keep].mean())
    sc['ece_10_equal_width']=float(ece);sc['calibration_bins']=[dict(n=int(((conf>=lo)&(conf<lo+.1 if lo<.9-1e-9 else conf<=1)).sum()),confidence_sum=float(conf[(conf>=lo)&(conf<lo+.1 if lo<.9-1e-9 else conf<=1)].sum()),correct_sum=int(ok[(conf>=lo)&(conf<lo+.1 if lo<.9-1e-9 else conf<=1)].sum())) for lo in np.arange(0,1,.1)];return sc


def boundaries(e):
    ss=e['annotation']['segments'];return [dict(index=i,time=b['start_frame']/e['fps'],from_stage=a['label'],to_stage=b['label'],short_to=(b['end_frame']-b['start_frame'])/e['fps']<=1.5,to_occurrence=sum(s['label']==b['label'] for s in ss[:i+2])) for i,(a,b) in enumerate(zip(ss,ss[1:])) if a['label']!=b['label']]


def warning_events(times,pc,pf,threshold=0.,k=1,gap=.75):
    """No labels: alert is onset of sustained (Current top1, Future top1) disagreement."""
    events=[];pair=None;count=0;emitted=False;start=None
    for i,t in enumerate(times):
        current,future=int(pc[i].argmax()),int(pf[i].argmax());p=(current,future) if current!=future and pf[i,future]>=threshold else None
        if i and t-times[i-1]>gap+1e-9:p0=None
        else:p0=pair
        if p!=p0:
            pair=p;count=0;emitted=False;start=float(t)
        if p is None:pair=None;count=0;emitted=False;continue
        count+=1
        if count>=k and not emitted:
            events.append(dict(anchor_index=i,time=float(t),start_time=start,from_stage=LABELS[p[0]],to_stage=LABELS[p[1]],confidence=float(pf[i,p[1]]),end_s=float(times[i+1]) if i+1<len(times) else float(t+.5)));emitted=True
        elif emitted:events[-1]['end_s']=float(times[i+1]) if i+1<len(times) else float(t+.5)
    return events


def match_events(gt,events,lo,hi):
    """Ordered maximum-cardinality pair match, then minimum absolute timing error."""
    @lru_cache(None)
    def dp(i,j):
        if i==len(gt) or j==len(events):return ()
        opts=[dp(i+1,j),dp(i,j+1)];d=events[j]['time']-gt[i]['time']
        if gt[i]['from_stage']==events[j]['from_stage'] and gt[i]['to_stage']==events[j]['to_stage'] and lo-1e-9<=d<=hi+1e-9:
            opts.append(((i,j),)+dp(i+1,j+1))
        return min(opts,key=lambda z:(-len(z),sum(abs(events[b]['time']-gt[a]['time']) for a,b in z),z))
    return dp(0,0)


def anticipation(entry,pc,pf,threshold=0.,k=1):
    gt=boundaries(entry['episode']);events=warning_events(entry['times'],pc,pf,threshold,k)
    # End is strictly excluded: do not allow a boundary-time alert as an advance success.
    pairs=match_events(gt,events,-1.,-1e-8);matched=dict(pairs);used={b for a,b in pairs};leads=[gt[a]['time']-events[b]['time'] for a,b in pairs]
    detail=[]
    for i,g in enumerate(gt):
        j=matched.get(i);ambiguous=any(0<abs(g['time']-h['time'])<1 for h in gt)
        detail.append(dict(**g,detected=j is not None,warning_time=events[j]['time'] if j is not None else None,lead_s=g['time']-events[j]['time'] if j is not None else None,ambiguous=ambiguous))
    duration=entry['episode']['length']/entry['episode']['fps'];false=[dict(**e,false_duration_s=max(0,min(e['end_s'],duration)-e['time'])) for i,e in enumerate(events) if i not in used]
    tp=len(pairs);fp=len(events)-tp;fn=len(gt)-tp
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,fp_per_min=fp/(duration/60),duration_s=duration,lead_p50_s=float(np.percentile(leads,50)) if leads else None,lead_p95_s=float(np.percentile(leads,95)) if leads else None,leads=leads,false_alarm_duration_s=sum(e['false_duration_s'] for e in false),events=detail,alerts=events,false_alarms=false,definition='Causal disagreement burst onset; from=predicted Current, to=predicted Future; stable warning emitted at kth support, no backdating; strict [-1,0), one-to-one. False duration holds to next anchor/last anchor+0.5 clipped to end.')


def score_episode(entry,pc,pf,n):
    e=entry['episode'];times=entry['times'];bs=np.array([b['time'] for b in boundaries(e)]);heads={}
    for h,p in [('current',pc),('future',pf)]:
        y=entry[h+'_labels'];valid=np.ones(len(y),bool) if h=='current' else entry['future_valid_mask'];target_time=times if h=='current' else entry['future_target_times_s']
        dist=np.min(abs(target_time[:,None]-bs[None,:]),axis=1) if len(bs) else np.full(len(y),np.inf)
        buckets={}
        for w in [.5,1.,2.]:
            keep=valid&(dist<=w+1e-9);buckets[f'boundary_pm_{w:g}']=classification(p[keep],y[keep])
        stable=valid&(dist>2);buckets['stable_gt2']=classification(p[stable],y[stable]);cold=np.arange(len(y))<n-1
        buckets['cold_start']=classification(p[valid&cold],y[valid&cold]);buckets['warm']=classification(p[valid&~cold],y[valid&~cold])
        segs=e['annotation']['segments'];frames=entry['anchors'] if h=='current' else entry['future_target_frames'];short=np.zeros(len(y),bool)
        for s in segs:
            if (s['end_frame']-s['start_frame'])/e['fps']<=1.5:short|=(frames>=s['start_frame'])&(frames<s['end_frame'])
        buckets['short_stage_le1.5']=classification(p[valid&short],y[valid&short]);buckets['long_stage']=classification(p[valid&~short],y[valid&~short])
        for occurrence in [1,2]:
            mask=np.zeros(len(y),bool);seen={}
            for s in segs:
                seen[s['label']]=seen.get(s['label'],0)+1
                if seen[s['label']]==occurrence:mask|=(frames>=s['start_frame'])&(frames<s['end_frame'])
            buckets[f'occurrence_{occurrence}']=classification(p[valid&mask],y[valid&mask])
        heads[h]=dict(overall=classification(p[valid],y[valid]),buckets=buckets)
    stable_raw=np.min(abs(times[:,None]-bs[None,:]),axis=1)>2 if len(bs) else np.ones(len(times),bool)
    flips=(pc[1:].argmax(1)!=pc[:-1].argmax(1))&stable_raw[1:]&stable_raw[:-1]
    return dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],heads=heads,stable_current_flips=int(flips.sum()))


def aggregate_classification(reports):
    reports=[r for r in reports if r is not None]
    if not reports:return None
    cm=np.sum([r['confusion_true_rows'] for r in reports],axis=0);n=int(cm.sum());tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
    return dict(samples=n,macro_f1=float(np.divide(2*tp,den,out=np.zeros(4),where=den>0).mean()),accuracy=float(tp.sum()/n),confusion_true_rows=cm.tolist(),nll=sum(r['nll']*r['samples'] for r in reports)/n,brier=sum(r['brier']*r['samples'] for r in reports)/n,ece_10_equal_width=sum(abs(sum(r['calibration_bins'][i]['correct_sum'] for r in reports)-sum(r['calibration_bins'][i]['confidence_sum'] for r in reports)) for i in range(10))/n)


def score_split(entries,pred,c,n):
    parts={h:[] for h in ['current','future']};ys={h:[] for h in parts};episodes=[];raw=[];stable=[];persistence=[];oracle=[]
    for e in entries:
        pp={h:probabilities(pred[e['episode']['episode_id']][h]) for h in parts};pc,pf=pp['current'],pp['future']
        episodes.append(score_episode(e,pc,pf,n));raw.append(anticipation(e,pc,pf));stable.append(anticipation(e,pc,pf,c['warning_threshold'],c['warning_k']))
        for h in parts:
            valid=np.ones(len(pc),bool) if h=='current' else e['future_valid_mask'];parts[h].append(pp[h][valid]);ys[h].append(e[h+'_labels'][valid])
        valid=e['future_valid_mask'];persistence.append(pc[valid]);oracle.append(np.eye(4)[e['current_labels'][valid]])
    def aggregate_events(rows):
        tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows);dur=sum(r['duration_s'] for r in rows);leads=[x for r in rows for x in r['leads']]
        types={}
        for r in rows:
            for ev in r['events']:
                key=ev['from_stage']+'->'+ev['to_stage'];v=types.setdefault(key,dict(tp=0,fn=0,short=0));v['tp']+=ev['detected'];v['fn']+=not ev['detected'];v['short']+=ev['short_to']
        return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,fp_per_min=fp/(dur/60),lead_p50_s=float(np.percentile(leads,50)) if leads else None,lead_p95_s=float(np.percentile(leads,95)) if leads else None,false_alarm_duration_s=sum(r['false_alarm_duration_s'] for r in rows),by_transition=types,episodes=rows)
    bucket_pooled={h:{key:aggregate_classification([ep['heads'][h]['buckets'][key] for ep in episodes]) for key in episodes[0]['heads'][h]['buckets']} for h in parts}
    return dict(bucket_pooled=bucket_pooled,heads={h:classification(np.concatenate(parts[h]),np.concatenate(ys[h])) for h in parts},episodes=episodes,future_events=dict(raw=aggregate_events(raw),stable=aggregate_events(stable)),future_persistence=dict(deployable_current=classification(np.concatenate(persistence),np.concatenate(ys['future'])),oracle_current_diagnostic_only=classification(np.concatenate(oracle),np.concatenate(ys['future']))))
