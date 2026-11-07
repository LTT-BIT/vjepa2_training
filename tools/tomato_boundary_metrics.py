"""Causal thresholded change events; matching uses truth only after event generation."""
import numpy as np
from . import tomato_metrics as metrics, tomato_dataset as ds

BINS = [(-2.,-1.),(-1.,-.5),(-.5,0.),(0.,.5),(.5,1.),(1.,2.)]


def change_events(probabilities, times, threshold, confirmations):
    if not 0 < threshold <= 1 or confirmations < 1:
        raise ValueError('Invalid event operating point')
    p=np.asarray(probabilities);t=np.asarray(times)
    if p.shape!=(len(t),4) or not np.isfinite(p).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid event timeline')
    events=[];accepted=None;pending=None;count=0
    for row,ts in zip(p,t):
        label=int(row.argmax())
        if row[label] < threshold:
            pending=None;count=0;continue
        if pending==label: count+=1
        else: pending=label;count=1
        if count>=confirmations and label!=accepted:
            if accepted is not None:
                events.append(dict(time=float(ts), previous=accepted, label=label))
            accepted=label
    return events


def boundaries(entry):
    seg=metrics.continuous_phases(entry['episode']['annotation']['segments'])
    fps=entry['episode']['fps']
    return [dict(time=b['start_frame']/fps, previous=ds.LABELS.index(a['label']),
                 label=ds.LABELS.index(b['label']), end=b['end_frame']/fps)
            for a,b in zip(seg,seg[1:])]


def event_report(entries, logits, head, horizon, threshold, confirmations):
    records=[];per_episode={};offset=0;false_total=0;duration=0.
    for r in entries:
        n=len(r['times']);p=metrics.probabilities(logits[offset:offset+n]);offset+=n
        # Future tail has no observable ground truth and is excluded consistently.
        keep=np.ones(n,dtype=bool) if head=='current' else r['future_valid_mask']
        ts=r['times'][keep];events=change_events(p[keep],ts,threshold,confirmations)
        used=set();rows=[]
        for b in boundaries(r):
            # Exact from/to matching prevents unrelated class changes receiving credit.
            lo=b['time'] if head=='current' else b['time']-horizon
            hi=min(b['end'],b['time']+2.) if head=='current' else b['time']
            candidates=[(i,v) for i,v in enumerate(events) if i not in used and
                        v['previous']==b['previous'] and v['label']==b['label'] and lo<=v['time']<hi]
            i,v=candidates[0] if candidates else (None,None)
            if i is not None: used.add(i)
            rows.append({**b,'detected':v is not None,'event_time':v['time'] if v else None,
                         'offset_s':(v['time']-b['time']) if v else None})
        false=len(events)-len(used);seconds=float(ts[-1]-ts[0]+.5) if len(ts) else 0.
        per_episode[r['episode']['episode_id']]=dict(boundaries=rows,events=events,unmatched_events=false,duration_s=seconds)
        records.extend(rows);false_total+=false;duration+=seconds
    detected=[v for v in records if v['detected']];offsets=[v['offset_s'] for v in detected]
    return dict(boundary_count=len(records),detected=len(detected),
                recall=len(detected)/len(records) if records else None,
                miss_rate=1-len(detected)/len(records) if records else None,
                unmatched_events=false_total,false_events_per_minute=false_total/(duration/60) if duration else None,
                false_discovery_rate=false_total/(false_total+len(detected)) if false_total+len(detected) else 0.,
                delay_p50_s=float(np.percentile(offsets,50)) if offsets and head=='current' else None,
                delay_p95_s=float(np.percentile(offsets,95)) if offsets and head=='current' else None,
                lead_p50_s=float(np.percentile(-np.array(offsets),50)) if offsets and head=='future' else None,
                lead_p10_s=float(np.percentile(-np.array(offsets),10)) if offsets and head=='future' else None,
                threshold=threshold,confirmations=confirmations,per_episode=per_episode,
                definition='Causal accepted-label changes; first stable label initializes without event; exact from/to one-to-one match; future strictly before boundary within horizon, current within 2s and segment; unmatched events are false alarms. 2Hz resolution.')


def distance_report(entries, logits, head):
    parts=[[] for _ in range(len(BINS)+1)];ys=[[] for _ in parts];offset=0
    per_episode={}
    for r in entries:
        n=len(r['times']);z=logits[offset:offset+n];offset+=n
        valid=np.ones(n,dtype=bool) if head=='current' else r['future_valid_mask']
        y=r[head+'_labels'];bs=np.array([b['time'] for b in boundaries(r)])
        distances=r['times'][:,None]-bs[None,:] if len(bs) else np.full((n,1),np.inf)
        signed=distances[np.arange(n),np.abs(distances).argmin(axis=1)]
        occupied=np.zeros(n,dtype=bool)
        for j,(lo,hi) in enumerate(BINS):
            keep=valid&(signed>=lo)&(signed<hi);occupied|=keep
            parts[j].append(z[keep]);ys[j].append(y[keep])
        keep=valid&~occupied;parts[-1].append(z[keep]);ys[-1].append(y[keep])
        per_episode[r['episode']['episode_id']]=metrics.scores(z[valid],y[valid])
    result={}
    for name,z,y in zip([str(b) for b in BINS]+['outside_2s'],parts,ys):
        z=np.concatenate(z);y=np.concatenate(y)
        result[name]=metrics.scores(z,y) if len(y) else None
    return dict(by_signed_nearest_current_boundary_s=result,per_episode=per_episode,
                note='Fixed four-class Macro-F1 assigns zero to absent classes; always inspect support/confusion. Future labels evaluated at t+h; distance refers to t.')
